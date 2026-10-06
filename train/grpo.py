"""TRL GRPO ablation trainer: fixed 64 rollouts/step, vary --G."""

import argparse
import json
import random
import sys
from pathlib import Path

from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envs.base import Example
from envs.countdown import CountdownEnv

ROLLOUTS_PER_STEP = 64
MICRO_BATCH_COMPLETIONS = 8
GRAD_ACCUM_STEPS = 8
DEFAULT_MAX_STEPS = 400
DEFAULT_SEED = 42
DEFAULT_LR = 2e-5
BETA = 0.001
TEMPERATURE = 1.0
MODEL_ID = "arcee-ai/AFM-4.5B-Base"
NUM_TRAIN_PROMPTS = 2048


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TRL GRPO on countdown (AFM + LoRA)")
    parser.add_argument("--G", type=int, required=True, choices=(4, 8, 16))
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--init-adapter",
        type=str,
        default=None,
        help="Path to an SFT LoRA adapter to continue from (SFT→GRPO arm).",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--num-train-prompts", type=int, default=NUM_TRAIN_PROMPTS)
    return parser.parse_args()


def build_countdown_dataset(n: int, seed: int) -> Dataset:
    env = CountdownEnv()
    rows = []
    for idx in range(n):
        rng = random.Random(f"{seed}:countdown:train:{idx}")
        ex = env.make_example(idx, rng, split="train")
        rows.append(
            {
                "prompt": ex.prompt,
                "answer": ex.answer,
                "numbers": ex.meta["numbers"],
                "target": ex.meta["target"],
                "id": ex.id,
            }
        )
    return Dataset.from_list(rows)


def make_countdown_reward():
    env = CountdownEnv()

    def reward_countdown(completions, numbers, target, **kwargs):
        rewards = []
        for completion, nums, tgt in zip(completions, numbers, target):
            if isinstance(nums, str):
                nums = json.loads(nums)
            example = Example(
                id="reward",
                prompt="",
                answer=str(tgt),
                meta={"numbers": list(nums), "target": int(tgt)},
            )
            if isinstance(completion, list):
                text = completion[-1].get("content", "") if completion else ""
            else:
                text = completion or ""
            rewards.append(float(env.score(example, text)["reward"]))
        return rewards

    return reward_countdown


def load_model(init_adapter: str | None):
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype="bfloat16",
        attn_implementation="sdpa",
    )
    if init_adapter:
        model = PeftModel.from_pretrained(model, init_adapter, is_trainable=True)
    else:
        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=16,
            lora_alpha=32,
            target_modules="all-linear",
        )
        model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def main() -> None:
    args = parse_args()
    if ROLLOUTS_PER_STEP % args.G != 0:
        raise ValueError(
            f"ROLLOUTS_PER_STEP ({ROLLOUTS_PER_STEP}) must be divisible by G ({args.G})"
        )
    if MICRO_BATCH_COMPLETIONS * GRAD_ACCUM_STEPS != ROLLOUTS_PER_STEP:
        raise ValueError("Microbatch × grad accum must equal ROLLOUTS_PER_STEP")

    prompts_per_step = ROLLOUTS_PER_STEP // args.G
    lr_tag = f"{args.lr:.0e}".replace("e-0", "e-").replace("e+0", "e+")
    run_name = f"grpo-G{args.G}-lr{lr_tag}-s{args.seed}"
    output_dir = args.output_dir or f"GRPO/{run_name}"
    print(
        f"{run_name} | rollouts/step={ROLLOUTS_PER_STEP} | "
        f"prompts/step={prompts_per_step} | max_steps={args.max_steps}"
    )

    dataset = build_countdown_dataset(args.num_train_prompts, args.seed)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_model(args.init_adapter)

    # Fixed step geometry across G; do not set generation_batch_size.
    training_args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=MICRO_BATCH_COMPLETIONS,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        num_generations=args.G,
        max_prompt_length=512,
        max_completion_length=1024,
        optim="adamw_8bit",
        max_steps=args.max_steps,
        bf16=True,
        beta=BETA,
        temperature=TEMPERATURE,
        seed=args.seed,
        data_seed=args.seed,
        gradient_checkpointing=True,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=None,
        logging_steps=1,
        log_completions=True,
        report_to=["wandb"],
        remove_unused_columns=False,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.3,
        run_name=run_name,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[make_countdown_reward()],
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(output_dir)


if __name__ == "__main__":
    main()
