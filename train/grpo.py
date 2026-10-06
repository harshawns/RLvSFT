"""TRL GRPO ablation trainer: fixed 64 rollouts/step, vary --G."""

import argparse
import json
import sys
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset
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
REQUIRED_COLUMNS = ("prompt", "numbers", "target")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TRL GRPO on countdown (AFM + LoRA)")
    parser.add_argument("--G", type=int, required=True, choices=(4, 8, 16))
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help=(
            "Train data: local .parquet/.json/.jsonl path, a directory, "
            "or a Hugging Face dataset id. Needs prompt, numbers, target "
            "(or prompt + meta with those fields)."
        ),
    )
    parser.add_argument("--dataset-split", type=str, default="train")
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
    return parser.parse_args()


def _parse_meta(meta):
    if meta is None:
        return {}
    if isinstance(meta, str):
        return json.loads(meta)
    if isinstance(meta, dict):
        return meta
    raise TypeError(f"Unsupported meta type: {type(meta)}")


def _normalize_row(row: dict) -> dict:
    out = dict(row)
    if "prompt" not in out:
        raise ValueError(f"Row missing 'prompt'. Keys={sorted(out)}")

    if "numbers" not in out or "target" not in out:
        meta = _parse_meta(out.get("meta"))
        if "numbers" not in out:
            out["numbers"] = meta["numbers"]
        if "target" not in out:
            out["target"] = meta["target"]

    if isinstance(out["numbers"], str):
        out["numbers"] = json.loads(out["numbers"])
    out["target"] = int(out["target"])
    return out


def load_train_dataset(source: str, split: str) -> Dataset:
    path = Path(source)
    if path.exists():
        if path.is_dir():
            dataset = load_dataset("parquet", data_dir=str(path), split=split)
        else:
            suffix = path.suffix.lower()
            if suffix == ".parquet":
                dataset = load_dataset("parquet", data_files=str(path), split="train")
            elif suffix in {".jsonl", ".json"}:
                dataset = load_dataset("json", data_files=str(path), split="train")
            else:
                raise ValueError(
                    f"Unsupported local file type '{suffix}'. Use .parquet, .json, or .jsonl."
                )
    else:
        dataset = load_dataset(source, split=split)

    if isinstance(dataset, DatasetDict):
        if split not in dataset:
            raise ValueError(f"Split '{split}' not in dataset. Available: {list(dataset)}")
        dataset = dataset[split]

    dataset = dataset.map(_normalize_row)
    missing = [c for c in REQUIRED_COLUMNS if c not in dataset.column_names]
    if missing:
        raise ValueError(
            f"Dataset missing required columns {missing}. "
            f"Have {dataset.column_names}. Provide prompt/numbers/target "
            "or prompt + meta{{numbers,target}}."
        )
    return dataset


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

    dataset = load_train_dataset(args.dataset, args.dataset_split)
    print(dataset)
    print(f"columns={dataset.column_names} | rows={len(dataset)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_model(args.init_adapter)

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
