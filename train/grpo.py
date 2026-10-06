# Note: imports currently expect configs/envs modules to be re-added.
"""
GRPO training script (TRL) for countdown-style tasks.

Ablation design:
  - Hold rollouts/optimizer-step fixed at 64 for every arm.
  - Vary only --G (completions per prompt): 4, 8, or 16.
  - Stop by --max-steps so compute matches across arms.
"""

import argparse
import json
import sys
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset
from peft import PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

# ---------------------------------------------------------------------------
# Imports from this repo when launched as: python train/grpo.py
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs import MODEL_ID, build_lora_config
from envs.base import Example
from envs.countdown import CountdownEnv

# ---------------------------------------------------------------------------
# Fixed experiment constants (same for every G arm)
# ---------------------------------------------------------------------------
# Completions used in one optimizer update. Must stay constant across --G.
ROLLOUTS_PER_STEP = 64
# How those 64 are built: microbatch × grad_accum (not via generation_batch_size).
MICRO_BATCH_COMPLETIONS = 8  # per_device_train_batch_size
GRAD_ACCUM_STEPS = 8  # 8 * 8 = 64 completions per optimizer step

DEFAULT_MAX_STEPS = 400
DEFAULT_SEED = 42
DEFAULT_LR = 2e-5

# Pin RL sampling/KL; TRL defaults have changed between versions.
BETA = 0.001  # KL coefficient vs reference policy
TEMPERATURE = 1.0

# Columns the countdown reward reads from the train dataset.
REQUIRED_COLUMNS = ("prompt", "numbers", "target")


# ---------------------------------------------------------------------------
# CLI — sweep knobs live here; fixed geometry stays in constants above
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TRL GRPO on countdown (AFM + LoRA)")
    parser.add_argument(
        "--G",
        type=int,
        required=True,
        choices=(4, 8, 16),
        help="Group size (completions per prompt). Rollouts/step stay fixed at 64.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help=(
            "Your train data: local .parquet / .json / .jsonl path, a directory of those, "
            "or a Hugging Face dataset id. Rows need prompt, numbers, target "
            "(or prompt + meta JSON with numbers/target)."
        ),
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="train",
        help="Split name when --dataset is a HF hub id or a DatasetDict (default: train).",
    )
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


# ---------------------------------------------------------------------------
# Dataset loading — bring your own file / HF id; normalize to reward columns
# ---------------------------------------------------------------------------
def _parse_meta(meta):
    """meta may be missing, a JSON string, or already a dict."""
    if meta is None:
        return {}
    if isinstance(meta, str):
        return json.loads(meta)
    if isinstance(meta, dict):
        return meta
    raise TypeError(f"Unsupported meta type: {type(meta)}")


def _normalize_row(row: dict) -> dict:
    """
    Map one raw row → {prompt, numbers, target, ...}.

    Accepts either flat columns or prompt + meta{numbers, target}.
    """
    out = dict(row)
    if "prompt" not in out:
        raise ValueError(f"Row missing 'prompt'. Keys={sorted(out)}")

    # Pull reward fields out of meta if they aren't top-level columns.
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
    """Load user data from disk or the HF Hub, then normalize + validate schema."""
    path = Path(source)

    # --- resolve source ---
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
        # Not a local path → treat as Hugging Face dataset id (e.g. org/name)
        dataset = load_dataset(source, split=split)

    # --- pick split if we got a DatasetDict ---
    if isinstance(dataset, DatasetDict):
        if split not in dataset:
            raise ValueError(f"Split '{split}' not in dataset. Available: {list(dataset)}")
        dataset = dataset[split]

    # --- normalize rows, then fail fast on missing reward columns ---
    dataset = dataset.map(_normalize_row)
    missing = [c for c in REQUIRED_COLUMNS if c not in dataset.column_names]
    if missing:
        raise ValueError(
            f"Dataset missing required columns {missing}. "
            f"Have {dataset.column_names}. Provide prompt/numbers/target "
            "or prompt + meta{{numbers,target}}."
        )

    # AFM-Base path assumes plain-string prompts (no chat template).
    assert len(dataset) > 0, "Train dataset is empty"
    assert isinstance(dataset[0]["prompt"], str), (
        "prompt must be a plain string; message lists make TRL apply a chat "
        f"template (AFM-Base template is unclear). Got {type(dataset[0]['prompt'])}"
    )
    return dataset


# ---------------------------------------------------------------------------
# Token accounting — matched rollouts ≠ matched tokens across G arms
# ---------------------------------------------------------------------------
class CompletionTokenCallback(TrainerCallback):
    """
    Accumulate generated completion tokens from TRL's completions/mean_length.

    step_tokens ≈ mean_length × rollouts_per_step
    Use total_completion_tokens (not just steps) when plotting accuracy vs compute.
    """

    def __init__(self, rollouts_per_step: int, output_dir: str | Path):
        self.rollouts_per_step = rollouts_per_step
        self.output_dir = Path(output_dir)
        self.total_completion_tokens = 0.0
        self.history: list[dict] = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        mean_len = logs.get("completions/mean_length")
        if mean_len is None:
            return

        # Approximate tokens generated this optimizer step (64 rollouts).
        step_tokens = float(mean_len) * self.rollouts_per_step
        self.total_completion_tokens += step_tokens
        row = {
            "step": int(state.global_step),
            "mean_completion_length": float(mean_len),
            "step_tokens": step_tokens,
            "total_completion_tokens": self.total_completion_tokens,
        }
        self.history.append(row)

        # on_log runs after the trainer already flushed `logs`; push derived
        # totals explicitly so W&B can plot accuracy vs tokens later.
        if state.is_world_process_zero:
            try:
                import wandb

                if wandb.run is not None:
                    wandb.log(
                        {
                            "completions/step_tokens": step_tokens,
                            "completions/total_tokens": self.total_completion_tokens,
                        },
                        step=state.global_step,
                    )
            except Exception:
                pass

    def on_train_end(self, args, state, control, **kwargs):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.output_dir / "completion_tokens.json"
        payload = {
            "rollouts_per_step": self.rollouts_per_step,
            "total_completion_tokens": self.total_completion_tokens,
            "num_logged_steps": len(self.history),
            "history": self.history,
        }
        out_path.write_text(json.dumps(payload, indent=2) + "\n")
        print(
            f"Wrote token accounting → {out_path} "
            f"(total_completion_tokens={self.total_completion_tokens:.0f})"
        )


# ---------------------------------------------------------------------------
# Reward — TRL calls this on generated completions; returns one float each
# ---------------------------------------------------------------------------
def make_countdown_reward():
    """Build a GRPO reward_fn that scores completions with CountdownEnv."""
    env = CountdownEnv()

    def reward_countdown(completions, numbers, target, **kwargs):
        # `numbers` / `target` come from matching dataset columns (see REQUIRED_COLUMNS).
        rewards = []
        for completion, nums, tgt in zip(completions, numbers, target):
            if isinstance(nums, str):
                nums = json.loads(nums)

            # CountdownEnv.score expects an Example with meta numbers/target.
            example = Example(
                id="reward",
                prompt="",
                answer=str(tgt),
                meta={"numbers": list(nums), "target": int(tgt)},
            )

            # TRL may pass plain strings or chat-style message lists.
            if isinstance(completion, list):
                text = completion[-1].get("content", "") if completion else ""
            else:
                text = completion or ""

            rewards.append(float(env.score(example, text)["reward"]))
        return rewards

    return reward_countdown


# ---------------------------------------------------------------------------
# Model — base AFM + LoRA (fresh or continued from SFT adapter)
# ---------------------------------------------------------------------------
def load_model(init_adapter: str | None):
    # No device_map: let the Trainer / vLLM colocate placement own the device.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype="bfloat16",
        attn_implementation="sdpa",  # safer than flash_attention_2 for AFM
    )

    if init_adapter:
        # SFT → GRPO arm: continue training an existing adapter.
        model = PeftModel.from_pretrained(model, init_adapter, is_trainable=True)
    else:
        # Fresh LoRA for GRPO-from-base (same knobs as SFT via configs.py).
        model = get_peft_model(model, build_lora_config())

    model.print_trainable_parameters()
    return model


# ---------------------------------------------------------------------------
# Main — wire data, model, config, trainer; run GRPO
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    # --- sanity checks on fixed batch geometry ---
    if ROLLOUTS_PER_STEP % args.G != 0:
        raise ValueError(
            f"ROLLOUTS_PER_STEP ({ROLLOUTS_PER_STEP}) must be divisible by G ({args.G})"
        )
    if MICRO_BATCH_COMPLETIONS * GRAD_ACCUM_STEPS != ROLLOUTS_PER_STEP:
        raise ValueError("Microbatch × grad accum must equal ROLLOUTS_PER_STEP")

    # prompts per optimizer step changes with G; rollouts (64) do not
    prompts_per_step = ROLLOUTS_PER_STEP // args.G
    lr_tag = f"{args.lr:.0e}".replace("e-0", "e-").replace("e+0", "e+")
    run_name = f"grpo-G{args.G}-lr{lr_tag}-s{args.seed}"
    output_dir = args.output_dir or f"GRPO/{run_name}"

    print(
        f"{run_name} | rollouts/step={ROLLOUTS_PER_STEP} | "
        f"prompts/step={prompts_per_step} | max_steps={args.max_steps} | "
        f"init_adapter={args.init_adapter!r}"
    )
    print(
        "Batch geometry: "
        f"per_device_train_batch_size={MICRO_BATCH_COMPLETIONS} × "
        f"gradient_accumulation_steps={GRAD_ACCUM_STEPS} = "
        f"{ROLLOUTS_PER_STEP} completions/optimizer step (all G)."
    )

    # --- data / tokenizer / model ---
    dataset = load_train_dataset(args.dataset, args.dataset_split)
    print(dataset)
    print(f"columns={dataset.column_names} | rows={len(dataset)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_model(args.init_adapter)

    # --- GRPO hyperparameters ---
    # Important: do NOT set generation_batch_size. That would generate 64 once and
    # then split them across several optimizer steps (stale rollouts + unequal compute).
    # With the defaults, each optimizer step generates a fresh set of 64 completions.
    training_args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=args.lr,
        # Fixed geometry across G arms:
        per_device_train_batch_size=MICRO_BATCH_COMPLETIONS,  # 8
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,  # 8 → 64 / step
        num_generations=args.G,  # only ablation knob in the batch math
        max_prompt_length=512,
        max_completion_length=1024,  # room for <think>…<answer>
        optim="adamw_8bit",
        max_steps=args.max_steps,  # matched compute; not num_train_epochs
        warmup_ratio=0.03,  # short warmup; LR still decays linearly to 0 by default
        bf16=True,
        beta=BETA,
        temperature=TEMPERATURE,
        seed=args.seed,
        data_seed=args.seed,
        gradient_checkpointing=True,  # needed at long completion length
        save_strategy="steps",
        save_steps=50,
        save_total_limit=None,  # keep early ckpts for accuracy-vs-compute curves
        logging_steps=1,
        log_completions=True,  # inspect samples in W&B (catch reward hacking)
        report_to=["wandb"],
        remove_unused_columns=False,  # keep numbers/target for the reward fn
        use_vllm=True,
        vllm_mode="colocate",  # vLLM shares the training GPU
        vllm_gpu_memory_utilization=0.3,  # leave headroom for the train model
        run_name=run_name,
    )

    # --- train ---
    # Loop inside TRL: sample G completions/prompt → score → group-normalize → update.
    token_cb = CompletionTokenCallback(ROLLOUTS_PER_STEP, output_dir)
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[make_countdown_reward()],
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[token_cb],
    )

    trainer.train()
    trainer.save_model(output_dir)


if __name__ == "__main__":
    main()
