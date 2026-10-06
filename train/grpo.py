#!/usr/bin/env python3
"""Tiny GRPO runner for the feasibility harness (4 gens/prompt, 10–20 updates).

Uses the real target_expr verifier reward (not length). For the fixed-geometry
G-ablation trainer previously on main, see train/grpo_ablation.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs.base import Example  # noqa: E402
from envs.target_expr import TargetExprEnv  # noqa: E402
from train.data_utils import load_jsonl, normalize_reward_row  # noqa: E402
from train.logging_utils import RunLogger, reset_peak_vram  # noqa: E402
from train.models import ModelConfig, write_run_metadata  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, choices=("qwen", "gemma", "afm"))
    p.add_argument("--config", type=Path, default=ROOT / "train" / "configs" / "grpo_smoke.yaml")
    p.add_argument("--dataset", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--init-adapter",
        type=str,
        default=None,
        help="SFT adapter path for the SFT→GRPO condition",
    )
    p.add_argument(
        "--condition",
        choices=("GRPO", "SFT→GRPO"),
        default=None,
        help="Logged condition name (default: GRPO or SFT→GRPO if --init-adapter)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configs/data without loading weights",
    )
    return p.parse_args()


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def make_reward_fn():
    env = TargetExprEnv()

    def reward_target_expr(completions, numbers, target, **kwargs):
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

    return reward_target_expr


def main() -> None:
    args = parse_args()
    smoke = _load_yaml(args.config)
    cfg = ModelConfig.from_alias(args.model)
    seed = int(args.seed if args.seed is not None else smoke.get("seed", 566))
    max_steps = int(args.max_steps if args.max_steps is not None else smoke.get("max_steps", 15))
    split_path = Path(args.dataset or smoke.get("split_path", "data/splits/target_expr_train.jsonl"))
    if not split_path.is_absolute():
        split_path = ROOT / split_path

    condition = args.condition or ("SFT→GRPO" if args.init_adapter else "GRPO")
    tag = "sft_grpo" if args.init_adapter else "grpo"
    output_dir = args.output_dir or ROOT / "outputs" / f"{tag}_{args.model}_s{seed}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = RunLogger(
        output_dir / "run.jsonl",
        context={
            "model": args.model,
            "model_id": cfg.model_id,
            "condition": condition,
            "seed": seed,
            "config": {**smoke, "model_config": cfg.alias, "init_adapter": args.init_adapter},
        },
    )
    reset_peak_vram()
    logger.log("run_start", dataset=str(split_path), max_steps=max_steps, num_generations=smoke.get("num_generations", 4))

    write_run_metadata(
        output_dir,
        cfg=cfg,
        condition=condition,
        seed=seed,
        extra={
            "split_path": str(split_path),
            "max_steps": max_steps,
            "init_adapter": args.init_adapter,
            "smoke_config": smoke,
        },
    )

    if args.dry_run:
        if not split_path.exists():
            raise SystemExit(f"Missing dataset: {split_path}. Run: python data/build_splits.py")
        rows = [normalize_reward_row(r) for r in load_jsonl(split_path)]
        cols = sorted(rows[0].keys()) if rows else []
        logger.log("dry_run_ok", n_rows=len(rows), columns=cols)
        logger.close(status="dry_run")
        print(f"dry-run ok | model={args.model} | rows={len(rows)} | out={output_dir}")
        return

    from trl import GRPOConfig, GRPOTrainer

    from train.data_utils import load_hf_split
    from train.logging_utils import MetricsCallback
    from train.models import apply_lora, load_base_model, load_tokenizer

    dataset = load_hf_split(str(split_path), "train").map(normalize_reward_row)
    tokenizer = load_tokenizer(cfg)
    model = load_base_model(cfg, device_map=None)
    model = apply_lora(model, cfg, init_adapter=args.init_adapter)
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    training_args = GRPOConfig(
        output_dir=str(output_dir),
        learning_rate=float(smoke.get("learning_rate", 2e-5)),
        per_device_train_batch_size=int(smoke.get("per_device_train_batch_size", 4)),
        gradient_accumulation_steps=int(smoke.get("gradient_accumulation_steps", 1)),
        num_generations=int(smoke.get("num_generations", 4)),
        max_prompt_length=int(smoke.get("max_prompt_length", 512)),
        max_completion_length=int(smoke.get("max_completion_length", 128)),
        max_steps=max_steps,
        bf16=bool(smoke.get("bf16", True)),
        beta=float(smoke.get("beta", 0.04)),
        temperature=float(smoke.get("temperature", 1.0)),
        seed=seed,
        data_seed=seed,
        logging_steps=int(smoke.get("logging_steps", 1)),
        report_to=list(smoke.get("report_to") or []),
        remove_unused_columns=False,
        use_vllm=bool(smoke.get("use_vllm", False)),
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[make_reward_fn()],
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    MetricsCallback(logger).attach(trainer)
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    logger.close(status="ok")
    print(f"{condition} done | model={args.model} | out={output_dir}")


if __name__ == "__main__":
    main()
