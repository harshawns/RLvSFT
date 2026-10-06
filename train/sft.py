#!/usr/bin/env python3
"""Tiny SFT runner for the feasibility harness (20–50 steps, LoRA save/reload)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train.data_utils import load_jsonl  # noqa: E402
from train.logging_utils import RunLogger, reset_peak_vram  # noqa: E402
from train.models import ModelConfig, write_run_metadata  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, choices=("qwen", "gemma", "afm"))
    p.add_argument("--config", type=Path, default=ROOT / "train" / "configs" / "sft_smoke.yaml")
    p.add_argument("--traces", type=Path, default=None, help="Override oracle JSONL path")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configs/data and write metadata without loading weights",
    )
    return p.parse_args()


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _messages_to_text(tokenizer, messages: list[dict]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        except Exception:
            pass
    parts = []
    for m in messages:
        parts.append(f"{m.get('role', 'user').upper()}: {m.get('content', '')}")
    return "\n".join(parts)


def build_sft_dataset(traces_path: Path, tokenizer):
    from datasets import Dataset

    rows = load_jsonl(traces_path)
    texts = []
    for row in rows:
        messages = row.get("messages")
        if messages:
            texts.append(_messages_to_text(tokenizer, messages))
        else:
            texts.append(
                f"USER: {row['prompt']}\nASSISTANT: {row.get('completion', row.get('answer', ''))}"
            )
    return Dataset.from_dict({"text": texts})


def main() -> None:
    args = parse_args()
    smoke = _load_yaml(args.config)
    cfg = ModelConfig.from_alias(args.model)
    seed = int(args.seed if args.seed is not None else smoke.get("seed", 566))
    max_steps = int(args.max_steps if args.max_steps is not None else smoke.get("max_steps", 40))
    traces_path = Path(args.traces or smoke.get("traces_path", "traces/out/target_expr_train_oracle.jsonl"))
    if not traces_path.is_absolute():
        traces_path = ROOT / traces_path
    output_dir = args.output_dir or ROOT / "outputs" / f"sft_{args.model}_s{seed}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = RunLogger(
        output_dir / "run.jsonl",
        context={
            "model": args.model,
            "model_id": cfg.model_id,
            "condition": "SFT",
            "seed": seed,
            "config": {**smoke, "model_config": cfg.alias},
        },
    )
    reset_peak_vram()
    logger.log("run_start", traces_path=str(traces_path), max_steps=max_steps)

    write_run_metadata(
        output_dir,
        cfg=cfg,
        condition="SFT",
        seed=seed,
        extra={"traces_path": str(traces_path), "max_steps": max_steps, "smoke_config": smoke},
    )

    if args.dry_run:
        if not traces_path.exists():
            raise SystemExit(f"Missing traces: {traces_path}. Run: python data/build_splits.py")
        n = sum(1 for _ in traces_path.open())
        logger.log("dry_run_ok", n_traces=n)
        logger.close(status="dry_run")
        print(f"dry-run ok | model={args.model} | traces={n} | out={output_dir}")
        return

    from trl import SFTConfig, SFTTrainer

    from train.logging_utils import MetricsCallback
    from train.models import apply_lora, load_adapter, load_base_model, load_tokenizer

    tokenizer = load_tokenizer(cfg)
    model = load_base_model(cfg, device_map="auto")
    model = apply_lora(model, cfg)
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    dataset = build_sft_dataset(traces_path, tokenizer)
    train_args = SFTConfig(
        output_dir=str(output_dir),
        max_steps=max_steps,
        per_device_train_batch_size=int(smoke.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(smoke.get("gradient_accumulation_steps", 4)),
        learning_rate=float(smoke.get("learning_rate", 2e-5)),
        logging_steps=int(smoke.get("logging_steps", 1)),
        save_steps=int(smoke.get("save_steps", 20)),
        bf16=bool(smoke.get("bf16", True)),
        seed=seed,
        data_seed=seed,
        report_to=list(smoke.get("report_to") or []),
        dataset_text_field="text",
        max_length=int(cfg.max_seq_length),
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=train_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    MetricsCallback(logger).attach(trainer)
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Reload smoke: adapter must load back from disk.
    base = load_base_model(cfg, device_map="auto")
    _ = load_adapter(base, output_dir, is_trainable=False)
    logger.log("adapter_reload_ok", adapter_dir=str(output_dir))
    logger.close(status="ok")
    print(f"SFT done | model={args.model} | out={output_dir}")


if __name__ == "__main__":
    main()
