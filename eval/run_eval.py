#!/usr/bin/env python3
"""Evaluate Base / adapter checkpoints on the frozen holdout split."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs import get_env  # noqa: E402
from eval.metrics import summarize  # noqa: E402
from train.data_utils import load_examples  # noqa: E402
from train.logging_utils import RunLogger, peak_vram_gb, reset_peak_vram  # noqa: E402
from train.models import (  # noqa: E402
    ModelConfig,
    load_adapter,
    load_base_model,
    load_tokenizer,
    write_run_metadata,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, choices=("qwen", "gemma", "afm"))
    p.add_argument(
        "--split-path",
        type=Path,
        default=ROOT / "data" / "splits" / "target_expr_holdout.jsonl",
    )
    p.add_argument("--adapter", type=Path, default=None, help="Optional LoRA adapter dir")
    p.add_argument(
        "--condition",
        default=None,
        help="Logged condition (Base / SFT / GRPO / SFT→GRPO). Default from adapter.",
    )
    p.add_argument("--seed", type=int, default=566)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--limit", type=int, default=0, help="0 = all rows")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument(
        "--oracle",
        action="store_true",
        help="Score gold oracle completions (no model load)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate split + write metadata without model generate",
    )
    return p.parse_args()


def _prompt_text(tokenizer, env, example) -> str:
    messages = env.build_messages(example)
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass
    return f"SYSTEM: {env.system_prompt()}\nUSER: {example.prompt}\nASSISTANT:"


def main() -> None:
    args = parse_args()
    cfg = ModelConfig.from_alias(args.model)
    examples = load_examples(args.split_path)
    if args.limit:
        examples = examples[: args.limit]

    condition = args.condition or ("Base" if args.adapter is None else "adapter")
    output_dir = args.output_dir or ROOT / "outputs" / f"eval_{args.model}_{condition.replace('→', '_').replace(' ', '_')}_s{args.seed}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = RunLogger(
        output_dir / "run.jsonl",
        context={
            "model": args.model,
            "model_id": cfg.model_id,
            "condition": condition,
            "seed": seed if (seed := args.seed) else 566,
        },
    )
    reset_peak_vram()
    write_run_metadata(
        output_dir,
        cfg=cfg,
        condition=condition,
        seed=args.seed,
        extra={"split_path": str(args.split_path), "adapter": str(args.adapter) if args.adapter else None},
    )

    env = get_env("target_expr")

    if args.oracle or args.dry_run:
        scored = []
        for ex in examples:
            completion = env.oracle_completion(ex)  # type: ignore[attr-defined]
            result = env.score(ex, completion)
            scored.append(
                {
                    "id": ex.id,
                    "env": "target_expr",
                    "completion": completion,
                    "gen_length": len(completion),
                    **result,
                }
            )
        report = {
            "overall": summarize(scored),
            "mode": "oracle" if args.oracle else "dry_run_oracle",
        }
        (output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
        logger.log("eval_done", **report["overall"])
        logger.close(status="ok")
        print(json.dumps(report, indent=2))
        return

    import torch

    tokenizer = load_tokenizer(cfg)
    model = load_base_model(cfg, device_map="auto")
    if args.adapter:
        model = load_adapter(model, args.adapter, is_trainable=False)
    model.eval()

    scored = []
    t0 = time.time()
    for ex in examples:
        prompt = _prompt_text(tokenizer, env, ex)
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        gen_ids = out[0][inputs["input_ids"].shape[-1] :]
        completion = tokenizer.decode(gen_ids, skip_special_tokens=True)
        result = env.score(ex, completion)
        scored.append(
            {
                "id": ex.id,
                "env": "target_expr",
                "completion": completion,
                "gen_length": int(gen_ids.numel()),
                **result,
            }
        )

    report = {
        "overall": summarize(scored),
        "wall_clock_s": round(time.time() - t0, 3),
        "peak_vram_gb": peak_vram_gb(),
        "n_examples": len(scored),
    }
    preds_path = output_dir / "predictions.jsonl"
    with preds_path.open("w", encoding="utf-8") as f:
        for row in scored:
            f.write(json.dumps(row) + "\n")
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    logger.log("eval_done", **report["overall"], wall_clock_s=report["wall_clock_s"])
    logger.close(status="ok")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
