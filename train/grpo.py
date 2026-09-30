#!/usr/bin/env python3
"""GRPO training with TRL on a frozen parquet split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs import Example, get_env  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())

    import pyarrow.parquet as pq
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    env = get_env(cfg["env"])
    rows = pq.read_table(cfg["split_path"]).to_pylist()
    prompts = []
    examples = []
    for row in rows:
        meta = row.get("meta")
        if isinstance(meta, str):
            meta = json.loads(meta)
        ex = Example(id=row["id"], prompt=row["prompt"], answer=row["answer"], meta=meta or {})
        examples.append(ex)
        prompts.append(ex.prompt)

    ds = Dataset.from_dict({"prompt": prompts})

    def reward_fn(completions: list[str], **kwargs):  # noqa: ANN003
        # TRL may pass prompts aligned with completions.
        rewards = []
        for i, completion in enumerate(completions):
            ex = examples[i % len(examples)]
            rewards.append(float(env.score(ex, completion)["reward"]))
        return rewards

    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg["model_name"])

    train_cfg = GRPOConfig(
        output_dir=cfg.get("output_dir", "outputs/grpo"),
        learning_rate=cfg.get("learning_rate", 1.0e-6),
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 8),
        num_generations=cfg.get("num_generations", 4),
        max_completion_length=cfg.get("max_completion_length", 256),
        logging_steps=cfg.get("logging_steps", 5),
        report_to=cfg.get("report_to", []),
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=train_cfg,
        train_dataset=ds,
        processing_class=tok,
    )
    trainer.train()
    trainer.save_model(cfg.get("output_dir", "outputs/grpo"))
    print(f"saved model to {cfg.get('output_dir', 'outputs/grpo')}")


if __name__ == "__main__":
    main()
