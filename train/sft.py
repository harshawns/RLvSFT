#!/usr/bin/env python3
"""Supervised fine-tuning with TRL SFTTrainer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_traces(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())

    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    traces = _load_traces(Path(cfg["traces_path"]))
    texts = []
    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    for row in traces:
        messages = row["messages"]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append({"text": text})

    ds = Dataset.from_list(texts)
    model = AutoModelForCausalLM.from_pretrained(cfg["model_name"])

    train_cfg = SFTConfig(
        output_dir=cfg.get("output_dir", "outputs/sft"),
        num_train_epochs=cfg.get("num_train_epochs", 1),
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 8),
        learning_rate=cfg.get("learning_rate", 2.0e-5),
        logging_steps=cfg.get("logging_steps", 10),
        save_steps=cfg.get("save_steps", 100),
        bf16=cfg.get("bf16", False),
        max_length=cfg.get("max_seq_length", 1024),
        report_to=cfg.get("report_to", []),
    )
    trainer = SFTTrainer(
        model=model,
        args=train_cfg,
        train_dataset=ds,
        processing_class=tok,
    )
    trainer.train()
    trainer.save_model(cfg.get("output_dir", "outputs/sft"))
    print(f"saved model to {cfg.get('output_dir', 'outputs/sft')}")


if __name__ == "__main__":
    main()
