"""Shared dataset helpers for harness train/eval scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from envs.base import Example


def parse_meta(meta: Any) -> dict:
    if meta is None:
        return {}
    if isinstance(meta, str):
        return json.loads(meta)
    if isinstance(meta, dict):
        return meta
    raise TypeError(f"Unsupported meta type: {type(meta)}")


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def row_to_example(row: dict) -> Example:
    meta = parse_meta(row.get("meta"))
    if "numbers" in row and "numbers" not in meta:
        nums = row["numbers"]
        if isinstance(nums, str):
            nums = json.loads(nums)
        meta["numbers"] = nums
    if "target" in row and "target" not in meta:
        meta["target"] = int(row["target"])
    return Example(
        id=str(row["id"]),
        prompt=str(row["prompt"]),
        answer=str(row.get("answer", meta.get("oracle_expr", meta.get("target", "")))),
        meta=meta,
    )


def load_examples(path: Path) -> list[Example]:
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        return [row_to_example(r) for r in load_jsonl(path)]
    if path.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq

        return [row_to_example(r) for r in pq.read_table(path).to_pylist()]
    raise ValueError(f"Unsupported split path: {path}")


def load_hf_split(source: str, split: str = "train"):
    from datasets import DatasetDict, load_dataset

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
                raise ValueError(f"Unsupported local file type '{suffix}'")
    else:
        dataset = load_dataset(source, split=split)

    if isinstance(dataset, DatasetDict):
        dataset = dataset[split]
    return dataset


def normalize_reward_row(row: dict) -> dict:
    """Ensure prompt/numbers/target columns exist for GRPO reward fns."""
    out = dict(row)
    if "prompt" not in out:
        raise ValueError(f"Row missing prompt. Keys={sorted(out)}")
    meta = parse_meta(out.get("meta"))
    if "numbers" not in out:
        out["numbers"] = meta["numbers"]
    if "target" not in out:
        out["target"] = meta["target"]
    if isinstance(out["numbers"], str):
        out["numbers"] = json.loads(out["numbers"])
    out["target"] = int(out["target"])
    return out
