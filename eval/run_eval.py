#!/usr/bin/env python3
"""Evaluate completions on frozen parquet splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs import Example, get_env  # noqa: E402
from eval.metrics import by_env, summarize  # noqa: E402


def _load_split(path: Path) -> list[tuple[str, Example]]:
    import pyarrow.parquet as pq

    rows = pq.read_table(path).to_pylist()
    out = []
    for row in rows:
        meta = row.get("meta")
        if isinstance(meta, str):
            meta = json.loads(meta)
        env_name = row.get("env") or path.name.split("_")[0]
        out.append(
            (
                env_name,
                Example(
                    id=row["id"],
                    prompt=row["prompt"],
                    answer=row["answer"],
                    meta=meta or {},
                ),
            )
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-path", type=Path, required=True)
    parser.add_argument(
        "--predictions",
        type=Path,
        help="JSONL with fields id, completion. If omitted, score gold answers as oracle.",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    pairs = _load_split(args.split_path)
    preds = {}
    if args.predictions:
        with args.predictions.open() as f:
            for line in f:
                row = json.loads(line)
                preds[row["id"]] = row["completion"]

    scored_rows = []
    for env_name, ex in pairs:
        env = get_env(env_name)
        if args.predictions:
            completion = preds.get(ex.id, "")
        else:
            completion = f"<think>oracle</think>\n<answer>{ex.answer}</answer>"
            if env_name == "mbpp":
                completion = (
                    f"<think>oracle</think>\n<answer>\n```python\n{ex.answer.strip()}\n```\n</answer>"
                )
        result = env.score(ex, completion)
        scored_rows.append({"id": ex.id, "env": env_name, **result})

    report = {
        "overall": summarize(scored_rows),
        "by_env": by_env(scored_rows),
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
