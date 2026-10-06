#!/usr/bin/env python3
"""Build frozen tiny train/holdout splits for the feasibility harness."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs import HARNESS_ENV, get_env  # noqa: E402
from envs.base import Example  # noqa: E402

DEFAULT_SEED = 566
DEFAULT_TRAIN = 200
DEFAULT_HOLDOUT = 80


def _rows(env_name: str, split: str, n: int, seed: int) -> list[dict]:
    env = get_env(env_name)
    rng = random.Random(f"{seed}:{env_name}:{split}")
    rows: list[dict] = []
    for i in range(n):
        ex = env.make_example(i, rng, split=split)
        rows.append(
            {
                "id": ex.id,
                "env": env_name,
                "split": split,
                "prompt": ex.prompt,
                "answer": ex.answer,
                "numbers": list(ex.meta["numbers"]),
                "target": int(ex.meta["target"]),
                "meta": ex.meta,
            }
        )
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _write_oracle_traces(env_name: str, train_rows: list[dict], out_path: Path) -> None:
    env = get_env(env_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in train_rows:
            example = Example(
                id=row["id"],
                prompt=row["prompt"],
                answer=row["answer"],
                meta=dict(row["meta"]),
            )
            completion = env.oracle_completion(example)  # type: ignore[attr-defined]
            scored = env.score(example, completion)
            f.write(
                json.dumps(
                    {
                        "id": example.id,
                        "env": env_name,
                        "prompt": example.prompt,
                        "messages": env.build_messages(example)
                        + [{"role": "assistant", "content": completion}],
                        "completion": completion,
                        "reward": scored["reward"],
                        "correct": scored["correct"],
                    }
                )
                + "\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "splits")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--train-size", type=int, default=DEFAULT_TRAIN)
    parser.add_argument("--holdout-size", type=int, default=DEFAULT_HOLDOUT)
    parser.add_argument("--env", default=HARNESS_ENV)
    args = parser.parse_args()

    if not (100 <= args.train_size <= 300):
        raise SystemExit("--train-size must be in [100, 300] for the harness")
    if not (50 <= args.holdout_size <= 100):
        raise SystemExit("--holdout-size must be in [50, 100] for the harness")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_rows = _rows(args.env, "train", args.train_size, args.seed)
    holdout_rows = _rows(args.env, "holdout", args.holdout_size, args.seed)

    train_path = args.out_dir / f"{args.env}_train.jsonl"
    holdout_path = args.out_dir / f"{args.env}_holdout.jsonl"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(holdout_path, holdout_rows)

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        for path, rows in ((train_path, train_rows), (holdout_path, holdout_rows)):
            parquet_path = path.with_suffix(".parquet")
            pq_rows = []
            for row in rows:
                pq_row = dict(row)
                pq_row["meta"] = json.dumps(row["meta"])
                pq_row["numbers"] = json.dumps(row["numbers"])
                pq_rows.append(pq_row)
            pq.write_table(pa.Table.from_pylist(pq_rows), parquet_path)
            print(f"wrote {parquet_path} ({len(pq_rows)} rows)")
    except ImportError:
        print("pyarrow not installed; wrote JSONL only")

    trace_path = ROOT / "traces" / "out" / f"{args.env}_train_oracle.jsonl"
    _write_oracle_traces(args.env, train_rows, trace_path)
    print(f"wrote {trace_path}")

    manifest = {
        "seed": args.seed,
        "env": args.env,
        "sizes": {"train": args.train_size, "holdout": args.holdout_size},
        "files": {
            "train_jsonl": train_path.name,
            "holdout_jsonl": holdout_path.name,
            "oracle_traces": str(trace_path.relative_to(ROOT)),
        },
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {train_path} ({len(train_rows)} rows)")
    print(f"wrote {holdout_path} ({len(holdout_rows)} rows)")
    print(f"manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
