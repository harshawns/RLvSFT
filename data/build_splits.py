#!/usr/bin/env python3
"""Build frozen train/val/test parquet splits for every env."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs import ENV_REGISTRY  # noqa: E402

DEFAULT_SIZES = {"train": 256, "val": 64, "test": 64}


def _rows_for_env(name: str, split: str, n: int, seed: int) -> list[dict]:
    env = ENV_REGISTRY[name]()
    rng = random.Random(f"{seed}:{name}:{split}")
    rows = []
    for i in range(n):
        ex = env.make_example(i, rng, split=split)
        rows.append(
            {
                "id": ex.id,
                "env": name,
                "split": split,
                "prompt": ex.prompt,
                "answer": ex.answer,
                "meta": json.dumps(ex.meta),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "splits")
    parser.add_argument("--seed", type=int, default=566)
    parser.add_argument("--train-size", type=int, default=DEFAULT_SIZES["train"])
    parser.add_argument("--val-size", type=int, default=DEFAULT_SIZES["val"])
    parser.add_argument("--test-size", type=int, default=DEFAULT_SIZES["test"])
    parser.add_argument(
        "--envs",
        nargs="*",
        default=sorted(ENV_REGISTRY),
        help="Subset of env names",
    )
    args = parser.parse_args()

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit(
            "pyarrow is required to freeze parquet splits. "
            "Install deps via ./setup_mlx_env.sh"
        ) from exc

    sizes = {"train": args.train_size, "val": args.val_size, "test": args.test_size}
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for env_name in args.envs:
        if env_name not in ENV_REGISTRY:
            raise SystemExit(f"Unknown env: {env_name}")
        for split, n in sizes.items():
            rows = _rows_for_env(env_name, split, n, args.seed)
            table = pa.Table.from_pylist(rows)
            out = args.out_dir / f"{env_name}_{split}.parquet"
            pq.write_table(table, out)
            print(f"wrote {out} ({len(rows)} rows)")

    manifest = {
        "seed": args.seed,
        "sizes": sizes,
        "envs": list(args.envs),
        "files": sorted(p.name for p in args.out_dir.glob("*.parquet")),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"manifest -> {args.out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
