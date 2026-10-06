#!/usr/bin/env python3
"""Generate oracle SFT traces (delegates to data/build_splits.py by default)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs import HARNESS_ENV, get_env  # noqa: E402
from envs.base import Example  # noqa: E402
from train.data_utils import load_examples  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env", default=HARNESS_ENV)
    p.add_argument("--split-path", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--teacher", default="oracle", help="Only 'oracle' is supported in the harness")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    if args.teacher != "oracle":
        raise SystemExit("Harness traces are oracle-only; use --teacher oracle")

    env = get_env(args.env)
    examples = load_examples(args.split_path)
    if args.limit:
        examples = examples[: args.limit]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    with args.out.open("w", encoding="utf-8") as f:
        for ex in examples:
            if not isinstance(ex, Example):
                raise TypeError(ex)
            completion = env.oracle_completion(ex)  # type: ignore[attr-defined]
            scored = env.score(ex, completion)
            n_ok += int(scored["correct"])
            f.write(
                json.dumps(
                    {
                        "id": ex.id,
                        "env": args.env,
                        "prompt": ex.prompt,
                        "messages": env.build_messages(ex)
                        + [{"role": "assistant", "content": completion}],
                        "completion": completion,
                        "reward": scored["reward"],
                        "correct": scored["correct"],
                    }
                )
                + "\n"
            )
    print(f"wrote {args.out} ({len(examples)} traces, {n_ok} correct)")


if __name__ == "__main__":
    main()
