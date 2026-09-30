#!/usr/bin/env python3
"""Generate SFT traces from a teacher model (or oracle completions)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs import Example, get_env  # noqa: E402


def _load_split(path: Path) -> list[Example]:
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    rows = table.to_pylist()
    examples = []
    for row in rows:
        meta = row.get("meta")
        if isinstance(meta, str):
            meta = json.loads(meta)
        examples.append(
            Example(
                id=row["id"],
                prompt=row["prompt"],
                answer=row["answer"],
                meta=meta or {},
            )
        )
    return examples


def _oracle_completion(env_name: str, example: Example) -> str:
    """Cheap deterministic teacher for scaffolding / offline smoke tests."""
    if env_name == "mbpp":
        body = example.answer.strip()
        return f"<think>Implement the requested function.</think>\n<answer>\n```python\n{body}\n```\n</answer>"
    return (
        f"<think>Reason carefully and report the final result.</think>\n"
        f"<answer>{example.answer}</answer>"
    )


def _generate_with_hf(model_name: str, messages: list[dict[str, str]], max_new_tokens: int) -> str:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    gen = out[0][inputs["input_ids"].shape[-1] :]
    return tok.decode(gen, skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", required=True, choices=sorted({"countdown", "knights_knaves", "zebra", "math_task", "mbpp"}))
    parser.add_argument("--split-path", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--teacher", default="oracle", help="'oracle' or HF model id")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0, help="0 = all rows")
    args = parser.parse_args()

    env = get_env(args.env)
    examples = _load_split(args.split_path)
    if args.limit:
        examples = examples[: args.limit]

    traces = []
    for ex in examples:
        if args.teacher == "oracle":
            completion = _oracle_completion(args.env, ex)
        else:
            completion = _generate_with_hf(args.teacher, env.build_messages(ex), args.max_new_tokens)
        scored = env.score(ex, completion)
        traces.append(
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

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in traces:
            f.write(json.dumps(row) + "\n")
    n_ok = sum(1 for r in traces if r["correct"])
    print(f"wrote {args.out} ({len(traces)} traces, {n_ok} correct)")


if __name__ == "__main__":
    main()
