"""Grade-school style arithmetic word problems."""

from __future__ import annotations

import re
from typing import Any

from .base import BaseEnv, Example, extract_tagged_answer, format_bonus


class MathTaskEnv(BaseEnv):
    name = "math_task"

    def make_example(self, idx: int, rng: Any, *, split: str = "train") -> Example:
        kind = rng.choice(["linear", "percent", "multi"])
        if kind == "linear":
            a, b = rng.randint(2, 20), rng.randint(1, 30)
            x = rng.randint(1, 15)
            answer = a * x + b
            prompt = f"Solve for x: {a}x + {b} = {answer}. What is x?"
            gold = str(x)
        elif kind == "percent":
            price = rng.randint(20, 200)
            pct = rng.choice([10, 15, 20, 25])
            answer = price * (100 - pct) // 100
            prompt = (
                f"A shirt costs ${price}. It is discounted by {pct}%. "
                f"What is the sale price in dollars (integer)?"
            )
            gold = str(answer)
        else:
            a, b, c = rng.randint(2, 12), rng.randint(2, 12), rng.randint(2, 12)
            gold_val = a * b + c
            prompt = (
                f"Maya has {a} boxes with {b} pencils each and finds {c} more. "
                f"How many pencils does she have in total?"
            )
            gold = str(gold_val)

        return Example(
            id=f"math-{split}-{idx:05d}",
            prompt=prompt,
            answer=gold,
            meta={"kind": kind},
        )

    def score(self, example: Example, completion: str) -> dict[str, Any]:
        gold = str(example.answer).strip()
        fmt = format_bonus(completion)
        tagged = extract_tagged_answer(completion)
        text = tagged if tagged is not None else (completion or "")
        format_ok = tagged is not None
        nums = re.findall(r"-?\d+", text.replace(",", ""))
        extracted = nums[-1] if nums else ""
        correct = extracted == gold
        reward = fmt + (1.0 if correct else 0.0)
        return {
            "reward": float(reward),
            "correct": correct,
            "format_ok": format_ok,
            "extracted": extracted,
        }
