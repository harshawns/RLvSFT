"""Mini MBPP-style coding tasks with simple unit checks."""

from __future__ import annotations

import re
import textwrap
from typing import Any

from .base import BaseEnv, Example, extract_tagged_answer, format_bonus

_TASKS = [
    {
        "prompt": "Write a Python function `add(a, b)` that returns the sum of a and b.",
        "entry": "add",
        "tests": [((1, 2), 3), ((0, 0), 0), ((-1, 5), 4)],
        "oracle": "def add(a, b):\n    return a + b\n",
    },
    {
        "prompt": "Write a Python function `is_even(n)` that returns True iff n is even.",
        "entry": "is_even",
        "tests": [((2,), True), ((3,), False), ((0,), True)],
        "oracle": "def is_even(n):\n    return n % 2 == 0\n",
    },
    {
        "prompt": "Write a Python function `factorial(n)` for n >= 0.",
        "entry": "factorial",
        "tests": [((0,), 1), ((1,), 1), ((5,), 120)],
        "oracle": "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n - 1)\n",
    },
    {
        "prompt": "Write a Python function `reverse_string(s)` that reverses s.",
        "entry": "reverse_string",
        "tests": [(("ab",), "ba"), (("",), ""), (("xyz",), "zyx")],
        "oracle": "def reverse_string(s):\n    return s[::-1]\n",
    },
]


def _extract_code(completion: str) -> str:
    tagged = extract_tagged_answer(completion)
    text = tagged if tagged is not None else (completion or "")
    fence = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        return fence.group(1).strip()
    return textwrap.dedent(text).strip()


class MbppEnv(BaseEnv):
    name = "mbpp"

    def make_example(self, idx: int, rng: Any, *, split: str = "train") -> Example:
        task = _TASKS[idx % len(_TASKS)]
        # Lightly shuffle test order in meta only; prompt stays stable.
        tests = list(task["tests"])
        rng.shuffle(tests)
        return Example(
            id=f"mbpp-{split}-{idx:05d}",
            prompt=task["prompt"]
            + " Put the full function in <answer>...</answer> (optionally inside a python code fence).",
            answer=task["oracle"],
            meta={"entry": task["entry"], "tests": tests},
        )

    def score(self, example: Example, completion: str) -> dict[str, Any]:
        fmt = format_bonus(completion)
        code = _extract_code(completion)
        format_ok = extract_tagged_answer(completion) is not None
        entry = example.meta["entry"]
        tests = example.meta["tests"]

        passed = 0
        error = None
        try:
            namespace: dict[str, Any] = {}
            exec(code, namespace, namespace)  # noqa: S102 — intentional for sandbox scoring
            fn = namespace.get(entry)
            if not callable(fn):
                raise TypeError(f"{entry} not defined")
            for args, expected in tests:
                if fn(*args) != expected:
                    raise AssertionError(f"{entry}{args} != {expected}")
                passed += 1
            correct = True
        except Exception as exc:  # noqa: BLE001 — collect failure for metrics
            correct = False
            error = f"{type(exc).__name__}: {exc}"

        partial = passed / max(len(tests), 1)
        reward = fmt + (1.0 if correct else 0.5 * partial)
        return {
            "reward": float(reward),
            "correct": correct,
            "format_ok": format_ok,
            "extracted": code,
            "error": error,
            "passed": passed,
            "n_tests": len(tests),
        }
