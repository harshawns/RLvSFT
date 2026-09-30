"""Unit tests for environment scorers."""

from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs import (  # noqa: E402
    CountdownEnv,
    KnightsKnavesEnv,
    MathTaskEnv,
    MbppEnv,
    ZebraEnv,
)


class ScorerTests(unittest.TestCase):
    def test_countdown_correct_expression(self) -> None:
        env = CountdownEnv()
        ex = env.make_example(0, random.Random(0), split="train")
        nums = ex.meta["numbers"]
        target = ex.meta["target"]
        # Construct a trivial correct completion when possible: single-number target.
        if len(nums) == 1:
            expr = str(nums[0])
        else:
            # Fall back to scoring a crafted example.
            ex.meta = {"numbers": [1, 2, 3], "target": 6}
            ex.answer = "6"
            expr = "1+2+3"
        completion = f"<think>ok</think>\n<answer>{expr}</answer>"
        result = env.score(ex, completion)
        self.assertTrue(result["format_ok"])
        self.assertTrue(result["correct"])
        self.assertGreaterEqual(result["reward"], 1.0)

    def test_countdown_rejects_wrong_numbers(self) -> None:
        env = CountdownEnv()
        ex = env.make_example(1, random.Random(1), split="train")
        ex.meta = {"numbers": [1, 2, 3], "target": 6}
        bad = "<answer>(1+2)*4</answer>"
        result = env.score(ex, bad)
        self.assertFalse(result["correct"])

    def test_knights_knaves_parses_labels(self) -> None:
        env = KnightsKnavesEnv()
        ex = env.make_example(0, random.Random(2), split="train")
        answer = ", ".join(f"{p}={r}" for p, r in ex.meta["roles"].items())
        completion = f"<think>logic</think>\n<answer>{answer}</answer>"
        result = env.score(ex, completion)
        self.assertTrue(result["correct"])

    def test_math_extracts_final_number(self) -> None:
        env = MathTaskEnv()
        ex = env.make_example(0, random.Random(3), split="train")
        completion = f"<think>calc</think>\n<answer>{ex.answer}</answer>"
        result = env.score(ex, completion)
        self.assertTrue(result["correct"])

    def test_mbpp_oracle_passes(self) -> None:
        env = MbppEnv()
        ex = env.make_example(0, random.Random(4), split="train")
        completion = (
            f"<think>code</think>\n<answer>\n```python\n{ex.answer.strip()}\n```\n</answer>"
        )
        result = env.score(ex, completion)
        self.assertTrue(result["correct"])
        self.assertEqual(result["passed"], result["n_tests"])

    def test_zebra_exact_answer(self) -> None:
        env = ZebraEnv()
        ex = env.make_example(0, random.Random(5), split="train")
        completion = f"<think>constraints</think>\n<answer>{ex.answer}</answer>"
        result = env.score(ex, completion)
        self.assertTrue(result["correct"])


if __name__ == "__main__":
    unittest.main()
