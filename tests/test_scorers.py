"""Backward-compatible scorer tests (harness task + countdown on main). """

from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.countdown import CountdownEnv
from envs.target_expr import TargetExprEnv


class ScorerTests(unittest.TestCase):
    def test_target_expr_oracle(self) -> None:
        env = TargetExprEnv()
        ex = env.make_example(0, random.Random(1), split="train")
        result = env.score(ex, env.oracle_completion(ex))
        self.assertTrue(result["correct"])
        self.assertTrue(result["format_ok"])

    def test_countdown_rejects_wrong_numbers(self) -> None:
        env = CountdownEnv()
        ex = env.make_example(1, random.Random(1), split="train")
        ex.meta = {"numbers": [1, 2, 3], "target": 6}
        result = env.score(ex, "<answer>(1+2)*4</answer>")
        self.assertFalse(result["correct"])


if __name__ == "__main__":
    unittest.main()
