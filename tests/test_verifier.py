"""Unit tests for the fresh target_expr verifier."""

from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.base import Example
from envs.target_expr import TargetExprEnv, safe_eval_arith


class TargetExprVerifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = TargetExprEnv()

    def _ex(self, numbers, target, oracle="1+2") -> Example:
        return Example(
            id="t",
            prompt="x",
            answer=oracle,
            meta={"numbers": numbers, "target": target, "oracle_expr": oracle},
        )

    def test_correct_expression_with_tags(self) -> None:
        ex = self._ex([1, 2, 3], 6, oracle="(1+2)*3")
        # Use a known-correct sum for simplicity.
        ex = self._ex([1, 2, 3], 6, oracle="1+2+3")
        completion = "<think>add</think>\n<answer>1+2+3</answer>"
        result = self.env.score(ex, completion)
        self.assertTrue(result["format_ok"])
        self.assertTrue(result["correct"])
        self.assertGreaterEqual(result["reward"], 1.0)

    def test_rejects_wrong_numbers(self) -> None:
        ex = self._ex([1, 2, 3], 6)
        result = self.env.score(ex, "<answer>(1+2)*4</answer>")
        self.assertFalse(result["correct"])

    def test_rejects_wrong_value(self) -> None:
        ex = self._ex([1, 2, 3], 6)
        result = self.env.score(ex, "<answer>1+2+3+0</answer>")
        # Uses extra 0 → multiset mismatch
        self.assertFalse(result["correct"])
        result2 = self.env.score(ex, "<answer>1*2*3</answer>")
        self.assertTrue(result2["format_ok"])
        self.assertTrue(result2["correct"])  # 6 with same numbers

    def test_bad_format_still_scored(self) -> None:
        ex = self._ex([1, 2, 3], 6)
        result = self.env.score(ex, "1+2+3")
        self.assertFalse(result["format_ok"])
        self.assertTrue(result["correct"])
        self.assertEqual(result["reward"], 1.0)

    def test_missing_tags_wrong_answer(self) -> None:
        ex = self._ex([1, 2, 3], 6)
        result = self.env.score(ex, "I think it is 7")
        self.assertFalse(result["format_ok"])
        self.assertFalse(result["correct"])

    def test_oracle_completion_scores_correct(self) -> None:
        ex = self.env.make_example(0, random.Random(0), split="train")
        completion = self.env.oracle_completion(ex)
        result = self.env.score(ex, completion)
        self.assertTrue(result["format_ok"])
        self.assertTrue(result["correct"])
        self.assertAlmostEqual(safe_eval_arith(ex.meta["oracle_expr"]), float(ex.meta["target"]))

    def test_make_example_deterministic(self) -> None:
        a = self.env.make_example(3, random.Random(566), split="holdout")
        b = self.env.make_example(3, random.Random(566), split="holdout")
        self.assertEqual(a.to_dict(), b.to_dict())


if __name__ == "__main__":
    unittest.main()
