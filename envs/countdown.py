"""Countdown: use each number exactly once to hit a target."""

from __future__ import annotations

import ast
import operator
import re
from typing import Any

from .base import BaseEnv, Example, extract_tagged_answer, format_bonus

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
}


def _eval_expr(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_expr(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_expr(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left = _eval_expr(node.left)
        right = _eval_expr(node.right)
        if isinstance(node.op, ast.Div) and abs(right) < 1e-12:
            raise ZeroDivisionError
        return _OPS[type(node.op)](left, right)
    raise ValueError("disallowed expression")


def safe_eval_arith(expr: str) -> float:
    tree = ast.parse(expr, mode="eval")
    return _eval_expr(tree)


class CountdownEnv(BaseEnv):
    name = "countdown"

    def make_example(self, idx: int, rng: Any, *, split: str = "train") -> Example:
        n_nums = 3 if split == "test" else rng.randint(3, 4)
        numbers = [rng.randint(1, 20) for _ in range(n_nums)]
        # Build a reachable target from a simple left-fold expression.
        expr_nums = list(numbers)
        rng.shuffle(expr_nums)
        value = float(expr_nums[0])
        for num in expr_nums[1:]:
            op = rng.choice(["+", "-", "*"])
            if op == "+":
                value += num
            elif op == "-":
                value -= num
            else:
                value *= num
        target = int(value)
        prompt = (
            f"Using each of the numbers {numbers} exactly once, and only the "
            f"operators +, -, *, and parentheses, write an expression that equals "
            f"{target}."
        )
        return Example(
            id=f"countdown-{split}-{idx:05d}",
            prompt=prompt,
            answer=str(target),
            meta={"numbers": numbers, "target": target},
        )

    def score(self, example: Example, completion: str) -> dict[str, Any]:
        numbers = list(example.meta["numbers"])
        target = int(example.meta["target"])
        fmt = format_bonus(completion)
        tagged = extract_tagged_answer(completion)
        candidate = tagged if tagged is not None else (completion or "").strip()
        format_ok = tagged is not None

        correct = False
        try:
            if not re.fullmatch(r"[0-9+\-*/()\s]+", candidate or ""):
                raise ValueError("bad chars")
            used = [int(x) for x in re.findall(r"\d+", candidate)]
            if sorted(used) != sorted(numbers):
                raise ValueError("number multiset mismatch")
            value = safe_eval_arith(candidate)
            correct = abs(value - target) < 1e-6
        except Exception:
            correct = False

        reward = fmt + (1.0 if correct else 0.0)
        return {
            "reward": float(reward),
            "correct": correct,
            "format_ok": format_ok,
            "extracted": candidate,
        }
