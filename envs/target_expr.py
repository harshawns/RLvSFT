"""Fresh tiny harness task: build an expression that hits a target.

Not recovered from git history — new deterministic generator + verifier for
the feasibility smoke (train 100–300 / holdout 50–100).
"""

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
    ast.USub: operator.neg,
}

_ALLOWED_CHARS = re.compile(r"^[0-9+\-*/()\s]+$")
_INT_RE = re.compile(r"\d+")


def _eval_ast(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return float(_OPS[type(node.op)](_eval_ast(node.operand)))
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left = _eval_ast(node.left)
        right = _eval_ast(node.right)
        return float(_OPS[type(node.op)](left, right))
    raise ValueError("disallowed expression node")


def safe_eval_arith(expr: str) -> float:
    """Evaluate a restricted arithmetic AST (+, -, *, unary -, ints/parens)."""
    tree = ast.parse(expr, mode="eval")
    return _eval_ast(tree)


def _fold_expression(nums: list[int], ops: list[str]) -> tuple[str, int]:
    """Left-fold nums with ops into a fully parenthesized expression + value."""
    expr = str(nums[0])
    value = nums[0]
    for num, op in zip(nums[1:], ops):
        if op == "+":
            value = value + num
        elif op == "-":
            value = value - num
        elif op == "*":
            value = value * num
        else:
            raise ValueError(f"unsupported op: {op}")
        expr = f"({expr}{op}{num})"
    return expr, int(value)


class TargetExprEnv(BaseEnv):
    """Use each listed number once to form an expression equal to the target."""

    name = "target_expr"

    def make_example(self, idx: int, rng: Any, *, split: str = "train") -> Example:
        n_nums = 3 if split != "train" else rng.choice([3, 4])
        numbers = [rng.randint(1, 12) for _ in range(n_nums)]
        order = list(numbers)
        rng.shuffle(order)
        ops = [rng.choice(["+", "-", "*"]) for _ in range(n_nums - 1)]
        oracle_expr, target = _fold_expression(order, ops)

        prompt = (
            f"Using each of these numbers exactly once: {numbers}. "
            f"You may use +, -, *, and parentheses. "
            f"Write one arithmetic expression that equals {target}."
        )
        return Example(
            id=f"target_expr-{split}-{idx:05d}",
            prompt=prompt,
            answer=oracle_expr,
            meta={
                "numbers": numbers,
                "target": target,
                "oracle_expr": oracle_expr,
            },
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
            if not candidate or not _ALLOWED_CHARS.fullmatch(candidate):
                raise ValueError("bad chars")
            used = [int(x) for x in _INT_RE.findall(candidate)]
            if sorted(used) != sorted(numbers):
                raise ValueError("number multiset mismatch")
            value = safe_eval_arith(candidate)
            correct = abs(value - float(target)) < 1e-6
        except Exception:
            correct = False

        # Main signal is correctness; small format shaping only.
        reward = (1.0 if correct else 0.0) + fmt
        return {
            "reward": float(reward),
            "correct": bool(correct),
            "format_ok": bool(format_ok),
            "extracted": candidate,
        }

    def oracle_completion(self, example: Example) -> str:
        expr = example.meta.get("oracle_expr") or example.answer
        return (
            "<think>Combine the numbers with the allowed operators to hit the target.</think>\n"
            f"<answer>{expr}</answer>"
        )
