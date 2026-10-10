"""
Countdown environment

Goal:
  Use each allowed number exactly once, with arithmetic operators, to build an
  equation that evaluates to the goal number.

Action space:

  Agent submits one equation string per episode.

Observation space:
  Agent sees the goal number, allowed arithmetic ops, and the numbers it may
  use exactly once (plus a text prompt for the model).

Internal state:
  Stores the equation submitted by the agent, acceptor flag, and reward.

Step / reward:
  Reward is assigned when the agent submits a completion.
  Verifier is check_equation(). Success → 1, scored failure → 0,
  unusable action / verifier crash → None (do not train on this sample).
  Episode terminates after the submit either way.

Failure handling (by severity; set explicitly, not via error-string matching):
  end_episode — bad/wrong answer for this problem (reward 0)
  end_rollout — empty action / verifier crash (reward None)
  stop_training — fatal infra error (reward None; caller should halt)

Dependencies:
  generate_problem() — create valid reachable puzzles
  check_equation()   — verify the agent's equation

Randomness:
  generate_problem samples numbers/ops via a seeded RNG.
"""

from __future__ import annotations

import ast
import operator
import random
import re
from fractions import Fraction
from typing import Any, Sequence

# Error severity → what the training loop should do.
SEVERITY_END_EPISODE = "end_episode"
SEVERITY_END_ROLLOUT = "end_rollout"
SEVERITY_STOP_TRAINING = "stop_training"

OPS = ["+", "-", "*", "/"]

_BINOPS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_OP_TO_AST: dict[str, type] = {
    "+": ast.Add,
    "-": ast.Sub,
    "*": ast.Mult,
    "/": ast.Div,
}


def _eval_ast(node: ast.AST, allowed_binops: set[type]) -> Fraction:
    """Evaluate a whitelisted arithmetic AST to an exact Fraction."""
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body, allowed_binops)
    if isinstance(node, ast.Constant):
        # bool is a subclass of int — reject True/False.
        if isinstance(node.value, int) and not isinstance(node.value, bool):
            return Fraction(node.value)
        raise ValueError("only integer literals allowed")
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _eval_ast(node.operand, allowed_binops)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and type(node.op) in allowed_binops:
        left = _eval_ast(node.left, allowed_binops)
        right = _eval_ast(node.right, allowed_binops)
        return _BINOPS[type(node.op)](left, right)
    raise ValueError("unsupported expression")


def generate_problem(seed: int, max_attempts: int = 1000) -> dict[str, Any]:
    """
    Create a valid countdown problem.

    Constructively left-folds four numbers with random ops so the target is
    reachable. Retries until the target is a positive integer in [1, 99999].
    Expected return shape:
      {"numbers": list[int], "target": int, "ops": list[str], "target_eq": str}
    """
    rng = random.Random(seed)
    math_fn = {
        "+": operator.add,
        "-": operator.sub,
        "*": operator.mul,
        "/": operator.truediv,
    }

    for _ in range(max_attempts):
        numbers = rng.sample(range(1, 100), 4)
        ordered_numbers = rng.sample(numbers, len(numbers))
        selected_ops = rng.choices(OPS, k=len(numbers) - 1)

        value = Fraction(ordered_numbers[0])
        expression = str(ordered_numbers[0])
        for op, number in zip(selected_ops, ordered_numbers[1:]):
            value = math_fn[op](value, Fraction(number))
            expression = f"({expression} {op} {number})"

        # Exact positive integer in an allowed range.
        if value.denominator == 1 and 1 <= value.numerator <= 99999:
            return {
                "numbers": numbers,
                "target": int(value),
                "ops": list(OPS),
                "target_eq": expression,  # parenthesized left-fold; do not show to agent
            }

    raise RuntimeError(
        f"Could not generate a valid puzzle after {max_attempts} attempts (seed={seed})"
    )

def check_equation(
    equation: str,
    numbers: list[int],
    target: int,
    ops: Sequence[str] | None = None,
) -> dict[str, Any]:
    """
    Check an equation submitted by the agent.

    1) only digits / allowed ops / ( ) whitespace; no ** or //
    2) each allowed number used exactly once
    3) AST-eval to Fraction equals target exactly
    """
    allowed_ops = list(ops) if ops is not None else list(OPS)
    eq = (equation or "").strip()
    fail = lambda error, value=None: {
        "correct": False,
        "value": value,
        "equation": equation,
        "error": error,
    }

    if "**" in eq or "//" in eq:
        return fail("bad operators")

    op_class = "".join(re.escape(op) for op in allowed_ops)
    if not re.fullmatch(rf"[0-9{op_class}()\s]+", eq):
        return fail("bad characters")

    # Each allowed number exactly once.
    remaining = list(numbers)
    for num in (int(x) for x in re.findall(r"\d+", eq)):
        if num not in remaining:
            return fail(f"unexpected number {num}")
        remaining.remove(num)
    if remaining:
        return fail(f"unused numbers {remaining}")

    allowed_binops = {_OP_TO_AST[op] for op in allowed_ops if op in _OP_TO_AST}
    try:
        tree = ast.parse(eq, mode="eval")
        value = _eval_ast(tree, allowed_binops)
    except Exception as exc:  # noqa: BLE001
        return fail(f"invalid expression: {exc}")

    if value != Fraction(target):
        return fail(f"wrong answer {value} != {target}", value=float(value))

    return {"correct": True, "value": float(value), "equation": equation, "error": None}


class CountdownEnv:
    name = "countdown"

    def __init__(self) -> None:
        # Goal: reach this number.
        self.target: int | None = None

        # Observation: numbers the agent may use exactly once.
        self.numbers: list[int] = []

        # Observation: arithmetic options available to the agent.
        self.ops: tuple[str, ...] = tuple(OPS)

        # Text prompt for the model (built in reset from the observation values).
        self.prompt: str | None = None

        # Action / internal state: last expression submitted by the agent.
        self.last_expression: str | None = None
        self.equation: str | None = None  # alias of last_expression

        # Episode progress.
        self.current_step: int = 0

        # Internal episode values.
        self.acceptor: bool = False  # whether the submitted equation was accepted
        self.reward: float | None = None  # 1 / 0 / None (unscored)
        self.terminated: bool = False
        self.severity: str | None = None

        # Dependencies used by this env (structure only; generate/check not filled yet).
        self.generate_problem = generate_problem
        self.check_equation = check_equation

    def build_prompt(self) -> str:
        """Turn current numbers / ops / target into a model prompt."""
        ops = ", ".join(self.ops)
        return (
            f"Using each of the numbers {list(self.numbers)} exactly once, "
            f"and only the operators {ops} and parentheses, write an expression "
            f"that equals {self.target}.\n"
            "Put step-by-step reasoning in <think>...</think> and the final "
            "expression in <answer>...</answer>."
        )

    def state(self) -> dict[str, Any]:
        return {
            "numbers": self.numbers,
            "target": self.target,
            "current_step": self.current_step,
            "last_expression": self.last_expression,
            "terminated": self.terminated,
            "prompt": self.prompt,
        }

    def reset(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """
        Start a new episode.

        1. Call generate_problem() for a fresh (numbers, target, ops).
        2. Reset internal values: acceptor, reward, equation.
        3. Build a prompt that includes those values for the agent/model.
        4. Return the starting observation (values + prompt).
        """
        if args or "seed" in kwargs:
            problem = self.generate_problem(*args, **kwargs)
        else:
            problem = self.generate_problem(seed=random.randrange(1 << 30))

        self.numbers = list(problem["numbers"])
        self.target = int(problem["target"])
        self.ops = tuple(problem.get("ops", OPS))
        self.prompt = self.build_prompt()

        # Reset internal episode values.
        self.last_expression = None
        self.equation = None
        self.current_step = 0
        self.acceptor = False
        self.reward = None
        self.terminated = False
        self.severity = None

        # Give the agent the start of the problem (values + prompt).
        # Note: target_eq is intentionally not returned (would leak the solution).
        return {
            "numbers": list(self.numbers),
            "target": self.target,
            "ops": list(self.ops),
            "prompt": self.prompt,
        }

    def step(self, equation: str) -> dict[str, Any]:
        """
        Agent submits a completion (reward timing = submit).

        Extracts <answer>...</answer>, then scores with check_equation().
        reward: 1 correct, 0 scored failure, None unscored (skip train).
        severity: set explicitly for failures (not via error-string matching).

        Rejects step-before-reset and step-after-terminate (reward None).
        """
        def _out(result: dict[str, Any], *, reward: float | None | object = ...) -> dict[str, Any]:
            return {
                "observation": {
                    "numbers": list(self.numbers),
                    "target": self.target,
                    "ops": list(self.ops),
                },
                "equation": self.equation,
                "reward": self.reward if reward is ... else reward,
                "acceptor": self.acceptor,
                "terminated": self.terminated,
                "severity": self.severity,
                "check": result,
                "state": self.state(),
            }

        # Episode lifecycle guards — check before mutating scored state.
        if self.target is None:
            result = {
                "correct": False,
                "value": None,
                "equation": "" if equation is None else str(equation),
                "error": "step before reset",
            }
            self.reward = None
            self.severity = SEVERITY_END_ROLLOUT
            self.terminated = True
            return _out(result)

        if self.terminated:
            result = {
                "correct": False,
                "value": None,
                "equation": "" if equation is None else str(equation),
                "error": "step after terminate",
            }
            # Keep prior self.reward/equation; this return is unscored.
            self.severity = SEVERITY_END_ROLLOUT
            return _out(result, reward=None)

        self.current_step += 1
        self.last_expression = equation
        self.equation = equation
        self.acceptor = False
        self.reward = None
        self.severity = None
        self.terminated = True
        result: dict[str, Any]

        if equation is None or str(equation).strip() == "":
            result = {
                "correct": False,
                "value": None,
                "equation": "" if equation is None else str(equation),
                "error": "empty equation",
            }
            self.severity = SEVERITY_END_ROLLOUT
            return _out(result)

        # Prompt asks for <answer>...</answer>; score only that body.
        tags = re.findall(
            r"<answer>(.*?)</answer>", str(equation), flags=re.DOTALL | re.IGNORECASE
        )
        expr = tags[-1].strip() if tags else ""
        if not expr:
            result = {
                "correct": False,
                "value": None,
                "equation": str(equation),
                "error": "missing <answer>...</answer>",
            }
            self.reward = 0.0
            self.severity = SEVERITY_END_EPISODE
            return _out(result)

        try:
            self.equation = expr
            result = self.check_equation(
                expr,
                numbers=self.numbers,
                target=self.target,
                ops=self.ops,
            )
        except NotImplementedError:
            raise
        except Exception as exc:  # noqa: BLE001
            result = {
                "correct": False,
                "value": None,
                "equation": expr,
                "error": str(exc),
            }
            # Typed fatal vs rollout crash — no substring matching on messages.
            if isinstance(exc, (MemoryError, SystemError)):
                self.severity = SEVERITY_STOP_TRAINING
            else:
                self.severity = SEVERITY_END_ROLLOUT
            return _out(result)  # reward stays None

        correct = bool(result.get("correct", False))
        self.acceptor = correct
        if correct:
            self.reward = 1.0
            self.severity = None
        else:
            self.reward = 0.0
            self.severity = SEVERITY_END_EPISODE
        return _out(result)
