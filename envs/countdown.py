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
  Reward is assigned when the agent submits an equation.
  Verifier is check_equation(). Success (nums exactly once + reach goal) → 1,
  otherwise → 0. Episode terminates after scoring either way.

Failure handling (by severity):
  end_episode — bad/wrong answer for this problem
  end_rollout — unusable action / verifier failure for this trajectory
  stop_training — fatal / unexpected error (caller should halt)

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

def check_equation(equation: str, numbers: list[int], target: int) -> dict[str, Any]:
    """
    Check an equation submitted by the agent.

    1) only digits / + - * / ( ) whitespace
    2) each allowed number used exactly once
    3) expression evaluates to target
    """
    eq = (equation or "").strip()
    fail = lambda error, value=None: {
        "correct": False,
        "value": value,
        "equation": equation,
        "error": error,
    }

    if not re.fullmatch(r"[0-9+\-*/()\s]+", eq):
        return fail("bad characters")

    # Each allowed number exactly once.
    remaining = list(numbers)
    for num in (int(x) for x in re.findall(r"\d+", eq)):
        if num not in remaining:
            return fail(f"unexpected number {num}")
        remaining.remove(num)
    if remaining:
        return fail(f"unused numbers {remaining}")

    # Whitelist already limits what eval can see; keep this simple.
    try:
        value = eval(eq, {"__builtins__": {}}, {})
    except Exception as exc:  # noqa: BLE001
        return fail(f"invalid expression: {exc}")

    if abs(float(value) - float(target)) > 1e-6:
        return fail(f"wrong answer {value} != {target}", value=float(value))

    return {"correct": True, "value": float(value), "equation": equation, "error": None}

def _error_severity(error: str | None, *, verifier_raised: bool = False) -> str | None:
    """
    Map a failure to a handling level.

    - no error + incorrect → end_episode (valid attempt, wrong answer)
    - format / constraint errors → end_episode
    - verifier crash / empty action → end_rollout
    - unexpected fatal → stop_training
    """
    if verifier_raised:
        return SEVERITY_END_ROLLOUT
    if error is None:
        return None
    fatal_markers = ("fatal", "internal", "oom", "cuda")
    rollout_markers = ("empty", "timeout", "decode", "unavailable")
    lowered = error.lower()
    if any(m in lowered for m in fatal_markers):
        return SEVERITY_STOP_TRAINING
    if any(m in lowered for m in rollout_markers):
        return SEVERITY_END_ROLLOUT
    return SEVERITY_END_EPISODE


class CountdownEnv:
    name = "countdown"

    def __init__(self) -> None:
        # Goal: reach this number.
        self.target: int | None = None

        # Observation: numbers the agent may use exactly once.
        self.numbers: list[int] = []

        # Observation: arithmetic options available to the agent.
        self.ops: tuple[str, ...] = OPS

        # Text prompt for the model (built in reset from the observation values).
        self.prompt: str | None = None

        # Action / internal state: last expression submitted by the agent.
        self.last_expression: str | None = None
        self.equation: str | None = None  # alias of last_expression

        # Episode progress.
        self.current_step: int = 0

        # Internal episode values.
        self.acceptor: bool = False  # whether the submitted equation was accepted
        self.reward: float = 0.0
        self.terminated: bool = False
        self.severity: str | None = None

        # Dependencies used by this env (structure only; generate/check not filled yet).
        self.generate_problem = generate_problem
        self.check_equation = check_equation

    def build_prompt(self) -> str:
        """Turn current numbers / ops / target into a model prompt."""
        ops = ", ".join(OPS)
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
        self.reward = 0.0
        self.terminated = False
        self.severity = None

        # Give the agent the start of the problem (values + prompt).
        # Note: target_eq is intentionally not returned (would leak the solution).
        return {
            "numbers": list(self.numbers),
            "target": self.target,
            "ops": list(OPS),
            "prompt": self.prompt,
        }

    def step(self, equation: str) -> dict[str, Any]:
        """
        Agent submits an equation (reward timing = submit).

        Success: uses each number exactly once AND reaches the goal → reward 1.
        Otherwise reward 0, but the episode still ends once scored.

        Verifier: check_equation().
        Termination: after a reward score is assigned for this equation.
        """
        self.current_step += 1
        self.last_expression = equation
        self.equation = equation
        self.severity = None
        verifier_raised = False
        result: dict[str, Any]

        try:
            if equation is None or str(equation).strip() == "":
                raise ValueError("empty equation")
            result = self.check_equation(
                equation,
                numbers=self.numbers,
                target=self.target,
            )
        except NotImplementedError:
            # Dependency not filled yet — treat as rollout-level failure for callers.
            raise
        except Exception as exc:  # noqa: BLE001
            verifier_raised = True
            result = {
                "correct": False,
                "value": None,
                "equation": str(equation) if equation is not None else "",
                "error": str(exc),
            }

        correct = bool(result.get("correct", False))

        # Reward fn: 1 if constraints + goal satisfied, else 0.
        # Wrong goal still ends the episode; reward just stays bad.
        self.acceptor = correct
        self.reward = 1.0 if correct else 0.0
        self.terminated = True  # episode over once we have a reward score

        if not correct:
            self.severity = _error_severity(
                result.get("error"),
                verifier_raised=verifier_raised,
            )
        else:
            self.severity = None

        return {
            "observation": {
                "numbers": list(self.numbers),
                "target": self.target,
                "ops": list(OPS),
            },
            "equation": self.equation,
            "reward": self.reward,
            "acceptor": self.acceptor,
            "terminated": self.terminated,
            "severity": self.severity,  # end_episode | end_rollout | stop_training | None
            "check": result,
            "state": self.state(),
        }
