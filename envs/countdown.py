"""
Countdown environment

Goal:
  Use each allowed number exactly once, with arithmetic operators, to build an
  equation that evaluates to the goal number.

Action space:
  Agent submits an equation string.

Observation space:
  Agent sees the goal number, allowed arithmetic ops, and the numbers it may
  use exactly once.

Internal state:
  Stores the equation submitted by the agent.

Dependencies (to implement later):
  generate_problem() — create valid problems
  check_equation()   — check the equation given by the agent

Randomness (to implement later):
  Random goal number and allowed numbers.
"""

from __future__ import annotations

from typing import Any


def generate_problem(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """
    Create a valid countdown problem.

    Should randomly sample allowed numbers and a reachable goal target.
    Expected return shape:
      {"numbers": list[int], "target": int, "ops": list[str]}
    """
    raise NotImplementedError


def check_equation(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """
    Check an equation submitted by the agent.

    Should verify:
      - only allowed arithmetic / parentheses
      - each allowed number used exactly once
      - expression evaluates to the goal
    Expected return shape:
      {"correct": bool, "value": float | None, "equation": str, "error": str | None}
    """
    raise NotImplementedError


class CountdownEnv:
    name = "countdown"

    def __init__(self) -> None:
        # Goal: reach this number.
        self.target: int | None = None

        # Observation: numbers the agent may use exactly once.
        self.numbers: list[int] = []

        # Observation: arithmetic options available to the agent.
        self.ops: tuple[str, ...] = ("+", "-", "*", "/")

        # Action / internal state: equation submitted by the agent.
        self.equation: str | None = None

        # Dependencies used by this env (structure only; not wired yet).
        self.generate_problem = generate_problem
        self.check_equation = check_equation
