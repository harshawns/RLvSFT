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
  Stores the equation submitted by the agent, acceptor flag, and reward.

Step / reward:
  Reward is assigned when the agent submits an equation.
  Verifier is check_equation(). Success (nums exactly once + reach goal) → 1,
  otherwise → 0. Episode still terminates after scoring either way.

Failure handling (by severity):
  end_episode — bad/wrong answer for this problem
  end_rollout — unusable action / verifier failure for this trajectory
  stop_training — fatal / unexpected error (caller should halt)

Dependencies (to implement later):
  generate_problem() — create valid problems
  check_equation()   — check the equation given by the agent

Randomness (to implement later):
  Random goal number and allowed numbers.
"""

from __future__ import annotations

from typing import Any

# Error severity → what the training loop should do.
SEVERITY_END_EPISODE = "end_episode"
SEVERITY_END_ROLLOUT = "end_rollout"
SEVERITY_STOP_TRAINING = "stop_training"


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
        self.ops: tuple[str, ...] = ("+", "-", "*", "/")

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

    def state(self) -> dict[str, Any]:
        return {
            "numbers": self.numbers,
            "target": self.target,
            "current_step": self.current_step,
            "last_expression": self.last_expression,
            "terminated": self.terminated,
        }

    def reset(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """
        Start a new episode.

        1. Call generate_problem() for a fresh (numbers, target, ops).
        2. Reset internal values: acceptor, reward, equation.
        3. Return the starting observation for the agent.
        """
        problem = self.generate_problem(*args, **kwargs)

        self.numbers = list(problem["numbers"])
        self.target = int(problem["target"])
        self.ops = tuple(problem.get("ops", self.ops))

        # Reset internal episode values.
        self.last_expression = None
        self.equation = None
        self.current_step = 0
        self.acceptor = False
        self.reward = 0.0
        self.terminated = False
        self.severity = None

        # Give the agent the start of the problem (observation).
        return {
            "numbers": list(self.numbers),
            "target": self.target,
            "ops": list(self.ops),
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
                "ops": list(self.ops),
            },
            "equation": self.equation,
            "reward": self.reward,
            "acceptor": self.acceptor,
            "terminated": self.terminated,
            "severity": self.severity,  # end_episode | end_rollout | stop_training | None
            "check": result,
            "state": self.state(),
        }
