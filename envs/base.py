"""Shared environment interface for RLvSFT tasks."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any


ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


@dataclass
class Example:
    """One frozen task instance."""

    id: str
    prompt: str
    answer: str
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "Example":
        return cls(
            id=str(row["id"]),
            prompt=str(row["prompt"]),
            answer=str(row["answer"]),
            meta=dict(row.get("meta") or {}),
        )


def extract_tagged_answer(text: str) -> str | None:
    match = ANSWER_RE.search(text or "")
    if not match:
        return None
    return match.group(1).strip()


def format_bonus(text: str) -> float:
    """Small shaping reward for think/answer tags."""
    reward = 0.0
    if THINK_RE.search(text or ""):
        reward += 0.1
    if ANSWER_RE.search(text or ""):
        reward += 0.1
    return reward


class BaseEnv(ABC):
    """Abstract task environment."""

    name: str = "base"

    @abstractmethod
    def make_example(self, idx: int, rng: Any, *, split: str = "train") -> Example:
        """Deterministically build example `idx` using `rng`."""

    @abstractmethod
    def score(self, example: Example, completion: str) -> dict[str, Any]:
        """
        Return at least:
          - reward: float
          - correct: bool
          - format_ok: bool
        """

    def system_prompt(self) -> str:
        return (
            "Solve the problem carefully. Put step-by-step reasoning inside "
            "<think>...</think> and the final answer inside <answer>...</answer>."
        )

    def build_messages(self, example: Example) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.system_prompt()},
            {"role": "user", "content": example.prompt},
        ]
