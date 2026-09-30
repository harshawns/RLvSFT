"""Task environments with deterministic scorers."""

from .base import BaseEnv, Example
from .countdown import CountdownEnv
from .knights_knaves import KnightsKnavesEnv
from .math_task import MathTaskEnv
from .mbpp import MbppEnv
from .zebra import ZebraEnv

ENV_REGISTRY: dict[str, type[BaseEnv]] = {
    CountdownEnv.name: CountdownEnv,
    KnightsKnavesEnv.name: KnightsKnavesEnv,
    ZebraEnv.name: ZebraEnv,
    MathTaskEnv.name: MathTaskEnv,
    MbppEnv.name: MbppEnv,
}


def get_env(name: str) -> BaseEnv:
    try:
        return ENV_REGISTRY[name]()
    except KeyError as exc:
        known = ", ".join(sorted(ENV_REGISTRY))
        raise ValueError(f"Unknown env '{name}'. Known: {known}") from exc


__all__ = [
    "BaseEnv",
    "Example",
    "ENV_REGISTRY",
    "get_env",
    "CountdownEnv",
    "KnightsKnavesEnv",
    "ZebraEnv",
    "MathTaskEnv",
    "MbppEnv",
]
