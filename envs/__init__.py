"""Task environments with deterministic scorers."""

from .base import BaseEnv, Example
from .countdown import CountdownEnv
from .target_expr import TargetExprEnv

ENV_REGISTRY: dict[str, type[BaseEnv]] = {
    TargetExprEnv.name: TargetExprEnv,
    CountdownEnv.name: CountdownEnv,
}

# Default harness task for feasibility smokes.
HARNESS_ENV = TargetExprEnv.name


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
    "HARNESS_ENV",
    "get_env",
    "TargetExprEnv",
    "CountdownEnv",
]
