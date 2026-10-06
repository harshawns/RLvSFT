"""Task environments with deterministic scorers."""

from .base import BaseEnv, Example
from .countdown import CountdownEnv

ENV_REGISTRY: dict[str, type[BaseEnv]] = {
    CountdownEnv.name: CountdownEnv,
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
]
