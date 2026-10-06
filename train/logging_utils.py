"""Local JSONL run logging + simple compute meters for harness smokes."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def peak_vram_gb() -> float | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return torch.cuda.max_memory_allocated() / (1024**3)
    except Exception:
        return None


def reset_peak_vram() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


@dataclass
class RunLogger:
    path: Path
    context: dict[str, Any] = field(default_factory=dict)
    _t0: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("", encoding="utf-8")

    def log(self, event: str, **fields: Any) -> None:
        row = {
            "event": event,
            "wall_clock_s": round(time.time() - self._t0, 3),
            "peak_vram_gb": peak_vram_gb(),
            **self.context,
            **fields,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")

    def close(self, **fields: Any) -> None:
        self.log("run_end", **fields)


class MetricsCallback:
    """Lightweight Trainer callback that records loss / grad / reward / tokens."""

    def __init__(self, logger: RunLogger):
        self.logger = logger
        self.train_tokens = 0
        self.rollout_tokens = 0

    def attach(self, trainer) -> None:
        from transformers import TrainerCallback

        logger = self.logger
        outer = self

        class _CB(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                logs = logs or {}
                payload: dict[str, Any] = {
                    "step": state.global_step,
                    "loss": logs.get("loss"),
                    "grad_norm": logs.get("grad_norm"),
                    "learning_rate": logs.get("learning_rate"),
                    "reward": logs.get("reward") or logs.get("rewards"),
                    "reward_mean": logs.get("reward_mean") or logs.get("rewards/mean"),
                    "reward_std": logs.get("reward_std") or logs.get("rewards/std"),
                    "train_tokens": outer.train_tokens,
                    "rollout_tokens": outer.rollout_tokens,
                }
                # TRL sometimes nests reward stats under completions/*.
                for key, value in logs.items():
                    if "reward" in key.lower() or "completion" in key.lower():
                        payload[key] = value
                logger.log("train_log", **payload)

        trainer.add_callback(_CB())
