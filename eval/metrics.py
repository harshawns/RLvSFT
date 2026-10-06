"""Aggregate evaluation metrics for harness smokes."""

from __future__ import annotations

from typing import Any


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "n": 0,
            "correct": 0,
            "total": 0,
            "accuracy": 0.0,
            "avg_reward": 0.0,
            "format_rate": 0.0,
            "avg_gen_length": 0.0,
        }

    n = len(rows)
    correct = sum(1 for r in rows if r.get("correct"))
    avg_reward = sum(float(r.get("reward", 0.0)) for r in rows) / n
    format_rate = sum(1 for r in rows if r.get("format_ok")) / n
    lengths = [float(r.get("gen_length", 0.0)) for r in rows]
    avg_gen_length = sum(lengths) / n
    return {
        "n": n,
        "correct": correct,
        "total": n,
        "accuracy": correct / n,
        "avg_reward": avg_reward,
        "format_rate": format_rate,
        "avg_gen_length": avg_gen_length,
    }
