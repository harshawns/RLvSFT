"""Aggregate evaluation metrics."""

from __future__ import annotations

from typing import Any


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0, "accuracy": 0.0, "avg_reward": 0.0, "format_rate": 0.0}

    n = len(rows)
    accuracy = sum(1 for r in rows if r.get("correct")) / n
    avg_reward = sum(float(r.get("reward", 0.0)) for r in rows) / n
    format_rate = sum(1 for r in rows if r.get("format_ok")) / n
    return {
        "n": n,
        "accuracy": accuracy,
        "avg_reward": avg_reward,
        "format_rate": format_rate,
    }


def by_env(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault(str(row.get("env", "unknown")), []).append(row)
    return {env: summarize(items) for env, items in sorted(buckets.items())}
