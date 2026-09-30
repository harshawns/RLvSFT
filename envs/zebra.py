"""Simplified zebra / Einstein-style constraint puzzles."""

from __future__ import annotations

import re
from typing import Any

from .base import BaseEnv, Example, extract_tagged_answer, format_bonus

COLORS = ["red", "green", "blue"]
PETS = ["cat", "dog", "bird"]
DRINKS = ["tea", "coffee", "water"]


class ZebraEnv(BaseEnv):
    name = "zebra"

    def make_example(self, idx: int, rng: Any, *, split: str = "train") -> Example:
        order = [0, 1, 2]
        rng.shuffle(order)
        colors = [COLORS[i] for i in order]
        pets = PETS[:]
        rng.shuffle(pets)
        drinks = DRINKS[:]
        rng.shuffle(drinks)

        houses = [
            {"position": i + 1, "color": colors[i], "pet": pets[i], "drink": drinks[i]}
            for i in range(3)
        ]

        bird_pos = next(h["position"] for h in houses if h["pet"] == "bird")
        if bird_pos == 1:
            neighbor = 2
        elif bird_pos == 3:
            neighbor = 2
        else:
            neighbor = int(rng.choice([1, 3]))
        neighbor_house = houses[neighbor - 1]
        coffee_color = next(h["color"] for h in houses if h["drink"] == "coffee")
        clues = [
            "There are three houses in a row, numbered 1 to 3 from left to right.",
            f"The {houses[0]['color']} house is on the far left.",
            f"The person in the {houses[2]['color']} house drinks {houses[2]['drink']}.",
            f"The {houses[1]['pet']} lives in house {houses[1]['position']}.",
            f"Coffee is drunk in the {coffee_color} house.",
            f"The bird lives in house {bird_pos}, next to the {neighbor_house['color']} house.",
        ]

        # Ask for one query attribute to keep scoring crisp.
        query_house = houses[rng.randint(0, 2)]
        attr = rng.choice(["pet", "drink", "color"])
        answer = str(query_house[attr])
        prompt = (
            "Solve the puzzle.\n"
            + "\n".join(f"- {c}" for c in clues)
            + f"\nQuestion: What is the {attr} of house {query_house['position']}?"
        )
        return Example(
            id=f"zebra-{split}-{idx:05d}",
            prompt=prompt,
            answer=answer,
            meta={"houses": houses, "attr": attr, "position": query_house["position"]},
        )

    def score(self, example: Example, completion: str) -> dict[str, Any]:
        gold = str(example.answer).strip().lower()
        fmt = format_bonus(completion)
        tagged = extract_tagged_answer(completion)
        text = (tagged if tagged is not None else completion or "").strip().lower()
        format_ok = tagged is not None
        # Accept bare answer or "house k: value".
        extracted = text
        match = re.search(rf"\b({re.escape(gold)})\b", text)
        correct = match is not None and (
            text == gold or text.endswith(gold) or bool(re.fullmatch(rf".*\b{re.escape(gold)}\b.*", text))
        )
        # Stricter: gold must appear and no conflicting known attributes as sole token mess.
        if tagged is not None:
            correct = tagged.strip().lower() == gold or tagged.strip().lower().endswith(gold)
            extracted = tagged.strip().lower()
        else:
            correct = text == gold
        reward = fmt + (1.0 if correct else 0.0)
        return {
            "reward": float(reward),
            "correct": correct,
            "format_ok": format_ok,
            "extracted": extracted,
        }
