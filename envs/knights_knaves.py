"""Knights and Knaves logic puzzles."""

from __future__ import annotations

import re
from typing import Any

from .base import BaseEnv, Example, extract_tagged_answer, format_bonus

NAMES = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank"]


class KnightsKnavesEnv(BaseEnv):
    name = "knights_knaves"

    def make_example(self, idx: int, rng: Any, *, split: str = "train") -> Example:
        n = 2 if split == "test" else rng.randint(2, 3)
        people = NAMES[:n]
        roles = {p: rng.choice(["knight", "knave"]) for p in people}
        # Each person makes a statement about another person's role.
        statements = []
        for i, speaker in enumerate(people):
            target = people[(i + 1) % n]
            claimed = rng.choice(["knight", "knave"])
            truth = roles[target] == claimed
            # Knights say true things; knaves say false things.
            if (roles[speaker] == "knight") != truth:
                claimed = "knave" if claimed == "knight" else "knight"
            statements.append(f'{speaker} says: "{target} is a {claimed}."')

        answer_map = ", ".join(f"{p}={roles[p]}" for p in people)
        prompt = (
            "Knights always tell the truth. Knaves always lie.\n"
            + "\n".join(statements)
            + "\nFor each person, decide whether they are a knight or a knave. "
            "Answer like: Alice=knight, Bob=knave"
        )
        return Example(
            id=f"kk-{split}-{idx:05d}",
            prompt=prompt,
            answer=answer_map,
            meta={"roles": roles, "people": people},
        )

    def score(self, example: Example, completion: str) -> dict[str, Any]:
        roles = dict(example.meta["roles"])
        fmt = format_bonus(completion)
        tagged = extract_tagged_answer(completion)
        text = (tagged if tagged is not None else completion or "").lower()
        format_ok = tagged is not None

        parsed: dict[str, str] = {}
        for person in roles:
            match = re.search(
                rf"{person.lower()}\s*[:=]\s*(knight|knave)",
                text,
            )
            if match:
                parsed[person] = match.group(1)

        correct = parsed == {k: v for k, v in roles.items()}
        # Partial credit for each correctly labeled person.
        hits = sum(1 for p, r in roles.items() if parsed.get(p) == r)
        partial = hits / max(len(roles), 1)
        reward = fmt + (1.0 if correct else 0.5 * partial)
        return {
            "reward": float(reward),
            "correct": correct,
            "format_ok": format_ok,
            "extracted": parsed,
        }
