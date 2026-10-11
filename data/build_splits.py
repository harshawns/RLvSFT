from envs.countdown import generate_problem
from envs.countdown import build_prompt

def build_splits_countdown(num_problems, split, **difficulty):
    seed_starts = {
        "train": 0,
        "val": 1_000_000,
        "test": 2_000_000
    }

    rows = []
    seen = set()
    seed = seed_starts[split]

    while len(rows) < num_problems:
        problem = generate_problem(seed)
        seed += 1

        key = (
            tuple(sorted(problem["numbers"])),
            problem["target"]
        )

        if key in seen:
            continue
        if key["target"] < difficulty["min_target"] or key["target"] > difficulty["max_target"]:
            continue
        if problem["numbers"][0] < difficulty["min_number"] or problem["numbers"][-1] > difficulty["max_number"]:
            continue
        seen.add(key)

        rows.append({
            "prompt": build_prompt(problem),
            "numbers": problem["numbers"],
            "target": problem["target"],
            "target_eq": problem["target_eq"]
        })

    return rows
