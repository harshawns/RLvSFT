import os
import json
from envs.countdown import CountdownEnv


def build_splits_countdown(path,num_problems, split, seen=None, **difficulty):
    env = CountdownEnv()
    seed_starts = {
        "train": 0,
        "val": 1_000_000,
        "test": 2_000_000
    }
    if seen is None:
        seen = set()
    rows = []
    seed = seed_starts[split]

    while len(rows) < num_problems:
        problem = env.generate_problem(seed, **difficulty)
        seed += 1
        key = (
            tuple(sorted(problem["numbers"])),
            problem["target"]
        )

        if key in seen:
            continue
        seen.add(key)

        rows.append({
            "prompt": env.build_prompt(problem["numbers"], problem["target"]),
            "numbers": problem["numbers"],
            "target": problem["target"],
            "target_eq": problem["target_eq"]
        })
    # at the path create a file with the rows
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return path, rows


if __name__ == "__main__":
    path = "data/countdown/train.jsonl"
    num_problems_train = 10000
    num_problems_val = 1000
    num_problems_test = 1000
    seen = set()
    difficulty = {
        "min_number": 1,
        "max_number": 100,
        "min_target": 1,
        "max_target": 1000
    }

    train_data= build_splits_countdown(path, num_problems, split, seen, **difficulty)
    val_data= build_splits_countdown("data/countdown/val.jsonl", num_problems_val, "val", seen, **difficulty)
    test_data= build_splits_countdown("data/countdown/test.jsonl", num_problems, "test", seen, **difficulty)
