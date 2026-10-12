import json
import os

from envs.countdown import CountdownEnv

DEFAULT_SEED_STARTS = {
    "train": 0,
    "val": 1_000_000,
    "test": 2_000_000,
}


def write_countdown_meta(
    path,
    *,
    difficulty,
    sizes,
    seed_starts=None,
    generator="countdown",
):
    """
    Write meta.json describing how the countdown splits were produced.

    Example shape:
      {
        "generator": "countdown",
        "difficulty": {...},
        "seed_starts": {"train": 0, "val": 1000000, "test": 2000000},
        "sizes": {"train": 7000, "val": 500, "test": 1000}
      }
    """
    if seed_starts is None:
        seed_starts = dict(DEFAULT_SEED_STARTS)

    meta = {
        "generator": generator,
        "difficulty": dict(difficulty),
        "seed_starts": dict(seed_starts),
        "sizes": dict(sizes),
    }

    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d)
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    return path


def build_splits_countdown(path, num_problems, split, seen=None, seed_starts=None, **difficulty):
    env = CountdownEnv()
    if seed_starts is None:
        seed_starts = dict(DEFAULT_SEED_STARTS)
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
    return rows, seed_starts


if __name__ == "__main__":
    path1 = "data/countdown/train.parquet"
    path2 = "data/countdown/val.parquet"
    path3 = "data/countdown/test.parquet"
    num_problems_train = 10000
    num_problems_val = 500
    num_problems_test = 1000
    seen = set()
    difficulty = {
        "min_number": 1,
        "max_number": 100,
        "min_target": 1,
        "max_target": 1000
    }

    train_data, seed_starts = build_splits_countdown(
        path1, num_problems_train, "train", seen, **difficulty
    )
    val_data, _ = build_splits_countdown(
        path2, num_problems_val, "val", seen, seed_starts=seed_starts, **difficulty
    )
    test_data, _ = build_splits_countdown(
        path3, num_problems_test, "test", seen, seed_starts=seed_starts, **difficulty
    )
    write_countdown_meta(
        "data/countdown/meta.json",
        difficulty={
            "num_numbers": 4,
            **difficulty,
        },
        seed_starts=seed_starts,
        sizes={
            "train": num_problems_train,
            "val": num_problems_val,
            "test": num_problems_test,
        },
    )
