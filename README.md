# RLvSFT

Compare **SFT** (teacher traces) vs **GRPO** (reward from env scorers) across synthetic reasoning / coding tasks.

## Layout

```
envs/        base.py, countdown.py, knights_knaves.py, zebra.py, math_task.py, mbpp.py
data/        build_splits.py  → frozen parquet files in data/splits/
traces/      gen_sft_traces.py
eval/        run_eval.py, metrics.py
train/       sft.py, grpo.py, configs/*.yaml
tests/       test_scorers.py
```

## Setup (Apple Silicon / MLX)

```bash
./setup_mlx_env.sh
source rl/bin/activate
```

## Pipeline

```bash
# 1) Freeze deterministic splits
python data/build_splits.py

# 2) Build oracle (or teacher-model) SFT traces
python traces/gen_sft_traces.py \
  --env countdown \
  --split-path data/splits/countdown_train.parquet \
  --out traces/out/countdown_train_oracle.jsonl \
  --teacher oracle

# 3) Evaluate scorers / predictions
python eval/run_eval.py --split-path data/splits/countdown_test.parquet

# 4) Train
python train/sft.py --config train/configs/sft_default.yaml
python train/grpo.py --config train/configs/grpo_default.yaml
```

## Tests

```bash
python -m unittest tests.test_scorers -v
```
