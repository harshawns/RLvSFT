# RLvSFT — feasibility harness

Tiny end-to-end smoke: **one dataset** (`target_expr`) × **three base models** × **Base / SFT / GRPO / SFT→GRPO** × **seed 566**, on **CUDA Linux** (USC CARC). Not the full study.

## Models

| Alias | Hugging Face ID |
| --- | --- |
| `qwen` | `Qwen/Qwen3.5-4B-Base` |
| `gemma` | `google/gemma-3n-E4B` (pretrained, **not** `-it`) |
| `afm` | `arcee-ai/AFM-4.5B-Base` |

Gemma is **gated**. Accept terms at https://huggingface.co/google/gemma-3n-E4B and set `HF_TOKEN` (or `huggingface-cli login`) before downloading. Loaders and smoke scripts are implemented even if the gate is not accepted yet.

## Layout

```
envs/           target_expr verifier (+ countdown kept for ablation)
data/           build_splits.py → frozen JSONL/parquet + oracle traces
train/          models.py, sft.py, grpo.py, logging_utils.py, configs/
train/grpo_ablation.py   fixed-geometry G ablation (from main; not the smoke)
eval/           run_eval.py, metrics.py
tests/          test_verifier.py
scripts/        smoke_carc.sh, carc_job_example.slurm
docs/go_no_go.md
```

## Install (CARC / CUDA)

```bash
# module load ... cuda ... python   # site-specific
python3 -m venv rl && source rl/bin/activate
pip install -U pip
# Match your CUDA module, e.g. cu124:
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

export HF_HOME=$PWD/.hf_cache
export HF_TOKEN=...   # required for Gemma; recommended for all downloads
# huggingface-cli login
```

Put caches on fast local / scratch disk, not a small home quota.

## Data + tests (CPU)

```bash
python data/build_splits.py --seed 566
python -m pytest tests/test_verifier.py -q
# oracle eval (no GPU weights):
python eval/run_eval.py --model qwen --oracle
```

## Smoke on GPU

```bash
# Dry config/data check (no weight download required beyond optional):
bash scripts/smoke_carc.sh qwen dry

# Full 4-condition smoke for one model:
bash scripts/smoke_carc.sh qwen
# or: gemma / afm

# Individual entrypoints:
python train/sft.py --model qwen
python train/grpo.py --model qwen
python train/grpo.py --model qwen --init-adapter outputs/sft_qwen_s566 --condition 'SFT→GRPO'
python eval/run_eval.py --model qwen --condition Base
python eval/run_eval.py --model qwen --adapter outputs/sft_qwen_s566 --condition SFT
```

Example Slurm skeleton: `scripts/carc_job_example.slurm` (partition/gres/account are placeholders — GPU SKU TBD).

After smokes, fill `docs/go_no_go.md`.

## Shared vs per-model config

- Shared schedules: `train/configs/sft_smoke.yaml`, `train/configs/grpo_smoke.yaml`
- Per-model quirks only: `train/configs/{qwen,gemma,afm}.yaml`
