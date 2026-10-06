#!/bin/bash
#SBATCH --job-name=rlvsft-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Example only — adjust partition / gres / account for your CARC allocation.
# GPU SKU is TBD; do not hard-code A100/L40/etc. in harness code.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs outputs

# module purge
# module load gcc/12.3.0 cuda/12.4 python/3.12   # example — use your site modules

python3 -m venv rl
source rl/bin/activate
python -m pip install -U pip
# Match the CUDA module you loaded:
# pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
# export HF_TOKEN=...   # or: huggingface-cli login
# Gemma gate: https://huggingface.co/google/gemma-3n-E4B

MODEL="${MODEL:-qwen}"
bash scripts/smoke_carc.sh "$MODEL"
