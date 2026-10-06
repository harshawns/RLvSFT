#!/usr/bin/env bash
# Feasibility smoke entrypoints for USC CARC (GPU SKU TBD).
# Usage:
#   bash scripts/smoke_carc.sh qwen          # full 4-condition smoke for one alias
#   bash scripts/smoke_carc.sh qwen dry      # data/config validation only
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL="${1:?model alias required: qwen|gemma|afm}"
MODE="${2:-full}"
SEED="${SEED:-566}"

PYTHON="${PYTHON:-python3}"

"$PYTHON" data/build_splits.py --seed "$SEED"

if [[ "$MODE" == "dry" ]]; then
  "$PYTHON" train/sft.py --model "$MODEL" --seed "$SEED" --dry-run
  "$PYTHON" train/grpo.py --model "$MODEL" --seed "$SEED" --dry-run
  "$PYTHON" eval/run_eval.py --model "$MODEL" --seed "$SEED" --oracle --limit 8
  echo "dry smoke ok for $MODEL"
  exit 0
fi

SFT_DIR="outputs/sft_${MODEL}_s${SEED}"
GRPO_DIR="outputs/grpo_${MODEL}_s${SEED}"
SFT_GRPO_DIR="outputs/sft_grpo_${MODEL}_s${SEED}"

"$PYTHON" eval/run_eval.py --model "$MODEL" --condition Base --seed "$SEED" \
  --output-dir "outputs/eval_${MODEL}_Base_s${SEED}"

"$PYTHON" train/sft.py --model "$MODEL" --seed "$SEED" --output-dir "$SFT_DIR"
"$PYTHON" eval/run_eval.py --model "$MODEL" --condition SFT --seed "$SEED" \
  --adapter "$SFT_DIR" --output-dir "outputs/eval_${MODEL}_SFT_s${SEED}"

"$PYTHON" train/grpo.py --model "$MODEL" --seed "$SEED" --output-dir "$GRPO_DIR"
"$PYTHON" eval/run_eval.py --model "$MODEL" --condition GRPO --seed "$SEED" \
  --adapter "$GRPO_DIR" --output-dir "outputs/eval_${MODEL}_GRPO_s${SEED}"

"$PYTHON" train/grpo.py --model "$MODEL" --seed "$SEED" \
  --init-adapter "$SFT_DIR" --condition "SFT→GRPO" --output-dir "$SFT_GRPO_DIR"
"$PYTHON" eval/run_eval.py --model "$MODEL" --condition "SFT→GRPO" --seed "$SEED" \
  --adapter "$SFT_GRPO_DIR" --output-dir "outputs/eval_${MODEL}_SFT_GRPO_s${SEED}"

echo "full smoke finished for $MODEL — fill docs/go_no_go.md"
