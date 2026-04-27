#!/usr/bin/env bash
# 5-fold disjoint K-fold CV を順次学習する。
# 既に学習済み fold は state file の存在で skip。

set -euo pipefail

cd "$(dirname "$0")"

NUM_FOLDS=${1:-5}
SEEDS=${2:-0}
MODELS_DIR=/mnt/c/GitHub/kotonoha-models
STATES_DIR=/tmp/v59_states
TEACHER_MODEL=$MODELS_DIR/accent_model_v24.onnx

mkdir -p "$STATES_DIR"

for FOLD in $(seq 0 $((NUM_FOLDS - 1))); do
  OUTPUT="$MODELS_DIR/accent_model_v59_fold$FOLD.onnx"
  STATE_FILE="$STATES_DIR/state_000.pt"
  TEACHER_CACHE="/tmp/v59_teacher_logits_fold$FOLD.pt"
  LOG="/tmp/v59_fold$FOLD.log"

  echo "===== fold $FOLD ====="
  echo "Output: $OUTPUT"
  if [[ -f "$OUTPUT" ]]; then
    echo "  Already exists, skipping."
    continue
  fi

  # state_dir を fold ごとに分けて衝突を避ける
  FOLD_STATES_DIR="$STATES_DIR/fold$FOLD"
  mkdir -p "$FOLD_STATES_DIR"

  PYTHONUNBUFFERED=1 uv run python -u train_onnx_v59.py \
    --fold-id "$FOLD" \
    --seeds "$SEEDS" \
    --state-dir "$FOLD_STATES_DIR" \
    --teacher-model "$TEACHER_MODEL" \
    --teacher-cache "$TEACHER_CACHE" \
    --output "$OUTPUT" 2>&1 | tee -a "$LOG"
done

echo
echo "===== ALL FOLDS DONE ====="
echo "Run OOF eval: uv run python v59_oof_eval.py"
