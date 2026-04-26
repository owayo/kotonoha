#!/bin/bash
# v54: Train v38 with multiple val_split_seed values for diverse ensemble
# Each model trained on different 90/10 split, then evaluated on val_split=0
set -e
cd /mnt/c/GitHub/kotonoha/training
for SPLIT in 1 2 3; do
    echo "=== val_split=${SPLIT} ==="
    rm -rf /tmp/v54_states_${SPLIT}
    mkdir -p /tmp/v54_states_${SPLIT}
    uv run python train_onnx_v38.py \
        --output /mnt/c/GitHub/kotonoha-models/accent_model_v54_split${SPLIT}.onnx \
        --seeds 0 \
        --val-split-seed ${SPLIT} \
        --teacher-cache /tmp/v54_teacher_split${SPLIT}.pt \
        --state-dir /tmp/v54_states_${SPLIT} >> /tmp/v54_train.log 2>&1
done
echo "=== all v54 splits done ==="
