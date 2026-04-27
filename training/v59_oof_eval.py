"""v59 OOF (Out-Of-Fold) evaluation across 5 disjoint folds.

各 fold で学習した model を、その fold の val set 上で eval する (OOF prediction)。
全 5 fold の OOF prediction を集計すれば、データ全体での真の精度が得られる
(disjoint K-fold で leak なし)。

Usage:
  uv run python v59_oof_eval.py --models-dir /mnt/c/GitHub/kotonoha-models \
    --num-folds 5 --fold-base-seed 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from train_onnx_v59 import (
    NUM_CLASSES,
    _enrich_utterances,
    _extract_morpheme_features,
    _kfold_val_idx,
    _load_accent_dicts,
    _load_dotenv,
)


def _accuracy(preds: list[np.ndarray], labels: list[np.ndarray]) -> tuple[int, int]:
    correct = sum(int((p == la).sum()) for p, la in zip(preds, labels, strict=True))
    total = sum(len(la) for la in labels)
    return correct, total


def main() -> None:
    """Run OOF evaluation across all folds and print per-fold + aggregate accuracy."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", default="/mnt/c/GitHub/kotonoha-models")
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--fold-base-seed", type=int, default=0)
    parser.add_argument(
        "--teacher-model",
        default="/mnt/c/GitHub/kotonoha-models/accent_model_v24.onnx",
    )
    parser.add_argument(
        "--model-name-template",
        default="accent_model_v59_fold{fold}.onnx",
        help="ONNX file name pattern (use {fold} placeholder)",
    )
    args = parser.parse_args()

    _load_dotenv()
    dict_paths = [
        Path("/mnt/c/GitHub/kotonoha/data/accent_dict.csv"),
        Path("/mnt/c/GitHub/kotonoha-training-data/train/accent_dict.csv"),
    ]
    accent_dict = _load_accent_dicts(dict_paths)

    jsut_path = "/home/owayo/kotonoha-training/data/jsut_accent_data_v3.json"
    with open(jsut_path, encoding="utf-8") as f:
        jsut = json.load(f)["utterances"]
    _enrich_utterances(jsut, accent_dict)
    print(f"Loaded {len(jsut)} JSUT utterances")

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess_v24 = ort.InferenceSession(args.teacher_model, providers=providers)
    print(f"Teacher: {args.teacher_model}")

    total_correct = 0
    total_morph = 0
    per_fold_results: list[tuple[int, float, int, int]] = []
    models_dir = Path(args.models_dir)

    for fold_id in range(args.num_folds):
        model_path = models_dir / args.model_name_template.format(fold=fold_id)
        if not model_path.exists():
            print(f"\n[fold {fold_id}] SKIP: {model_path} not found")
            continue
        sess = ort.InferenceSession(str(model_path), providers=providers)
        val_idx = _kfold_val_idx(
            len(jsut), fold_id, args.num_folds, args.fold_base_seed
        )
        val_utts = [u for i, u in enumerate(jsut) if i in val_idx]

        preds_list: list[np.ndarray] = []
        labels_list: list[np.ndarray] = []
        for utt in val_utts:
            ms = utt.get("morphemes", [])
            if not ms:
                continue
            n = len(ms)
            feats13 = np.array(
                [
                    _extract_morpheme_features(m, i / max(n - 1, 1))
                    for i, m in enumerate(ms)
                ],
                dtype=np.float32,
            )
            v24_arg = sess_v24.run(None, {"input": feats13[:, :11]})[0].argmax(-1)
            feats14 = np.concatenate(
                [feats13[:, :13], (v24_arg.astype(np.float32) / 20.0).reshape(-1, 1)],
                axis=1,
            )
            log = sess.run(None, {"input": feats14})[0]
            preds_list.append(log.argmax(-1))
            labels = np.array(
                [min(m.get("accent_type", 0), NUM_CLASSES - 1) for m in ms],
                dtype=np.int64,
            )
            labels_list.append(labels)

        correct, total = _accuracy(preds_list, labels_list)
        acc = correct / total if total else 0.0
        per_fold_results.append((fold_id, acc, correct, total))
        total_correct += correct
        total_morph += total
        print(
            f"[fold {fold_id}] {len(val_utts)} utts, {total} morphemes: "
            f"{acc * 100:.2f}% ({correct}/{total})"
        )

    print("\n=== OOF Aggregate ===")
    for fold_id, acc, correct, total in per_fold_results:
        print(f"  fold {fold_id}: {acc * 100:.2f}% ({correct}/{total})")
    if total_morph > 0:
        agg_acc = total_correct / total_morph
        print(
            f"  Aggregate (no leak): {agg_acc * 100:.2f}% "
            f"({total_correct}/{total_morph})"
        )


if __name__ == "__main__":
    main()
