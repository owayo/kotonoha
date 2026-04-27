"""v59 5-fold ensemble evaluation across the entire dataset.

各 utt は OOF predict (自分の fold の model のみで予測) → 集計、
または別 split 上で 5 fold 全 model の ensemble を評価する。

3 つの評価モード:
  1. OOF aggregate: 各 fold の utt を対応 model で predict → leak なし
  2. 5-fold ensemble (no-leak per utt): 各 utt は 4 model (own fold を除外) の ensemble
  3. Full 5-fold ensemble (leak あり): 全 5 model の ensemble
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


def _softmax(x: np.ndarray) -> np.ndarray:
    mx = np.max(x, axis=-1, keepdims=True)
    e = np.exp(x - mx)
    return e / np.sum(e, axis=-1, keepdims=True)


def main() -> None:
    """Run 5-fold OOF + ensemble evaluation across the full dataset."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", default="/mnt/c/GitHub/kotonoha-models")
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--fold-base-seed", type=int, default=0)
    parser.add_argument(
        "--teacher-model",
        default="/mnt/c/GitHub/kotonoha-models/accent_model_v24.onnx",
    )
    parser.add_argument(
        "--model-name-template", default="accent_model_v59_fold{fold}.onnx"
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
    n_utts = len(jsut)
    print(f"Loaded {n_utts} JSUT utterances")

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess_v24 = ort.InferenceSession(args.teacher_model, providers=providers)
    models_dir = Path(args.models_dir)
    fold_sessions: dict[int, ort.InferenceSession] = {}
    for fold_id in range(args.num_folds):
        path = models_dir / args.model_name_template.format(fold=fold_id)
        if not path.exists():
            print(f"  fold {fold_id}: NOT FOUND ({path})")
            continue
        fold_sessions[fold_id] = ort.InferenceSession(str(path), providers=providers)
        print(f"  fold {fold_id}: loaded")
    if not fold_sessions:
        print("No fold models found. Exiting.")
        return

    # 各 utt の所属 fold を計算
    utt_fold: list[int] = [-1] * n_utts
    for fold_id in range(args.num_folds):
        val_idx = _kfold_val_idx(n_utts, fold_id, args.num_folds, args.fold_base_seed)
        for i in val_idx:
            utt_fold[i] = fold_id

    # 全 utt に対し softmax を計算 (各 fold model)
    # 大きいので、pre-allocate せずに dict で保持
    n_morph_total = 0
    flat_labels: list[int] = []
    # per_fold_softmax[fold_id][utt_index] = np.ndarray [seq, NUM_CLASSES]
    per_fold_softmax: dict[int, list[np.ndarray | None]] = {
        f: [None] * n_utts for f in fold_sessions
    }
    # Also store v24 argmax features per utt
    feats14_cache: list[np.ndarray | None] = [None] * n_utts
    labels_cache: list[np.ndarray | None] = [None] * n_utts

    print("\nComputing per-utt softmax for each fold model...")
    for utt_i, utt in enumerate(jsut):
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
        labels = np.array(
            [min(m.get("accent_type", 0), NUM_CLASSES - 1) for m in ms],
            dtype=np.int64,
        )
        labels_cache[utt_i] = labels
        flat_labels.extend(labels.tolist())
        n_morph_total += len(labels)
        v24_arg = sess_v24.run(None, {"input": feats13[:, :11]})[0].argmax(-1)
        feats14 = np.concatenate(
            [feats13[:, :13], (v24_arg.astype(np.float32) / 20.0).reshape(-1, 1)],
            axis=1,
        )
        feats14_cache[utt_i] = feats14
        for fold_id, sess in fold_sessions.items():
            log = sess.run(None, {"input": feats14})[0]
            per_fold_softmax[fold_id][utt_i] = _softmax(log)
        if (utt_i + 1) % 1000 == 0:
            print(f"  processed {utt_i + 1}/{n_utts}")

    print(f"\nTotal morphemes: {n_morph_total}")

    # 1. OOF aggregate: own fold model only
    print("\n--- 1. OOF aggregate (own fold model only, no leak) ---")
    correct = 0
    total = 0
    for utt_i in range(n_utts):
        f = utt_fold[utt_i]
        if f not in fold_sessions or labels_cache[utt_i] is None:
            continue
        sm = per_fold_softmax[f][utt_i]
        if sm is None:
            continue
        preds = sm.argmax(-1)
        labels = labels_cache[utt_i]
        correct += int((preds == labels).sum())
        total += len(labels)
    print(f"  OOF: {correct / total * 100:.2f}% ({correct}/{total})")

    # 2. 5-fold ensemble per utt, OOF (own fold は除外)
    print("\n--- 2. 5-fold ensemble per utt (exclude own fold, no leak) ---")
    correct = 0
    total = 0
    for utt_i in range(n_utts):
        f = utt_fold[utt_i]
        if labels_cache[utt_i] is None:
            continue
        # Average over all OTHER folds (each model has utt_i in its train)
        # Only include folds with model
        other_folds = [fid for fid in fold_sessions if fid != f]
        if not other_folds:
            continue
        sms = [per_fold_softmax[fid][utt_i] for fid in other_folds]
        sms = [s for s in sms if s is not None]
        if not sms:
            continue
        avg = np.stack(sms).mean(0)
        preds = avg.argmax(-1)
        labels = labels_cache[utt_i]
        correct += int((preds == labels).sum())
        total += len(labels)
    print(f"  Excl-own-fold ensemble: {correct / total * 100:.2f}% ({correct}/{total})")
    print("  (Note: 4-of-5 models LEAK on this utt; only 1 utt is OOF for each)")

    # 3. Full ensemble (all 5 folds, leak)
    print("\n--- 3. Full ensemble (all folds, leak) ---")
    correct = 0
    total = 0
    for utt_i in range(n_utts):
        if labels_cache[utt_i] is None:
            continue
        sms = [per_fold_softmax[fid][utt_i] for fid in fold_sessions]
        sms = [s for s in sms if s is not None]
        if not sms:
            continue
        avg = np.stack(sms).mean(0)
        preds = avg.argmax(-1)
        labels = labels_cache[utt_i]
        correct += int((preds == labels).sum())
        total += len(labels)
    print(f"  Full ensemble: {correct / total * 100:.2f}% ({correct}/{total})")


if __name__ == "__main__":
    main()
