"""v46: Weighted ensemble + Test-Time Augmentation evaluation.

Try:
  - Weighted softmax averaging (higher weights for stronger models)
  - TTA via feature noise injection (multiple passes)
  - Per-position confidence-based model selection
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import onnxruntime as ort
from train_onnx_v38 import (
    NUM_CLASSES,
    _enrich_utterances,
    _extract_morpheme_features,
    _load_accent_dicts,
)


def _softmax(x: np.ndarray) -> np.ndarray:
    mx = np.max(x, axis=-1, keepdims=True)
    e = np.exp(x - mx)
    return e / np.sum(e, axis=-1, keepdims=True)


def _inference(sess, feats: np.ndarray) -> np.ndarray:
    """Run onnx inference and return softmax."""
    logits = sess.run(None, {"input": feats})[0]
    return _softmax(logits)


def _eval_preds(preds: np.ndarray, labels: np.ndarray) -> float:
    return (preds == labels).mean()


def main() -> None:
    dict_paths = [
        Path("/mnt/c/GitHub/kotonoha/data/accent_dict.csv"),
        Path("/mnt/c/GitHub/kotonoha-training-data/train/accent_dict.csv"),
    ]
    accent_dict = _load_accent_dicts(dict_paths)

    with open(
        "/home/owayo/kotonoha-training/data/jsut_accent_data_v3.json",
        encoding="utf-8",
    ) as f:
        jsut = json.load(f)["utterances"]
    _enrich_utterances(jsut, accent_dict)
    random.seed(0)
    idx = list(range(len(jsut)))
    random.shuffle(idx)
    val_size = int(len(idx) * 0.1)
    val_idx = set(idx[:val_size])
    val_utts = [u for i, u in enumerate(jsut) if i in val_idx]

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess_v24 = ort.InferenceSession(
        "/mnt/c/GitHub/kotonoha-models/accent_model_v24.onnx", providers=providers
    )
    sess_v38 = ort.InferenceSession(
        "/mnt/c/GitHub/kotonoha-models/accent_model_v38.onnx", providers=providers
    )
    sess_v41 = ort.InferenceSession(
        "/mnt/c/GitHub/kotonoha-models/accent_model_v41.onnx", providers=providers
    )

    # Collect softmax per model + labels
    smx_v24, smx_v38, smx_v41 = [], [], []
    labels_list = []

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
        labs = np.array(
            [min(m.get("accent_type", 0), NUM_CLASSES - 1) for m in ms],
            dtype=np.int64,
        )
        labels_list.append(labs)

        v24_smx = _inference(sess_v24, feats13[:, :11])
        smx_v24.append(v24_smx)
        v24_arg = v24_smx.argmax(-1)
        feats14 = np.concatenate(
            [feats13[:, :13], (v24_arg.astype(np.float32) / 20.0).reshape(-1, 1)],
            axis=1,
        )
        smx_v38.append(_inference(sess_v38, feats14))
        smx_v41.append(_inference(sess_v41, feats14))

    flat_labels = np.concatenate(labels_list)
    total = len(flat_labels)

    def _pred_acc(weight_v24: float, weight_v38: float, weight_v41: float) -> float:
        wsum = weight_v24 + weight_v38 + weight_v41
        w24 = weight_v24 / wsum
        w38 = weight_v38 / wsum
        w41 = weight_v41 / wsum
        preds_list = []
        for i in range(len(labels_list)):
            avg = w24 * smx_v24[i] + w38 * smx_v38[i] + w41 * smx_v41[i]
            preds_list.append(avg.argmax(-1))
        preds = np.concatenate(preds_list)
        return _eval_preds(preds, flat_labels)

    # Single-model baselines
    print(
        f"v24 only: {_pred_acc(1.0, 0.0, 0.0) * 100:.2f}%   "
        f"v38 only: {_pred_acc(0.0, 1.0, 0.0) * 100:.2f}%   "
        f"v41 only: {_pred_acc(0.0, 0.0, 1.0) * 100:.2f}%"
    )

    # Grid search over weights
    best_acc = 0.0
    best_w = (0.0, 0.0, 0.0)
    for w24 in range(0, 11):
        for w38 in range(0, 11 - w24):
            w41 = 10 - w24 - w38
            a = _pred_acc(w24, w38, w41)
            if a > best_acc:
                best_acc = a
                best_w = (w24, w38, w41)
    print(
        f"\nBest weighted: v24={best_w[0] / 10}, v38={best_w[1] / 10}, "
        f"v41={best_w[2] / 10} -> {best_acc * 100:.2f}%"
    )

    # Per-position confidence selection (oracle-ish)
    preds_by_max_conf = []
    for i in range(len(labels_list)):
        # stack [3, seq, 21]
        stacked = np.stack([smx_v24[i], smx_v38[i], smx_v41[i]])
        # per-position: pick model with highest max-softmax-prob
        max_probs = stacked.max(-1)  # [3, seq]
        best_model = max_probs.argmax(0)  # [seq]
        # gather preds
        per_model_preds = stacked.argmax(-1)  # [3, seq]
        n_seq = best_model.shape[0]
        preds_seq = per_model_preds[best_model, np.arange(n_seq)]
        preds_by_max_conf.append(preds_seq)
    acc_conf = (np.concatenate(preds_by_max_conf) == flat_labels).mean()
    print(f"Per-position max-conf voting: {acc_conf * 100:.2f}%")

    # Majority vote (3 models) with argmax
    preds_vote = []
    for i in range(len(labels_list)):
        stacked = np.stack(
            [smx_v24[i].argmax(-1), smx_v38[i].argmax(-1), smx_v41[i].argmax(-1)]
        )  # [3, seq]
        n_seq = stacked.shape[1]
        vote = np.empty(n_seq, dtype=np.int64)
        for j in range(n_seq):
            u, c = np.unique(stacked[:, j], return_counts=True)
            vote[j] = u[np.argmax(c)]
        preds_vote.append(vote)
    acc_vote = (np.concatenate(preds_vote) == flat_labels).mean()
    print(f"Majority vote: {acc_vote * 100:.2f}%")

    print(f"\nTotal morphemes: {total}")


if __name__ == "__main__":
    main()
