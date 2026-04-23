"""TTA (Test-Time Augmentation) evaluation on v24."""

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
    sess = ort.InferenceSession(
        "/mnt/c/GitHub/kotonoha-models/accent_model_v24.onnx", providers=providers
    )

    # Baseline
    labels_all = []
    preds_base = []
    smx_base_list = []
    feats_list = []
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
        labels_all.append(labs)
        feats11 = feats13[:, :11]
        feats_list.append(feats11)
        smx = _softmax(sess.run(None, {"input": feats11})[0])
        smx_base_list.append(smx)
        preds_base.append(smx.argmax(-1))
    flat_labels = np.concatenate(labels_all)
    acc_base = (np.concatenate(preds_base) == flat_labels).mean()
    print(f"Baseline (no TTA): {acc_base * 100:.2f}%")

    # TTA: add Gaussian noise to continuous features (dim 5..10) multiple times
    for sigma in [0.01, 0.02, 0.03, 0.05]:
        for n_aug in [4, 8, 16]:
            rng = np.random.default_rng(0)
            smx_avg_list = []
            for feats_orig in feats_list:
                smx_sum = np.zeros((feats_orig.shape[0], NUM_CLASSES), dtype=np.float32)
                # Include original
                smx_sum += _softmax(sess.run(None, {"input": feats_orig})[0])
                for _ in range(n_aug):
                    feats_aug = feats_orig.copy()
                    noise = rng.normal(0, sigma, size=feats_aug[:, 5:].shape).astype(
                        np.float32
                    )
                    feats_aug[:, 5:] += noise
                    smx_sum += _softmax(sess.run(None, {"input": feats_aug})[0])
                smx_avg = smx_sum / (1 + n_aug)
                smx_avg_list.append(smx_avg)
            preds_tta = np.concatenate([s.argmax(-1) for s in smx_avg_list])
            acc_tta = (preds_tta == flat_labels).mean()
            print(
                f"TTA sigma={sigma} n_aug={n_aug}: {acc_tta * 100:.2f}% "
                f"(delta {(acc_tta - acc_base) * 100:+.2f}%)"
            )


if __name__ == "__main__":
    main()
