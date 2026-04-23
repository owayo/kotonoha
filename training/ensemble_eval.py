"""v44: Ensemble evaluation across multiple ONNX models.

Tries combinations of (v24, v33, v38, v41) to find best ensemble.
Inputs with different dim are handled:
  - v24: 11 dim
  - v33: 13 dim (uses feats13)
  - v38: 14 dim (feats13 + v24_argmax/20)
  - v41: 14 dim (same layout as v38)
"""

from __future__ import annotations

import itertools
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

    jsut_path = "/home/owayo/kotonoha-training/data/jsut_accent_data_v3.json"
    with open(jsut_path, encoding="utf-8") as f:
        jsut = json.load(f)["utterances"]
    _enrich_utterances(jsut, accent_dict)
    random.seed(0)
    idx = list(range(len(jsut)))
    random.shuffle(idx)
    val_size = int(len(idx) * 0.1)
    val_idx = set(idx[:val_size])
    val_utts = [u for i, u in enumerate(jsut) if i in val_idx]
    print(f"val utts: {len(val_utts)}")

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    model_info = [
        ("v24", "/mnt/c/GitHub/kotonoha-models/accent_model_v24.onnx", 11),
        ("v33", "/mnt/c/GitHub/kotonoha-models/accent_model_v33.onnx", 13),
        ("v38", "/mnt/c/GitHub/kotonoha-models/accent_model_v38.onnx", 14),
        ("v41", "/mnt/c/GitHub/kotonoha-models/accent_model_v41.onnx", 14),
    ]
    sessions = {}
    for name, path, _dim in model_info:
        try:
            sessions[name] = ort.InferenceSession(path, providers=providers)
            print(f"loaded {name}: {path}")
        except Exception as e:
            print(f"skip {name}: {e}")

    # Precompute softmax for each utterance for each model
    all_softmax: dict[str, list[np.ndarray]] = {n: [] for n in sessions}
    all_labels: list[np.ndarray] = []

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
        labels = np.array(
            [min(m.get("accent_type", 0), NUM_CLASSES - 1) for m in ms],
            dtype=np.int64,
        )
        all_labels.append(labels)

        # v24 first (needed as input for v38/v41)
        if "v24" in sessions:
            v24_log = sessions["v24"].run(None, {"input": feats13[:, :11]})[0]
            all_softmax["v24"].append(_softmax(v24_log))
            v24_arg = v24_log.argmax(-1)
        else:
            v24_arg = None

        if "v33" in sessions:
            v33_log = sessions["v33"].run(None, {"input": feats13[:, :13]})[0]
            all_softmax["v33"].append(_softmax(v33_log))

        feats14: np.ndarray | None = None
        if v24_arg is not None:
            feats14 = np.concatenate(
                [feats13[:, :13], (v24_arg.astype(np.float32) / 20.0).reshape(-1, 1)],
                axis=1,
            )

        if "v38" in sessions and feats14 is not None:
            v38_log = sessions["v38"].run(None, {"input": feats14})[0]
            all_softmax["v38"].append(_softmax(v38_log))

        if "v41" in sessions and feats14 is not None:
            v41_log = sessions["v41"].run(None, {"input": feats14})[0]
            all_softmax["v41"].append(_softmax(v41_log))

    # Concatenate labels for per-position metrics
    flat_labels = np.concatenate(all_labels)
    total = len(flat_labels)
    print(f"\nTotal morphemes: {total}")

    # Single-model accuracies
    print("\nSingle-model accuracies:")
    single_accs = {}
    for name, smx_list in all_softmax.items():
        if not smx_list:
            continue
        preds = np.concatenate([sm.argmax(-1) for sm in smx_list])
        acc = (preds == flat_labels).mean()
        single_accs[name] = acc
        print(f"  {name}: {acc * 100:.2f}%")

    # All non-empty ensembles
    names = [n for n in all_softmax if all_softmax[n]]
    print("\nEnsemble accuracies (equal-weight softmax avg):")
    best_acc = max(single_accs.values())
    best_combo: tuple[str, ...] = ()
    for r in range(2, len(names) + 1):
        for combo in itertools.combinations(names, r):
            # Per-utt average softmax; then flatten
            avg_preds_list = []
            for i in range(len(all_labels)):
                stacked = np.stack([all_softmax[n][i] for n in combo])  # [R, seq, 21]
                avg = stacked.mean(0)
                avg_preds_list.append(avg.argmax(-1))
            preds = np.concatenate(avg_preds_list)
            acc = (preds == flat_labels).mean()
            tag = "+".join(combo)
            marker = ""
            if acc > best_acc:
                best_acc = acc
                best_combo = combo
                marker = " *"
            print(f"  {tag}: {acc * 100:.2f}%{marker}")

    print(
        f"\nBest ensemble: {'+'.join(best_combo) or 'single'} = {best_acc * 100:.2f}%"
    )


if __name__ == "__main__":
    main()
