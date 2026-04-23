"""v46: Full ensemble evaluation across ALL available onnx models."""

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

    # All available models with their input dims
    onnx_dir = Path("/mnt/c/GitHub/kotonoha-models")
    # Exclude v1-v4 (very old), test models
    candidates = [
        ("v5", 11),
        ("v6", 11),
        ("v7", 11),
        ("v8", 11),
        ("v10", 11),
        ("v11", 11),
        ("v12", 11),
        ("v13", 11),
        ("v14", 11),
        ("v15", 11),
        ("v15b", 11),
        ("v16", 11),
        ("v17", 11),
        ("v18", 11),
        ("v19", 11),
        ("v20", 11),
        ("v24", 11),
        ("v31_soup", 11),
        ("v33", 13),
        ("v38", 14),
        ("v41", 14),
    ]
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    # Load all sessions, probe output dim, discard incompatible
    sessions: dict[str, ort.InferenceSession] = {}
    dims: dict[str, int] = {}
    for name, dim in candidates:
        path = onnx_dir / f"accent_model_{name}.onnx"
        if not path.exists():
            continue
        try:
            s = ort.InferenceSession(str(path), providers=providers)
            inp_shape = s.get_inputs()[0].shape
            actual_dim = inp_shape[1] if len(inp_shape) >= 2 else None
            if isinstance(actual_dim, int):
                dims[name] = actual_dim
            else:
                dims[name] = dim
            sessions[name] = s
            print(f"loaded {name}: dim={dims[name]}")
        except Exception as e:
            print(f"skip {name}: {e}")

    # Precompute softmax per model per utt
    softmax_all: dict[str, list[np.ndarray]] = {n: [] for n in sessions}
    labels_list: list[np.ndarray] = []
    v24_args_list: list[np.ndarray] = []

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

        # v24 first for 14 dim models
        v24_arg = None
        if "v24" in sessions:
            v24_log = sessions["v24"].run(None, {"input": feats13[:, :11]})[0]
            softmax_all["v24"].append(_softmax(v24_log))
            v24_arg = v24_log.argmax(-1)
        v24_args_list.append(v24_arg)

        feats14 = None
        if v24_arg is not None:
            feats14 = np.concatenate(
                [
                    feats13[:, :13],
                    (v24_arg.astype(np.float32) / 20.0).reshape(-1, 1),
                ],
                axis=1,
            )

        for name, sess in sessions.items():
            if name == "v24":
                continue
            d = dims[name]
            if d == 11:
                inp = feats13[:, :11]
            elif d == 13:
                inp = feats13[:, :13]
            elif d == 14 and feats14 is not None:
                inp = feats14
            else:
                softmax_all[name].append(
                    np.zeros((len(ms), NUM_CLASSES), dtype=np.float32)
                )
                continue
            logits = sess.run(None, {"input": inp})[0]
            softmax_all[name].append(_softmax(logits))

    flat_labels = np.concatenate(labels_list)
    total = len(flat_labels)

    # Single model accuracies
    print("\nSingle accuracies (val_split=0):")
    single = {}
    for name, sml in softmax_all.items():
        if not sml:
            continue
        preds = np.concatenate([s.argmax(-1) for s in sml])
        acc = (preds == flat_labels).mean()
        single[name] = acc
        print(f"  {name}: {acc * 100:.2f}%")

    # Sort by accuracy
    sorted_models = sorted(single.items(), key=lambda x: -x[1])
    print(f"\nTop 5: {sorted_models[:5]}")

    # Try ensembles of top N models
    for top_n in [2, 3, 5, 7, 10]:
        members = [n for n, _ in sorted_models[:top_n]]
        avg_preds_list = []
        for i in range(len(labels_list)):
            stacked = np.stack([softmax_all[n][i] for n in members])
            avg = stacked.mean(0)
            avg_preds_list.append(avg.argmax(-1))
        preds = np.concatenate(avg_preds_list)
        acc = (preds == flat_labels).mean()
        print(f"\nTop-{top_n} uniform avg: {acc * 100:.2f}%")

    # Greedy selection
    best_members = [sorted_models[0][0]]
    best_acc = sorted_models[0][1]
    print(f"\nGreedy start: {best_members} -> {best_acc * 100:.2f}%")
    remaining = [n for n, _ in sorted_models[1:]]
    while remaining:
        best_candidate = None
        best_new_acc = best_acc
        for cand in remaining:
            trial = best_members + [cand]
            avg_preds_list = []
            for i in range(len(labels_list)):
                stacked = np.stack([softmax_all[n][i] for n in trial])
                avg = stacked.mean(0)
                avg_preds_list.append(avg.argmax(-1))
            preds = np.concatenate(avg_preds_list)
            acc = (preds == flat_labels).mean()
            if acc > best_new_acc:
                best_new_acc = acc
                best_candidate = cand
        if best_candidate is None:
            break
        best_members.append(best_candidate)
        best_acc = best_new_acc
        remaining.remove(best_candidate)
        print(
            f"  added {best_candidate}: {best_acc * 100:.2f}% "
            f"(members: {len(best_members)})"
        )

    print(f"\nBest greedy ensemble: {'+'.join(best_members)} = {best_acc * 100:.2f}%")
    print(f"Total morphemes: {total}")


if __name__ == "__main__":
    main()
