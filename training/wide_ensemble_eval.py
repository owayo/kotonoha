"""Wide ensemble: all 21 models on CPU (since GPU busy)."""

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

    onnx_dir = Path("/mnt/c/GitHub/kotonoha-models")
    providers = ["CPUExecutionProvider"]  # GPU busy

    # Only strong models
    candidates = [
        "v20",
        "v24",
        "v17",
        "v13",
        "v14",
        "v19",
        "v18",
        "v15",
        "v15b",
        "v16",
        "v12",
        "v38",
        "v41",
    ]
    sessions: dict[str, ort.InferenceSession] = {}
    dims: dict[str, int] = {}
    for name in candidates:
        path = onnx_dir / f"accent_model_{name}.onnx"
        if not path.exists():
            continue
        s = ort.InferenceSession(str(path), providers=providers)
        inp_shape = s.get_inputs()[0].shape
        actual_dim = inp_shape[1] if len(inp_shape) >= 2 else None
        dims[name] = actual_dim if isinstance(actual_dim, int) else 11
        sessions[name] = s

    # Compute softmax for all models
    softmax_all: dict[str, list[np.ndarray]] = {n: [] for n in sessions}
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
        labs = np.array(
            [min(m.get("accent_type", 0), NUM_CLASSES - 1) for m in ms], dtype=np.int64
        )
        labels_list.append(labs)

        v24_arg = None
        if "v24" in sessions:
            v24_log = sessions["v24"].run(None, {"input": feats13[:, :11]})[0]
            v24_arg = v24_log.argmax(-1)
        feats14 = None
        if v24_arg is not None:
            feats14 = np.concatenate(
                [feats13[:, :13], (v24_arg.astype(np.float32) / 20.0).reshape(-1, 1)],
                axis=1,
            )

        for name, sess in sessions.items():
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
    print(f"Total morphemes: {total}")

    single = {}
    for name, sml in softmax_all.items():
        if not sml:
            continue
        preds = np.concatenate([s.argmax(-1) for s in sml])
        acc = (preds == flat_labels).mean()
        single[name] = acc

    # Sort
    sorted_models = sorted(single.items(), key=lambda x: -x[1])
    print("\nSingle model accs:")
    for n, a in sorted_models:
        print(f"  {n}: {a * 100:.2f}%")

    # Greedy ensemble (aggressive)
    best_members = [sorted_models[0][0]]
    best_acc = sorted_models[0][1]
    remaining = [n for n, _ in sorted_models[1:]]
    print(f"\nGreedy start: {best_members} -> {best_acc * 100:.2f}%")
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
        print(f"+{best_candidate}: {best_acc * 100:.2f}%")

    print(f"\nGreedy best: {'+'.join(best_members)} = {best_acc * 100:.2f}%")

    # Try weighted via random search
    print("\nRandom weighted search (1000 trials) on greedy members...")
    rng = np.random.default_rng(42)
    best_r_acc = best_acc
    best_r_weights = np.ones(len(best_members)) / len(best_members)
    for _ in range(1000):
        ws = rng.dirichlet(np.ones(len(best_members)))
        avg_preds_list = []
        for i in range(len(labels_list)):
            avg = sum(
                ws[k] * softmax_all[best_members[k]][i]
                for k in range(len(best_members))
            )
            avg_preds_list.append(avg.argmax(-1))
        preds = np.concatenate(avg_preds_list)
        acc = (preds == flat_labels).mean()
        if acc > best_r_acc:
            best_r_acc = acc
            best_r_weights = ws
    print(f"Best random weighted: {best_r_acc * 100:.2f}%")
    print(f"  weights: {dict(zip(best_members, [f'{w:.3f}' for w in best_r_weights]))}")


if __name__ == "__main__":
    main()
