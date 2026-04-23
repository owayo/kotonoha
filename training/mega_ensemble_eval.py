"""Mega ensemble: all strong models + multiple TTA seeds + aggressive search."""

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
    providers = ["CPUExecutionProvider"]

    all_models = ["v20", "v24", "v17", "v13", "v14", "v19", "v18", "v15", "v15b", "v38", "v46"]
    sessions: dict[str, ort.InferenceSession] = {}
    dims: dict[str, int] = {}
    for name in all_models:
        path = onnx_dir / f"accent_model_{name}.onnx"
        s = ort.InferenceSession(str(path), providers=providers)
        inp_shape = s.get_inputs()[0].shape
        actual_dim = inp_shape[1] if len(inp_shape) >= 2 else None
        dims[name] = actual_dim if isinstance(actual_dim, int) else 11
        sessions[name] = s

    softmax_all: dict[str, list[np.ndarray]] = {n: [] for n in sessions}
    # Multi-seed TTA for top models
    tta_configs = []
    for m in ["v20", "v17", "v24", "v13"]:
        for seed_idx in range(3):
            tta_configs.append((m, f"{m}_tta{seed_idx}", seed_idx))
            softmax_all[f"{m}_tta{seed_idx}"] = []

    labels_list: list[np.ndarray] = []

    for utt in val_utts:
        ms = utt.get("morphemes", [])
        if not ms:
            continue
        n = len(ms)
        feats13 = np.array(
            [_extract_morpheme_features(m, i / max(n - 1, 1)) for i, m in enumerate(ms)],
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
                softmax_all[name].append(np.zeros((len(ms), NUM_CLASSES), dtype=np.float32))
                continue
            logits = sess.run(None, {"input": inp})[0]
            softmax_all[name].append(_softmax(logits))

        # TTA with different seeds
        for base_name, key, seed_idx in tta_configs:
            if base_name not in sessions:
                softmax_all[key].append(np.zeros((len(ms), NUM_CLASSES), dtype=np.float32))
                continue
            sess = sessions[base_name]
            d = dims[base_name]
            base_inp = feats13[:, :11] if d == 11 else (feats13[:, :13] if d == 13 else feats14)
            if base_inp is None:
                softmax_all[key].append(np.zeros((len(ms), NUM_CLASSES), dtype=np.float32))
                continue
            rng2 = np.random.default_rng(seed_idx * 100 + 7)
            sm_sum = _softmax(sess.run(None, {"input": base_inp})[0])
            n_aug = 6
            for _ in range(n_aug):
                aug = base_inp.copy()
                noise = rng2.normal(0, 0.02, size=aug[:, 5:].shape).astype(np.float32)
                aug[:, 5:] += noise
                sm_sum = sm_sum + _softmax(sess.run(None, {"input": aug})[0])
            softmax_all[key].append(sm_sum / (1 + n_aug))

    flat_labels = np.concatenate(labels_list)
    single = {}
    for k, sml in softmax_all.items():
        if not sml:
            continue
        preds = np.concatenate([s.argmax(-1) for s in sml])
        acc = (preds == flat_labels).mean()
        single[k] = acc

    sorted_m = sorted(single.items(), key=lambda x: -x[1])
    print(f"Total morphemes: {len(flat_labels)}")
    print("Top variants:")
    for n, a in sorted_m[:20]:
        print(f"  {n}: {a * 100:.2f}%")

    # Greedy with full pool
    best_members = [sorted_m[0][0]]
    best_acc = sorted_m[0][1]
    remaining = [n for n, _ in sorted_m[1:]]
    print(f"\nGreedy:")
    while remaining:
        cand_best = None
        cand_acc = best_acc
        for c in remaining:
            trial = best_members + [c]
            avg_preds_list = []
            for i in range(len(labels_list)):
                stacked = np.stack([softmax_all[n][i] for n in trial])
                avg = stacked.mean(0)
                avg_preds_list.append(avg.argmax(-1))
            preds = np.concatenate(avg_preds_list)
            acc = (preds == flat_labels).mean()
            if acc > cand_acc:
                cand_acc = acc
                cand_best = c
        if cand_best is None:
            break
        best_members.append(cand_best)
        best_acc = cand_acc
        remaining.remove(cand_best)
        print(f"  +{cand_best}: {best_acc * 100:.2f}%")

    print(f"\nGreedy best: {len(best_members)} members = {best_acc * 100:.2f}%")
    print(f"  {best_members}")

    # Random weighted search
    rng = np.random.default_rng(42)
    print(f"\nLarge weighted search (10000 trials)...")
    best_wacc = best_acc
    best_ws = np.ones(len(best_members)) / len(best_members)
    for _ in range(10000):
        ws = rng.dirichlet(np.ones(len(best_members)))
        avg_preds_list = []
        for i in range(len(labels_list)):
            avg = sum(ws[k] * softmax_all[best_members[k]][i] for k in range(len(best_members)))
            avg_preds_list.append(avg.argmax(-1))
        preds = np.concatenate(avg_preds_list)
        acc = (preds == flat_labels).mean()
        if acc > best_wacc:
            best_wacc = acc
            best_ws = ws
    print(f"Best weighted: {best_wacc * 100:.2f}%")
    for n, w in zip(best_members, best_ws):
        print(f"  {n}: {w:.3f}")

    # Also try ALL models weighted search (not just greedy members)
    all_keys = [k for k in softmax_all if softmax_all[k]]
    print(f"\nHuge weighted search on ALL {len(all_keys)} variants (5000 trials)...")
    best_all_acc = best_wacc
    best_all_ws = None
    for _ in range(5000):
        ws = rng.dirichlet(np.ones(len(all_keys)))
        avg_preds_list = []
        for i in range(len(labels_list)):
            avg = sum(ws[k] * softmax_all[all_keys[k]][i] for k in range(len(all_keys)))
            avg_preds_list.append(avg.argmax(-1))
        preds = np.concatenate(avg_preds_list)
        acc = (preds == flat_labels).mean()
        if acc > best_all_acc:
            best_all_acc = acc
            best_all_ws = ws
    print(f"Best all-weighted: {best_all_acc * 100:.2f}%")


if __name__ == "__main__":
    main()
