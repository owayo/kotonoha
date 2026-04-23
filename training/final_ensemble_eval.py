"""v46: Final aggressive ensemble with weighted + TTA."""

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

    onnx_dir = Path("/mnt/c/GitHub/kotonoha-models")
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    # Strong candidates (>80%) from prior eval
    candidates = ["v20", "v24", "v17", "v13", "v14", "v19", "v18", "v38", "v15b", "v15"]
    sessions: dict[str, ort.InferenceSession] = {}
    dims: dict[str, int] = {}
    for name in candidates:
        path = onnx_dir / f"accent_model_{name}.onnx"
        if not path.exists():
            continue
        s = ort.InferenceSession(str(path), providers=providers)
        inp_shape = s.get_inputs()[0].shape
        actual_dim = inp_shape[1] if len(inp_shape) >= 2 else None
        if isinstance(actual_dim, int):
            dims[name] = actual_dim
        else:
            dims[name] = 11
        sessions[name] = s

    softmax_all: dict[str, list[np.ndarray]] = {n: [] for n in sessions}
    labels_list: list[np.ndarray] = []

    # Also do TTA on v20 (best single)
    rng = np.random.default_rng(0)
    softmax_v20_tta: list[np.ndarray] = []

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

        if "v24" in sessions:
            v24_log = sessions["v24"].run(None, {"input": feats13[:, :11]})[0]
            v24_arg = v24_log.argmax(-1)
            feats14 = np.concatenate(
                [
                    feats13[:, :13],
                    (v24_arg.astype(np.float32) / 20.0).reshape(-1, 1),
                ],
                axis=1,
            )
        else:
            feats14 = None

        for name, sess in sessions.items():
            d = dims[name]
            if d == 11:
                inp = feats13[:, :11]
            elif d == 13:
                inp = feats13[:, :13]
            elif d == 14:
                if feats14 is None:
                    softmax_all[name].append(
                        np.zeros((len(ms), NUM_CLASSES), dtype=np.float32)
                    )
                    continue
                inp = feats14
            else:
                softmax_all[name].append(
                    np.zeros((len(ms), NUM_CLASSES), dtype=np.float32)
                )
                continue
            logits = sess.run(None, {"input": inp})[0]
            softmax_all[name].append(_softmax(logits))

        # TTA on v20
        if "v20" in sessions:
            v20_sess = sessions["v20"]
            sm_sum = _softmax(v20_sess.run(None, {"input": feats13[:, :11]})[0])
            n_aug = 8
            for _ in range(n_aug):
                feats_aug = feats13[:, :11].copy()
                noise = rng.normal(0, 0.02, size=feats_aug[:, 5:].shape).astype(np.float32)
                feats_aug[:, 5:] += noise
                sm_sum = sm_sum + _softmax(v20_sess.run(None, {"input": feats_aug})[0])
            softmax_v20_tta.append(sm_sum / (1 + n_aug))

    flat_labels = np.concatenate(labels_list)
    total = len(flat_labels)
    print(f"Total morphemes: {total}")

    # Single accuracies
    single = {}
    for name, sml in softmax_all.items():
        if not sml:
            continue
        preds = np.concatenate([s.argmax(-1) for s in sml])
        acc = (preds == flat_labels).mean()
        single[name] = acc

    # Add TTA-v20
    v20_tta_preds = np.concatenate([s.argmax(-1) for s in softmax_v20_tta])
    acc_v20_tta = (v20_tta_preds == flat_labels).mean()
    softmax_all["v20_tta"] = softmax_v20_tta
    single["v20_tta"] = acc_v20_tta
    print(f"\nv20 baseline: {single.get('v20', 0) * 100:.2f}%")
    print(f"v20 TTA: {acc_v20_tta * 100:.2f}%")

    sorted_models = sorted(single.items(), key=lambda x: -x[1])
    print("\nTop models:")
    for n, a in sorted_models[:8]:
        print(f"  {n}: {a * 100:.2f}%")

    # Greedy ensemble
    best_members = [sorted_models[0][0]]
    best_acc = sorted_models[0][1]
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
            f"+{best_candidate}: {best_acc * 100:.2f}% "
            f"(members: {len(best_members)})"
        )

    # Weighted grid search on greedy members (if 2-3 members)
    if len(best_members) <= 4:
        print(f"\nWeighted grid search on {best_members}")
        step_count = 6
        best_wacc = best_acc
        best_weights = tuple([1.0] * len(best_members))
        from itertools import product

        grids = list(
            product(range(0, step_count + 1), repeat=len(best_members))
        )
        for ws in grids:
            if sum(ws) == 0:
                continue
            wsum = sum(ws)
            norm_ws = [w / wsum for w in ws]
            avg_preds_list = []
            for i in range(len(labels_list)):
                avg = sum(
                    norm_ws[k] * softmax_all[best_members[k]][i]
                    for k in range(len(best_members))
                )
                avg_preds_list.append(avg.argmax(-1))
            preds = np.concatenate(avg_preds_list)
            acc = (preds == flat_labels).mean()
            if acc > best_wacc:
                best_wacc = acc
                best_weights = norm_ws
        print(
            f"Best weighted: {best_weights} -> {best_wacc * 100:.2f}%"
        )

    print(
        f"\nFinal best: {'+'.join(best_members)} greedy = "
        f"{best_acc * 100:.2f}%"
    )


if __name__ == "__main__":
    main()
