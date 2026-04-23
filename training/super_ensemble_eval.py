"""v46: Super aggressive ensemble - TTA on multiple strong models + weighted."""

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

    # All models >= 79% from full_ensemble_eval
    candidates = ["v20", "v24", "v17", "v13", "v14", "v19", "v18", "v15", "v15b", "v38"]
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

    softmax_all: dict[str, list[np.ndarray]] = {n: [] for n in sessions}
    # Add TTA variants for top models
    tta_models = ["v20", "v17", "v13", "v24"]
    for tm in tta_models:
        softmax_all[f"{tm}_tta"] = []
    labels_list: list[np.ndarray] = []

    rng = np.random.default_rng(0)

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

        v24_arg = None
        if "v24" in sessions:
            v24_log = sessions["v24"].run(None, {"input": feats13[:, :11]})[0]
            v24_arg = v24_log.argmax(-1)
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

        # TTA for top models (11 dim only, simple case)
        for tm in tta_models:
            if tm not in sessions:
                softmax_all[f"{tm}_tta"].append(
                    np.zeros((len(ms), NUM_CLASSES), dtype=np.float32)
                )
                continue
            sess = sessions[tm]
            d = dims[tm]
            if d == 11:
                base_inp = feats13[:, :11]
            elif d == 13:
                base_inp = feats13[:, :13]
            else:
                # 14 dim, use feats14
                base_inp = feats14
            if base_inp is None:
                softmax_all[f"{tm}_tta"].append(
                    np.zeros((len(ms), NUM_CLASSES), dtype=np.float32)
                )
                continue
            sm_sum = _softmax(sess.run(None, {"input": base_inp})[0])
            n_aug = 6
            for _ in range(n_aug):
                feats_aug = base_inp.copy()
                noise = rng.normal(0, 0.015, size=feats_aug[:, 5:].shape).astype(
                    np.float32
                )
                feats_aug[:, 5:] += noise
                sm_sum = sm_sum + _softmax(sess.run(None, {"input": feats_aug})[0])
            softmax_all[f"{tm}_tta"].append(sm_sum / (1 + n_aug))

    flat_labels = np.concatenate(labels_list)
    total = len(flat_labels)

    single = {}
    for name, sml in softmax_all.items():
        if not sml:
            continue
        preds = np.concatenate([s.argmax(-1) for s in sml])
        acc = (preds == flat_labels).mean()
        single[name] = acc

    print(f"Total morphemes: {total}")
    sorted_models = sorted(single.items(), key=lambda x: -x[1])
    for n, a in sorted_models:
        print(f"  {n}: {a * 100:.2f}%")

    # Greedy ensemble
    best_members = [sorted_models[0][0]]
    best_acc = sorted_models[0][1]
    remaining = [n for n, _ in sorted_models[1:]]
    print(f"\nGreedy: start {best_members} = {best_acc * 100:.2f}%")
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
        print(f"  +{best_candidate}: {best_acc * 100:.2f}%")

    print(f"\nGreedy best: {best_members} = {best_acc * 100:.2f}%")

    # Weighted grid search on greedy members
    print(f"\nWeighted grid search on {best_members} (step=8)")
    step_count = 8
    best_wacc = best_acc
    best_weights = tuple([1.0] * len(best_members))
    from itertools import product

    if len(best_members) <= 5:
        total_grids = (step_count + 1) ** len(best_members)
        print(f"  {total_grids} combinations to try")
        for ws in product(range(0, step_count + 1), repeat=len(best_members)):
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
            f"  Best weighted: acc={best_wacc * 100:.2f}% weights={[f'{w:.3f}' for w in best_weights]}"
        )


if __name__ == "__main__":
    main()
