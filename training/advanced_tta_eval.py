"""Advanced TTA: multiple aug types (feature noise, position noise, reading shuffle)."""

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

    # Strong 11-dim models for TTA
    candidates = ["v20", "v17", "v13", "v14", "v19", "v24", "v18"]
    sessions: dict[str, ort.InferenceSession] = {}
    for name in candidates:
        path = onnx_dir / f"accent_model_{name}.onnx"
        s = ort.InferenceSession(str(path), providers=providers)
        sessions[name] = s

    labels_list: list[np.ndarray] = []
    sm_variants: dict[str, list[np.ndarray]] = {}

    # Variants: base, feat_noise (3 sigmas), pos_shift, morph_drop
    variant_names = []
    for m in candidates:
        sm_variants[m] = []
        variant_names.append(m)
    for m in ["v20", "v17"]:
        for suf in ["_fn01", "_fn02", "_ps", "_md"]:
            key = f"{m}{suf}"
            sm_variants[key] = []
            variant_names.append(key)

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
            [min(m.get("accent_type", 0), NUM_CLASSES - 1) for m in ms], dtype=np.int64
        )
        labels_list.append(labs)

        for name in candidates:
            sess = sessions[name]
            smx = _softmax(sess.run(None, {"input": feats13[:, :11]})[0])
            sm_variants[name].append(smx)

        # TTA variants on v20 and v17
        for m in ["v20", "v17"]:
            sess = sessions[m]
            # fn01: feat_noise sigma=0.01 n_aug=4
            for suf, sigma, n_aug, aug_type in [
                ("_fn01", 0.01, 4, "feat_noise"),
                ("_fn02", 0.02, 4, "feat_noise"),
                ("_ps", None, 4, "pos_shift"),
                ("_md", None, 4, "morph_drop"),
            ]:
                key = f"{m}{suf}"
                base = feats13[:, :11]
                sm_sum = _softmax(sess.run(None, {"input": base})[0])
                for _ in range(n_aug):
                    aug = base.copy()
                    if aug_type == "feat_noise":
                        noise = rng.normal(0, sigma, size=aug[:, 5:].shape).astype(
                            np.float32
                        )
                        aug[:, 5:] += noise
                    elif aug_type == "pos_shift":
                        # shift position column (index 9) slightly
                        shift = rng.normal(0, 0.02, size=aug.shape[0]).astype(
                            np.float32
                        )
                        aug[:, 9] = np.clip(aug[:, 9] + shift, 0, 1)
                    elif aug_type == "morph_drop":
                        # zero out ~10% random positions
                        drop_mask = rng.random(aug.shape[0]) < 0.1
                        aug[drop_mask] = 0.0
                    sm_sum = sm_sum + _softmax(sess.run(None, {"input": aug})[0])
                sm_variants[key].append(sm_sum / (1 + n_aug))

    flat_labels = np.concatenate(labels_list)
    total = len(flat_labels)

    # Single accs
    single = {}
    for k, sml in sm_variants.items():
        if not sml:
            continue
        preds = np.concatenate([s.argmax(-1) for s in sml])
        acc = (preds == flat_labels).mean()
        single[k] = acc

    sorted_m = sorted(single.items(), key=lambda x: -x[1])
    print(f"Total morphemes: {total}")
    print("\nVariants sorted by accuracy:")
    for n, a in sorted_m[:20]:
        print(f"  {n}: {a * 100:.2f}%")

    # Greedy
    best_members = [sorted_m[0][0]]
    best_acc = sorted_m[0][1]
    remaining = [n for n, _ in sorted_m[1:]]
    print(f"\nGreedy start: {best_members} = {best_acc * 100:.2f}%")
    while remaining:
        cand_best = None
        cand_acc = best_acc
        for c in remaining:
            trial = best_members + [c]
            avg_preds_list = []
            for i in range(len(labels_list)):
                stacked = np.stack([sm_variants[n][i] for n in trial])
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

    print(f"\nGreedy best: {best_members} = {best_acc * 100:.2f}%")

    # Weighted random search on greedy members
    print(f"\nRandom weighted search (3000 trials)...")
    best_wacc = best_acc
    best_ws = np.ones(len(best_members)) / len(best_members)
    for _ in range(3000):
        ws = rng.dirichlet(np.ones(len(best_members)))
        avg_preds_list = []
        for i in range(len(labels_list)):
            avg = sum(
                ws[k] * sm_variants[best_members[k]][i]
                for k in range(len(best_members))
            )
            avg_preds_list.append(avg.argmax(-1))
        preds = np.concatenate(avg_preds_list)
        acc = (preds == flat_labels).mean()
        if acc > best_wacc:
            best_wacc = acc
            best_ws = ws
    print(f"Best weighted: {best_wacc * 100:.2f}%")
    for n, w in zip(best_members, best_ws):
        print(f"  {n}: {w:.3f}")


if __name__ == "__main__":
    main()
