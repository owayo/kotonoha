"""v63 multi-seed ensemble evaluation on val_split=0 val 500 utts.

Loads multiple v63 ONNX models (different seeds) and computes ensemble
softmax average. Goal: push beyond v63 single-seed 83.35% toward 85%.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path

import numpy as np
import onnxruntime as ort

from precompute_v61_meta import _meta_features
from train_onnx_v60 import (
    NUM_CLASSES,
    _enrich_utterances,
    _extract_morpheme_features,
    _load_accent_dicts,
    _load_dotenv,
    _morpheme_dict_accent,
    _softmax_1d,
    _teacher_soft_stats,
)


def main() -> None:
    """Multi-seed v63 ensemble eval; supports adding v61, v62 too."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-split-seed", type=int, default=0)
    parser.add_argument("--models-dir", default="/mnt/c/GitHub/kotonoha-models")
    parser.add_argument(
        "--targets",
        default=(
            "v63:/mnt/c/GitHub/kotonoha-models/accent_model_v63.onnx,"
            "v63s1:/tmp/v63_seed1.onnx,v63s2:/tmp/v63_seed2.onnx,"
            "v61:/mnt/c/GitHub/kotonoha-models/accent_model_v61.onnx,"
            "v62:/mnt/c/GitHub/kotonoha-models/accent_model_v62.onnx"
        ),
        help="comma-separated name:path pairs",
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

    random.seed(args.val_split_seed)
    indices = list(range(len(jsut)))
    random.shuffle(indices)
    val_size = int(len(indices) * 0.1)
    val_idx = set(indices[:val_size])
    val_utts = [u for i, u in enumerate(jsut) if i in val_idx]
    print(f"Val: {len(val_utts)} utts")

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    models_dir = Path(args.models_dir)
    sess_v24 = ort.InferenceSession(
        str(models_dir / "accent_model_v24.onnx"), providers=providers
    )
    student_specs = [
        "accent_model_v38.onnx",
        "accent_model_v54_split1.onnx",
        "accent_model_v54_split2.onnx",
        "accent_model_v54_split3.onnx",
        "accent_model_v59_fold0.onnx",
        "accent_model_v59_fold1.onnx",
        "accent_model_v59_fold2.onnx",
        "accent_model_v59_fold3.onnx",
        "accent_model_v59_fold4.onnx",
    ]
    student_sessions = [
        ort.InferenceSession(str(models_dir / fname), providers=providers)
        for fname in student_specs
    ]

    target_sessions: dict[str, ort.InferenceSession] = {}
    for spec in args.targets.split(","):
        if ":" not in spec:
            continue
        name, path = spec.split(":", 1)
        if not Path(path).exists():
            print(f"  SKIP {name}: not found ({path})")
            continue
        target_sessions[name] = ort.InferenceSession(path, providers=providers)
        print(f"  loaded {name}: {path}")

    target_softmax: dict[str, list[np.ndarray]] = {n: [] for n in target_sessions}
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
        labels = np.array(
            [min(m.get("accent_type", 0), NUM_CLASSES - 1) for m in ms],
            dtype=np.int64,
        )
        labels_list.append(labels)

        v24_log = sess_v24.run(None, {"input": feats13[:, :11]})[0]
        v24_arg = v24_log.argmax(-1)

        seq_len = feats13.shape[0]
        feats14 = np.concatenate(
            [feats13[:, :13], (v24_arg.astype(np.float32) / 20.0).reshape(-1, 1)],
            axis=1,
        )
        stacks = []
        for sess in student_sessions:
            log = sess.run(None, {"input": feats14})[0]
            stacks.append(_softmax_1d(log))
        sm_stack = np.stack(stacks, axis=0)
        meta = np.empty((seq_len, 5), dtype=np.float32)
        for t in range(seq_len):
            meta[t] = _meta_features(sm_stack[:, t, :])

        feats24 = np.empty((seq_len, 24), dtype=np.float32)
        for t in range(seq_len):
            logits_row = v24_log[t]
            tp_argmax = float(int(np.argmax(logits_row))) / 20.0
            dict_acc = _morpheme_dict_accent(ms[t])
            exp_y, pmax, margin, entropy, p_dict = _teacher_soft_stats(
                logits_row, dict_acc
            )
            feats24[t, :13] = feats13[t]
            feats24[t, 13] = tp_argmax
            feats24[t, 14:19] = [exp_y, pmax, margin, entropy, p_dict]
            feats24[t, 19:24] = meta[t]

        for name, sess in target_sessions.items():
            log = sess.run(None, {"input": feats24})[0]
            target_softmax[name].append(_softmax_1d(log))

    flat_labels = np.concatenate(labels_list)
    total = len(flat_labels)
    print(f"\nTotal morphemes: {total}")

    print("\nSingle:")
    accs: dict[str, float] = {}
    for name in target_sessions:
        preds = np.concatenate([sm.argmax(-1) for sm in target_softmax[name]])
        acc = float((preds == flat_labels).mean())
        accs[name] = acc
        print(f"  {name}: {acc * 100:.2f}%")

    print("\nEnsemble (equal-weight softmax avg):")
    names = list(target_sessions.keys())
    best_acc = max(accs.values())
    best_combo: tuple[str, ...] = ()
    for r in range(2, len(names) + 1):
        for combo in itertools.combinations(names, r):
            avg_preds_list = []
            for i in range(len(labels_list)):
                stacked = np.stack([target_softmax[n][i] for n in combo])
                avg_preds_list.append(stacked.mean(0).argmax(-1))
            preds = np.concatenate(avg_preds_list)
            acc = float((preds == flat_labels).mean())
            mark = ""
            if acc > best_acc:
                best_acc = acc
                best_combo = combo
                mark = " *"
            print(f"  {'+'.join(combo)}: {acc * 100:.2f}%{mark}")

    print(
        f"\nBest: {'+'.join(best_combo) if best_combo else 'single'} = "
        f"{best_acc * 100:.2f}%"
    )


if __name__ == "__main__":
    main()
