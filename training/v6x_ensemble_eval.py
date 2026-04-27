"""Ensemble evaluation across v60-v63 models on val_split=0 val 500 utts.

All v60-v63 models share the same 24-dim or 19-dim input (with v60 = 19 dim).
This script supports models with FEATURE_DIM=24 (v61, v62, v63) and computes
several ensemble combinations to find the best on the no-leak val set.
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
    """Run ensemble eval across v60..v63 (24 dim) on val_split=0 val 500 utts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-split-seed", type=int, default=0)
    parser.add_argument("--models-dir", default="/mnt/c/GitHub/kotonoha-models")
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
    # 9 student models for stacking meta-features
    student_specs = [
        ("v38", "accent_model_v38.onnx"),
        ("v54_s1", "accent_model_v54_split1.onnx"),
        ("v54_s2", "accent_model_v54_split2.onnx"),
        ("v54_s3", "accent_model_v54_split3.onnx"),
        ("v59_f0", "accent_model_v59_fold0.onnx"),
        ("v59_f1", "accent_model_v59_fold1.onnx"),
        ("v59_f2", "accent_model_v59_fold2.onnx"),
        ("v59_f3", "accent_model_v59_fold3.onnx"),
        ("v59_f4", "accent_model_v59_fold4.onnx"),
    ]
    student_sessions = [
        ort.InferenceSession(str(models_dir / fname), providers=providers)
        for _name, fname in student_specs
    ]

    # Target models to ensemble (24 dim input)
    targets = [
        ("v61", "accent_model_v61.onnx"),
        ("v62", "accent_model_v62.onnx"),
        ("v63", "accent_model_v63.onnx"),
    ]
    target_sessions: dict[str, ort.InferenceSession] = {}
    for name, fname in targets:
        path = models_dir / fname
        if not path.exists():
            print(f"  SKIP {name}: not found")
            continue
        target_sessions[name] = ort.InferenceSession(str(path), providers=providers)
        print(f"  loaded {name}")

    # Pre-compute val softmax for each target model
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

        # v24 logits → argmax + soft stats
        v24_log = sess_v24.run(None, {"input": feats13[:, :11]})[0]
        v24_arg = v24_log.argmax(-1)

        # 9-student softmax for meta features
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

        # Build 24-dim feature for v61/v62/v63
        feats24 = np.empty((seq_len, 24), dtype=np.float32)
        for t in range(seq_len):
            feat_13 = feats13[t]
            logits_row = v24_log[t]
            tp_argmax = float(int(np.argmax(logits_row))) / 20.0
            dict_acc = _morpheme_dict_accent(ms[t])
            exp_y, pmax, margin, entropy, p_dict = _teacher_soft_stats(
                logits_row, dict_acc
            )
            feats24[t, :13] = feat_13
            feats24[t, 13] = tp_argmax
            feats24[t, 14:19] = [exp_y, pmax, margin, entropy, p_dict]
            feats24[t, 19:24] = meta[t]

        for name, sess in target_sessions.items():
            log = sess.run(None, {"input": feats24})[0]
            target_softmax[name].append(_softmax_1d(log))

    flat_labels = np.concatenate(labels_list)
    total = len(flat_labels)
    print(f"\nTotal morphemes: {total}")

    # Single-model accuracies
    print("\nSingle:")
    accs: dict[str, float] = {}
    for name in target_sessions:
        preds = np.concatenate([sm.argmax(-1) for sm in target_softmax[name]])
        acc = float((preds == flat_labels).mean())
        accs[name] = acc
        print(f"  {name}: {acc * 100:.2f}%")

    # All ensemble combinations (>=2 models)
    print("\nEnsemble (equal-weight softmax avg):")
    names = list(target_sessions.keys())
    best_acc = max(accs.values())
    best_combo = ()
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
