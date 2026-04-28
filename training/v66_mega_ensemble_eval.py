"""Mega-ensemble eval: 11 base students + v66 + v68 + hybrid options.

Computes student softmax for 12 models on val_split=0 val 500 utts and
explores all ensemble combinations to maximize accuracy toward 90%.

Models loaded:
  v24, v38, v54_split{1,2,3}, v59_fold{0..4}: 14-dim input
  v61, v63, v68: 24-dim input
  v66: 103-dim input (uses precomputed 84-dim stacker)
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

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
    """Run mega-ensemble eval on val_split=0 val 500 utts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-split-seed", type=int, default=0)
    parser.add_argument("--models-dir", default="/mnt/c/GitHub/kotonoha-models")
    parser.add_argument("--stacker-cache", default="/tmp/v66_stacker.pt")
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
    val_idx_in_order = [i for i in range(len(jsut)) if i in val_idx]
    print(f"Val: {len(val_utts)} utts")

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    models_dir = Path(args.models_dir)
    sess_v24 = ort.InferenceSession(
        str(models_dir / "accent_model_v24.onnx"), providers=providers
    )
    student_14d_specs = [
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
    student_14d_sessions = {
        name: ort.InferenceSession(str(models_dir / fname), providers=providers)
        for name, fname in student_14d_specs
    }
    student_24d_specs = [
        ("v61", "accent_model_v61.onnx"),
        ("v63", "accent_model_v63.onnx"),
        ("v68", "accent_model_v68.onnx"),
    ]
    student_24d_sessions: dict[str, ort.InferenceSession] = {}
    for name, fname in student_24d_specs:
        path = models_dir / fname
        if not path.exists():
            print(f"  SKIP missing 24d: {fname}")
            continue
        student_24d_sessions[name] = ort.InferenceSession(
            str(path), providers=providers
        )
    sess_v66_path = models_dir / "accent_model_v66.onnx"
    sess_v66 = (
        ort.InferenceSession(str(sess_v66_path), providers=providers)
        if sess_v66_path.exists()
        else None
    )
    if sess_v66 is None:
        print("  SKIP v66: not found")
    print(
        f"Loaded v24, {len(student_14d_sessions)} 14d, "
        f"{len(student_24d_sessions)} 24d, v66={sess_v66 is not None}"
    )

    stacker_all = torch.load(args.stacker_cache, weights_only=False)

    target_softmax: dict[str, list[np.ndarray]] = {}
    target_softmax["v24"] = []
    for name in student_14d_sessions:
        target_softmax[name] = []
    for name in student_24d_sessions:
        target_softmax[name] = []
    if sess_v66 is not None:
        target_softmax["v66"] = []
    labels_list: list[np.ndarray] = []

    for utt_idx, utt in zip(val_idx_in_order, val_utts, strict=True):
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
        target_softmax["v24"].append(_softmax_1d(v24_log))
        v24_arg = v24_log.argmax(-1)

        seq_len = feats13.shape[0]
        feats14 = np.concatenate(
            [feats13[:, :13], (v24_arg.astype(np.float32) / 20.0).reshape(-1, 1)],
            axis=1,
        )
        sm_14_list: list[np.ndarray] = []
        for name, sess in student_14d_sessions.items():
            log = sess.run(None, {"input": feats14})[0]
            sm = _softmax_1d(log)
            target_softmax[name].append(sm)
            sm_14_list.append(sm)
        sm_stack_9 = np.stack(sm_14_list, axis=0)
        meta_9 = np.empty((seq_len, 5), dtype=np.float32)
        for t in range(seq_len):
            meta_9[t] = _meta_features(sm_stack_9[:, t, :])

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
            feats24[t, 19:24] = meta_9[t]

        for name, sess in student_24d_sessions.items():
            log = sess.run(None, {"input": feats24})[0]
            target_softmax[name].append(_softmax_1d(log))

        if sess_v66 is not None:
            stacker_t = stacker_all[utt_idx]
            feats103 = np.empty((seq_len, 103), dtype=np.float32)
            feats103[:, :19] = feats24[:, :19]
            feats103[:, 19:103] = stacker_t[:seq_len, :84]
            log_v66 = sess_v66.run(None, {"input": feats103})[0]
            target_softmax["v66"].append(_softmax_1d(log_v66))

    flat_labels = np.concatenate(labels_list)
    total = len(flat_labels)
    print(f"\nTotal morphemes: {total}")

    print("\nSingle-model accuracies:")
    accs = {}
    for name, sm_list in target_softmax.items():
        if not sm_list:
            continue
        preds = np.concatenate([sm.argmax(-1) for sm in sm_list])
        acc = float((preds == flat_labels).mean())
        accs[name] = acc
        print(f"  {name}: {acc * 100:.2f}%")

    # Try selected ensemble combinations
    names = [n for n in target_softmax if target_softmax[n]]
    print(f"\n--- Selected ensembles (out of {len(names)} models) ---")

    # All 12 students
    print("\nAll-12 ensemble:")
    if len(names) >= 2:
        avg_preds_list = []
        for i in range(len(labels_list)):
            stacked = np.stack([target_softmax[n][i] for n in names])
            avg_preds_list.append(stacked.mean(0).argmax(-1))
        preds = np.concatenate(avg_preds_list)
        acc = float((preds == flat_labels).mean())
        print(f"  {'+'.join(names)}: {acc * 100:.2f}%")

    # Best single + add others
    best_single = max(accs, key=accs.get)
    print(f"\nBest single: {best_single} = {accs[best_single] * 100:.2f}%")
    print("Greedy add (start with best single, add the model that helps most):")
    cur_set = [best_single]
    cur_avg_per_utt = [
        target_softmax[best_single][i].copy() for i in range(len(labels_list))
    ]
    cur_acc = accs[best_single]
    remaining = [n for n in names if n != best_single]
    while remaining:
        best_add = None
        best_acc = cur_acc
        for cand in remaining:
            new_avgs = []
            for i in range(len(labels_list)):
                new_avg = (
                    cur_avg_per_utt[i] * len(cur_set) + target_softmax[cand][i]
                ) / (len(cur_set) + 1)
                new_avgs.append(new_avg)
            preds = np.concatenate([a.argmax(-1) for a in new_avgs])
            acc = float((preds == flat_labels).mean())
            if acc > best_acc:
                best_acc = acc
                best_add = cand
                best_avgs = new_avgs
        if best_add is None:
            break
        cur_set.append(best_add)
        cur_avg_per_utt = best_avgs
        cur_acc = best_acc
        print(f"  + {best_add}: {best_acc * 100:.2f}% (set={cur_set})")
    print(f"\nGreedy ensemble best: {cur_acc * 100:.2f}% with {cur_set}")

    # Top-N by single accuracy
    print("\nTop-N by single accuracy:")
    sorted_names = sorted(names, key=lambda n: -accs[n])
    for k in range(2, min(len(sorted_names) + 1, 13)):
        combo = sorted_names[:k]
        avg_preds_list = []
        for i in range(len(labels_list)):
            stacked = np.stack([target_softmax[n][i] for n in combo])
            avg_preds_list.append(stacked.mean(0).argmax(-1))
        preds = np.concatenate(avg_preds_list)
        acc = float((preds == flat_labels).mean())
        print(f"  Top-{k}: {acc * 100:.2f}% ({'+'.join(combo)})")


if __name__ == "__main__":
    main()
