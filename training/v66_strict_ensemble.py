"""Strict no-leak ensemble eval: only models trained on val_split=0 train.

Excludes v54_split{1,2,3} (trained on val_split=N≠0, val 500 utts in their train)
and v59_fold{1,2,3,4} (trained on disjoint folds containing most val 500 utts).

Strict no-leak set:
- v38, v59_f0, v61, v63, v66, v68: trained on val_split=0 train (val 500 OOF)

Also evaluates README-convention ensembles (with teacher leak) for comparison.
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
    """Strict-no-leak ensemble eval on val_split=0 val 500 utts."""
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

    # All 14-dim students (need both leak and no-leak ones for stacker compute)
    all_14d_specs = [
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
    sessions_14d = {
        name: ort.InferenceSession(str(models_dir / fname), providers=providers)
        for name, fname in all_14d_specs
    }
    sessions_24d: dict[str, ort.InferenceSession] = {}
    for name, fname in [
        ("v61", "accent_model_v61.onnx"),
        ("v63", "accent_model_v63.onnx"),
        ("v68", "accent_model_v68.onnx"),
    ]:
        path = models_dir / fname
        if path.exists():
            sessions_24d[name] = ort.InferenceSession(str(path), providers=providers)
    sess_v66_path = models_dir / "accent_model_v66.onnx"
    sess_v66 = (
        ort.InferenceSession(str(sess_v66_path), providers=providers)
        if sess_v66_path.exists()
        else None
    )
    sess_v66_split1_path = models_dir / "accent_model_v66_split1.onnx"
    sess_v66_split1 = (
        ort.InferenceSession(str(sess_v66_split1_path), providers=providers)
        if sess_v66_split1_path.exists()
        else None
    )
    sess_v66_split2_path = models_dir / "accent_model_v66_split2.onnx"
    sess_v66_split2 = (
        ort.InferenceSession(str(sess_v66_split2_path), providers=providers)
        if sess_v66_split2_path.exists()
        else None
    )

    stacker_all = torch.load(args.stacker_cache, weights_only=False)

    # Compute softmax for all models on val 500
    target_softmax: dict[str, list[np.ndarray]] = {n: [] for n in sessions_14d}
    target_softmax["v24"] = []
    for n in sessions_24d:
        target_softmax[n] = []
    if sess_v66 is not None:
        target_softmax["v66"] = []
    if sess_v66_split1 is not None:
        target_softmax["v66_s1"] = []
    if sess_v66_split2 is not None:
        target_softmax["v66_s2"] = []
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
        for name, sess in sessions_14d.items():
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
        for name, sess in sessions_24d.items():
            log = sess.run(None, {"input": feats24})[0]
            target_softmax[name].append(_softmax_1d(log))
        feats103 = None
        if (
            sess_v66 is not None
            or sess_v66_split1 is not None
            or sess_v66_split2 is not None
        ):
            stacker_t = stacker_all[utt_idx]
            feats103 = np.empty((seq_len, 103), dtype=np.float32)
            feats103[:, :19] = feats24[:, :19]
            feats103[:, 19:103] = stacker_t[:seq_len, :84]
        if sess_v66 is not None and feats103 is not None:
            log_v66 = sess_v66.run(None, {"input": feats103})[0]
            target_softmax["v66"].append(_softmax_1d(log_v66))
        if sess_v66_split1 is not None and feats103 is not None:
            log_v66s1 = sess_v66_split1.run(None, {"input": feats103})[0]
            target_softmax["v66_s1"].append(_softmax_1d(log_v66s1))
        if sess_v66_split2 is not None and feats103 is not None:
            log_v66s2 = sess_v66_split2.run(None, {"input": feats103})[0]
            target_softmax["v66_s2"].append(_softmax_1d(log_v66s2))

    flat_labels = np.concatenate(labels_list)
    total = len(flat_labels)

    def acc_of(combo: list[str]) -> float:
        avg_preds_list = []
        for i in range(len(labels_list)):
            stacked = np.stack([target_softmax[n][i] for n in combo])
            avg_preds_list.append(stacked.mean(0).argmax(-1))
        preds = np.concatenate(avg_preds_list)
        return float((preds == flat_labels).mean())

    print(f"\nTotal morphemes: {total}")
    print("\nSingle-model accuracies (* = strict no-leak on val_split=0):")
    no_leak_set = {"v38", "v59_f0", "v61", "v63", "v66", "v68"}
    accs = {}
    for name in target_softmax:
        if not target_softmax[name]:
            continue
        preds = np.concatenate([sm.argmax(-1) for sm in target_softmax[name]])
        acc = float((preds == flat_labels).mean())
        accs[name] = acc
        mark = " *" if name in no_leak_set else ""
        print(f"  {name}: {acc * 100:.2f}%{mark}")

    print("\n=== Strict no-leak ensembles (only val_split=0 train models) ===")
    candidates_no_leak = [n for n in no_leak_set if n in target_softmax]
    print(f"Candidates: {candidates_no_leak}")
    # All
    all_acc = acc_of(candidates_no_leak)
    print(f"\nAll {len(candidates_no_leak)} no-leak: {all_acc * 100:.2f}%")
    # Greedy add starting from v66 (best no-leak)
    sorted_nl = sorted(candidates_no_leak, key=lambda n: -accs.get(n, 0.0))
    cur = [sorted_nl[0]]
    cur_acc = accs[cur[0]]
    print(f"\nGreedy from {cur[0]} ({cur_acc * 100:.2f}%):")
    for cand in sorted_nl[1:]:
        new = cur + [cand]
        a = acc_of(new)
        if a > cur_acc:
            cur = new
            cur_acc = a
            print(f"  + {cand}: {a * 100:.2f}% [{','.join(cur)}]")
    print(f"  Best strict no-leak: {cur_acc * 100:.2f}% [{','.join(cur)}]")

    print("\n=== README-convention ensembles (allows teacher leak) ===")
    all_models = list(target_softmax.keys())
    print(f"All models: {all_models}")
    sorted_all = sorted(all_models, key=lambda n: -accs[n])
    cur = [sorted_all[0]]
    cur_acc = accs[cur[0]]
    print(f"\nGreedy from {cur[0]} ({cur_acc * 100:.2f}%):")
    for cand in sorted_all[1:]:
        new = cur + [cand]
        a = acc_of(new)
        if a > cur_acc:
            cur = new
            cur_acc = a
            print(f"  + {cand}: {a * 100:.2f}% [{','.join(cur)}]")
    print(f"  Best README-convention: {cur_acc * 100:.2f}% [{','.join(cur)}]")


if __name__ == "__main__":
    main()
