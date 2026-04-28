"""Confidence-gated hybrid: v66_split1 + 11-student consensus on uncertain tokens.

For tokens where v66_split1 confidence is high, use v66_split1 directly.
Otherwise, fall back to 11-student consensus argmax (majority vote).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

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
    """Hybrid eval."""
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

    random.seed(0)
    indices = list(range(len(jsut)))
    random.shuffle(indices)
    val_size = int(len(indices) * 0.1)
    val_idx = set(indices[:val_size])
    val_utts = [u for i, u in enumerate(jsut) if i in val_idx]
    val_idx_in_order = [i for i in range(len(jsut)) if i in val_idx]
    print(f"Val: {len(val_utts)} utts (val_split=0)")

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess_v24 = ort.InferenceSession(
        "/mnt/c/GitHub/kotonoha-models/accent_model_v24.onnx", providers=providers
    )
    stacker_all = torch.load("/tmp/v66_stacker.pt", weights_only=False)
    cons_all = torch.load("/tmp/v64_consensus.pt", weights_only=False)
    sess_v66s1 = ort.InferenceSession(
        "/mnt/c/GitHub/kotonoha-models/accent_model_v66_split1.onnx",
        providers=providers,
    )

    sm_s1: list[np.ndarray] = []
    cons_per_utt: list[np.ndarray] = []
    labels_per_utt: list[np.ndarray] = []

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
        labels_per_utt.append(labels)
        v24_log = sess_v24.run(None, {"input": feats13[:, :11]})[0]
        v24_arg = v24_log.argmax(-1)
        seq_len = feats13.shape[0]
        stacker_t = stacker_all[utt_idx]
        feats103 = np.empty((seq_len, 103), dtype=np.float32)
        feats103[:, :13] = feats13
        feats103[:, 13] = v24_arg.astype(np.float32) / 20.0
        for t in range(seq_len):
            dict_acc = _morpheme_dict_accent(ms[t])
            exp_y, pmax, margin, entropy, p_dict = _teacher_soft_stats(
                v24_log[t], dict_acc
            )
            feats103[t, 14:19] = [exp_y, pmax, margin, entropy, p_dict]
        feats103[:, 19:103] = stacker_t[:seq_len, :84]
        log = sess_v66s1.run(None, {"input": feats103})[0]
        sm_s1.append(_softmax_1d(log))
        cons_per_utt.append(cons_all[utt_idx])

    flat_labels = np.concatenate(labels_per_utt)
    print(f"Total: {len(flat_labels)} morphemes")

    # v66_split1 alone
    preds_s1 = np.concatenate([sm.argmax(-1) for sm in sm_s1])
    s1_acc = float((preds_s1 == flat_labels).mean())
    print(f"v66_split1 alone: {s1_acc * 100:.2f}%")

    # 11-student consensus alone
    cons_argmax_per_utt = []
    for cons in cons_per_utt:
        cons_argmax_per_utt.append(cons[:, 0].astype(np.int64))
    cons_preds = np.concatenate(cons_argmax_per_utt)
    cons_acc = float((cons_preds == flat_labels).mean())
    print(f"11-stu consensus alone: {cons_acc * 100:.2f}%")

    # Hybrid: use v66_split1 if its max_prob >= threshold, else consensus
    print("\nConfidence-gated hybrid (v66s1 high-conf + consensus fallback):")
    for thr in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
        correct = 0
        total = 0
        n_use_s1 = 0
        for sm, cons, labels in zip(sm_s1, cons_per_utt, labels_per_utt, strict=True):
            seq = len(labels)
            if seq == 0:
                continue
            argmax_s1 = sm.argmax(-1)
            max_p_s1 = sm.max(-1)
            cons_arg = cons[:seq, 0].astype(np.int64)
            for t in range(seq):
                if max_p_s1[t] >= thr:
                    pred = int(argmax_s1[t])
                    n_use_s1 += 1
                else:
                    pred = int(cons_arg[t])
                if pred == int(labels[t]):
                    correct += 1
                total += 1
        acc = correct / total
        usage = n_use_s1 / total * 100
        print(f"  thr={thr}: {acc * 100:.2f}% (s1 used in {usage:.1f}%)")

    # Also try: v66_split1 if agrees with consensus, else use whichever is more conf
    print("\nAgreement-based hybrid (use v66s1 if v66s1 == cons, else higher conf):")
    correct = 0
    total = 0
    for sm, cons, labels in zip(sm_s1, cons_per_utt, labels_per_utt, strict=True):
        seq = len(labels)
        if seq == 0:
            continue
        argmax_s1 = sm.argmax(-1)
        max_p_s1 = sm.max(-1)
        cons_arg = cons[:seq, 0].astype(np.int64)
        cons_agree = cons[:seq, 1]
        for t in range(seq):
            if int(argmax_s1[t]) == int(cons_arg[t]):
                pred = int(argmax_s1[t])
            elif max_p_s1[t] >= cons_agree[t]:
                pred = int(argmax_s1[t])
            else:
                pred = int(cons_arg[t])
            if pred == int(labels[t]):
                correct += 1
            total += 1
    acc = correct / total
    print(f"  agreement-based: {acc * 100:.2f}%")


if __name__ == "__main__":
    main()
