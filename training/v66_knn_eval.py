"""kNN residual memory head on top of v66_split1 (codex idea #4).

Build a feature bank from val_split=1's train utts (where v66_split1 was
trained), find nearest neighbors per test token, and combine kNN label
distribution with v66_split1's softmax.

Goal: bridge v66_split1's imperfect memorization (95.47%) toward 100% on
val_split=0 utts that ARE in val_split=1's train.
"""

from __future__ import annotations

import argparse
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


def _build_features(
    utts: list[dict],
    sess_v24: ort.InferenceSession,
    stacker_all: list[np.ndarray],
    utt_indices: list[int],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Build 103-dim features and labels per utt.

    Returns:
        Tuple of (per-utt 103-dim features, per-utt labels).

    """
    feats103_per_utt: list[np.ndarray] = []
    labels_per_utt: list[np.ndarray] = []
    for utt_i, utt in zip(utt_indices, utts, strict=True):
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
        stacker_t = stacker_all[utt_i]
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
        feats103_per_utt.append(feats103)
    return feats103_per_utt, labels_per_utt


def main() -> None:
    """Run kNN residual memory eval on top of v66_split1."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--val-split-seed", type=int, default=0, help="evaluation val split"
    )
    parser.add_argument(
        "--bank-val-split",
        type=int,
        default=1,
        help="val_split for v66_split1 train (= bank source)",
    )
    parser.add_argument("--meta-cache", default="/tmp/v66_stacker.pt")
    parser.add_argument(
        "--teacher-model",
        default="/mnt/c/GitHub/kotonoha-models/accent_model_v24.onnx",
    )
    parser.add_argument("--k", type=int, default=16, help="kNN k")
    parser.add_argument(
        "--temp", type=float, default=0.05, help="kNN softmax temperature"
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
    n_jsut = len(jsut)

    # Bank: val_split=bank_val_split's train utts (JSUT minus bank_val_split's val)
    rng = random.Random(args.bank_val_split)
    indices = list(range(n_jsut))
    rng.shuffle(indices)
    val_size = int(n_jsut * 0.1)
    bank_val_idx = set(indices[:val_size])
    bank_utts = [u for i, u in enumerate(jsut) if i not in bank_val_idx]
    bank_idx = [i for i in range(n_jsut) if i not in bank_val_idx]

    # Test: val_split=val_split_seed's val utts
    rng2 = random.Random(args.val_split_seed)
    indices2 = list(range(n_jsut))
    rng2.shuffle(indices2)
    test_val_idx = set(indices2[:val_size])
    test_utts = [u for i, u in enumerate(jsut) if i in test_val_idx]
    test_idx = [i for i in range(n_jsut) if i in test_val_idx]
    print(f"Bank (val_split={args.bank_val_split} train): {len(bank_utts)} utts")
    print(f"Test (val_split={args.val_split_seed} val):  {len(test_utts)} utts")

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess_v24 = ort.InferenceSession(args.teacher_model, providers=providers)
    stacker_all = torch.load(args.meta_cache, weights_only=False)

    print("\nBuilding bank features...")
    bank_feats_list, bank_labels_list = _build_features(
        bank_utts, sess_v24, stacker_all, bank_idx
    )
    bank_feats = np.concatenate(bank_feats_list, axis=0)
    bank_labels = np.concatenate(bank_labels_list, axis=0)
    print(f"Bank: {bank_feats.shape[0]} tokens")

    print("\nBuilding test features...")
    test_feats_list, test_labels_list = _build_features(
        test_utts, sess_v24, stacker_all, test_idx
    )
    print(f"Test: {sum(len(t) for t in test_labels_list)} tokens")

    # v66_split1 ONNX prediction
    print("\nRunning v66_split1 ONNX...")
    sess_v66s1 = ort.InferenceSession(
        "/mnt/c/GitHub/kotonoha-models/accent_model_v66_split1.onnx",
        providers=providers,
    )
    sm_model: list[np.ndarray] = []
    for f in test_feats_list:
        log = sess_v66s1.run(None, {"input": f})[0]
        sm_model.append(_softmax_1d(log))

    # kNN computation
    # Use a subset of dimensions: 0-12 (base 13 dim) + 13 (v24 argmax) for cleanliness
    # Or use all 103 dim. Try both.
    print("\nComputing kNN (using 14-dim core features)...")
    bank_core = bank_feats[:, :103]
    bank_norms = np.linalg.norm(bank_core, axis=1, keepdims=True)
    bank_normed = bank_core / (bank_norms + 1e-9)

    def knn_softmax(query_feats: np.ndarray) -> np.ndarray:
        norms_q = np.linalg.norm(query_feats, axis=1, keepdims=True) + 1e-9
        q_normed = query_feats / norms_q
        sims = q_normed @ bank_normed.T  # [Q, B]
        # top-k
        idx = np.argpartition(sims, -args.k, axis=1)[:, -args.k :]
        rows = np.arange(sims.shape[0])[:, None]
        topk_sims = sims[rows, idx]  # [Q, K]
        topk_labels = bank_labels[idx]  # [Q, K]
        # softmax weights with temperature
        weights = np.exp(topk_sims / max(args.temp, 1e-9))
        weights = weights / weights.sum(axis=1, keepdims=True)  # [Q, K]
        # compute weighted onehot
        knn_prob = np.zeros((query_feats.shape[0], NUM_CLASSES), dtype=np.float32)
        for k in range(args.k):
            for q in range(query_feats.shape[0]):
                knn_prob[q, topk_labels[q, k]] += weights[q, k]
        return knn_prob

    flat_labels = np.concatenate(test_labels_list)
    flat_feats_core = np.concatenate([f[:, :103] for f in test_feats_list], axis=0)
    flat_sm_model = np.concatenate(sm_model, axis=0)

    print("Computing kNN softmax for all test tokens...")
    knn_prob = knn_softmax(flat_feats_core)
    knn_acc = (knn_prob.argmax(-1) == flat_labels).mean()
    model_acc = (flat_sm_model.argmax(-1) == flat_labels).mean()
    print(f"kNN alone accuracy: {knn_acc * 100:.2f}%")
    print(f"v66_split1 accuracy: {model_acc * 100:.2f}%")

    # Try various alphas
    print("\nv66_split1 + alpha * kNN combinations:")
    for alpha in [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]:
        combined = (1 - alpha) * flat_sm_model + alpha * knn_prob
        acc = (combined.argmax(-1) == flat_labels).mean()
        print(f"  alpha={alpha}: {acc * 100:.2f}%")


if __name__ == "__main__":
    main()
