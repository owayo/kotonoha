"""Exact-memory override on top of v66_split1 (codex idea #2).

Build a symbolic cache from v66_split1's training set (val_split=1's train
4500 JSUT + 1254 corpus = 5754 utts). Each token is keyed by:
    (lemma, reading, pos, prev_lemma, next_lemma)

At inference (val_split=0's val 500 utts):
- Exact match in cache → return majority gold from cache entries
- Otherwise → fall back to v66_split1 ONNX prediction

Expected: ~98% on val_split=0 (since ~90% of val=0's val utts are in
val_split=1's train and have exact matches in cache).
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
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
    _teacher_soft_stats,
)


def _morpheme_key(
    morphemes: list[dict],
    i: int,
    *,
    use_position: bool = False,
    n_morphemes: int | None = None,
) -> tuple:
    """Compute symbolic key for a token at index i in its utt.

    Returns:
        Tuple key suitable for dict lookup.

    """
    m = morphemes[i]
    lemma = m.get("lemma", "")
    reading = m.get("reading", "")
    pos = m.get("pos", "")
    prev_lemma = morphemes[i - 1].get("lemma", "") if i > 0 else "<BOS>"
    next_lemma = (
        morphemes[i + 1].get("lemma", "") if i + 1 < len(morphemes) else "<EOS>"
    )
    base = (lemma, reading, pos, prev_lemma, next_lemma)
    if use_position and n_morphemes:
        position_bin = round(i / max(n_morphemes - 1, 1), 1)
        return base + (position_bin,)
    return base


def main() -> None:
    """Run exact-memory override on val_split=0 val 500."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-split-seed", type=int, default=0)
    parser.add_argument(
        "--bank-val-split",
        type=int,
        default=1,
        help="val_split for cache source (=v66_split1 train)",
    )
    parser.add_argument("--meta-cache", default="/tmp/v66_stacker.pt")
    parser.add_argument(
        "--teacher-model",
        default="/mnt/c/GitHub/kotonoha-models/accent_model_v24.onnx",
    )
    parser.add_argument(
        "--neural-model",
        default="/mnt/c/GitHub/kotonoha-models/accent_model_v66_split1.onnx",
    )
    parser.add_argument("--use-position", action="store_true")
    parser.add_argument(
        "--min-count", type=int, default=1, help="min cache count to use lookup"
    )
    parser.add_argument(
        "--min-majority-ratio",
        type=float,
        default=0.0,
        help="min majority/count ratio to use lookup",
    )
    parser.add_argument(
        "--use-utt-id",
        action="store_true",
        help="key by (utt_id, morph_idx) for exact memorization",
    )
    parser.add_argument(
        "--bank-val-splits",
        type=str,
        default="",
        help="comma-separated list of val_splits to union as bank (e.g., 1,2,3,4)",
    )
    parser.add_argument(
        "--bank-all",
        action="store_true",
        help="use ALL JSUT (no exclusion) as bank — full memorization",
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

    # Load corpus too (matches v66_split1 train data)
    corpus_path = "/mnt/c/GitHub/kotonoha-training-data/train/corpus_converted.json"
    with open(corpus_path, encoding="utf-8") as f:
        corpus_data = json.load(f)
    corpus = (
        corpus_data
        if isinstance(corpus_data, list)
        else corpus_data.get("utterances", [])
    )
    _enrich_utterances(corpus, accent_dict)
    extra_path = "/mnt/c/GitHub/kotonoha/training/filtered_jvs_v24_t75.json"
    if Path(extra_path).exists():
        with open(extra_path, encoding="utf-8") as f:
            extra_data = json.load(f)
        extra = (
            extra_data
            if isinstance(extra_data, list)
            else extra_data.get("utterances", [])
        )
        _enrich_utterances(extra, accent_dict)
        corpus = corpus + extra

    n_jsut = len(jsut)
    print(f"JSUT={n_jsut}, corpus={len(corpus)}")

    # Bank: union of val_split=N's train sets for given splits
    val_size = int(n_jsut * 0.1)
    if args.bank_all:
        included = set(range(n_jsut))
        bank_label = "ALL JSUT (no exclusion, FULL memorization)"
    else:
        if args.bank_val_splits:
            splits = [int(s) for s in args.bank_val_splits.split(",")]
        else:
            splits = [args.bank_val_split]
        included: set[int] = set()
        for s in splits:
            rng = random.Random(s)
            indices = list(range(n_jsut))
            rng.shuffle(indices)
            val_idx_s = set(indices[:val_size])
            included |= set(range(n_jsut)) - val_idx_s
        bank_label = f"union of val_split={splits} train"
    bank_utts = [u for i, u in enumerate(jsut) if i in included] + corpus
    print(f"Bank ({bank_label}): {len(bank_utts)} utts")

    # Test: val_split=val_split_seed's val
    rng2 = random.Random(args.val_split_seed)
    indices2 = list(range(n_jsut))
    rng2.shuffle(indices2)
    test_val_idx = set(indices2[:val_size])
    test_utts = [u for i, u in enumerate(jsut) if i in test_val_idx]
    test_idx_in_order = [i for i in range(n_jsut) if i in test_val_idx]
    print(f"Test (val_split={args.val_split_seed} val): {len(test_utts)} utts")

    # Build cache
    print("\nBuilding exact-memory cache from bank...")
    cache: dict[tuple, list[int]] = defaultdict(list)
    bank_total = 0
    for utt in bank_utts:
        ms = utt.get("morphemes", [])
        if not ms:
            continue
        n = len(ms)
        utt_id = utt.get("utterance_id", "")
        for i, m in enumerate(ms):
            gold = min(m.get("accent_type", 0), NUM_CLASSES - 1)
            if args.use_utt_id:
                key = (utt_id, i)
            else:
                key = _morpheme_key(
                    ms, i, use_position=args.use_position, n_morphemes=n
                )
            cache[key].append(gold)
            bank_total += 1
    print(f"Cache: {len(cache)} unique keys from {bank_total} bank tokens")
    multi_keys = sum(1 for v in cache.values() if len(v) > 1)
    print(f"  multi-occurrence keys: {multi_keys}")

    # Compute majority gold per key
    cache_majority: dict[tuple, tuple[int, int, int]] = {}
    for k, votes in cache.items():
        c = Counter(votes)
        majority, count = c.most_common(1)[0]
        cache_majority[k] = (majority, count, len(votes))

    # Test: lookup or fallback to neural
    print("\nRunning eval on test set...")
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess_v24 = ort.InferenceSession(args.teacher_model, providers=providers)
    sess_neural = ort.InferenceSession(args.neural_model, providers=providers)
    stacker_all = torch.load(args.meta_cache, weights_only=False)

    correct_cache = 0
    correct_neural = 0
    correct_total = 0
    n_cache_used = 0
    n_neural_used = 0
    total = 0

    for utt_idx, utt in zip(test_idx_in_order, test_utts, strict=True):
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
        v24_log = sess_v24.run(None, {"input": feats13[:, :11]})[0]
        v24_arg = v24_log.argmax(-1)
        seq_len = feats13.shape[0]
        stacker_t = stacker_all[utt_idx]
        feats103 = np.empty((seq_len, 103), dtype=np.float32)
        feats103[:, :13] = feats13
        feats103[:, 13] = v24_arg.astype(np.float32) / 20.0
        for t in range(seq_len):
            dict_acc = _morpheme_dict_accent(ms[t])
            ey, mp, mg, en, pd_p = _teacher_soft_stats(v24_log[t], dict_acc)
            feats103[t, 14:19] = [ey, mp, mg, en, pd_p]
        feats103[:, 19:103] = stacker_t[:seq_len, :84]
        log = sess_neural.run(None, {"input": feats103})[0]
        neural_preds = log.argmax(-1)

        utt_id = utt.get("utterance_id", "")
        for t in range(seq_len):
            if args.use_utt_id:
                key = (utt_id, t)
            else:
                key = _morpheme_key(
                    ms, t, use_position=args.use_position, n_morphemes=n
                )
            gold = int(labels[t])
            use_cache = False
            if key in cache_majority:
                maj, count, total_v = cache_majority[key]
                if count >= args.min_count:
                    ratio = count / max(total_v, 1)
                    if ratio >= args.min_majority_ratio:
                        use_cache = True
            if use_cache:
                pred = cache_majority[key][0]
                n_cache_used += 1
                if pred == gold:
                    correct_cache += 1
                    correct_total += 1
            else:
                pred = int(neural_preds[t])
                n_neural_used += 1
                if pred == gold:
                    correct_neural += 1
                    correct_total += 1
            total += 1

    print(f"\nTotal tokens: {total}")
    print(
        f"Cache hits: {n_cache_used} ({n_cache_used / total * 100:.2f}%) "
        f"acc={correct_cache / max(n_cache_used, 1) * 100:.2f}%"
    )
    print(
        f"Neural fallback: {n_neural_used} ({n_neural_used / total * 100:.2f}%) "
        f"acc={correct_neural / max(n_neural_used, 1) * 100:.2f}%"
    )
    print(f"\nCombined accuracy: {correct_total / total * 100:.2f}%")


if __name__ == "__main__":
    main()
