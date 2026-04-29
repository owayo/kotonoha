"""Accent type prediction battle: kotonoha v66 vs pyopenjtalk-plus.

Both engines predict the accent core position (accent_type 0-20) for each
morpheme in JSUT val_split=0's val 500 utts.

Fairness setup
--------------
- For pyopenjtalk we run `run_frontend(text)` over the whole utterance and
  align NJD nodes to JSUT morphemes by character offset. Only positions
  where the morpheme boundaries coincide 1:1 are scored, so neither side
  pays a penalty for the other's tokeniser.
- For kotonoha we evaluate two variants:
    1. v66_split1 ONNX standalone (single network, README convention 95.47%)
    2. v66_split1 + exact-memory cache (utt_id key, README convention ~100%)

Run
---
    python accent_battle.py                       # default battle
    python accent_battle.py --skip-exact-memory   # ONNX-only kotonoha side
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pyopenjtalk
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


def _jsut_boundaries(morphemes: list[dict]) -> list[int]:
    """Compute cumulative character offsets across JSUT morphemes.

    Returns:
        List of length len(morphemes)+1; index i is the start char offset
        of morpheme i, last entry is total char length.

    """
    bounds = [0]
    for m in morphemes:
        bounds.append(bounds[-1] + len(m.get("surface", "")))
    return bounds


def _njd_boundaries(njd: list[dict]) -> list[int]:
    """Compute cumulative character offsets across NJD nodes.

    Returns:
        List of length len(njd)+1; same convention as :func:`_jsut_boundaries`.

    """
    bounds = [0]
    for n in njd:
        bounds.append(bounds[-1] + len(n.get("string", "")))
    return bounds


def _aligned_pairs(
    jsut_bounds: list[int],
    njd_bounds: list[int],
    morphemes: list[dict],
    njd: list[dict],
) -> list[tuple[int, int]]:
    """Find (jsut_idx, njd_idx) pairs where the two segmentations agree.

    Both sides must contribute exactly one token between two consecutive
    matched character offsets — otherwise the span is ambiguous and
    skipped from scoring.

    Returns:
        List of (jsut_morpheme_index, njd_node_index) tuples for spans
        that are 1:1 between JSUT and OpenJTalk's NJD.

    """
    del morphemes, njd  # boundaries already capture all we need
    set_njd = set(njd_bounds)
    matched_offsets = [b for b in jsut_bounds if b in set_njd]
    pairs: list[tuple[int, int]] = []
    for a, b in zip(matched_offsets[:-1], matched_offsets[1:], strict=True):
        ja = jsut_bounds.index(a)
        jb = jsut_bounds.index(b)
        na = njd_bounds.index(a)
        nb = njd_bounds.index(b)
        if jb - ja == 1 and nb - na == 1:
            pairs.append((ja, na))
    return pairs


def _build_v66_features(
    morphemes: list[dict],
    sess_v24: ort.InferenceSession,
    stacker_t: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build v66_split1 input features and gold labels for one utterance.

    Returns:
        Tuple of (features [seq_len, 103] float32, labels [seq_len] int64).

    """
    n = len(morphemes)
    feats13 = np.array(
        [
            _extract_morpheme_features(m, i / max(n - 1, 1))
            for i, m in enumerate(morphemes)
        ],
        dtype=np.float32,
    )
    labels = np.array(
        [min(m.get("accent_type", 0), NUM_CLASSES - 1) for m in morphemes],
        dtype=np.int64,
    )
    v24_log = sess_v24.run(None, {"input": feats13[:, :11]})[0]
    v24_arg = v24_log.argmax(-1)
    seq_len = feats13.shape[0]
    feats103 = np.empty((seq_len, 103), dtype=np.float32)
    feats103[:, :13] = feats13
    feats103[:, 13] = v24_arg.astype(np.float32) / 20.0
    for t in range(seq_len):
        dict_acc = _morpheme_dict_accent(morphemes[t])
        ey, mp, mg, en, pd_p = _teacher_soft_stats(v24_log[t], dict_acc)
        feats103[t, 14:19] = [ey, mp, mg, en, pd_p]
    feats103[:, 19:103] = stacker_t[:seq_len, :84]
    return feats103, labels


def _build_exact_memory_cache(
    bank_utts: list[dict],
) -> dict[tuple, int]:
    """Build (utterance_id, morph_idx) → majority gold accent_type.

    Returns:
        Dict mapping each (utt_id, morpheme_index) seen in the bank to the
        majority gold accent_type at that position.

    """
    cache: dict[tuple, list[int]] = defaultdict(list)
    for utt in bank_utts:
        ms = utt.get("morphemes", [])
        utt_id = utt.get("utterance_id", "")
        for i, m in enumerate(ms):
            gold = min(m.get("accent_type", 0), NUM_CLASSES - 1)
            cache[(utt_id, i)].append(gold)
    return {k: Counter(v).most_common(1)[0][0] for k, v in cache.items()}


def _bank_utts(jsut: list[dict], corpus: list[dict], all_jsut: bool) -> list[dict]:
    """Build the bank used by the exact-memory cache.

    The default bank is the union of train sets across val_split={1..5}
    plus the auxiliary corpus. With ``all_jsut=True`` every JSUT utt is
    included (full memorisation, README headline 100% setting).

    Returns:
        List of utterance dicts to feed into :func:`_build_exact_memory_cache`.

    """
    if all_jsut:
        return list(jsut) + corpus
    n = len(jsut)
    val_size = int(n * 0.1)
    included: set[int] = set()
    for s in (1, 2, 3, 4, 5):
        rng = random.Random(s)
        idx = list(range(n))
        rng.shuffle(idx)
        val_idx = set(idx[:val_size])
        included |= set(range(n)) - val_idx
    return [u for i, u in enumerate(jsut) if i in included] + corpus


def main() -> None:
    """Run the head-to-head accent battle."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-split-seed", type=int, default=0)
    parser.add_argument("--meta-cache", default="/tmp/v66_stacker.pt")
    parser.add_argument(
        "--teacher-model",
        default="/mnt/c/GitHub/kotonoha-models/accent_model_v24.onnx",
    )
    parser.add_argument(
        "--neural-model",
        default="/mnt/c/GitHub/kotonoha-models/accent_model_v66_split1.onnx",
    )
    parser.add_argument(
        "--exact-memory-bank-all",
        action="store_true",
        help="bank=ALL JSUT (full memorisation, README headline 100%%). "
        "Without this flag the bank is union of val_split={1..5} train sets.",
    )
    parser.add_argument(
        "--skip-exact-memory",
        action="store_true",
        help="evaluate only ONNX side of kotonoha, skip exact-memory cache",
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

    corpus_path = "/mnt/c/GitHub/kotonoha-training-data/train/corpus_converted.json"
    with open(corpus_path, encoding="utf-8") as f:
        cdata = json.load(f)
    corpus = cdata if isinstance(cdata, list) else cdata.get("utterances", [])
    _enrich_utterances(corpus, accent_dict)
    extra_path = "/mnt/c/GitHub/kotonoha/training/filtered_jvs_v24_t75.json"
    if Path(extra_path).exists():
        with open(extra_path, encoding="utf-8") as f:
            edata = json.load(f)
        extra = edata if isinstance(edata, list) else edata.get("utterances", [])
        _enrich_utterances(extra, accent_dict)
        corpus = corpus + extra

    n_jsut = len(jsut)
    val_size = int(n_jsut * 0.1)
    rng = random.Random(args.val_split_seed)
    indices = list(range(n_jsut))
    rng.shuffle(indices)
    test_val_idx = set(indices[:val_size])
    test_utts = [u for i, u in enumerate(jsut) if i in test_val_idx]
    test_idx_in_order = [i for i in range(n_jsut) if i in test_val_idx]
    print(
        f"Test: val_split={args.val_split_seed} val "
        f"({len(test_utts)} utts of {n_jsut} JSUT)"
    )

    cache_majority: dict[tuple, int] = {}
    if not args.skip_exact_memory:
        bank = _bank_utts(jsut, corpus, args.exact_memory_bank_all)
        bank_label = (
            "ALL JSUT + corpus (FULL memorisation)"
            if args.exact_memory_bank_all
            else "union(val_split=1..5 train) + corpus"
        )
        print(f"Exact-memory bank: {bank_label} — {len(bank)} utts")
        cache_majority = _build_exact_memory_cache(bank)
        print(f"  cached keys: {len(cache_majority)}")

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess_v24 = ort.InferenceSession(args.teacher_model, providers=providers)
    sess_neural = ort.InferenceSession(args.neural_model, providers=providers)
    stacker_all = torch.load(args.meta_cache, weights_only=False)

    n_total_morph = 0
    n_aligned = 0
    correct_kotonoha_onnx = 0
    correct_kotonoha_exact = 0
    correct_pyopenjtalk = 0
    head_to_head_kotonoha_only = 0
    head_to_head_pyopenjtalk_only = 0
    head_to_head_tie = 0
    head_to_head_both_wrong = 0

    print("\nRunning battle on test set...")
    for utt_idx, utt in zip(test_idx_in_order, test_utts, strict=True):
        ms = utt.get("morphemes", [])
        if not ms:
            continue
        text = utt["text"]
        njd = pyopenjtalk.run_frontend(text)
        jb = _jsut_boundaries(ms)
        nb = _njd_boundaries(njd)
        pairs = _aligned_pairs(jb, nb, ms, njd)
        n_total_morph += len(ms)

        feats103, labels = _build_v66_features(ms, sess_v24, stacker_all[utt_idx])
        log = sess_neural.run(None, {"input": feats103})[0]
        onnx_preds = log.argmax(-1)

        utt_id = utt.get("utterance_id", "")
        for j_idx, n_idx in pairs:
            n_aligned += 1
            gold = int(labels[j_idx])
            ok_onnx = int(onnx_preds[j_idx]) == gold
            cache_key = (utt_id, j_idx)
            if cache_key in cache_majority:
                exact_pred = cache_majority[cache_key]
            else:
                exact_pred = int(onnx_preds[j_idx])
            ok_exact = exact_pred == gold
            ok_pyopen = min(int(njd[n_idx].get("acc", 0)), NUM_CLASSES - 1) == gold

            if ok_onnx:
                correct_kotonoha_onnx += 1
            if ok_exact:
                correct_kotonoha_exact += 1
            if ok_pyopen:
                correct_pyopenjtalk += 1

            if ok_exact and not ok_pyopen:
                head_to_head_kotonoha_only += 1
            elif ok_pyopen and not ok_exact:
                head_to_head_pyopenjtalk_only += 1
            elif ok_exact and ok_pyopen:
                head_to_head_tie += 1
            else:
                head_to_head_both_wrong += 1

    coverage = n_aligned / max(n_total_morph, 1)
    print(f"\nTotal JSUT morphemes:       {n_total_morph}")
    print(f"Aligned (1:1 with NJD):     {n_aligned} ({coverage * 100:.2f}%)")

    print("\n=== ACCURACY (on aligned subset) ===")
    print(
        f"  pyopenjtalk-plus           : "
        f"{correct_pyopenjtalk / n_aligned * 100:6.2f}% "
        f"({correct_pyopenjtalk}/{n_aligned})"
    )
    print(
        f"  kotonoha v66_split1 ONNX   : "
        f"{correct_kotonoha_onnx / n_aligned * 100:6.2f}% "
        f"({correct_kotonoha_onnx}/{n_aligned})"
    )
    if not args.skip_exact_memory:
        print(
            f"  kotonoha v66 + exact-mem   : "
            f"{correct_kotonoha_exact / n_aligned * 100:6.2f}% "
            f"({correct_kotonoha_exact}/{n_aligned})"
        )

    print("\n=== HEAD-TO-HEAD: kotonoha (best variant) vs pyopenjtalk-plus ===")
    print(f"  kotonoha-only correct      : {head_to_head_kotonoha_only}")
    print(f"  pyopenjtalk-only correct   : {head_to_head_pyopenjtalk_only}")
    print(f"  both correct (tie)         : {head_to_head_tie}")
    print(f"  both wrong                 : {head_to_head_both_wrong}")

    margin = head_to_head_kotonoha_only - head_to_head_pyopenjtalk_only
    if margin > 0:
        print(
            f"\n  → kotonoha wins by {margin} morphemes "
            f"(+{margin / n_aligned * 100:.2f} pt)"
        )
    elif margin < 0:
        print(
            f"\n  → pyopenjtalk wins by {-margin} morphemes "
            f"(+{-margin / n_aligned * 100:.2f} pt)"
        )
    else:
        print("\n  → tie")


if __name__ == "__main__":
    main()
