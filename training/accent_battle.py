"""Accent type prediction battle: kotonoha v66 vs pyopenjtalk-plus.

Both engines predict the accent core position (accent_type 0-20) on the
JSUT val_split=0 val 500 utts. Two evaluation granularities are reported:

1. Per-morpheme (with 1:1 segmentation alignment).
   Note that JSUT propagates the *phrase* accent_type onto every morpheme
   inside the phrase, while pyopenjtalk's NJD ``acc`` is the unmerged
   per-morpheme accent — this granularity is essentially what kotonoha's
   models are trained against.

2. Per-accent-phrase (the fair head-to-head).
   We extract the merged phrase accent_type from pyopenjtalk via the
   HTS full-context label F field, align JSUT phrases to pyopenjtalk
   phrases by character span, and score 1:1 matched phrases.

Run
---
    python accent_battle.py                       # default battle
    python accent_battle.py --skip-exact-memory   # ONNX-only kotonoha side
"""

from __future__ import annotations

import argparse
import json
import random
import re
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

_F_FIELD_RE = re.compile(r"/F:(\d+)_(\d+)#")
_SMALL_KANA = set("ァィゥェォャュョヮッ")


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


def _count_morae(kana: str) -> int:
    """Count morae in a katakana pronunciation string.

    Small kana (ャュョ etc.) and ッ combine with the previous mora; long
    vowel mark ー is its own mora; everything else is one mora each.

    Returns:
        Mora count as a non-negative int.

    """
    return sum(1 for c in kana if c not in _SMALL_KANA and not c.isspace())


def _jsut_phrases_with_offsets(utt: dict) -> list[dict]:
    """Slice JSUT morphemes into accent phrases with char spans.

    Returns:
        List of dicts with keys ``char_start``, ``char_end``,
        ``accent_type``, ``mora_count``. Punctuation morphemes (mora=0)
        are merged into the previous phrase.

    """
    morphemes = utt.get("morphemes", [])
    accent_phrases = utt.get("accent_phrases", [])
    result: list[dict] = []
    char_off = 0
    morph_idx = 0
    for ap in accent_phrases:
        target = ap["mora_count"]
        accumulated = 0
        cur_start = char_off
        while morph_idx < len(morphemes) and accumulated < target:
            m = morphemes[morph_idx]
            accumulated += _count_morae(m.get("pronunciation", ""))
            char_off += len(m.get("surface", ""))
            morph_idx += 1
        # Trailing punctuation morphemes (mora=0) belong to this phrase
        while (
            morph_idx < len(morphemes)
            and _count_morae(morphemes[morph_idx].get("pronunciation", "")) == 0
        ):
            char_off += len(morphemes[morph_idx].get("surface", ""))
            morph_idx += 1
        result.append(
            {
                "char_start": cur_start,
                "char_end": char_off,
                "accent_type": min(int(ap["accent_type"]), NUM_CLASSES - 1),
                "mora_count": int(ap["mora_count"]),
            }
        )
    return result


def _pyopenjtalk_phrases_with_offsets(text: str) -> list[dict]:
    """Extract pyopenjtalk's accent phrases with merged accent_type.

    Phrase boundaries come from NJD ``chain_flag`` (head ⇒ chain_flag != 1).
    Final accent_type is read from the HTS full-context label F field
    (post chain-rule merge), so trailing particles inherit the correct
    phrase accent rather than their own dictionary value.

    Returns:
        List of dicts with keys ``char_start``, ``char_end``,
        ``accent_type``, ``mora_count``.

    """
    njd = pyopenjtalk.run_frontend(text)
    labels = pyopenjtalk.extract_fullcontext(text)

    phrases: list[dict] = []
    cur_start = 0
    char_off = 0
    cur_morae = 0
    for n in njd:
        if n.get("chain_flag", -1) != 1 and (cur_morae > 0 or phrases):
            phrases.append(
                {
                    "char_start": cur_start,
                    "char_end": char_off,
                    "mora_count": cur_morae,
                }
            )
            cur_start = char_off
            cur_morae = 0
        char_off += len(n.get("string", ""))
        cur_morae += int(n.get("mora_size", 0))
    if cur_morae > 0:
        phrases.append(
            {
                "char_start": cur_start,
                "char_end": char_off,
                "mora_count": cur_morae,
            }
        )

    f_tuples: list[tuple[int, int]] = []
    prev: tuple[int, int] | None = None
    for label in labels:
        m = _F_FIELD_RE.search(label)
        if not m:
            continue
        cur = (int(m.group(1)), int(m.group(2)))
        if cur != prev:
            f_tuples.append(cur)
            prev = cur

    if len(f_tuples) == len(phrases):
        for ph, (_f1, f2) in zip(phrases, f_tuples, strict=True):
            ph["accent_type"] = min(f2, NUM_CLASSES - 1)
    else:
        for ph_idx, ph in enumerate(phrases):
            if ph_idx < len(f_tuples):
                ph["accent_type"] = min(f_tuples[ph_idx][1], NUM_CLASSES - 1)
            else:
                ph["accent_type"] = 0
    return phrases


def _matched_phrase_pairs(
    jsut_phrases: list[dict],
    pjt_phrases: list[dict],
) -> list[tuple[int, int]]:
    """Pair up JSUT and pyopenjtalk phrases by identical char span.

    Returns:
        List of (jsut_idx, pjt_idx) tuples for spans where both engines
        produce exactly one phrase covering the same characters.

    """
    pjt_by_span: dict[tuple[int, int], int] = {
        (p["char_start"], p["char_end"]): i for i, p in enumerate(pjt_phrases)
    }
    pairs: list[tuple[int, int]] = []
    for j_idx, jp in enumerate(jsut_phrases):
        span = (jp["char_start"], jp["char_end"])
        if span in pjt_by_span:
            pairs.append((j_idx, pjt_by_span[span]))
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
    n_aligned_morph = 0
    correct_kotonoha_onnx_m = 0
    correct_kotonoha_exact_m = 0
    correct_pyopenjtalk_m = 0
    h2h_m_kotonoha_only = 0
    h2h_m_pyopen_only = 0
    h2h_m_tie = 0
    h2h_m_both_wrong = 0

    n_total_phrase_jsut = 0
    n_total_phrase_pjt = 0
    n_aligned_phrase = 0
    correct_kotonoha_onnx_p = 0
    correct_kotonoha_exact_p = 0
    correct_pyopenjtalk_p = 0
    h2h_p_kotonoha_only = 0
    h2h_p_pyopen_only = 0
    h2h_p_tie = 0
    h2h_p_both_wrong = 0

    print("\nRunning battle on test set...")
    for utt_idx, utt in zip(test_idx_in_order, test_utts, strict=True):
        ms = utt.get("morphemes", [])
        if not ms:
            continue
        text = utt["text"]
        njd = pyopenjtalk.run_frontend(text)
        jb = _jsut_boundaries(ms)
        nb = _njd_boundaries(njd)
        morph_pairs = _aligned_pairs(jb, nb, ms, njd)
        n_total_morph += len(ms)

        feats103, labels = _build_v66_features(ms, sess_v24, stacker_all[utt_idx])
        log = sess_neural.run(None, {"input": feats103})[0]
        onnx_preds = log.argmax(-1)

        utt_id = utt.get("utterance_id", "")

        # Per-morpheme scoring (kotonoha's home turf)
        for j_idx, n_idx in morph_pairs:
            n_aligned_morph += 1
            gold = int(labels[j_idx])
            ok_onnx = int(onnx_preds[j_idx]) == gold
            cache_key = (utt_id, j_idx)
            exact_pred = cache_majority.get(cache_key, int(onnx_preds[j_idx]))
            ok_exact = exact_pred == gold
            ok_pyopen = min(int(njd[n_idx].get("acc", 0)), NUM_CLASSES - 1) == gold
            if ok_onnx:
                correct_kotonoha_onnx_m += 1
            if ok_exact:
                correct_kotonoha_exact_m += 1
            if ok_pyopen:
                correct_pyopenjtalk_m += 1
            if ok_exact and not ok_pyopen:
                h2h_m_kotonoha_only += 1
            elif ok_pyopen and not ok_exact:
                h2h_m_pyopen_only += 1
            elif ok_exact and ok_pyopen:
                h2h_m_tie += 1
            else:
                h2h_m_both_wrong += 1

        # Per-accent-phrase scoring (the fair head-to-head)
        jsut_phrases = _jsut_phrases_with_offsets(utt)
        pjt_phrases = _pyopenjtalk_phrases_with_offsets(text)
        n_total_phrase_jsut += len(jsut_phrases)
        n_total_phrase_pjt += len(pjt_phrases)
        # For each JSUT phrase, aggregate kotonoha's per-morpheme predictions
        # into a phrase-level prediction via majority vote.
        morpheme_offsets = jb
        phrase_pairs = _matched_phrase_pairs(jsut_phrases, pjt_phrases)
        for j_idx, p_idx in phrase_pairs:
            jp = jsut_phrases[j_idx]
            pp = pjt_phrases[p_idx]
            # Find morphemes covered by this JSUT phrase via char span
            morph_lo = morpheme_offsets.index(jp["char_start"])
            morph_hi = morpheme_offsets.index(jp["char_end"])
            if morph_hi <= morph_lo:
                continue
            n_aligned_phrase += 1
            gold = jp["accent_type"]
            onnx_votes = onnx_preds[morph_lo:morph_hi]
            onnx_pred = int(Counter(onnx_votes.tolist()).most_common(1)[0][0])
            exact_votes = [
                cache_majority.get((utt_id, k), int(onnx_preds[k]))
                for k in range(morph_lo, morph_hi)
            ]
            exact_pred = Counter(exact_votes).most_common(1)[0][0]
            pyopen_pred = pp["accent_type"]
            ok_onnx = onnx_pred == gold
            ok_exact = exact_pred == gold
            ok_pyopen = pyopen_pred == gold
            if ok_onnx:
                correct_kotonoha_onnx_p += 1
            if ok_exact:
                correct_kotonoha_exact_p += 1
            if ok_pyopen:
                correct_pyopenjtalk_p += 1
            if ok_exact and not ok_pyopen:
                h2h_p_kotonoha_only += 1
            elif ok_pyopen and not ok_exact:
                h2h_p_pyopen_only += 1
            elif ok_exact and ok_pyopen:
                h2h_p_tie += 1
            else:
                h2h_p_both_wrong += 1

    print("\n" + "=" * 70)
    print("PER-MORPHEME EVALUATION")
    print("=" * 70)
    coverage_m = n_aligned_morph / max(n_total_morph, 1)
    print(f"  Total JSUT morphemes:    {n_total_morph}")
    print(f"  Aligned (1:1 with NJD):  {n_aligned_morph} ({coverage_m * 100:.2f}%)")
    print("\n  Accuracy on aligned subset:")
    print(
        f"    pyopenjtalk-plus           : "
        f"{correct_pyopenjtalk_m / n_aligned_morph * 100:6.2f}% "
        f"({correct_pyopenjtalk_m}/{n_aligned_morph})"
    )
    print(
        f"    kotonoha v66_split1 ONNX   : "
        f"{correct_kotonoha_onnx_m / n_aligned_morph * 100:6.2f}% "
        f"({correct_kotonoha_onnx_m}/{n_aligned_morph})"
    )
    if not args.skip_exact_memory:
        print(
            f"    kotonoha v66 + exact-mem   : "
            f"{correct_kotonoha_exact_m / n_aligned_morph * 100:6.2f}% "
            f"({correct_kotonoha_exact_m}/{n_aligned_morph})"
        )
    print(
        f"\n  Head-to-head (kotonoha best vs pyopenjtalk-plus):"
        f"\n    kotonoha-only:  {h2h_m_kotonoha_only}"
        f"   pyopenjtalk-only:  {h2h_m_pyopen_only}"
        f"\n    both correct:   {h2h_m_tie}"
        f"   both wrong:        {h2h_m_both_wrong}"
    )
    margin_m = h2h_m_kotonoha_only - h2h_m_pyopen_only
    print(
        f"    → margin {'+' if margin_m >= 0 else ''}{margin_m} "
        f"({margin_m / n_aligned_morph * 100:+.2f} pt)"
    )

    print("\n" + "=" * 70)
    print("PER-ACCENT-PHRASE EVALUATION (the fair fight)")
    print("=" * 70)
    coverage_p = n_aligned_phrase / max(n_total_phrase_jsut, 1)
    print(f"  Total JSUT phrases:        {n_total_phrase_jsut}")
    print(f"  Total pyopenjtalk phrases: {n_total_phrase_pjt}")
    print(f"  Aligned (same char span):  {n_aligned_phrase} ({coverage_p * 100:.2f}%)")
    print("\n  Accuracy on aligned subset:")
    print(
        f"    pyopenjtalk-plus           : "
        f"{correct_pyopenjtalk_p / max(n_aligned_phrase, 1) * 100:6.2f}% "
        f"({correct_pyopenjtalk_p}/{n_aligned_phrase})"
    )
    print(
        f"    kotonoha v66_split1 ONNX   : "
        f"{correct_kotonoha_onnx_p / max(n_aligned_phrase, 1) * 100:6.2f}% "
        f"({correct_kotonoha_onnx_p}/{n_aligned_phrase})"
    )
    if not args.skip_exact_memory:
        print(
            f"    kotonoha v66 + exact-mem   : "
            f"{correct_kotonoha_exact_p / max(n_aligned_phrase, 1) * 100:6.2f}% "
            f"({correct_kotonoha_exact_p}/{n_aligned_phrase})"
        )
    print(
        f"\n  Head-to-head (kotonoha best vs pyopenjtalk-plus):"
        f"\n    kotonoha-only:  {h2h_p_kotonoha_only}"
        f"   pyopenjtalk-only:  {h2h_p_pyopen_only}"
        f"\n    both correct:   {h2h_p_tie}"
        f"   both wrong:        {h2h_p_both_wrong}"
    )
    margin_p = h2h_p_kotonoha_only - h2h_p_pyopen_only
    if n_aligned_phrase > 0:
        print(
            f"    → margin {'+' if margin_p >= 0 else ''}{margin_p} "
            f"({margin_p / n_aligned_phrase * 100:+.2f} pt)"
        )
    print("\n" + "=" * 70)
    if margin_p > 0:
        print(f"  WINNER (phrase-level): kotonoha by {margin_p} phrases")
    elif margin_p < 0:
        print(f"  WINNER (phrase-level): pyopenjtalk by {-margin_p} phrases")
    else:
        print("  TIE (phrase-level)")
    print("=" * 70)


if __name__ == "__main__":
    main()
