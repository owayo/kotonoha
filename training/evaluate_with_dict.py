"""v44: Evaluate v38 ONNX + dictionary post-processing on JSUT val split.

For each morpheme in val:
  - Run v38 inference to get argmax
  - If (lemma, reading) is in accent_dict: override with dict_accent_type
  - Compute accuracy vs hard labels
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path

import numpy as np
import onnxruntime as ort
from train_onnx_v38 import (
    NUM_CLASSES,
    _enrich_utterances,
    _extract_morpheme_features,
    _load_accent_dicts,
)


def _parse_dict_val(val: str) -> int | None:
    """Parse dict accent type string into integer 0-20 (or None if unknown)."""
    if val in ("*", ""):
        return None
    m = re.match(r'^"?(\d+)$', val)
    if m:
        v = int(m.group(1))
        if 0 <= v <= 20:
            return v
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--teacher-v24",
        default="/mnt/c/GitHub/kotonoha-models/accent_model_v24.onnx",
    )
    ap.add_argument(
        "--model",
        default="/mnt/c/GitHub/kotonoha-models/accent_model_v38.onnx",
        help="Model to evaluate (expects 14 dim input if v38-style)",
    )
    ap.add_argument(
        "--model-dim",
        type=int,
        default=14,
        help="Input dim of --model (14 for v38, 11 for v24, etc.)",
    )
    ap.add_argument(
        "--finetune-data",
        default="/home/owayo/kotonoha-training/data/jsut_accent_data_v3.json",
    )
    ap.add_argument(
        "--accent-dict",
        default=(
            "/mnt/c/GitHub/kotonoha/data/accent_dict.csv:"
            "/mnt/c/GitHub/kotonoha-training-data/train/accent_dict.csv"
        ),
    )
    ap.add_argument("--val-split-seed", type=int, default=0)
    args = ap.parse_args()

    dict_paths = [Path(p) for p in args.accent_dict.split(":") if p]
    accent_dict = _load_accent_dicts(dict_paths)
    print(f"accent_dict: {len(accent_dict)} entries")

    with open(args.finetune_data, encoding="utf-8") as f:
        jsut = json.load(f)["utterances"]
    _enrich_utterances(jsut, accent_dict)
    random.seed(args.val_split_seed)
    idx = list(range(len(jsut)))
    random.shuffle(idx)
    val_size = int(len(idx) * 0.1)
    val_idx = set(idx[:val_size])
    val_utts = [u for i, u in enumerate(jsut) if i in val_idx]
    print(f"val utts: {len(val_utts)}")

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess_v24 = ort.InferenceSession(args.teacher_v24, providers=providers)
    sess_m = ort.InferenceSession(args.model, providers=providers)

    total = 0
    corr_model = 0
    corr_dict = 0  # model + dict override
    corr_dict_only = 0  # only override when dict exists and matches (oracle)
    dict_hits = 0  # count of morphemes with valid dict entry
    dict_correct = 0  # dict value == label

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
        # v24 for building v38 14-dim input
        v24_logits = sess_v24.run(None, {"input": feats13[:, :11]})[0]
        v24_arg = v24_logits.argmax(-1)
        if args.model_dim == 11:
            feats_m = feats13[:, :11]
        elif args.model_dim == 13:
            feats_m = feats13[:, :13]
        elif args.model_dim == 14:
            feats_m = np.concatenate(
                [feats13[:, :13], (v24_arg.astype(np.float32) / 20.0).reshape(-1, 1)],
                axis=1,
            )
        else:
            raise ValueError(f"unsupported model_dim {args.model_dim}")

        m_logits = sess_m.run(None, {"input": feats_m})[0]
        m_arg = m_logits.argmax(-1)

        for i, morph in enumerate(ms):
            label = min(morph.get("accent_type", 0), NUM_CLASSES - 1)
            total += 1
            if m_arg[i] == label:
                corr_model += 1

            # Dictionary lookup: (lemma, reading) key
            lemma = morph.get("lemma", "")
            reading = morph.get("reading", "")
            key = (lemma, reading)
            dict_val = accent_dict.get(key)
            if dict_val is None and "-" in lemma:
                base = lemma.split("-")[0]
                dict_val = accent_dict.get((base, reading))
            dval_int = _parse_dict_val(dict_val) if dict_val else None

            if dval_int is not None:
                dict_hits += 1
                if dval_int == label:
                    dict_correct += 1
                # Override
                pred_with_dict = dval_int
            else:
                pred_with_dict = int(m_arg[i])
            if pred_with_dict == label:
                corr_dict += 1

            # Oracle: override only when dict matches label (upper bound)
            if dval_int is not None and dval_int == label:
                oracle_pred = dval_int
            else:
                oracle_pred = int(m_arg[i])
            if oracle_pred == label:
                corr_dict_only += 1

    print(f"\nTotal morphemes: {total}")
    print(f"Model only:           {corr_model / total * 100:.2f}%")
    print(f"Model + Dict override: {corr_dict / total * 100:.2f}%")
    print(
        f"Oracle (dict only when correct): {corr_dict_only / total * 100:.2f}% "
        "(upper bound)"
    )
    print(f"\nDict lookup hits: {dict_hits} ({dict_hits / total * 100:.2f}%)")
    if dict_hits > 0:
        print(f"Dict accuracy on hits: {dict_correct / dict_hits * 100:.2f}%")


if __name__ == "__main__":
    main()
