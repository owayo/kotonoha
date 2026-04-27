"""Evaluate raw 11-student consensus accuracy on val_split=0 val 500 utts.

Sanity check: how much does v64's "teacher signal" actually know?
Theoretical upper bound for student v64 trained with this teacher.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch

from train_onnx_v60 import (
    NUM_CLASSES,
    _enrich_utterances,
    _load_accent_dicts,
    _load_dotenv,
)


def main() -> None:
    """Compare consensus argmax to gold on val 500."""
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

    cons_all = torch.load("/tmp/v64_consensus.pt", weights_only=False)

    correct = 0
    total = 0
    correct_high_agree = 0
    total_high_agree = 0
    correct_low_agree = 0
    total_low_agree = 0

    n_jsut = len(jsut)
    val_idx_in_order = [i for i in range(n_jsut) if i in val_idx]

    for i, utt_i in enumerate(val_idx_in_order):
        utt = val_utts[i]
        ms = utt.get("morphemes", [])
        if not ms:
            continue
        cons = cons_all[utt_i]
        for t, m in enumerate(ms):
            if t >= len(cons):
                break
            gold = min(m.get("accent_type", 0), NUM_CLASSES - 1)
            cons_arg = int(cons[t, 0])
            agree = float(cons[t, 1])
            total += 1
            if cons_arg == gold:
                correct += 1
            if agree >= 0.7:
                total_high_agree += 1
                if cons_arg == gold:
                    correct_high_agree += 1
            else:
                total_low_agree += 1
                if cons_arg == gold:
                    correct_low_agree += 1

    print(
        f"11-student consensus argmax accuracy on val 500: "
        f"{correct / total * 100:.2f}% ({correct}/{total})"
    )
    if total_high_agree:
        print(
            f"  High-agree (>=0.7): {correct_high_agree / total_high_agree * 100:.2f}% "
            f"({correct_high_agree}/{total_high_agree})"
        )
    if total_low_agree:
        print(
            f"  Low-agree  (<0.7):  {correct_low_agree / total_low_agree * 100:.2f}% "
            f"({correct_low_agree}/{total_low_agree})"
        )


if __name__ == "__main__":
    main()
