"""v68: high-agree token relabel for label noise cleaning (train-only).

Identify training tokens where:
  1. 11-student consensus argmax != gold label
  2. agreement >= 0.82 (9+/11 students agree)
  3. ens_prob[consensus] - ens_prob[gold] >= 0.3

These tokens likely have wrong gold labels. Replace gold with consensus
ONLY for training utterances (val_split=0's val utts are kept unchanged
so evaluation remains on original gold labels).

Used by v68 trainer.
"""

from __future__ import annotations

import argparse
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
    """Generate cleaned JSUT JSON with high-confidence relabels."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", default="/tmp/v68_jsut_relabeled.json", help="output JSON path"
    )
    parser.add_argument("--ens-prob-cache", default="/tmp/v65_ens_prob.pt")
    parser.add_argument("--consensus-cache", default="/tmp/v64_consensus.pt")
    parser.add_argument(
        "--agree-threshold", type=float, default=0.82, help="min agreement (0..1)"
    )
    parser.add_argument(
        "--margin-threshold",
        type=float,
        default=0.3,
        help="min ens_prob[consensus] - ens_prob[gold]",
    )
    parser.add_argument(
        "--val-split-seed",
        type=int,
        default=0,
        help="val_split_seed used during training (these utts are NOT relabeled)",
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
        jsut_data = json.load(f)
    jsut = jsut_data["utterances"]
    _enrich_utterances(jsut, accent_dict)

    ens_all = torch.load(args.ens_prob_cache, weights_only=False)
    cons_all = torch.load(args.consensus_cache, weights_only=False)
    print(
        f"Loaded ens_prob ({len(ens_all)} entries) "
        f"and consensus ({len(cons_all)} entries)"
    )

    n_jsut = len(jsut)
    # Identify val_split=0 val utts to skip (so eval stays on original gold)
    rng = random.Random(args.val_split_seed)
    indices = list(range(n_jsut))
    rng.shuffle(indices)
    val_size = int(n_jsut * 0.1)
    val_idx = set(indices[:val_size])
    print(f"Skipping val_split={args.val_split_seed} val ({len(val_idx)} utts)")

    relabeled = 0
    total = 0
    skipped_val = 0
    relabeled_class_dist: dict[int, int] = {}

    for utt_i, utt in enumerate(jsut):
        ms = utt.get("morphemes", [])
        if not ms or utt_i >= n_jsut:
            continue
        if utt_i in val_idx:
            skipped_val += len(ms)
            continue
        ens_p = ens_all[utt_i]
        cons = cons_all[utt_i]
        if len(ens_p) < len(ms) or len(cons) < len(ms):
            continue
        for t, m in enumerate(ms):
            gold = min(m.get("accent_type", 0), NUM_CLASSES - 1)
            cons_arg = int(cons[t, 0])
            agree = float(cons[t, 1])
            total += 1
            if agree < args.agree_threshold:
                continue
            if cons_arg == gold:
                continue
            margin = float(ens_p[t, cons_arg] - ens_p[t, gold])
            if margin < args.margin_threshold:
                continue
            # Relabel to consensus
            m["accent_type"] = cons_arg
            relabeled += 1
            relabeled_class_dist[gold] = relabeled_class_dist.get(gold, 0) + 1

    print(
        f"Relabeled {relabeled}/{total} tokens "
        f"({relabeled / total * 100:.2f}%) "
        f"(agree >= {args.agree_threshold}, margin >= {args.margin_threshold})"
    )
    print(
        "Top-5 most-relabeled gold classes: "
        + ", ".join(
            f"{c}:{n}"
            for c, n in sorted(relabeled_class_dist.items(), key=lambda x: -x[1])[:5]
        )
    )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(jsut_data, f, ensure_ascii=False)
    print(f"Saved cleaned JSUT to {args.out}")


if __name__ == "__main__":
    main()
