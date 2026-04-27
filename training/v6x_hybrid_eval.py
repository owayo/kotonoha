"""Hybrid inference eval: trust 11-student consensus when agreement is high.

For tokens where 11-student consensus agreement >= threshold, use consensus
argmax. Otherwise use v63 (or other strong student) prediction. Measure
accuracy on val_split=0 val 500 utts.

Goal: bridge the gap between v63 (~83%) and 11-student consensus (~86%).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F  # noqa: N812
from torch.utils.data import DataLoader

from train_onnx_v60 import (
    NUM_CLASSES,
    _enrich_utterances,
    _load_accent_dicts,
    _load_dotenv,
)
from train_onnx_v63 import (
    AccentModel,
    _AccentDataset,
    _collate_fn,
)


@torch.no_grad()
def _predict_softmax(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Run model on val loader; return per-utt softmax + labels lists.

    Returns:
        Tuple of (softmax list per utt, labels list per utt).

    """
    model.eval()
    sm_per_utt: list[np.ndarray] = []
    label_per_utt: list[np.ndarray] = []
    for batch in loader:
        feats = batch["features"].to(device)
        labels = batch["labels"].to(device)
        lengths = batch["lengths"]
        r_ids = batch["reading_ids"].to(device)
        logits = model(feats, lengths, r_ids)
        for b in range(feats.size(0)):
            length = int(lengths[b])
            sm = F.softmax(logits[b, :length], dim=-1).cpu().numpy()
            sm_per_utt.append(sm)
            label_per_utt.append(labels[b, :length].cpu().numpy())
    return sm_per_utt, label_per_utt


def main() -> None:
    """Run hybrid eval and report per-threshold accuracy."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-split-seed", type=int, default=0)
    parser.add_argument("--state", default="/tmp/v63_states/state_000.pt")
    parser.add_argument("--meta-cache", default="/tmp/v64_meta_features.pt")
    parser.add_argument("--consensus-cache", default="/tmp/v64_consensus.pt")
    parser.add_argument(
        "--teacher-model",
        default="/mnt/c/GitHub/kotonoha-models/accent_model_v24.onnx",
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

    random.seed(args.val_split_seed)
    indices = list(range(len(jsut)))
    random.shuffle(indices)
    val_size = int(len(indices) * 0.1)
    val_idx = set(indices[:val_size])
    val_utts = [u for i, u in enumerate(jsut) if i in val_idx]
    val_idx_in_order = [i for i in range(len(jsut)) if i in val_idx]

    # Compute teacher val logits
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess_v24 = ort.InferenceSession(args.teacher_model, providers=providers)
    teacher_val_logits: list[np.ndarray] = []
    from train_onnx_v60 import _extract_morpheme_features

    for utt in val_utts:
        ms = utt.get("morphemes", [])
        if not ms:
            teacher_val_logits.append(np.zeros((0, NUM_CLASSES), dtype=np.float32))
            continue
        n = len(ms)
        feats = np.array(
            [
                _extract_morpheme_features(m, i / max(n - 1, 1))
                for i, m in enumerate(ms)
            ],
            dtype=np.float32,
        )
        teacher_val_logits.append(sess_v24.run(None, {"input": feats[:, :11]})[0])

    meta_all = torch.load(args.meta_cache, weights_only=False)
    cons_all = torch.load(args.consensus_cache, weights_only=False)
    meta_val = [meta_all[i] for i in val_idx_in_order]
    cons_val = [cons_all[i] for i in val_idx_in_order]

    val_ds = _AccentDataset(
        val_utts,
        augment=False,
        teacher_logits_per_morpheme=teacher_val_logits,
        meta_features_per_utt=meta_val,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=64,
        shuffle=False,
        collate_fn=_collate_fn,
        num_workers=0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = torch.load(args.state, map_location="cpu", weights_only=False)
    print(f"Loaded {args.state}: train_acc={bundle['acc'] * 100:.2f}%")
    model = AccentModel(
        embed_dim=64,
        hidden_dim=256,
        num_layers=3,
        num_classes=NUM_CLASSES,
        dropout=0.4,
        attention_heads=4,
        reading_dropout=0.0,
    ).to(device)
    model.load_state_dict(bundle["state"], strict=False)
    sm_per_utt, label_per_utt = _predict_softmax(model, val_loader, device)

    print(
        f"\nVal utts: {len(sm_per_utt)}, total morphemes: "
        f"{sum(len(s) for s in sm_per_utt)}"
    )

    student_correct = 0
    consensus_correct = 0
    total = 0
    thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    hybrid_correct: list[int] = [0] * len(thresholds)
    cons_correct_high: list[int] = [0] * len(thresholds)
    cons_total_high: list[int] = [0] * len(thresholds)
    student_correct_low: list[int] = [0] * len(thresholds)
    student_total_low: list[int] = [0] * len(thresholds)

    for sm, labels, cons in zip(sm_per_utt, label_per_utt, cons_val, strict=False):
        student_pred = sm.argmax(-1)
        seq = len(labels)
        if seq == 0:
            continue
        for t in range(seq):
            gold = int(labels[t])
            sp = int(student_pred[t])
            cons_arg = int(cons[t, 0])
            agree = float(cons[t, 1])
            total += 1
            if sp == gold:
                student_correct += 1
            if cons_arg == gold:
                consensus_correct += 1
            for k, thr in enumerate(thresholds):
                if agree >= thr:
                    cons_total_high[k] += 1
                    if cons_arg == gold:
                        cons_correct_high[k] += 1
                    hybrid_pred = cons_arg
                else:
                    student_total_low[k] += 1
                    if sp == gold:
                        student_correct_low[k] += 1
                    hybrid_pred = sp
                if hybrid_pred == gold:
                    hybrid_correct[k] += 1

    print(f"\nStudent (v63 state) alone: {student_correct / total * 100:.2f}%")
    print(f"Consensus alone: {consensus_correct / total * 100:.2f}%")
    print("\nHybrid (use consensus when agree >= threshold else student):")
    for k, thr in enumerate(thresholds):
        ch = cons_correct_high[k]
        th_h = cons_total_high[k]
        sh = student_correct_low[k]
        tl = student_total_low[k]
        cons_ratio = th_h / total
        cons_acc = ch / th_h if th_h > 0 else 0.0
        st_acc = sh / tl if tl > 0 else 0.0
        h_acc = hybrid_correct[k] / total
        print(
            f"  thr={thr}: hybrid={h_acc * 100:.2f}% "
            f"(cons {cons_ratio * 100:.1f}% @ {cons_acc * 100:.2f}% + "
            f"student {(1 - cons_ratio) * 100:.1f}% @ {st_acc * 100:.2f}%)"
        )


if __name__ == "__main__":
    main()
