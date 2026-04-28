"""v66_split1 base + confidence-gated hybrid + MC dropout TTA.

Combines codex idea #3 (confidence-gated hybrid) and #5 (MC dropout TTA)
on top of v66_split1 (95.47% baseline). Goal: 96%+ on val_split=0.

Strategy:
  1. Use v66_split1 as primary predictor
  2. For low-confidence tokens (max_prob < threshold), fallback to:
     - 11-student consensus, OR
     - v66 (strict no-leak), OR
     - average of leaked v66_split{1,2,3}
  3. MC dropout TTA: run v66_split1 with dropout=on multiple times, average
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
    _extract_morpheme_features,
    _load_accent_dicts,
    _load_dotenv,
    _morpheme_dict_accent,
    _softmax_1d,
    _teacher_soft_stats,
)
from train_onnx_v66 import (
    AccentModel,
    _AccentDataset,
    _collate_fn,
)


@torch.no_grad()
def _mc_dropout_softmax(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_passes: int = 12,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Run model with dropout enabled, average softmax across n passes.

    Returns:
        Tuple of (per-utt softmax list, per-utt labels list).

    """
    sm_per_utt: list[np.ndarray] = []
    label_per_utt: list[np.ndarray] = []
    # Pre-collect all batches to enable multi-pass
    batches = []
    for batch in loader:
        batches.append(
            {
                "features": batch["features"].to(device),
                "labels": batch["labels"].to(device),
                "lengths": batch["lengths"],
                "reading_ids": batch["reading_ids"].to(device),
            }
        )
    model.train()
    for batch in batches:
        feats = batch["features"]
        labels = batch["labels"]
        lengths = batch["lengths"]
        r_ids = batch["reading_ids"]
        sm_acc = None
        for _ in range(n_passes):
            logits = model(feats, lengths, r_ids)
            sm = F.softmax(logits, dim=-1)
            sm_acc = sm if sm_acc is None else sm_acc + sm
        sm_acc = sm_acc / n_passes
        for b in range(feats.size(0)):
            length = int(lengths[b])
            sm_per_utt.append(sm_acc[b, :length].cpu().numpy())
            label_per_utt.append(labels[b, :length].cpu().numpy())
    return sm_per_utt, label_per_utt


def main() -> None:
    """Test confidence-gated hybrid + MC dropout on v66_split1."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-split-seed", type=int, default=0)
    parser.add_argument(
        "--state-s1",
        default="/tmp/v66_split1_states/state_000.pt",
        help="v66_split1 state (highest leak-augmented model)",
    )
    parser.add_argument(
        "--state-strict",
        default="/tmp/v66_states/state_000.pt",
        help="v66 strict no-leak state",
    )
    parser.add_argument("--meta-cache", default="/tmp/v66_stacker.pt")
    parser.add_argument(
        "--teacher-model",
        default="/mnt/c/GitHub/kotonoha-models/accent_model_v24.onnx",
    )
    parser.add_argument("--mc-passes", type=int, default=12)
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
    sess_v24 = ort.InferenceSession(args.teacher_model, providers=providers)

    teacher_val_logits: list[np.ndarray] = []
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

    # Build val 103-dim features using stacker cache
    stacker_all = torch.load(args.meta_cache, weights_only=False)

    feats103_per_utt: list[np.ndarray] = []
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
        feats14 = np.concatenate(
            [feats13[:, :13], (v24_arg.astype(np.float32) / 20.0).reshape(-1, 1)],
            axis=1,
        )
        # Skip v66 strict path; use v66_split1 path only.
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
        feats103_per_utt.append(feats103)
        _ = feats14  # unused but kept for clarity

    flat_labels = np.concatenate(labels_per_utt)
    total = len(flat_labels)
    print(f"Total morphemes: {total}")

    # 1) v66_split1 standard inference (no MC) via ONNX
    print("\n--- v66_split1 standard inference ---")
    sess_v66s1 = ort.InferenceSession(
        "/mnt/c/GitHub/kotonoha-models/accent_model_v66_split1.onnx",
        providers=providers,
    )
    sm_s1: list[np.ndarray] = []
    for f in feats103_per_utt:
        log = sess_v66s1.run(None, {"input": f})[0]
        sm_s1.append(_softmax_1d(log))
    preds = np.concatenate([s.argmax(-1) for s in sm_s1])
    acc_s1 = float((preds == flat_labels).mean())
    print(f"v66_split1 ONNX: {acc_s1 * 100:.2f}%")

    # 2) MC dropout TTA via PyTorch state
    print("\n--- v66_split1 + MC dropout TTA ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = torch.load(args.state_s1, map_location="cpu", weights_only=False)
    print(f"Loaded {args.state_s1}: train_acc={bundle['acc'] * 100:.2f}%")
    model_s1 = AccentModel(
        embed_dim=64,
        hidden_dim=256,
        num_layers=3,
        num_classes=NUM_CLASSES,
        dropout=0.4,
        attention_heads=4,
        reading_dropout=0.0,
    ).to(device)
    model_s1.load_state_dict(bundle["state"], strict=False)

    val_ds = _AccentDataset(
        val_utts,
        augment=False,
        teacher_logits_per_morpheme=teacher_val_logits,
        meta_features_per_utt=[stacker_all[i] for i in val_idx_in_order],
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=64,
        shuffle=False,
        collate_fn=_collate_fn,
        num_workers=0,
    )
    sm_mc, labels_mc = _mc_dropout_softmax(
        model_s1, val_loader, device, n_passes=args.mc_passes
    )
    flat_labels_mc = np.concatenate(labels_mc)
    preds_mc = np.concatenate([s.argmax(-1) for s in sm_mc])
    acc_mc = float((preds_mc == flat_labels_mc).mean())
    print(f"v66_split1 + MC dropout ({args.mc_passes} passes): {acc_mc * 100:.2f}%")

    # 3) Confidence-gated hybrid: use v66_split1 if confidence high, else MC
    print("\n--- Confidence-gated hybrid (v66_split1 std + MC fallback) ---")
    for thr in [0.5, 0.6, 0.7, 0.8, 0.9]:
        correct = 0
        total_t = 0
        for sm, sm2, labels in zip(sm_s1, sm_mc, labels_mc, strict=True):
            seq = len(labels)
            if seq == 0:
                continue
            argmax_s1 = sm.argmax(-1)
            max_p_s1 = sm.max(-1)
            argmax_mc = sm2.argmax(-1)
            for t in range(seq):
                if max_p_s1[t] >= thr:
                    pred = int(argmax_s1[t])
                else:
                    pred = int(argmax_mc[t])
                if pred == int(labels[t]):
                    correct += 1
                total_t += 1
        acc = correct / total_t
        print(f"  thr={thr}: {acc * 100:.2f}%")


if __name__ == "__main__":
    main()
