"""v63 multi-seed PyTorch state ensemble eval.

ONNX export で ~0.5% 精度が落ちる現象を回避するため、
.pt state file 直接ロードして PyTorch GPU 上で ensemble する。

Usage:
  uv run python v63_state_ensemble.py \
    --states /tmp/v63_states/state_000.pt /tmp/v63_states/state_001.pt
"""

from __future__ import annotations

import argparse
import itertools
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
)
from train_onnx_v63 import (
    AccentModel,
    _AccentDataset,
    _build_mask,
    _collate_fn,
)


@torch.no_grad()
def _predict_softmax(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Run model on val loader; return per-token softmax + flat labels.

    Returns:
        Tuple of (softmax probabilities [N, NUM_CLASSES], flat labels [N]).

    """
    model.eval()
    sm_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    for batch in loader:
        feats = batch["features"].to(device)
        labels = batch["labels"].to(device)
        lengths = batch["lengths"]
        r_ids = batch["reading_ids"].to(device)
        logits = model(feats, lengths, r_ids)
        mask = _build_mask(labels, lengths, device)
        flat_logits = logits[mask]
        sm = F.softmax(flat_logits, dim=-1).cpu().numpy()
        sm_chunks.append(sm)
        label_chunks.append(labels[mask].cpu().numpy())
    return np.concatenate(sm_chunks, axis=0), np.concatenate(label_chunks, axis=0)


def main() -> None:
    """Multi-state PyTorch ensemble eval on val_split=0 val 500 utts.

    Raises:
        FileNotFoundError: meta-features cache is missing.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-split-seed", type=int, default=0)
    parser.add_argument(
        "--states", nargs="+", required=True, help=".pt state file paths"
    )
    parser.add_argument("--meta-cache", default="/tmp/v61_meta_features.pt")
    parser.add_argument(
        "--teacher-model", default="/mnt/c/GitHub/kotonoha-models/accent_model_v24.onnx"
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

    meta_cache_path = Path(args.meta_cache)
    if not meta_cache_path.exists():
        msg = f"meta-features cache missing: {meta_cache_path}"
        raise FileNotFoundError(msg)
    meta_all = torch.load(meta_cache_path, weights_only=False)
    n_jsut = len(jsut)
    meta_val = [meta_all[i] for i in range(n_jsut) if i in val_idx]
    print(f"meta_features val={len(meta_val)}")

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

    state_softmax: dict[str, np.ndarray] = {}
    flat_labels = None
    for state_path in args.states:
        bundle = torch.load(state_path, map_location="cpu", weights_only=False)
        acc = float(bundle["acc"])
        print(f"  loading {state_path}: train_time_acc={acc * 100:.2f}%")
        model = AccentModel(
            embed_dim=64,
            hidden_dim=256,
            num_layers=3,
            num_classes=NUM_CLASSES,
            dropout=0.4,
            attention_heads=4,
            reading_dropout=0.0,
        ).to(device)
        # strict=False: v61 state lacks ord_head (only present in v62/v63)
        model.load_state_dict(bundle["state"], strict=False)
        sm, labels = _predict_softmax(model, val_loader, device)
        state_softmax[state_path] = sm
        if flat_labels is None:
            flat_labels = labels
        single_acc = float((sm.argmax(-1) == labels).mean())
        print(f"    PyTorch single acc: {single_acc * 100:.2f}%")

    print("\nSingle-state acc (re-eval):")
    accs = {}
    for path, sm in state_softmax.items():
        acc = float((sm.argmax(-1) == flat_labels).mean())
        accs[path] = acc

    print("\nEnsemble (equal-weight softmax avg):")
    paths = list(state_softmax.keys())
    best_acc = max(accs.values())
    for r in range(2, len(paths) + 1):
        for combo in itertools.combinations(paths, r):
            avg_sm = np.mean([state_softmax[p] for p in combo], axis=0)
            preds = avg_sm.argmax(-1)
            acc = float((preds == flat_labels).mean())
            mark = ""
            if acc > best_acc:
                best_acc = acc
                mark = " *"
            short = "+".join(Path(p).stem for p in combo)
            print(f"  {short}: {acc * 100:.2f}%{mark}")

    print(f"\nBest ensemble: {best_acc * 100:.2f}%")


if __name__ == "__main__":
    main()
