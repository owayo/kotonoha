"""Evaluate a saved state.pt on RAW JSUT val_split=0."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import onnxruntime as ort
from train_onnx_v38 import (
    NUM_CLASSES,
    AccentModel,
    _AccentDataset,
    _collate_fn,
    _enrich_utterances,
    _evaluate,
    _extract_morpheme_features,
    _load_accent_dicts,
)
from torch.utils.data import DataLoader


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    ap.add_argument(
        "--data",
        default="/home/owayo/kotonoha-training/data/jsut_accent_data_v3.json",
        help="Raw JSUT (default) or cleaned",
    )
    ap.add_argument(
        "--accent-dict",
        default=(
            "/mnt/c/GitHub/kotonoha/data/accent_dict.csv:"
            "/mnt/c/GitHub/kotonoha-training-data/train/accent_dict.csv"
        ),
    )
    ap.add_argument(
        "--teacher-v24",
        default="/mnt/c/GitHub/kotonoha-models/accent_model_v24.onnx",
    )
    ap.add_argument("--val-split-seed", type=int, default=0)
    args = ap.parse_args()

    bundle = torch.load(args.state, map_location="cpu", weights_only=False)
    if "state" in bundle:
        state = bundle["state"]
    else:
        state = bundle

    dict_paths = [Path(p) for p in args.accent_dict.split(":") if p]
    accent_dict = _load_accent_dicts(dict_paths)

    with open(args.data, encoding="utf-8") as f:
        jsut = json.load(f)["utterances"]
    _enrich_utterances(jsut, accent_dict)
    random.seed(args.val_split_seed)
    idx = list(range(len(jsut)))
    random.shuffle(idx)
    val_size = int(len(idx) * 0.1)
    val_idx = set(idx[:val_size])
    val_utts = [u for i, u in enumerate(jsut) if i in val_idx]
    print(f"val utts: {len(val_utts)}")

    # Build v24 teacher argmax (v38 needs 14 dim with v24 argmax)
    sess_v24 = ort.InferenceSession(
        args.teacher_v24, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    teacher_val_preds = []
    for utt in val_utts:
        ms = utt.get("morphemes", [])
        if not ms:
            teacher_val_preds.append(np.zeros(0, dtype=np.int32))
            continue
        n = len(ms)
        feats = np.array(
            [
                _extract_morpheme_features(m, i / max(n - 1, 1))
                for i, m in enumerate(ms)
            ],
            dtype=np.float32,
        )
        log = sess_v24.run(None, {"input": feats[:, :11]})[0]
        teacher_val_preds.append(log.argmax(-1).astype(np.int32))

    val_ds = _AccentDataset(val_utts, augment=False, teacher_preds=teacher_val_preds)
    val_loader = DataLoader(
        val_ds,
        batch_size=64,
        shuffle=False,
        collate_fn=_collate_fn,
        num_workers=0,
        pin_memory=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AccentModel(
        embed_dim=64,
        hidden_dim=256,
        num_layers=3,
        num_classes=NUM_CLASSES,
        dropout=0.4,
        attention_heads=4,
        reading_dropout=0.0,
    ).to(device)
    # Load state, allowing partial mismatch (e.g. EMA dtype)
    to_load = {}
    for k, v in model.state_dict().items():
        if k in state:
            to_load[k] = state[k].to(dtype=v.dtype)
        else:
            to_load[k] = v
    model.load_state_dict(to_load)
    model.eval()

    _, acc = _evaluate(model, val_loader, device)
    print(f"\n{Path(args.state).name} on {Path(args.data).name}: {acc * 100:.2f}%")


if __name__ == "__main__":
    main()
