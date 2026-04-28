"""v66 trained on FULL JSUT (no JSUT val held out).

Use corpus utts as val instead of JSUT. Train on all 5000 JSUT + corpus
training utts. Evaluate on val_split=0's val 500 utts (FULL leak —
all 500 utts are in training set).

Goal: structural ceiling on val_split=0 evaluation (memorization-based).
This is 'train on test' by README convention's leak augmented strict read.
Result is for analytic purposes (deployment ceiling), NOT for production.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch.utils.data import DataLoader

from train_onnx_v60 import (
    NUM_CLASSES,
    _enrich_utterances,
    _extract_morpheme_features,
    _load_accent_dicts,
    _load_dotenv,
)
from train_onnx_v66 import (
    AccentModel,
    _AccentDataset,
    _collate_fn,
    _evaluate,
    _export_onnx,
    _train_epoch,
)


def main() -> None:
    """Train v66 on FULL JSUT, eval on val_split=0 val 500."""
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
    print(f"JSUT={len(jsut)}, corpus={len(corpus)}")

    # FULL JSUT train + most corpus train. Use last 200 corpus utts as val
    # (just for convergence monitoring; not used for selection).
    train_utts = list(jsut) + list(corpus[:-200])
    val_corpus = list(corpus[-200:])
    print(f"train={len(train_utts)} (full JSUT + most corpus)")
    print(f"val_corpus={len(val_corpus)} (corpus only, NOT JSUT)")

    # JVS pretrain
    jvs_path = "/home/owayo/kotonoha-training/data/jvs_accent_data.json"
    with open(jvs_path, encoding="utf-8") as f:
        jvs_data = json.load(f)
    jvs_utts = (
        jvs_data if isinstance(jvs_data, list) else jvs_data.get("utterances", [])
    )
    _enrich_utterances(jvs_utts, accent_dict)
    print(f"JVS={len(jvs_utts)}")

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess_v24 = ort.InferenceSession(
        "/mnt/c/GitHub/kotonoha-models/accent_model_v24.onnx", providers=providers
    )

    print("\nComputing teacher logits for train+val...")
    teacher_train_logits: list[np.ndarray] = []
    for j, utt in enumerate(train_utts):
        ms = utt.get("morphemes", [])
        if not ms:
            teacher_train_logits.append(np.zeros((0, NUM_CLASSES), dtype=np.float32))
            continue
        n = len(ms)
        feats = np.array(
            [
                _extract_morpheme_features(m, i / max(n - 1, 1))
                for i, m in enumerate(ms)
            ],
            dtype=np.float32,
        )
        teacher_train_logits.append(sess_v24.run(None, {"input": feats[:, :11]})[0])
        if (j + 1) % 1000 == 0:
            print(f"  teacher: {j + 1}/{len(train_utts)}")
    teacher_val_logits: list[np.ndarray] = []
    for utt in val_corpus:
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

    stacker_all = torch.load("/tmp/v66_stacker.pt", weights_only=False)
    n_jsut = len(jsut)
    # Train: full JSUT (indices 0..4999) + corpus[:-200] (indices 5000..n-200)
    n_corpus_keep = len(corpus) - 200
    stacker_train = [stacker_all[i] for i in range(n_jsut)] + [
        stacker_all[n_jsut + i] for i in range(n_corpus_keep)
    ]
    stacker_val = [stacker_all[n_jsut + n_corpus_keep + i] for i in range(200)]

    train_ds = _AccentDataset(
        train_utts,
        augment=True,
        teacher_logits_per_morpheme=teacher_train_logits,
        meta_features_per_utt=stacker_train,
    )
    val_ds = _AccentDataset(
        val_corpus,
        augment=False,
        teacher_logits_per_morpheme=teacher_val_logits,
        meta_features_per_utt=stacker_val,
    )
    train_loader = DataLoader(
        train_ds, batch_size=64, shuffle=True, collate_fn=_collate_fn, num_workers=2
    )
    val_loader = DataLoader(
        val_ds, batch_size=64, shuffle=False, collate_fn=_collate_fn
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    random.seed(0)
    model = AccentModel(
        embed_dim=64,
        hidden_dim=256,
        num_layers=3,
        num_classes=NUM_CLASSES,
        dropout=0.4,
        attention_heads=4,
        reading_dropout=0.0,
    ).to(device)

    # JVS pretrain (Stage 1)
    jvs_ds = _AccentDataset(jvs_utts, augment=True)
    jvs_loader = DataLoader(
        jvs_ds, batch_size=64, shuffle=True, collate_fn=_collate_fn, num_workers=2
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    print("\n[Stage 1] JVS pretrain (20 ep)")
    for ep in range(1, 21):
        loss, acc = _train_epoch(model, jvs_loader, optimizer, device, scaler)
        if ep % 5 == 0:
            print(f"  ep {ep}: loss={loss:.4f} acc={acc * 100:.2f}%")

    # Stage 2: full JSUT fine-tune (NO val_split=0 holdout; corpus 200 used)
    print("\n[Stage 2] Full JSUT fine-tune (100 ep, fixed schedule)")
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    for ep in range(1, 101):
        loss, tr_acc = _train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            scaler,
            label_smoothing=0.1,
            sam_rho=0.05,
            mixup_alpha=0.3,
            kd_alpha=0.3,
            kd_temperature=2.0,
        )
        scheduler.step()
        va_loss, va_acc = _evaluate(model, val_loader, device)
        if ep % 5 == 0 or ep == 100:
            print(f"  ep {ep}: tr={tr_acc * 100:.2f}% va_corpus={va_acc * 100:.2f}%")
        if ep == 100:
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    # Save and export
    out_state = Path("/tmp/v66_full_jsut_state.pt")
    torch.save({"acc": va_acc, "state": best_state, "seed": 0}, out_state)
    out_onnx = Path("/mnt/c/GitHub/kotonoha-models/accent_model_v66_full.onnx")
    model.load_state_dict(best_state)
    _export_onnx(model, out_onnx, device)
    print(f"\nSaved {out_onnx}")


if __name__ == "__main__":
    main()
