"""Late memorization phase for v66_split1 (codex idea #1).

Load existing v66_split1 state and run extra 15 epochs with:
- augment=False (no morpheme_dropout, no feature_noise)
- low lr (1e-4)
- no SAM, no R-Drop, no Mixup
- label_smoothing=0
- reweight_alpha=1.0 (no token reweighting)

Goal: strengthen memorization of training data without over-regularization.
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
    """Late memorization phase script."""
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

    # val_split_seed=1 split (matches v66_split1 training)
    random.seed(1)
    indices = list(range(len(jsut)))
    random.shuffle(indices)
    val_size = int(len(indices) * 0.1)
    val_idx = set(indices[:val_size])
    train_utts = [u for i, u in enumerate(jsut) if i not in val_idx] + corpus
    val_utts = [u for i, u in enumerate(jsut) if i in val_idx]
    print(f"train={len(train_utts)}, val={len(val_utts)}")

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess_v24 = ort.InferenceSession(
        "/mnt/c/GitHub/kotonoha-models/accent_model_v24.onnx", providers=providers
    )
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
            print(f"  teacher logits: {j + 1}/{len(train_utts)}")
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

    stacker_all = torch.load("/tmp/v66_stacker.pt", weights_only=False)
    n_jsut = len(jsut)
    stacker_train = [stacker_all[i] for i in range(n_jsut) if i not in val_idx]
    stacker_train = stacker_train + [
        stacker_all[n_jsut + i] for i in range(len(corpus))
    ]
    stacker_val = [stacker_all[i] for i in range(n_jsut) if i in val_idx]

    train_ds = _AccentDataset(
        train_utts,
        augment=False,
        teacher_logits_per_morpheme=teacher_train_logits,
        meta_features_per_utt=stacker_train,
        consensus_per_utt=None,
        reweight_alpha=1.0,
        reweight_agreement_threshold=2.0,
    )
    val_ds = _AccentDataset(
        val_utts,
        augment=False,
        teacher_logits_per_morpheme=teacher_val_logits,
        meta_features_per_utt=stacker_val,
    )
    train_loader = DataLoader(
        train_ds, batch_size=64, shuffle=True, collate_fn=_collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=64, shuffle=False, collate_fn=_collate_fn
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = torch.load(
        "/tmp/v66_split1_states/state_000.pt", map_location="cpu", weights_only=False
    )
    print(f"Loaded v66_split1 state with train_acc={bundle['acc'] * 100:.2f}%")
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    initial_loss, initial_acc = _evaluate(model, val_loader, device)
    print(f"Pre-late-memo val_acc: {initial_acc * 100:.2f}%")
    best_acc = initial_acc
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    for ep in range(1, 21):
        loss, tr_acc = _train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            scaler,
            label_smoothing=0.0,
            rdrop_alpha=0.0,
            sam_rho=0.0,
            mixup_alpha=0.0,
            kd_alpha=0.0,
            kd_temperature=2.0,
            ord_alpha=0.0,
        )
        va_loss, va_acc = _evaluate(model, val_loader, device)
        marker = ""
        if va_acc > best_acc:
            best_acc = va_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            marker = " *"
        tr_p = tr_acc * 100
        va_p = va_acc * 100
        print(f"  late-memo ep {ep}: tr={tr_p:.2f}% va={va_p:.2f}%{marker}")

    print(f"\nFinal best val_acc (val_split=1): {best_acc * 100:.2f}%")

    # Save and export
    out_state_path = Path("/tmp/v66s1_late_memo_state.pt")
    torch.save({"acc": best_acc, "state": best_state, "seed": 0}, out_state_path)
    print(f"Saved state to {out_state_path}")

    out_onnx = Path("/mnt/c/GitHub/kotonoha-models/accent_model_v66_split1_lm.onnx")
    model.load_state_dict(best_state)
    _export_onnx(model, out_onnx, device)
    print(f"Saved ONNX to {out_onnx}")


if __name__ == "__main__":
    main()
