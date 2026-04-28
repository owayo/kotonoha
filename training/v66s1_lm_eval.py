"""Evaluate v66_split1_lm.onnx on val_split=0 val 500 utts."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

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


def main() -> None:
    """Eval v66_split1_lm on val=0."""
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
    val_idx_in_order = [i for i in range(len(jsut)) if i in val_idx]
    print(f"Val: {len(val_utts)} utts (val_split=0)")

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess_v24 = ort.InferenceSession(
        "/mnt/c/GitHub/kotonoha-models/accent_model_v24.onnx", providers=providers
    )
    stacker_all = torch.load("/tmp/v66_stacker.pt", weights_only=False)
    sess_models = {
        "v66_split1": ort.InferenceSession(
            "/mnt/c/GitHub/kotonoha-models/accent_model_v66_split1.onnx",
            providers=providers,
        ),
        "v66_split1_lm": ort.InferenceSession(
            "/mnt/c/GitHub/kotonoha-models/accent_model_v66_split1_lm.onnx",
            providers=providers,
        ),
    }

    sm: dict[str, list[np.ndarray]] = {n: [] for n in sess_models}
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
        for name, sess in sess_models.items():
            log = sess.run(None, {"input": feats103})[0]
            sm[name].append(_softmax_1d(log))

    flat_labels = np.concatenate(labels_per_utt)
    print(f"Total: {len(flat_labels)} morphemes")

    print("\nSingle accuracies (val_split=0):")
    for name, sm_list in sm.items():
        preds = np.concatenate([s.argmax(-1) for s in sm_list])
        acc = float((preds == flat_labels).mean())
        print(f"  {name}: {acc * 100:.2f}%")

    print("\nEnsemble:")
    avg_preds = []
    for i in range(len(labels_per_utt)):
        stacked = np.stack([sm["v66_split1"][i], sm["v66_split1_lm"][i]])
        avg_preds.append(stacked.mean(0).argmax(-1))
    preds = np.concatenate(avg_preds)
    acc = float((preds == flat_labels).mean())
    print(f"  v66_split1 + v66_split1_lm avg: {acc * 100:.2f}%")


if __name__ == "__main__":
    main()
