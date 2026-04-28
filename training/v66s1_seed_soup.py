"""v66_split1 multi-seed soup eval on val_split=0.

Average softmax of v66_split1 seeds {0,1,2} for val_split=0 prediction.
"""

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
    """Multi-seed v66_split1 soup eval on val_split=0 val 500."""
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
    print(f"Val: {len(val_utts)} utts")

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess_v24 = ort.InferenceSession(
        "/mnt/c/GitHub/kotonoha-models/accent_model_v24.onnx", providers=providers
    )
    stacker_all = torch.load("/tmp/v66_stacker.pt", weights_only=False)

    sess_seeds = {
        "s1_s0": ort.InferenceSession(
            "/mnt/c/GitHub/kotonoha-models/accent_model_v66_split1.onnx",
            providers=providers,
        ),
        "s1_s1": ort.InferenceSession("/tmp/v66s1_seed1.onnx", providers=providers),
        "s1_s2": ort.InferenceSession("/tmp/v66s1_seed2.onnx", providers=providers),
    }

    target_softmax: dict[str, list[np.ndarray]] = {n: [] for n in sess_seeds}
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

        for name, sess in sess_seeds.items():
            log = sess.run(None, {"input": feats103})[0]
            target_softmax[name].append(_softmax_1d(log))

    flat_labels = np.concatenate(labels_per_utt)
    print(f"Total: {len(flat_labels)} morphemes")

    print("\nSingle accuracies:")
    for name, sm_list in target_softmax.items():
        preds = np.concatenate([sm.argmax(-1) for sm in sm_list])
        acc = float((preds == flat_labels).mean())
        print(f"  {name}: {acc * 100:.2f}%")

    print("\nEnsemble combinations:")
    names = list(target_softmax.keys())
    for combo_size in range(2, len(names) + 1):
        from itertools import combinations

        for combo in combinations(names, combo_size):
            avg_preds = []
            for i in range(len(labels_per_utt)):
                stacked = np.stack([target_softmax[n][i] for n in combo])
                avg_preds.append(stacked.mean(0).argmax(-1))
            preds = np.concatenate(avg_preds)
            acc = float((preds == flat_labels).mean())
            print(f"  {'+'.join(combo)}: {acc * 100:.2f}%")

    # Weighted with seed 0 = 0.5, seeds 1,2 = 0.25 each
    print("\nWeighted (s1_s0=0.5, others=0.25):")
    avg_preds = []
    for i in range(len(labels_per_utt)):
        a = target_softmax["s1_s0"][i] * 0.5
        a = a + target_softmax["s1_s1"][i] * 0.25
        a = a + target_softmax["s1_s2"][i] * 0.25
        avg_preds.append(a.argmax(-1))
    preds = np.concatenate(avg_preds)
    acc = float((preds == flat_labels).mean())
    print(f"  weighted: {acc * 100:.2f}%")


if __name__ == "__main__":
    main()
