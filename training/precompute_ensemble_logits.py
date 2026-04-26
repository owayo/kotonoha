"""Compute ensemble softmax logits (v17+v20+v24 mean) per utterance.

Output: list of np.ndarray [seq_len, 21] - probabilities, NOT logits, since
KD uses softmax. We save log-probabilities so the KD code can read them as
"logits" with same KL behavior.
Actually we save the average softmax probs, then compute log to get fake logits.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from train_onnx_v38 import (
    NUM_CLASSES,
    _enrich_utterances,
    _extract_morpheme_features,
    _load_accent_dicts,
)


def _softmax(x: np.ndarray) -> np.ndarray:
    mx = np.max(x, axis=-1, keepdims=True)
    e = np.exp(x - mx)
    return e / np.sum(e, axis=-1, keepdims=True)


def main() -> None:
    """Cache ensemble (v17+v20+v24) average logits per utterance."""
    dict_paths = [
        Path("/mnt/c/GitHub/kotonoha/data/accent_dict.csv"),
        Path("/mnt/c/GitHub/kotonoha-training-data/train/accent_dict.csv"),
    ]
    accent_dict = _load_accent_dicts(dict_paths)

    files = [
        (
            "/home/owayo/kotonoha-training/data/jsut_accent_data_v3.json",
            "/tmp/v53_ens_logits_jsut.pt",
        ),
        (
            "/home/owayo/kotonoha-training/data/jvs_accent_data.json",
            "/tmp/v53_ens_logits_jvs.pt",
        ),
        (
            "/mnt/c/GitHub/kotonoha-training-data/train/corpus_converted.json",
            "/tmp/v53_ens_logits_corpus.pt",
        ),
        (
            "/mnt/c/GitHub/kotonoha/training/filtered_jvs_v24_t75.json",
            "/tmp/v53_ens_logits_filtered_jvs.pt",
        ),
    ]

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess_v17 = ort.InferenceSession(
        "/mnt/c/GitHub/kotonoha-models/accent_model_v17.onnx", providers=providers
    )
    sess_v20 = ort.InferenceSession(
        "/mnt/c/GitHub/kotonoha-models/accent_model_v20.onnx", providers=providers
    )
    sess_v24 = ort.InferenceSession(
        "/mnt/c/GitHub/kotonoha-models/accent_model_v24.onnx", providers=providers
    )

    for src_path, out_path in files:
        if not Path(src_path).exists():
            continue
        if Path(out_path).exists():
            print(f"already cached: {out_path}")
            continue
        with open(src_path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            utts = data.get("utterances", [])
        else:
            utts = data
        _enrich_utterances(utts, accent_dict)
        print(f"\n[{Path(src_path).name}] {len(utts)} utts")

        results: list[np.ndarray] = []
        for i, utt in enumerate(utts):
            ms = utt.get("morphemes", [])
            if not ms:
                results.append(np.zeros((0, NUM_CLASSES), dtype=np.float32))
                continue
            n = len(ms)
            feats = np.array(
                [
                    _extract_morpheme_features(m, j / max(n - 1, 1))
                    for j, m in enumerate(ms)
                ],
                dtype=np.float32,
            )
            inp11 = feats[:, :11]
            sm17 = _softmax(sess_v17.run(None, {"input": inp11})[0])
            sm20 = _softmax(sess_v20.run(None, {"input": inp11})[0])
            sm24 = _softmax(sess_v24.run(None, {"input": inp11})[0])
            avg = (sm17 + sm20 + sm24) / 3
            # Save as fake logits (log of probs); KD code expects logits to softmax
            log_avg = np.log(avg + 1e-9).astype(np.float32)
            results.append(log_avg)
            if (i + 1) % 1000 == 0:
                print(f"  encoded {i + 1}/{len(utts)}")

        torch.save(results, out_path)
        print(f"  saved to {out_path}")


if __name__ == "__main__":
    main()
