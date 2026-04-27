"""v63: precompute 9-student consensus argmax + agreement per token.

For each utt × token, run 9 student ONNX models and save:
  - consensus argmax: most-voted accent type (0..NUM_CLASSES-1)
  - agreement: fraction of models agreeing with consensus (0..1)

Used by v63 trainer to detect potential label noise:
if gold != consensus AND agreement >= threshold → reduce sample weight.

Output cache: per-utt arrays of shape [seq_len, 2] (consensus, agreement).
"""

from __future__ import annotations

import argparse
import json
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
    _softmax_1d,
)


def main() -> None:
    """Compute v63 consensus + agreement over JSUT + corpus utterances."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="/tmp/v63_consensus.pt")
    parser.add_argument(
        "--models-dir", type=str, default="/mnt/c/GitHub/kotonoha-models"
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

    utts = list(jsut)
    corpus_paths = [
        Path("/mnt/c/GitHub/kotonoha-training-data/train/corpus_converted.json"),
        Path("/mnt/c/GitHub/kotonoha/training/filtered_jvs_v24_t75.json"),
    ]
    for cp in corpus_paths:
        if not cp.exists():
            continue
        with open(cp, encoding="utf-8") as f:
            cdata = json.load(f)
        cu = cdata if isinstance(cdata, list) else cdata.get("utterances", [])
        _enrich_utterances(cu, accent_dict)
        utts = utts + list(cu)
    n_corpus = len(utts) - len(jsut)
    print(f"Total utts: {len(utts)} (JSUT {len(jsut)} + corpus {n_corpus})")

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    models_dir = Path(args.models_dir)
    sess_v24 = ort.InferenceSession(
        str(models_dir / "accent_model_v24.onnx"), providers=providers
    )
    student_specs = [
        "accent_model_v38.onnx",
        "accent_model_v54_split1.onnx",
        "accent_model_v54_split2.onnx",
        "accent_model_v54_split3.onnx",
        "accent_model_v59_fold0.onnx",
        "accent_model_v59_fold1.onnx",
        "accent_model_v59_fold2.onnx",
        "accent_model_v59_fold3.onnx",
        "accent_model_v59_fold4.onnx",
    ]
    student_sessions = []
    for fname in student_specs:
        path = models_dir / fname
        if not path.exists():
            print(f"  SKIP missing: {fname}")
            continue
        student_sessions.append(ort.InferenceSession(str(path), providers=providers))
    print(f"Loaded {len(student_sessions)} student models")

    out_per_utt: list[np.ndarray] = []
    for j, utt in enumerate(utts):
        ms = utt.get("morphemes", [])
        if not ms:
            out_per_utt.append(np.zeros((0, 2), dtype=np.float32))
            continue
        n = len(ms)
        feats13 = np.array(
            [
                _extract_morpheme_features(m, i / max(n - 1, 1))
                for i, m in enumerate(ms)
            ],
            dtype=np.float32,
        )
        v24_arg = sess_v24.run(None, {"input": feats13[:, :11]})[0].argmax(-1)
        feats14 = np.concatenate(
            [feats13[:, :13], (v24_arg.astype(np.float32) / 20.0).reshape(-1, 1)],
            axis=1,
        )
        argmaxes = []
        for sess in student_sessions:
            log = sess.run(None, {"input": feats14})[0]
            argmaxes.append(_softmax_1d(log).argmax(-1))
        argmax_stack = np.stack(argmaxes, axis=0)  # [M, seq_len]
        seq_len = feats13.shape[0]
        out = np.empty((seq_len, 2), dtype=np.float32)
        m_count = argmax_stack.shape[0]
        for t in range(seq_len):
            counts = np.bincount(argmax_stack[:, t], minlength=NUM_CLASSES)
            consensus = int(counts.argmax())
            agreement = float(counts[consensus]) / m_count
            out[t] = [float(consensus), agreement]
        out_per_utt.append(out)
        if (j + 1) % 1000 == 0:
            print(f"  processed {j + 1}/{len(utts)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_per_utt, out_path)
    print(f"\nSaved {len(out_per_utt)} entries to {out_path}")
    print(f"Total morphemes covered: {sum(len(m) for m in out_per_utt)}")


if __name__ == "__main__":
    main()
