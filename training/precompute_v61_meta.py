"""v61: precompute multi-model OOF stacking meta-features.

For each utt × token, run 9 student ONNX models (v38, v54_split{1,2,3},
v59_fold{0..4}) and aggregate their softmax distributions into 5 meta-features:

  meta[0]: mean of E[y] / 20  across 9 models
  meta[1]: std  of E[y] / 20  across 9 models
  meta[2]: agreement_count / 9  (how many models predict the consensus argmax)
  meta[3]: mean entropy / log(NUM_CLASSES)
  meta[4]: max p(consensus) across models

Output cache: per-utt arrays of shape [seq_len, 5] saved as torch list.

Note: each student model uses 14-dim feature input requiring v24 argmax as dim 13.
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


def _meta_features(softmax_stack: np.ndarray) -> np.ndarray:
    """Compute 5 meta-features per token from a stack of M model softmax outputs.

    Args:
        softmax_stack: shape [M, NUM_CLASSES] for one token.

    Returns:
        np.ndarray shape [5].

    """
    cls = np.arange(NUM_CLASSES, dtype=np.float32)
    exp_per = (softmax_stack * cls).sum(axis=-1) / 20.0  # [M]
    mean_exp = float(exp_per.mean())
    std_exp = float(exp_per.std())
    argmaxes = softmax_stack.argmax(axis=-1)  # [M]
    consensus = int(np.bincount(argmaxes, minlength=NUM_CLASSES).argmax())
    agreement = float((argmaxes == consensus).sum()) / softmax_stack.shape[0]
    entropy_per = -(softmax_stack * np.log(softmax_stack + 1e-9)).sum(axis=-1)
    mean_entropy = float(entropy_per.mean() / np.log(NUM_CLASSES))
    max_p_consensus = float(softmax_stack[:, consensus].max())
    return np.array(
        [mean_exp, std_exp, agreement, mean_entropy, max_p_consensus],
        dtype=np.float32,
    )


def main() -> None:
    """Compute v61 meta-features over JSUT + corpus utterances."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", type=str, default="/tmp/v61_meta_features.pt", help="output cache path"
    )
    parser.add_argument(
        "--models-dir", type=str, default="/mnt/c/GitHub/kotonoha-models"
    )
    parser.add_argument(
        "--include-corpus",
        action="store_true",
        default=True,
        help="include corpus utts in cache",
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
    if args.include_corpus:
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
        ("v38", "accent_model_v38.onnx"),
        ("v54_s1", "accent_model_v54_split1.onnx"),
        ("v54_s2", "accent_model_v54_split2.onnx"),
        ("v54_s3", "accent_model_v54_split3.onnx"),
        ("v59_f0", "accent_model_v59_fold0.onnx"),
        ("v59_f1", "accent_model_v59_fold1.onnx"),
        ("v59_f2", "accent_model_v59_fold2.onnx"),
        ("v59_f3", "accent_model_v59_fold3.onnx"),
        ("v59_f4", "accent_model_v59_fold4.onnx"),
    ]
    student_sessions = []
    for name, fname in student_specs:
        path = models_dir / fname
        if not path.exists():
            print(f"  SKIP missing: {fname}")
            continue
        student_sessions.append(
            (name, ort.InferenceSession(str(path), providers=providers))
        )
    print(f"Loaded {len(student_sessions)} student models")

    meta_per_utt: list[np.ndarray] = []
    for j, utt in enumerate(utts):
        ms = utt.get("morphemes", [])
        if not ms:
            meta_per_utt.append(np.zeros((0, 5), dtype=np.float32))
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
        # Stack all student softmax outputs: shape [M, seq_len, NUM_CLASSES]
        stacks = []
        for _name, sess in student_sessions:
            log = sess.run(None, {"input": feats14})[0]  # [seq_len, NUM_CLASSES]
            stacks.append(_softmax_1d(log))
        sm_stack = np.stack(stacks, axis=0)  # [M, seq, C]
        seq_len = feats13.shape[0]
        meta = np.empty((seq_len, 5), dtype=np.float32)
        for t in range(seq_len):
            meta[t] = _meta_features(sm_stack[:, t, :])
        meta_per_utt.append(meta)
        if (j + 1) % 1000 == 0:
            print(f"  processed {j + 1}/{len(utts)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(meta_per_utt, out_path)
    print(f"\nSaved {len(meta_per_utt)} entries to {out_path}")
    print(f"Total morphemes covered: {sum(len(m) for m in meta_per_utt)}")


if __name__ == "__main__":
    main()
