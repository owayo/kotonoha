"""v65: precompute 11-student average softmax probability per token.

For each utt × token, run 11 students (9 14-dim + v61 + v63) and save
the AVERAGE softmax distribution as [seq_len, NUM_CLASSES].

Used by v65 trainer for agreement-aware soft target distillation:
  soft_target = (1-w) * onehot(gold) + w * ens_prob
  w = clip((agree - thr) / (1 - thr), 0, 1)

Output: per-utt list of [seq_len, NUM_CLASSES] float32.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from precompute_v61_meta import _meta_features
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
    """Compute 11-student mean softmax per token for soft-target training."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/tmp/v65_ens_prob.pt")
    parser.add_argument("--models-dir", default="/mnt/c/GitHub/kotonoha-models")
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
    student_14d_specs = [
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
    student_14d_sessions = [
        ort.InferenceSession(str(models_dir / fname), providers=providers)
        for fname in student_14d_specs
    ]
    student_24d_specs = [
        "accent_model_v61.onnx",
        "accent_model_v63.onnx",
    ]
    student_24d_sessions = [
        ort.InferenceSession(str(models_dir / fname), providers=providers)
        for fname in student_24d_specs
    ]
    print(
        f"Loaded {len(student_14d_sessions)} 14-dim + "
        f"{len(student_24d_sessions)} 24-dim student models"
    )

    ens_prob_per_utt: list[np.ndarray] = []
    for j, utt in enumerate(utts):
        ms = utt.get("morphemes", [])
        if not ms:
            ens_prob_per_utt.append(np.zeros((0, NUM_CLASSES), dtype=np.float32))
            continue
        n = len(ms)
        feats13 = np.array(
            [
                _extract_morpheme_features(m, i / max(n - 1, 1))
                for i, m in enumerate(ms)
            ],
            dtype=np.float32,
        )
        v24_log = sess_v24.run(None, {"input": feats13[:, :11]})[0]
        v24_arg = v24_log.argmax(-1)
        feats14 = np.concatenate(
            [feats13[:, :13], (v24_arg.astype(np.float32) / 20.0).reshape(-1, 1)],
            axis=1,
        )
        sm_list = []
        for sess in student_14d_sessions:
            log = sess.run(None, {"input": feats14})[0]
            sm_list.append(_softmax_1d(log))

        seq_len = feats13.shape[0]
        sm_stack_9 = np.stack(sm_list, axis=0)
        meta_9 = np.empty((seq_len, 5), dtype=np.float32)
        for t in range(seq_len):
            meta_9[t] = _meta_features(sm_stack_9[:, t, :])

        feats24 = np.empty((seq_len, 24), dtype=np.float32)
        for t in range(seq_len):
            logits_row = v24_log[t]
            tp_argmax = float(int(np.argmax(logits_row))) / 20.0
            dict_acc = _morpheme_dict_accent(ms[t])
            exp_y, pmax, margin, entropy, p_dict = _teacher_soft_stats(
                logits_row, dict_acc
            )
            feats24[t, :13] = feats13[t]
            feats24[t, 13] = tp_argmax
            feats24[t, 14:19] = [exp_y, pmax, margin, entropy, p_dict]
            feats24[t, 19:24] = meta_9[t]

        for sess in student_24d_sessions:
            log = sess.run(None, {"input": feats24})[0]
            sm_list.append(_softmax_1d(log))

        sm_stack = np.stack(sm_list, axis=0)  # [11, seq, NUM_CLASSES]
        ens_prob = sm_stack.mean(axis=0).astype(np.float32)  # [seq, NUM_CLASSES]
        ens_prob_per_utt.append(ens_prob)
        if (j + 1) % 1000 == 0:
            print(f"  processed {j + 1}/{len(utts)}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ens_prob_per_utt, args.out)
    print(f"\nSaved to {args.out}")
    print(f"Total morphemes: {sum(len(p) for p in ens_prob_per_utt)}")


if __name__ == "__main__":
    main()
