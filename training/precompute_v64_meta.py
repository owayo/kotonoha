"""v64: precompute 11-student stacking meta + consensus.

v61 (9 students) に加え、v61 自身と v63 (現状の best 単独) を teacher として
追加した 11-student の meta-features と consensus を計算。

- meta-features (5 dim, 形式は v61 と同じ): mean E[y]/20, std, agreement,
  mean entropy, max p(consensus)
- consensus (2 dim): consensus argmax, agreement

v64 trainer (= v63 trainer with new caches) で利用。
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
    """Compute 11-student meta + consensus using v61 / v63 added to 9 students."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta-out", default="/tmp/v64_meta_features.pt")
    parser.add_argument("--consensus-out", default="/tmp/v64_consensus.pt")
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
    # 9 base 14-dim students
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
    # 24-dim students that will be evaluated AFTER 14-dim meta is computed
    # (v61, v63 use 24-dim input, so we compute their predictions later)
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

    meta_per_utt: list[np.ndarray] = []
    consensus_per_utt: list[np.ndarray] = []
    for j, utt in enumerate(utts):
        ms = utt.get("morphemes", [])
        if not ms:
            meta_per_utt.append(np.zeros((0, 5), dtype=np.float32))
            consensus_per_utt.append(np.zeros((0, 2), dtype=np.float32))
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
        # 14-dim student softmaxes
        sm_list = []
        for sess in student_14d_sessions:
            log = sess.run(None, {"input": feats14})[0]
            sm_list.append(_softmax_1d(log))
        # Build 24-dim feature for 24-dim students using the 9-student meta first
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

        sm_stack = np.stack(sm_list, axis=0)  # [11, seq, C]
        m_count = sm_stack.shape[0]

        meta = np.empty((seq_len, 5), dtype=np.float32)
        cons = np.empty((seq_len, 2), dtype=np.float32)
        for t in range(seq_len):
            meta[t] = _meta_features(sm_stack[:, t, :])
            argmaxes = sm_stack[:, t, :].argmax(axis=-1)
            counts = np.bincount(argmaxes, minlength=NUM_CLASSES)
            consensus_arg = int(counts.argmax())
            agreement = float(counts[consensus_arg]) / m_count
            cons[t] = [float(consensus_arg), agreement]
        meta_per_utt.append(meta)
        consensus_per_utt.append(cons)
        if (j + 1) % 1000 == 0:
            print(f"  processed {j + 1}/{len(utts)}")

    Path(args.meta_out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(meta_per_utt, args.meta_out)
    torch.save(consensus_per_utt, args.consensus_out)
    print(f"\nSaved meta to {args.meta_out}")
    print(f"Saved consensus to {args.consensus_out}")
    print(f"Total morphemes: {sum(len(m) for m in meta_per_utt)}")


if __name__ == "__main__":
    main()
