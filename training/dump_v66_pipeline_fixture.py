"""v66_split1 standalone (no exact-memory) 推論結果を fixture として dump する.

Rust 側 `tests/v66_pipeline_match.rs` で、同じ utts に対する V66Pipeline 出力と
argmax 一致を検証するために使う。
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import onnxruntime as ort
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
    """Run v66_split1 standalone on the same fixture utts."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="/mnt/c/GitHub/kotonoha/kotonoha/tests/fixtures/v66_pipeline.json",
    )
    parser.add_argument("--num-utts", type=int, default=20)
    parser.add_argument("--val-split-seed", type=int, default=0)
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

    rng = random.Random(args.val_split_seed)
    indices = list(range(len(jsut)))
    rng.shuffle(indices)
    test_idx = sorted(indices[: args.num_utts])
    selected = [jsut[i] for i in test_idx]

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
    student_sessions = [
        ort.InferenceSession(str(models_dir / fname), providers=providers)
        for fname in student_specs
    ]
    sess_v61 = ort.InferenceSession(
        str(models_dir / "accent_model_v61.onnx"), providers=providers
    )
    sess_v63 = ort.InferenceSession(
        str(models_dir / "accent_model_v63.onnx"), providers=providers
    )
    sess_v66 = ort.InferenceSession(
        str(models_dir / "accent_model_v66_split1.onnx"), providers=providers
    )

    fixture = []
    for utt in selected:
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
        v24_log = sess_v24.run(None, {"input": feats13[:, :11]})[0]
        v24_arg = v24_log.argmax(-1)

        feats14 = np.concatenate(
            [feats13, (v24_arg.astype(np.float32) / 20.0).reshape(-1, 1)], axis=1
        )
        sm_list = []
        for sess in student_sessions:
            log = sess.run(None, {"input": feats14})[0]
            sm_list.append(_softmax_1d(log))
        sm_stack_9 = np.stack(sm_list, axis=0)
        meta_9 = np.empty((n, 5), dtype=np.float32)
        for t in range(n):
            meta_9[t] = _meta_features(sm_stack_9[:, t, :])

        feats24 = np.empty((n, 24), dtype=np.float32)
        for t in range(n):
            tp_argmax = float(int(np.argmax(v24_log[t]))) / 20.0
            dict_acc = _morpheme_dict_accent(ms[t])
            ey, mp, mg, en, pd_p = _teacher_soft_stats(v24_log[t], dict_acc)
            feats24[t, :13] = feats13[t]
            feats24[t, 13] = tp_argmax
            feats24[t, 14:19] = [ey, mp, mg, en, pd_p]
            feats24[t, 19:24] = meta_9[t]
        log_v61 = sess_v61.run(None, {"input": feats24})[0]
        log_v63 = sess_v63.run(None, {"input": feats24})[0]
        sm_v61 = _softmax_1d(log_v61)
        sm_v63 = _softmax_1d(log_v63)
        sm_stack_11 = np.stack(sm_list + [sm_v61, sm_v63], axis=0)

        stacker = np.empty((n, 84), dtype=np.float32)
        ens_mean = sm_stack_11.mean(axis=0)
        ens_std = sm_stack_11.std(axis=0)
        argmaxes = sm_stack_11.argmax(axis=-1)
        for t in range(n):
            counts = np.bincount(argmaxes[:, t], minlength=NUM_CLASSES).astype(
                np.float32
            )
            counts /= max(counts.sum(), 1.0)
            stacker[t, 0:21] = ens_mean[t]
            stacker[t, 21:42] = ens_std[t]
            stacker[t, 42:63] = counts
            stacker[t, 63:84] = sm_v63[t]

        feats103 = np.empty((n, 103), dtype=np.float32)
        feats103[:, :13] = feats13
        feats103[:, 13] = v24_arg.astype(np.float32) / 20.0
        for t in range(n):
            dict_acc = _morpheme_dict_accent(ms[t])
            ey, mp, mg, en, pd_p = _teacher_soft_stats(v24_log[t], dict_acc)
            feats103[t, 14:19] = [ey, mp, mg, en, pd_p]
        feats103[:, 19:103] = stacker
        log_v66 = sess_v66.run(None, {"input": feats103})[0]
        preds = log_v66.argmax(-1).astype(int).tolist()

        morphemes_dump = []
        for m in ms:
            morphemes_dump.append(
                {
                    "surface": m.get("surface", ""),
                    "pos": m.get("pos", ""),
                    "pos_detail1": m.get("pos_detail1", "*"),
                    "pos_detail2": m.get("pos_detail2", "*"),
                    "pos_detail3": m.get("pos_detail3", "*"),
                    "conjugation_type": m.get("conjugation_type", "*"),
                    "conjugation_form": m.get("conjugation_form", "*"),
                    "lemma": m.get("lemma", ""),
                    "reading": m.get("reading", ""),
                    "pronunciation": m.get("pronunciation", ""),
                    "dict_accent_type": m.get("dict_accent_type", "*"),
                }
            )
        fixture.append(
            {
                "utterance_id": utt.get("utterance_id", ""),
                "morphemes": morphemes_dump,
                "predicted_accent_types": preds,
            }
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"utterances": fixture}, f, ensure_ascii=False, indent=None)
    print(f"Wrote {len(fixture)} utts to {out}")


if __name__ == "__main__":
    main()
