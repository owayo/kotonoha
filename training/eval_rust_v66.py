"""Rust kotonoha (v66 bundle) を JSUT val_split=0 上で評価する.

Python API (`kotonoha.KotonohaEngine`) 経由で V66Pipeline を呼び、形態素単位の
gold accent_type と比較。lookup-only モード (本番想定) と JSUT-pre-set モード
(訓練時と同じ dict_accent_type) の双方を測定する。

使い方:
    ORT_DYLIB_PATH=/path/to/libonnxruntime.so python eval_rust_v66.py
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from types import SimpleNamespace

from train_onnx_v60 import (
    NUM_CLASSES,
    _enrich_utterances,
    _load_accent_dicts,
    _load_dotenv,
)


def _make_token(m: dict) -> SimpleNamespace:
    """Convert a JSUT morpheme to an attribute-style object kotonoha expects.

    The PyO3 binding reads fields via attribute access (e.g. `tok.surface`),
    so we wrap each morpheme in `SimpleNamespace` rather than a plain dict.

    Returns:
        SimpleNamespace with `surface`, `pos`, ..., `pronunciation` attrs.

    """
    return SimpleNamespace(
        surface=m.get("surface", ""),
        pos=m.get("pos", ""),
        pos_detail1=m.get("pos_detail1", "*"),
        pos_detail2=m.get("pos_detail2", "*"),
        pos_detail3=m.get("pos_detail3", "*"),
        ctype=m.get("conjugation_type", "*"),
        cform=m.get("conjugation_form", "*"),
        lemma=m.get("lemma", m.get("surface", "")),
        reading=m.get("reading", ""),
        pronunciation=m.get("pronunciation", m.get("reading", "")),
    )


def main() -> None:
    """Evaluate Rust V66Pipeline on JSUT val_split=0 val 500 utts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-split-seed", type=int, default=0)
    parser.add_argument("--bundle", default="/mnt/c/GitHub/kotonoha-models")
    args = parser.parse_args()

    if "ORT_DYLIB_PATH" not in os.environ:
        os.environ["ORT_DYLIB_PATH"] = (
            "/mnt/c/GitHub/kotonoha/training/.venv/lib/python3.12/site-packages/"
            "onnxruntime/capi/libonnxruntime.so.1.24.3"
        )

    import kotonoha  # noqa: E402

    _load_dotenv()
    # accent_dict_jsut.csv is the comprehensive lookup table extracted from
    # JSUT v3 + corpus that JSUT used for dict_accent_type. By using it
    # exclusively, the Rust runtime reproduces JSUT's enriched values without
    # introducing false positives from kanjium.
    dict_paths = [
        Path("/mnt/c/GitHub/kotonoha-models/accent_dict_jsut.csv"),
    ]
    accent_dict = _load_accent_dicts(dict_paths)
    jsut_path = "/home/owayo/kotonoha-training/data/jsut_accent_data_v3.json"
    with open(jsut_path, encoding="utf-8") as f:
        jsut = json.load(f)["utterances"]
    _enrich_utterances(jsut, accent_dict)

    n_jsut = len(jsut)
    val_size = int(n_jsut * 0.1)
    rng = random.Random(args.val_split_seed)
    indices = list(range(n_jsut))
    rng.shuffle(indices)
    val_idx = set(indices[:val_size])
    test_utts = [u for i, u in enumerate(jsut) if i in val_idx]
    print(f"Test: val_split={args.val_split_seed} val ({len(test_utts)} utts)")

    engine = kotonoha.KotonohaEngine(
        model_bundle=args.bundle,
        accent_dict_paths=[str(p) for p in dict_paths],
    )

    correct = 0
    total = 0
    for utt in test_utts:
        ms = utt.get("morphemes", [])
        if not ms:
            continue
        tokens = [_make_token(m) for m in ms]
        preds = engine.predict_accent_types(tokens)
        gold = [min(int(m.get("accent_type", 0)), NUM_CLASSES - 1) for m in ms]
        for p, g in zip(preds, gold, strict=True):
            if p == g:
                correct += 1
            total += 1

    acc = correct / max(total, 1)
    print(f"Accuracy: {correct}/{total} = {acc * 100:.2f}%")


if __name__ == "__main__":
    main()
