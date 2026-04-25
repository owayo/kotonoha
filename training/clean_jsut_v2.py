"""Clean JSUT using merged dict (kotonoha+kanjium) and v20 predictions.

Fixes morphemes where:
  - dict_val (from any of 3 dict files) matches v20 prediction
  - but differs from label
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import onnxruntime as ort
from train_onnx_v38 import (
    NUM_CLASSES,
    _enrich_utterances,
    _extract_morpheme_features,
    _load_accent_dicts,
)


def _parse_dict_val(val: str) -> int | None:
    if val in ("*", ""):
        return None
    m = re.match(r'^"?(\d+)$', val)
    if m:
        v = int(m.group(1))
        if 0 <= v <= 20:
            return v
    return None


def main() -> None:
    dict_paths = [
        Path("/mnt/c/GitHub/kotonoha/data/accent_dict.csv"),
        Path("/mnt/c/GitHub/kotonoha-training-data/train/accent_dict.csv"),
        Path("/mnt/c/GitHub/kotonoha-training-data/train/accent_dict_kanjium.csv"),
    ]
    accent_dict = _load_accent_dicts(dict_paths)
    src = Path("/home/owayo/kotonoha-training/data/jsut_accent_data_v3.json")
    with open(src, encoding="utf-8") as f:
        data = json.load(f)
    jsut_utts = data["utterances"]
    _enrich_utterances(jsut_utts, accent_dict)

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(
        "/mnt/c/GitHub/kotonoha-models/accent_model_v20.onnx", providers=providers
    )

    fixed = 0
    total = 0
    for utt in jsut_utts:
        ms = utt.get("morphemes", [])
        if not ms:
            continue
        n = len(ms)
        feats13 = np.array(
            [_extract_morpheme_features(m, i / max(n - 1, 1)) for i, m in enumerate(ms)],
            dtype=np.float32,
        )
        logits = sess.run(None, {"input": feats13[:, :11]})[0]
        preds = logits.argmax(-1)
        for i, morph in enumerate(ms):
            label = min(morph.get("accent_type", 0), NUM_CLASSES - 1)
            pred = int(preds[i])
            total += 1
            if pred == label:
                continue
            lemma = morph.get("lemma", "")
            reading = morph.get("reading", "")
            key = (lemma, reading)
            dict_val_str = accent_dict.get(key)
            if dict_val_str is None and "-" in lemma:
                base = lemma.split("-")[0]
                dict_val_str = accent_dict.get((base, reading))
            dict_val = _parse_dict_val(dict_val_str) if dict_val_str else None
            if dict_val is not None and dict_val == pred:
                morph["accent_type"] = pred
                fixed += 1

    print(f"Fixed {fixed} / {total} ({fixed / total * 100:.3f}%)")
    out = Path("/mnt/c/GitHub/kotonoha-training-data/train/jsut_accent_data_v3_cleaned_v2.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
