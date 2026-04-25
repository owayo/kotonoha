"""Re-analyze label noise with kotonoha+kanjium combined dict."""

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
    print(f"Total dict entries: {len(accent_dict)}")

    with open(
        "/home/owayo/kotonoha-training/data/jsut_accent_data_v3.json",
        encoding="utf-8",
    ) as f:
        jsut = json.load(f)["utterances"]
    _enrich_utterances(jsut, accent_dict)

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(
        "/mnt/c/GitHub/kotonoha-models/accent_model_v20.onnx", providers=providers
    )

    total = 0
    no_dict = 0
    dict_matches_label = 0
    dict_matches_pred = 0  # noise candidate

    noise_candidates: list[dict] = []

    for utt_idx, utt in enumerate(jsut):
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
            lemma = morph.get("lemma", "")
            reading = morph.get("reading", "")
            key = (lemma, reading)
            dict_val_str = accent_dict.get(key)
            if dict_val_str is None and "-" in lemma:
                base = lemma.split("-")[0]
                dict_val_str = accent_dict.get((base, reading))
            dict_val = _parse_dict_val(dict_val_str) if dict_val_str else None

            if dict_val is None:
                no_dict += 1
                continue
            if dict_val == label:
                dict_matches_label += 1
            if dict_val == pred and pred != label:
                dict_matches_pred += 1
                noise_candidates.append(
                    {
                        "utt_idx": utt_idx,
                        "morph_idx": i,
                        "lemma": lemma,
                        "reading": reading,
                        "label": label,
                        "pred": pred,
                        "dict": dict_val,
                    }
                )

    print(f"Total morphemes: {total}")
    dict_covered = total - no_dict
    print(f"Dict covered: {dict_covered} ({dict_covered / total * 100:.2f}%)")
    print(f"  Dict matches label: {dict_matches_label} ({dict_matches_label / dict_covered * 100:.2f}%)")
    print(f"  Dict matches pred (NOISE): {dict_matches_pred} ({dict_matches_pred / dict_covered * 100:.2f}%)")

    out_path = Path("/tmp/v48_label_noise_candidates.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(noise_candidates, f, ensure_ascii=False)
    print(f"\nSaved {len(noise_candidates)} candidates to {out_path}")


if __name__ == "__main__":
    main()
