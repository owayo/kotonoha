"""Filter JVS utterances by teacher model predictions.

v24 などの強力な teacher ONNX モデルで JVS 発話を予測し、
予測と真ラベルの一致率が閾値以上の発話のみを抽出して保存する。

Usage:
  uv run python filter_jvs_by_teacher.py \
      --jvs /home/owayo/kotonoha-training/data/jvs_accent_data.json \
      --teacher /mnt/c/GitHub/kotonoha-models/accent_model_v24.onnx \
      --threshold 0.90 \
      --output /mnt/c/GitHub/kotonoha/training/filtered_jvs_v24.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from train_onnx_v24 import (
    NUM_CLASSES,
    _enrich_utterances,
    _extract_morpheme_features,
    _load_accent_dicts,
)


def filter_jvs_with_teacher(
    jvs_utterances: list[dict],
    onnx_path: Path,
    threshold: float,
) -> tuple[list[dict], dict]:
    """Teacher ONNX で JVS 発話を予測し、一致率が閾値以上のものを抽出.

    Returns:
        (filtered_utterances, stats_dict).

    """
    sess = ort.InferenceSession(str(onnx_path))
    filtered: list[dict] = []
    total_morphemes = 0
    matched_morphemes = 0
    agreements = []

    for utt in jvs_utterances:
        morphemes = utt.get("morphemes", [])
        if not morphemes:
            continue

        n = len(morphemes)
        features = []
        labels = []

        for i, m in enumerate(morphemes):
            position = i / max(n - 1, 1)
            feat = _extract_morpheme_features(m, position)
            features.append(feat)
            accent = m.get("accent_type", 0)
            labels.append(min(accent, NUM_CLASSES - 1))

        feat_array = np.array(features, dtype=np.float32)
        logits = sess.run(None, {"input": feat_array})[0]
        preds = logits.argmax(axis=-1)
        labels_array = np.array(labels)
        agreement = float((preds == labels_array).mean())
        agreements.append(agreement)
        total_morphemes += n
        matched_morphemes += int((preds == labels_array).sum())

        if agreement >= threshold:
            filtered.append(utt)

    stats = {
        "total_utterances": len(jvs_utterances),
        "filtered_utterances": len(filtered),
        "total_morphemes": total_morphemes,
        "matched_morphemes": matched_morphemes,
        "overall_morpheme_acc": matched_morphemes / max(total_morphemes, 1),
        "threshold": threshold,
        "agreement_p50": float(np.percentile(agreements, 50)) if agreements else 0.0,
        "agreement_p90": float(np.percentile(agreements, 90)) if agreements else 0.0,
        "agreement_mean": float(np.mean(agreements)) if agreements else 0.0,
    }
    return filtered, stats


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Filter JVS by teacher ONNX")
    parser.add_argument("--jvs", type=Path, required=True, help="JVS accent JSON")
    parser.add_argument(
        "--teacher", type=Path, required=True, help="Teacher ONNX model"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.90,
        help="Agreement threshold (0-1)",
    )
    parser.add_argument(
        "--accent-dict",
        type=str,
        default="",
        help="Accent dict CSV paths (colon-separated, to enrich dict_accent_type)",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    print(f"Loading JVS: {args.jvs}")
    with open(args.jvs, encoding="utf-8") as f:
        jvs_data = json.load(f)
    jvs_utterances = jvs_data["utterances"]
    print(f"  {len(jvs_utterances)} utterances")

    if args.accent_dict:
        dict_paths = [Path(p) for p in args.accent_dict.split(":") if p]
        print(f"Loading accent dicts ({len(dict_paths)} files)")
        accent_dict = _load_accent_dicts(dict_paths)
        n = _enrich_utterances(jvs_utterances, accent_dict)
        print(f"  Enriched {n} morphemes")

    print(f"\nFiltering with teacher {args.teacher} (threshold={args.threshold})")
    filtered, stats = filter_jvs_with_teacher(
        jvs_utterances, args.teacher, args.threshold
    )

    print("\nStats:")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    print(f"\nWriting filtered utterances: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"utterances": filtered, "stats": stats}, f, ensure_ascii=False)
    print(f"  {len(filtered)} utterances saved")


if __name__ == "__main__":
    main()
