r"""Analyze validation errors of v33 best model.

v33 seed 0 (75.58% on split=0) で val を予測し、予測誤りの
パターンを分析。ラベル誤り vs モデル誤りの判定に使う。

Usage:
  uv run python analyze_val_errors.py \
      --state /tmp/v33_states/state_000.pt \
      --val-split-seed 0
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path

import torch

from train_onnx_v33 import (
    NUM_CLASSES,
    AccentModel,
    _enrich_utterances,
    _extract_morpheme_features,
    _load_accent_dicts,
)


def _load_dotenv() -> None:
    """Load .env file into environment variables."""
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())


def main() -> None:
    """Entry point."""
    _load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True, help="state_*.pt path")
    parser.add_argument("--val-split-seed", type=int, default=0)
    parser.add_argument(
        "--jsut", type=Path, default=Path(os.environ.get("FINETUNE_DATA", ""))
    )
    parser.add_argument(
        "--accent-dict", type=str, default=os.environ.get("ACCENT_DICT", "")
    )
    parser.add_argument("--top-errors", type=int, default=30)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dict_paths = [Path(p) for p in args.accent_dict.split(":") if p]
    accent_dict = _load_accent_dicts(dict_paths)

    with open(args.jsut, encoding="utf-8") as f:
        jsut = json.load(f)["utterances"]
    _enrich_utterances(jsut, accent_dict)

    random.seed(args.val_split_seed)
    indices = list(range(len(jsut)))
    random.shuffle(indices)
    val_size = int(len(indices) * 0.1)
    val_idx = set(indices[:val_size])
    val_utts = [u for i, u in enumerate(jsut) if i in val_idx]
    print(f"Val utts: {len(val_utts)}")

    # Load model
    d = torch.load(args.state, map_location="cpu", weights_only=False)
    model = AccentModel(
        embed_dim=64,
        hidden_dim=256,
        num_layers=3,
        num_classes=NUM_CLASSES,
        dropout=0.4,
        attention_heads=4,
        reading_dropout=0.0,
    ).to(device)
    model.load_state_dict(d["state"])
    model.eval()
    print(f"Loaded state: orig acc {d['acc'] * 100:.2f}%")

    total = 0
    correct = 0
    error_details: list[dict] = []
    pred_true_pairs: Counter = Counter()

    with torch.no_grad():
        for utt in val_utts:
            morphemes = utt.get("morphemes", [])
            if not morphemes:
                continue
            n = len(morphemes)
            feats = [
                _extract_morpheme_features(m, i / max(n - 1, 1))
                for i, m in enumerate(morphemes)
            ]
            feats_t = torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(device)
            logits = model(feats_t).squeeze(0)
            preds = logits.argmax(dim=-1).cpu().numpy()
            for i, m in enumerate(morphemes):
                true_label = min(m.get("accent_type", 0), NUM_CLASSES - 1)
                pred = int(preds[i])
                total += 1
                if pred == true_label:
                    correct += 1
                else:
                    pred_true_pairs[(true_label, pred)] += 1
                    if len(error_details) < args.top_errors:
                        prev_m = morphemes[i - 1] if i > 0 else None
                        next_m = morphemes[i + 1] if i + 1 < len(morphemes) else None
                        error_details.append(
                            {
                                "utt_id": utt.get("utterance_id", "?"),
                                "text": utt.get("text", ""),
                                "surface": m.get("surface", ""),
                                "pos": m.get("pos", ""),
                                "reading": m.get("reading", ""),
                                "true": true_label,
                                "pred": pred,
                                "dict_acc": m.get("dict_accent_type", "*"),
                                "prev": prev_m.get("surface", "") if prev_m else "",
                                "next": next_m.get("surface", "") if next_m else "",
                            }
                        )

    acc = correct / total if total > 0 else 0
    print(f"\nAccuracy: {acc * 100:.2f}% ({correct}/{total})")
    print(f"Errors: {total - correct}")

    print("\n=== Top 10 error pattern (true -> pred) ===")
    for (t, p), n in pred_true_pairs.most_common(10):
        print(f"  {t} -> {p}: {n} times")

    print(f"\n=== Top {args.top_errors} error samples ===")
    for e in error_details[: args.top_errors]:
        print(
            f"  [{e['utt_id']}] {e['text'][:30]}... "
            f"'{e['prev']}'<{e['surface']}({e['pos']}, r={e['reading']}, "
            f"dict={e['dict_acc']})>'{e['next']}' "
            f"true={e['true']}, pred={e['pred']}"
        )


if __name__ == "__main__":
    main()
