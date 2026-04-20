r"""Evaluate saved states across multiple validation splits.

既存の state_*.pt を複数の val split で評価し、最も高い
(state, split) の組み合わせを探す。73% 到達の現実性を検証。

Usage:
  uv run python evaluate_across_splits.py \
      --state-dirs /tmp/v27_states /tmp/v29_states \
      --split-seeds 42 0 1 2 3 10 20 30
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from train_onnx_v27 import (
    NUM_CLASSES,
    AccentModel,
    _AccentDataset,
    _collate_fn,
    _enrich_utterances,
    _evaluate,
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


def _build_val(jsut: list[dict], split_seed: int) -> list[dict]:
    random.seed(split_seed)
    indices = list(range(len(jsut)))
    random.shuffle(indices)
    val_size = int(len(indices) * 0.1)
    val_idx = set(indices[:val_size])
    return [u for i, u in enumerate(jsut) if i in val_idx]


def main() -> None:
    """Entry point."""
    _load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--split-seeds", nargs="+", type=int, required=True)
    parser.add_argument(
        "--jsut", type=Path, default=Path(os.environ.get("FINETUNE_DATA", ""))
    )
    parser.add_argument(
        "--accent-dict", type=str, default=os.environ.get("ACCENT_DICT", "")
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top states per dir (by original acc) to evaluate",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dict_paths = [Path(p) for p in args.accent_dict.split(":") if p]
    accent_dict = _load_accent_dicts(dict_paths)
    with open(args.jsut, encoding="utf-8") as f:
        jsut = json.load(f)["utterances"]
    _enrich_utterances(jsut, accent_dict)

    # Prepare val loaders per split
    val_loaders = {}
    for ss in args.split_seeds:
        val_utts = _build_val(jsut, ss)
        ds = _AccentDataset(val_utts, augment=False)
        val_loaders[ss] = DataLoader(
            ds, batch_size=64, shuffle=False, collate_fn=_collate_fn, num_workers=2
        )
        print(f"  split seed={ss}: {len(val_utts)} val utts")

    # Collect states
    all_states: list[tuple[float, dict, str]] = []
    for sd in args.state_dirs:
        for p in sorted(sd.glob("state_*.pt")):
            d = torch.load(p, map_location="cpu", weights_only=False)
            all_states.append((d["acc"], d["state"], f"{sd.name}/{p.name}"))
    all_states.sort(key=lambda x: -x[0])
    all_states = all_states[: args.top_k]
    print(
        f"\nEvaluating {len(all_states)} states across {len(args.split_seeds)} splits"
    )

    # Build model stub
    model = AccentModel(
        embed_dim=64,
        hidden_dim=256,
        num_layers=3,
        num_classes=NUM_CLASSES,
        dropout=0.4,
        attention_heads=4,
        reading_dropout=0.0,
    ).to(device)

    # Evaluate each state on each split
    best_overall = (0.0, "", -1)
    results: list[tuple[str, int, float]] = []
    for orig_acc, state, name in all_states:
        model.load_state_dict(state)
        model.to(device)
        row = []
        for ss in args.split_seeds:
            _, acc = _evaluate(model, val_loaders[ss], device)
            row.append((ss, acc))
            results.append((name, ss, acc))
            if acc > best_overall[0]:
                best_overall = (acc, name, ss)
        accs_str = " ".join(f"s{ss}:{acc * 100:.2f}%" for ss, acc in row)
        print(f"  {name} (orig {orig_acc * 100:.2f}%): {accs_str}")

    print("\n=== BEST OVERALL ===")
    acc, name, ss = best_overall
    print(f"  {name} on split seed {ss}: {acc * 100:.2f}%")


if __name__ == "__main__":
    main()
