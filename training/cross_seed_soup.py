r"""Cross-seed greedy soup across multiple state directories.

v27, v29, v29b 等の複数バージョンの state_*.pt をプールし、
greedy soup で最適な重み平均を探索する。

Usage:
  uv run python cross_seed_soup.py \
      --state-dirs /tmp/v27_states /tmp/v29_states \
      --output /mnt/c/GitHub/kotonoha-models/accent_model_v30_soup.onnx
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
    _export_onnx,
    _greedy_soup,
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
    parser.add_argument(
        "--state-dirs",
        nargs="+",
        type=Path,
        required=True,
        help="state_XXX.pt directories to pool",
    )
    parser.add_argument("--output", type=Path, required=True, help="output ONNX path")
    parser.add_argument(
        "--jsut", type=Path, default=Path(os.environ.get("FINETUNE_DATA", ""))
    )
    parser.add_argument(
        "--accent-dict", type=str, default=os.environ.get("ACCENT_DICT", "")
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Use only top-K states by acc (0 = all)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.0,
        help="Greedy soup tolerance",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load accent dicts + JSUT val split ──
    dict_paths = [Path(p) for p in args.accent_dict.split(":") if p]
    accent_dict = _load_accent_dicts(dict_paths)

    with open(args.jsut, encoding="utf-8") as f:
        jsut = json.load(f)["utterances"]
    _enrich_utterances(jsut, accent_dict)

    random.seed(42)
    indices = list(range(len(jsut)))
    random.shuffle(indices)
    val_size = int(len(indices) * 0.1)
    val_idx = set(indices[:val_size])
    val_utts = [u for i, u in enumerate(jsut) if i in val_idx]
    print(f"Val utts: {len(val_utts)}")

    val_ds = _AccentDataset(val_utts, augment=False)
    val_loader = DataLoader(
        val_ds, batch_size=64, shuffle=False, collate_fn=_collate_fn, num_workers=2
    )

    # ── Collect all state dicts ──
    all_states: list[tuple[float, dict, str]] = []
    for sd in args.state_dirs:
        for p in sorted(sd.glob("state_*.pt")):
            d = torch.load(p, map_location="cpu", weights_only=False)
            all_states.append((d["acc"], d["state"], f"{sd.name}/{p.name}"))
    all_states.sort(key=lambda x: -x[0])
    print(f"Collected {len(all_states)} states")

    if args.top_k > 0:
        all_states = all_states[: args.top_k]
        print(f"  using top-{args.top_k}")

    print("\nTop 10 individual accuracies:")
    for acc, _, name in all_states[:10]:
        print(f"  {name}: {acc * 100:.2f}%")

    # ── Build dummy model for greedy soup evaluation ──
    model = AccentModel(
        embed_dim=64,
        hidden_dim=256,
        num_layers=3,
        num_classes=NUM_CLASSES,
        dropout=0.4,
        attention_heads=4,
        reading_dropout=0.0,
    ).to(device)

    # Verify best single on val
    best_acc_single = all_states[0][0]
    print(f"\nBest single (train split val): {best_acc_single * 100:.2f}%")
    # Reevaluate on val to confirm
    model.load_state_dict(all_states[0][1])
    model.to(device)
    _, acc_confirm = _evaluate(model, val_loader, device)
    print(f"Best single (fresh eval): {acc_confirm * 100:.2f}%")

    # ── Greedy soup over all states ──
    print(f"\nGreedy soup over {len(all_states)} states...")
    candidates = [(acc, state) for acc, state, _ in all_states]
    soup_acc, soup_state = _greedy_soup(
        candidates, model, val_loader, device, tolerance=args.tolerance
    )
    print(f"Greedy soup: {soup_acc * 100:.2f}%")

    # ── Load soup_state and export ONNX ──
    model.load_state_dict(soup_state)
    model.to(device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _export_onnx(model, args.output, device)
    print(f"Saved: {args.output} ({args.output.stat().st_size / 1_048_576:.1f} MB)")
    print(f"\nFinal soup val accuracy: {soup_acc * 100:.2f}%")


if __name__ == "__main__":
    main()
