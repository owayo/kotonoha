r"""Precompute ensemble teacher logits for v29 KD.

v24 ONNX + v27 top-K state_dict の logits 平均を教師として事前計算する。

Outputs: torch.save([np.ndarray[seq_len, NUM_CLASSES], ...], cache_path)

Usage:
  uv run python precompute_ensemble_teacher.py \
      --teacher-onnx /mnt/c/GitHub/kotonoha-models/accent_model_v24.onnx \
      --state-dir /tmp/v27_states \
      --top-k 5 \
      --cache /tmp/v29_ensemble_teacher.pt
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from train_onnx_v27 import (
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


def _build_train_split(
    jsut_utterances: list[dict],
    corpus_utterances: list[dict],
) -> list[dict]:
    random.seed(42)
    indices = list(range(len(jsut_utterances)))
    random.shuffle(indices)
    val_size = int(len(indices) * 0.1)
    val_idx = set(indices[:val_size])
    train_utts = [u for i, u in enumerate(jsut_utterances) if i not in val_idx]
    return train_utts + corpus_utterances


def _utt_features(utt: dict) -> np.ndarray:
    morphemes = utt.get("morphemes", [])
    n = len(morphemes)
    if n == 0:
        return np.zeros((0, 11), dtype=np.float32)
    feats = [
        _extract_morpheme_features(m, i / max(n - 1, 1))
        for i, m in enumerate(morphemes)
    ]
    return np.array(feats, dtype=np.float32)


def main() -> None:
    """Entry point."""
    _load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-onnx", type=Path, required=True)
    parser.add_argument(
        "--state-dir", type=Path, required=True, help="v27 state_XXX.pt directory"
    )
    parser.add_argument("--top-k", type=int, default=5, help="top seeds to ensemble")
    parser.add_argument("--cache", type=Path, required=True, help="output cache path")
    parser.add_argument(
        "--jsut", type=Path, default=Path(os.environ.get("FINETUNE_DATA", ""))
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path(__file__).parent / "combined_corpus.json",
    )
    parser.add_argument(
        "--extra-corpus",
        type=str,
        default=str(Path(__file__).parent / "filtered_jvs_v24_t75.json"),
    )
    parser.add_argument(
        "--accent-dict",
        type=str,
        default=os.environ.get("ACCENT_DICT", ""),
    )
    parser.add_argument("--weight-onnx", type=float, default=1.0)
    parser.add_argument("--weight-per-seed", type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load accent dicts
    dict_paths = [Path(p) for p in args.accent_dict.split(":") if p]
    accent_dict = _load_accent_dicts(dict_paths)
    print(f"Accent dict: {len(accent_dict)} entries")

    # Load data
    print(f"Loading JSUT: {args.jsut}")
    with open(args.jsut, encoding="utf-8") as f:
        jsut_utterances = json.load(f)["utterances"]
    _enrich_utterances(jsut_utterances, accent_dict)
    print(f"  {len(jsut_utterances)} utterances")

    print(f"Loading corpus: {args.corpus}")
    with open(args.corpus, encoding="utf-8") as f:
        corpus_raw = json.load(f)
    corpus_utts = (
        corpus_raw if isinstance(corpus_raw, list) else corpus_raw.get("utterances", [])
    )
    _enrich_utterances(corpus_utts, accent_dict)

    if args.extra_corpus:
        extra_path = Path(args.extra_corpus)
        if extra_path.exists():
            with open(extra_path, encoding="utf-8") as f:
                extra_raw = json.load(f)
            extra = (
                extra_raw
                if isinstance(extra_raw, list)
                else extra_raw.get("utterances", [])
            )
            _enrich_utterances(extra, accent_dict)
            corpus_utts = corpus_utts + extra
    print(f"  corpus combined: {len(corpus_utts)}")

    train_utts = _build_train_split(jsut_utterances, corpus_utts)
    print(f"Train split: {len(train_utts)} utterances")

    # ── Teacher ONNX (v24) ──
    import onnxruntime as ort

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if device.type == "cuda"
        else ["CPUExecutionProvider"]
    )
    sess = ort.InferenceSession(str(args.teacher_onnx), providers=providers)

    # ── Top-K state_dict (v27 student best seeds) ──
    state_files = sorted(args.state_dir.glob("state_*.pt"))
    acc_list = []
    for p in state_files:
        d = torch.load(p, map_location="cpu", weights_only=False)
        acc_list.append((d["acc"], p, d["state"]))
    acc_list.sort(key=lambda x: -x[0])
    top_states = acc_list[: args.top_k]
    print(f"\nTop-{args.top_k} states for ensemble:")
    for acc, p, _ in top_states:
        print(f"  {p.name}: {acc * 100:.2f}%")

    # Build PyTorch models for top-K
    models = []
    for _, _, state in top_states:
        m = AccentModel(
            embed_dim=64,
            hidden_dim=256,
            num_layers=3,
            num_classes=NUM_CLASSES,
            dropout=0.4,
            attention_heads=4,
            reading_dropout=0.0,
        ).to(device)
        m.load_state_dict(state)
        m.eval()
        models.append(m)

    # ── Compute ensemble logits ──
    print(f"\nComputing ensemble logits for {len(train_utts)} utterances...")
    ensemble_logits_list: list[np.ndarray] = []
    w_onnx = args.weight_onnx
    w_seed = args.weight_per_seed
    total_weight = w_onnx + w_seed * len(models)

    with torch.no_grad():
        for j, utt in enumerate(train_utts):
            morphemes = utt.get("morphemes", [])
            if not morphemes:
                ensemble_logits_list.append(
                    np.zeros((0, NUM_CLASSES), dtype=np.float32)
                )
                continue
            feats_np = _utt_features(utt)  # [seq_len, 11]

            # ONNX teacher (v24)
            logits_onnx = sess.run(None, {"input": feats_np})[0]  # [seq_len, 21]

            # PyTorch models
            feats_t = torch.from_numpy(feats_np).unsqueeze(0).to(device)
            # reading_ids: use zeros (teacher path without reading)
            # but the model supports reading_ids=None -> fallback
            logits_sum = torch.zeros_like(torch.from_numpy(logits_onnx).to(device))
            for m in models:
                seeds_logits = m(feats_t).squeeze(0)
                logits_sum = logits_sum + seeds_logits

            ensemble = (
                w_onnx * torch.from_numpy(logits_onnx).to(device) + w_seed * logits_sum
            ) / total_weight
            ensemble_logits_list.append(ensemble.cpu().numpy().astype(np.float32))

            if (j + 1) % 1000 == 0:
                print(f"  processed {j + 1}/{len(train_utts)}")

    # Save
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ensemble_logits_list, args.cache)
    print(f"\nSaved ensemble logits to {args.cache}")
    print(f"  entries: {len(ensemble_logits_list)}")

    # Sanity: measure ensemble val accuracy? Skip for speed.


if __name__ == "__main__":
    main()
