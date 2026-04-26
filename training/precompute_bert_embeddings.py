"""Precompute BERT embeddings for each morpheme in JSUT/JVS/corpus.

Strategy: encode each morpheme's surface with BERT independently and
mean-pool hidden states (excluding CLS/SEP) → 768 dim per morpheme.

Output: list of np.ndarray (one per utterance), shape [seq_len, 768]
Saved as torch tensor for compactness.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def main() -> None:
    print("Loading BERT...")
    model_name = "tohoku-nlp/bert-base-japanese-v3"
    tk = AutoTokenizer.from_pretrained(model_name)
    m = AutoModel.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m.to(device).eval()
    print(f"  device={device} hidden={m.config.hidden_size}")

    files_to_process = [
        (
            "/home/owayo/kotonoha-training/data/jsut_accent_data_v3.json",
            "/tmp/bert_emb_jsut.pt",
        ),
        (
            "/home/owayo/kotonoha-training/data/jvs_accent_data.json",
            "/tmp/bert_emb_jvs.pt",
        ),
        (
            "/mnt/c/GitHub/kotonoha-training-data/train/corpus_converted.json",
            "/tmp/bert_emb_corpus.pt",
        ),
        (
            "/mnt/c/GitHub/kotonoha/training/filtered_jvs_v24_t75.json",
            "/tmp/bert_emb_filtered_jvs.pt",
        ),
    ]

    for src_path, out_path in files_to_process:
        if not Path(src_path).exists():
            print(f"skip missing: {src_path}")
            continue
        if Path(out_path).exists():
            print(f"already cached: {out_path}")
            continue
        with open(src_path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            utts = data.get("utterances", [])
        else:
            utts = data
        print(f"\n[{Path(src_path).name}] {len(utts)} utterances")

        # Collect all unique surface forms first
        surface_set: set[str] = set()
        for utt in utts:
            for m_dict in utt.get("morphemes", []):
                surface = m_dict.get("surface", "")
                if surface:
                    surface_set.add(surface)
        surfaces = sorted(surface_set)
        print(f"  unique surfaces: {len(surfaces)}")

        # Encode in batches
        BATCH = 128
        emb_dict: dict[str, np.ndarray] = {}
        for i in range(0, len(surfaces), BATCH):
            chunk = surfaces[i : i + BATCH]
            enc = tk(
                chunk,
                padding=True,
                truncation=True,
                max_length=16,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                out = m(**enc)
                hidden = out.last_hidden_state  # [B, T, 768]
                # Mean over non-padding tokens excluding [CLS] (0) and [SEP] (last non-pad)
                attn = enc["attention_mask"]
                lens = attn.sum(dim=1) - 2  # exclude CLS, SEP
                lens = lens.clamp(min=1).unsqueeze(-1)
                mask = attn.clone()
                mask[:, 0] = 0  # exclude CLS
                idx_last_nonpad = attn.sum(dim=1) - 1
                # exclude SEP
                for bi in range(mask.size(0)):
                    if idx_last_nonpad[bi] > 0:
                        mask[bi, idx_last_nonpad[bi]] = 0
                masked = hidden * mask.unsqueeze(-1).float()
                mean = masked.sum(dim=1) / lens.float()
                mean = mean.cpu().numpy().astype(np.float32)
            for k, surf in enumerate(chunk):
                emb_dict[surf] = mean[k]
            if (i + BATCH) % (BATCH * 10) == 0:
                print(f"    encoded {i + BATCH}/{len(surfaces)}")

        # Build per-utterance arrays
        utt_arrays: list[np.ndarray] = []
        for utt in utts:
            ms = utt.get("morphemes", [])
            if not ms:
                utt_arrays.append(np.zeros((0, 768), dtype=np.float32))
                continue
            arr = np.stack(
                [
                    emb_dict.get(m.get("surface", ""), np.zeros(768, dtype=np.float32))
                    for m in ms
                ]
            )
            utt_arrays.append(arr.astype(np.float32))

        torch.save(utt_arrays, out_path)
        print(f"  saved: {out_path} ({len(utt_arrays)} utts)")


if __name__ == "__main__":
    main()
