"""Precompute context-aware BERT embeddings using char-level BERT.

Uses tohoku-nlp/bert-base-japanese-char-v3, where each token corresponds to
~1 character. Encodes the whole utterance text at once and pools per-morpheme
hidden states by char-range alignment.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def encode_utterance(
    morphemes: list[dict],
    tk,
    model,
    device,
    hidden_dim: int = 768,
) -> np.ndarray:
    """Encode utterance with char-level BERT and pool per-morpheme.

    Returns:
        Array of shape [n_morphemes, hidden_dim].

    """
    surfaces = [m.get("surface", "") for m in morphemes]
    text = "".join(surfaces)
    if not text:
        return np.zeros((len(morphemes), hidden_dim), dtype=np.float32)

    # Compute char offsets per morpheme
    morph_ranges: list[tuple[int, int]] = []
    pos = 0
    for s in surfaces:
        morph_ranges.append((pos, pos + len(s)))
        pos += len(s)

    enc = tk(text, truncation=True, max_length=512, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc)
        hidden = out.last_hidden_state[0]  # [T, 768]
    ids = enc["input_ids"][0].cpu().tolist()
    tokens = tk.convert_ids_to_tokens(ids)

    # Walk: map each token to char position in text
    char_pos = 0
    token_to_char: list[int] = []
    for tok in tokens:
        if tok in ("[CLS]", "[SEP]", "[PAD]"):
            token_to_char.append(-1)
            continue
        clean = tok[2:] if tok.startswith("##") else tok
        if tok == "[UNK]":
            if char_pos < len(text):
                token_to_char.append(char_pos)
                char_pos += 1
            else:
                token_to_char.append(-1)
            continue
        if char_pos < len(text) and text[char_pos : char_pos + len(clean)] == clean:
            token_to_char.append(char_pos)
            char_pos += len(clean)
        else:
            found = text.find(clean, char_pos) if clean else -1
            if found >= 0:
                token_to_char.append(found)
                char_pos = found + len(clean)
            else:
                token_to_char.append(-1)

    embs = []
    for m_start, m_end in morph_ranges:
        if m_start >= m_end:
            embs.append(np.zeros(hidden_dim, dtype=np.float32))
            continue
        idxs = [
            ti for ti, c in enumerate(token_to_char) if c >= 0 and m_start <= c < m_end
        ]
        if not idxs:
            embs.append(np.zeros(hidden_dim, dtype=np.float32))
            continue
        emb = hidden[idxs].mean(0).cpu().numpy().astype(np.float32)
        embs.append(emb)
    return np.stack(embs)


def main() -> None:
    """Encode all training/finetune utterances and cache embeddings."""
    print("Loading BERT (char-v3)...")
    model_name = "tohoku-nlp/bert-base-japanese-char-v3"
    tk = AutoTokenizer.from_pretrained(model_name)
    m = AutoModel.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m.to(device).eval()

    files = [
        (
            "/home/owayo/kotonoha-training/data/jsut_accent_data_v3.json",
            "/tmp/bert_v2_emb_jsut.pt",
        ),
        (
            "/home/owayo/kotonoha-training/data/jvs_accent_data.json",
            "/tmp/bert_v2_emb_jvs.pt",
        ),
        (
            "/mnt/c/GitHub/kotonoha-training-data/train/corpus_converted.json",
            "/tmp/bert_v2_emb_corpus.pt",
        ),
        (
            "/mnt/c/GitHub/kotonoha/training/filtered_jvs_v24_t75.json",
            "/tmp/bert_v2_emb_filtered_jvs.pt",
        ),
    ]

    for src_path, out_path in files:
        if not Path(src_path).exists():
            print(f"skip: {src_path}")
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

        results = []
        for i, utt in enumerate(utts):
            ms = utt.get("morphemes", [])
            if not ms:
                results.append(np.zeros((0, 768), dtype=np.float32))
                continue
            emb = encode_utterance(ms, tk, m, device)
            results.append(emb)
            if (i + 1) % 500 == 0:
                print(f"  encoded {i + 1}/{len(utts)}")

        torch.save(results, out_path)
        print(f"  saved to {out_path}")


if __name__ == "__main__":
    main()
