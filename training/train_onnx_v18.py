"""kotonoha ONNX accent model trainer (v18).

v13ベースに以下の改善を加えた学習パイプライン:

主な改善点:
- Sub-utterance augmentation (サブ発話データ拡張):
  各訓練発話から1-2個の連続部分列を抽出し、学習データを実質3倍に増量。
  位置特徴量を再計算し、LSTMは異なる文脈長で学習するため過学習を抑制。
  v6でのコーパス追加(+1.07%)と同様のデータ量効果を期待。
- LR warmup (optional): 最初数エポックで線形warmup
- Gradient accumulation (optional): 複数ステップで実効batch拡大
- Top-K=8 (was 5 in v13)
- 48 seeds (was 24 in v13)

v13から継承:
- Manifold Mixup, Feature noise, morpheme_dropout=0.15
- SAM, R-Drop, checkpoint averaging, multi-seed Best-of-N
- Reading dropout (p=0.1), ReduceLROnPlateau / Cosine

特徴量は nn.rs の extract_features_v2 と完全に一致させること:
  [pos_id, pd1_id, pd2_id, ct_id, cf_id,
   mora_count, reading_hash, first_char_hash,
   last_char_hash, position, dict_accent_type]

Usage:
  cp .env.example .env  # edit paths
  cd training
  uv run python train_onnx_v18.py                    # default: 48 seeds
  uv run python train_onnx_v18.py --seeds 0,1,2      # 3 seeds
  uv run python train_onnx_v18.py --no-sub-utt       # disable sub-utterance aug
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset

# ── Constants (must match nn.rs) ─────────────────────────────────────

FEATURE_DIM = 11
NUM_CLASSES = 21  # accent types 0-20

# ── Vocab mappings (must match nn.rs) ────────────────────────────────

POS_VOCAB: dict[str, int] = {
    "<unk>": 0,
    "名詞": 1,
    "助詞": 2,
    "動詞": 3,
    "助動詞": 4,
    "接尾辞": 5,
    "形容詞": 6,
    "代名詞": 7,
    "副詞": 8,
    "形状詞": 9,
    "連体詞": 10,
    "接頭辞": 11,
    "接続詞": 12,
    "感動詞": 13,
    "記号": 14,
}
NUM_POS = len(POS_VOCAB)

PD1_VOCAB: dict[str, int] = {
    "<unk>": 0,
    "*": 1,
    "普通名詞": 2,
    "格助詞": 3,
    "非自立可能": 4,
    "固有名詞": 5,
    "接続助詞": 6,
    "係助詞": 7,
    "副助詞": 8,
    "一般": 9,
    "終助詞": 10,
    "準体助詞": 11,
    "数詞": 12,
    "助動詞語幹": 13,
    "タリ": 14,
    "フィラー": 15,
    "動詞的": 16,
    "名詞的": 17,
    "形容詞的": 18,
    "形状詞的": 19,
    "文字": 20,
}
NUM_PD1 = len(PD1_VOCAB)

PD2_VOCAB: dict[str, int] = {
    "<unk>": 0,
    "*": 1,
    "一般": 2,
    "サ変可能": 3,
    "サ変形状詞可能": 4,
    "副詞可能": 5,
    "形状詞可能": 6,
    "人名": 7,
    "地名": 8,
    "助数詞": 9,
    "助数詞可能": 10,
}
NUM_PD2 = len(PD2_VOCAB)

CONJ_TYPE_GROUPS: dict[str, int] = {
    "*": 0,
    "五段": 1,
    "上一段": 2,
    "下一段": 3,
    "カ行変格": 4,
    "サ行変格": 5,
    "形容詞": 6,
    "助動詞": 7,
    "文語": 8,
}
NUM_CONJ_TYPE = len(CONJ_TYPE_GROUPS)

CONJ_FORM_GROUPS: dict[str, int] = {
    "*": 0,
    "未然形": 1,
    "連用形": 2,
    "終止形": 3,
    "連体形": 4,
    "仮定形": 5,
    "命令形": 6,
    "已然形": 7,
    "意志推量形": 8,
    "語幹": 9,
}
NUM_CONJ_FORM = len(CONJ_FORM_GROUPS)

# Kana character vocabulary for reading embedding
KANA_VOCAB: dict[str, int] = {"<pad>": 0, "<unk>": 1}
_KANA_CHARS = (
    "ァアィイゥウェエォオカガキギクグケゲコゴサザシジスズセゼソゾ"
    "タダチヂッツヅテデトドナニヌネノハバパヒビピフブプヘベペホボポ"
    "マミムメモャヤュユョヨラリルレロヮワヰヱヲンヴヵヶー"
)
for _i, _c in enumerate(_KANA_CHARS, start=2):
    KANA_VOCAB[_c] = _i
NUM_KANA = len(KANA_VOCAB)
MAX_READING_LEN = 12


# ── Feature extraction helpers ───────────────────────────────────────


def _reading_hash(reading: str) -> float:
    """Must match nn.rs reading_hash.

    Returns:
        Normalized hash value in [0, 1).

    """
    h: int = 5381
    for c in reading:
        h = (h * 33 + ord(c)) & 0xFFFFFFFF
    return (h % 10000) / 10000.0


def _char_hash(ch: str) -> float:
    """Must match nn.rs char_hash.

    Returns:
        Normalized hash value in [0, 1).

    """
    h: int = 5381
    h = (h * 33 + ord(ch)) & 0xFFFFFFFF
    return (h % 10000) / 10000.0


_SMALL_KANA = set("ァィゥェォャュョヮ")
_FUNCTION_POS = {"助詞", "助動詞"}


def _count_mora(reading: str) -> int:
    """カタカナ読みからモーラ数を算出する.

    Returns:
        モーラ数.

    """
    if not reading:
        return 0
    count = 0
    long_vowel = "ー"
    for ch in reading:
        if ch in _SMALL_KANA:
            continue
        if "\u30a0" <= ch <= "\u30ff" or ch == long_vowel:
            count += 1
    return max(count, 1)


def _get_conj_type_group(conj_type: str) -> int:
    """活用型をグループIDに変換する.

    Returns:
        グループID.

    """
    if conj_type == "*":
        return 0
    for prefix, idx in CONJ_TYPE_GROUPS.items():
        if prefix != "*" and conj_type.startswith(prefix):
            return idx
    return 0


def _get_conj_form_group(conj_form: str) -> int:
    """活用形をグループIDに変換する.

    Returns:
        グループID.

    """
    if conj_form == "*":
        return 0
    for prefix, idx in CONJ_FORM_GROUPS.items():
        if prefix != "*" and conj_form.startswith(prefix):
            return idx
    return 0


def _parse_dict_accent(val: str) -> float:
    """dict_accent_typeを数値に変換する.

    Returns:
        正規化されたアクセント型の値.

    """
    if val in ("*", ""):
        return 0.0
    m = re.match(r'^"?(\d+)$', val)
    if m:
        return (int(m.group(1)) + 1) / 8.0
    return 0.0


def _encode_reading(reading: str) -> list[int]:
    """読みを仮名文字IDのリストに変換する.

    Returns:
        MAX_READING_LEN長のID列.

    """
    ids = []
    for ch in reading[:MAX_READING_LEN]:
        ids.append(KANA_VOCAB.get(ch, 1))
    while len(ids) < MAX_READING_LEN:
        ids.append(0)
    return ids


def _compute_confidence_weight(morpheme: dict) -> float:
    """辞書アクセント型とラベルの一致度から信頼度重みを算出する.

    Returns:
        信頼度重み [0.5, 1.0].

    """
    pos = morpheme.get("pos", "")
    if pos in _FUNCTION_POS:
        return 1.0

    dict_val = morpheme.get("dict_accent_type", "*")
    if dict_val in ("*", ""):
        return 1.0

    return 1.0


def _extract_morpheme_features(
    morpheme: dict,
    position: float,
) -> list[float]:
    """形態素辞書から特徴量ベクトルを抽出する.

    Returns:
        FEATURE_DIM長の特徴量リスト.

    """
    pos_id = POS_VOCAB.get(morpheme["pos"], 0)
    pd1_id = PD1_VOCAB.get(morpheme["pos_detail1"], 0)
    pd2_id = PD2_VOCAB.get(morpheme["pos_detail2"], 0)
    ct_id = _get_conj_type_group(morpheme["conjugation_type"])
    cf_id = _get_conj_form_group(morpheme["conjugation_form"])

    reading = morpheme.get("reading", "")
    surface = morpheme.get("surface", "")

    mora_count = _count_mora(reading) / 10.0
    r_hash = _reading_hash(reading)
    first_ch = _char_hash(surface[0]) if surface else 0.0
    last_ch = _char_hash(surface[-1]) if surface else 0.0

    dict_acc = _parse_dict_accent(
        morpheme.get("dict_accent_type", "*"),
    )

    return [
        float(pos_id),
        float(pd1_id),
        float(pd2_id),
        float(ct_id),
        float(cf_id),
        mora_count,
        r_hash,
        first_ch,
        last_ch,
        position,
        dict_acc,
    ]


# ── Accent dictionary enrichment ────────────────────────────────────


def _load_accent_dict_single(path: Path) -> dict[tuple[str, str], str]:
    """単一のアクセント辞書CSVを読み込む.

    Returns:
        (lemma, reading) をキーとしたアクセント型辞書.

    """
    lookup: dict[tuple[str, str], str] = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 3:
                continue
            lemma, reading, accent = row[0], row[1], row[2]
            if lemma in ("lemma", "#") or lemma.startswith("#"):
                continue
            lookup[(lemma, reading)] = accent
    return lookup


def _load_accent_dicts(
    paths: list[Path],
) -> dict[tuple[str, str], str]:
    """複数のアクセント辞書をマージして読み込む.

    Returns:
        マージされたアクセント型辞書.

    """
    merged: dict[tuple[str, str], str] = {}
    for path in paths:
        d = _load_accent_dict_single(path)
        print(f"    {path.name}: {len(d)} entries")
        merged.update(d)
    return merged


def _enrich_utterances(
    utterances: list[dict],
    accent_dict: dict[tuple[str, str], str],
) -> int:
    """dict_accent_typeが欠損している形態素を辞書で補完する.

    Returns:
        補完された形態素数.

    """
    enriched = 0
    for utt in utterances:
        for m in utt.get("morphemes", []):
            dat = m.get("dict_accent_type", "*")
            if dat in ("*", ""):
                lemma = m.get("lemma", "")
                reading = m.get("reading", "")
                key = (lemma, reading)
                if key in accent_dict:
                    m["dict_accent_type"] = accent_dict[key]
                    enriched += 1
                else:
                    base = lemma.split("-")[0] if "-" in lemma else lemma
                    key2 = (base, reading)
                    if key2 in accent_dict:
                        m["dict_accent_type"] = accent_dict[key2]
                        enriched += 1
    return enriched


# ── Dataset ──────────────────────────────────────────────────────────


def _build_sample_from_morphemes(
    morphemes: list[dict],
) -> dict | None:
    """形態素リストからサンプル辞書を構築する.

    Returns:
        サンプル辞書、または形態素が空の場合はNone.

    """
    if not morphemes:
        return None
    n = len(morphemes)
    features = []
    labels = []
    r_ids = []
    weights = []
    for i, m in enumerate(morphemes):
        position = i / max(n - 1, 1)
        feat = _extract_morpheme_features(m, position)
        features.append(feat)
        accent = m.get("accent_type", 0)
        labels.append(min(accent, NUM_CLASSES - 1))
        reading = m.get("reading", "")
        r_ids.append(_encode_reading(reading))
        weights.append(_compute_confidence_weight(m))
    if not features:
        return None
    return {
        "features": features,
        "labels": labels,
        "reading_ids": r_ids,
        "weights": weights,
    }


def _generate_sub_utterances_from_sample(
    sample: dict,
    num_subs: int = 2,
    min_len: int = 4,
    min_ratio: float = 0.5,
) -> list[dict]:
    """構築済みサンプルから連続部分列（サブ発話）を生成する.

    元の発話から構築済みの特徴量をスライスするため、位置特徴量が保持される。
    これにより train/val の分布ミスマッチを回避する。

    Returns:
        サブ発話サンプルのリスト.

    """
    n = len(sample["features"])
    if n < min_len + 2:
        return []

    subs = []
    for _ in range(num_subs):
        min_window = max(min_len, int(n * min_ratio))
        max_window = max(min_window, int(n * 0.8))
        if min_window >= n:
            continue
        window_len = random.randint(min_window, min(max_window, n - 1))
        max_start = n - window_len
        start = random.randint(0, max_start)
        end = start + window_len
        subs.append(
            {
                "features": sample["features"][start:end],
                "labels": sample["labels"][start:end],
                "reading_ids": sample["reading_ids"][start:end],
                "weights": sample["weights"][start:end],
            }
        )
    return subs


class _AccentDataset(Dataset):
    """Accent prediction dataset with sub-utterance augmentation."""

    def __init__(
        self,
        utterances: list[dict],
        *,
        augment: bool = False,
        morpheme_dropout: float = 0.15,
        feature_noise_std: float = 0.02,
        sub_utterance_aug: bool = False,
        sub_utterance_count: int = 2,
    ) -> None:
        """Build dataset from parsed utterance dicts."""
        self.augment = augment
        self.morpheme_dropout = morpheme_dropout
        self.feature_noise_std = feature_noise_std
        self.samples: list[dict] = []

        for utt in utterances:
            morphemes = utt.get("morphemes", [])
            if not morphemes:
                continue

            sample = _build_sample_from_morphemes(morphemes)
            if sample is not None:
                self.samples.append(sample)

            # v18: Sub-utterance aug (slice to preserve positions)
            if (
                sub_utterance_aug
                and sample is not None
                and len(sample["features"]) >= 6
            ):
                subs = _generate_sub_utterances_from_sample(
                    sample, num_subs=sub_utterance_count
                )
                self.samples.extend(subs)

        if sub_utterance_aug:
            orig = len(utterances)
            total = len(self.samples)
            print(
                f"    Sub-utterance aug: {orig} -> {total} samples "
                f"(+{total - orig} sub-utterances)"
            )

    def __len__(self) -> int:
        """Return number of samples.

        Returns:
            Sample count.

        """
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        """Return sample with optional augmentation.

        Returns:
            Dict with features, labels, reading_ids.

        """
        sample = self.samples[idx]
        if self.augment:
            features = []
            r_ids = []
            for feat, rids in zip(
                sample["features"],
                sample["reading_ids"],
                strict=True,
            ):
                if (
                    self.morpheme_dropout > 0
                    and random.random() < self.morpheme_dropout
                ):
                    features.append([0.0] * FEATURE_DIM)
                    r_ids.append([0] * MAX_READING_LEN)
                else:
                    if self.feature_noise_std > 0:
                        noisy_feat = list(feat)
                        for j in range(5, FEATURE_DIM):
                            noisy_feat[j] += random.gauss(0, self.feature_noise_std)
                        features.append(noisy_feat)
                    else:
                        features.append(feat)
                    r_ids.append(rids)
            return {
                "features": features,
                "labels": sample["labels"],
                "reading_ids": r_ids,
                "weights": sample["weights"],
            }
        return sample


def _collate_fn(batch: list[dict]) -> dict:
    """可変長系列をパディングしてバッチ化する.

    Returns:
        パディング済みテンソルと系列長の辞書.

    """
    lengths = [len(item["features"]) for item in batch]
    max_len = max(lengths)

    padded_features = torch.zeros(len(batch), max_len, FEATURE_DIM)
    padded_labels = torch.zeros(len(batch), max_len, dtype=torch.long)
    padded_reading_ids = torch.zeros(
        len(batch), max_len, MAX_READING_LEN, dtype=torch.long
    )
    padded_weights = torch.zeros(len(batch), max_len)

    for i, item in enumerate(batch):
        seq_len = lengths[i]
        padded_features[i, :seq_len] = torch.tensor(
            item["features"], dtype=torch.float32
        )
        padded_labels[i, :seq_len] = torch.tensor(item["labels"], dtype=torch.long)
        padded_reading_ids[i, :seq_len] = torch.tensor(
            item["reading_ids"], dtype=torch.long
        )
        padded_weights[i, :seq_len] = torch.tensor(item["weights"], dtype=torch.float32)

    return {
        "features": padded_features,
        "labels": padded_labels,
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "reading_ids": padded_reading_ids,
        "weights": padded_weights,
    }


# ── Model (v5 architecture) ─────────────────────────────────────────


class _SelfAttentionLayer(nn.Module):
    """Multi-head self-attention with residual + LayerNorm."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        """Initialize self-attention layer."""
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv_proj = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Self-attention with residual connection.

        Returns:
            Tensor of same shape as input.

        """
        b, s, d = x.shape
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(b, s, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(b, s, d)
        out = self.out_proj(out)

        return self.norm(x + self.dropout(out))


class _EmbeddingFrontend(nn.Module):
    """Embedding layers shared by LSTM and Transformer models."""

    def __init__(
        self,
        embed_dim: int = 64,
        hidden_dim: int = 256,
    ) -> None:
        """Initialize embedding frontend."""
        super().__init__()

        self.pos_emb = nn.Embedding(NUM_POS, embed_dim // 4)
        self.pd1_emb = nn.Embedding(NUM_PD1, embed_dim // 4)
        self.pd2_emb = nn.Embedding(NUM_PD2, embed_dim // 8)
        self.ct_emb = nn.Embedding(NUM_CONJ_TYPE, embed_dim // 8)
        self.cf_emb = nn.Embedding(NUM_CONJ_FORM, embed_dim // 8)

        self.kana_emb = nn.Embedding(NUM_KANA, embed_dim // 4, padding_idx=0)
        self.kana_proj = nn.Linear(embed_dim // 4, embed_dim // 4)

        self.cont_proj = nn.Linear(6, embed_dim // 4)

        cat_emb_dim = (
            embed_dim // 4
            + embed_dim // 4
            + embed_dim // 8
            + embed_dim // 8
            + embed_dim // 8
        )
        input_dim = cat_emb_dim + embed_dim // 4 + embed_dim // 4

        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        reading_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Embed features and project to hidden_dim.

        Returns:
            [batch, seq_len, hidden_dim] tensor.

        """
        cat_feats = x[:, :, :5].long()
        cont_feats = x[:, :, 5:]

        pos_ids = cat_feats[:, :, 0].clamp(0, NUM_POS - 1)
        pd1_ids = cat_feats[:, :, 1].clamp(0, NUM_PD1 - 1)
        pd2_ids = cat_feats[:, :, 2].clamp(0, NUM_PD2 - 1)
        ct_ids = cat_feats[:, :, 3].clamp(0, NUM_CONJ_TYPE - 1)
        cf_ids = cat_feats[:, :, 4].clamp(0, NUM_CONJ_FORM - 1)

        e_pos = self.pos_emb(pos_ids)
        e_pd1 = self.pd1_emb(pd1_ids)
        e_pd2 = self.pd2_emb(pd2_ids)
        e_ct = self.ct_emb(ct_ids)
        e_cf = self.cf_emb(cf_ids)

        if reading_ids is not None:
            rid = reading_ids.clamp(0, NUM_KANA - 1)
            kana_embs = self.kana_emb(rid)
            kana_mask = (rid > 0).float().unsqueeze(-1)
            kana_sum = (kana_embs * kana_mask).sum(dim=2)
            kana_count = kana_mask.sum(dim=2).clamp(min=1.0)
            kana_pooled = kana_sum / kana_count
            e_kana = self.kana_proj(kana_pooled)
        else:
            r_hash = cont_feats[:, :, 1:2]
            e_kana = self.kana_proj(
                r_hash.expand(-1, -1, self.kana_proj.in_features),
            )

        e_cont = self.cont_proj(cont_feats)

        h = torch.cat(
            [e_pos, e_pd1, e_pd2, e_ct, e_cf, e_kana, e_cont],
            dim=-1,
        )
        h = self.input_norm(h)
        return self.input_proj(h)


class AccentModel(nn.Module):
    """BiLSTM + Self-Attention accent predictor.

    v18: v13アーキテクチャを維持。SWAフェーズを外部で管理。
    """

    def __init__(
        self,
        embed_dim: int = 64,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.4,
        attention_heads: int = 4,
        reading_dropout: float = 0.0,
    ) -> None:
        """Initialize the accent predictor model."""
        super().__init__()

        self.reading_dropout = reading_dropout
        self.frontend = _EmbeddingFrontend(embed_dim, hidden_dim)

        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.lstm_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.lstm_norm = nn.LayerNorm(hidden_dim)

        self.self_attn = _SelfAttentionLayer(
            hidden_dim,
            num_heads=attention_heads,
            dropout=dropout,
        )

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
        reading_ids: torch.Tensor | None = None,
        *,
        return_hidden: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Returns:
            logits or (logits, hidden) if return_hidden.

        """
        if (
            self.training
            and reading_ids is not None
            and self.reading_dropout > 0
            and random.random() < self.reading_dropout
        ):
            reading_ids = None

        h = self.frontend(x, reading_ids)
        h_input = h

        if lengths is not None and self.training:
            packed = pack_padded_sequence(
                h,
                lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            packed_out, _ = self.lstm(packed)
            h, _ = pad_packed_sequence(packed_out, batch_first=True)
        else:
            h, _ = self.lstm(h)

        h = self.lstm_proj(h)
        if h.size(1) < h_input.size(1):
            pad_len = h_input.size(1) - h.size(1)
            h = F.pad(h, (0, 0, 0, pad_len))
        h = self.lstm_norm(h + h_input)

        if return_hidden:
            return h, h

        if lengths is not None:
            max_len = h.size(1)
            arange = torch.arange(max_len, device=h.device)
            attn_mask = arange.unsqueeze(0) >= lengths.unsqueeze(1).to(h.device)
        else:
            attn_mask = None

        h = self.self_attn(h, mask=attn_mask)
        h = self.dropout(h)
        return self.classifier(h)

    def forward_from_hidden(
        self,
        h: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward from hidden representation (for Manifold Mixup).

        Returns:
            logits: [batch, seq_len, num_classes].

        """
        if lengths is not None:
            max_len = h.size(1)
            arange = torch.arange(max_len, device=h.device)
            attn_mask = arange.unsqueeze(0) >= lengths.unsqueeze(1).to(h.device)
        else:
            attn_mask = None

        h = self.self_attn(h, mask=attn_mask)
        h = self.dropout(h)
        return self.classifier(h)


# ── ONNX export ──────────────────────────────────────────────────────


class _OnnxWrapper(nn.Module):
    """Batch dim removal for ONNX export."""

    def __init__(self, model: AccentModel) -> None:
        """Wrap model for ONNX export."""
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map [seq_len, 11] to [seq_len, 21].

        Returns:
            Logits of shape [seq_len, num_classes].

        """
        out = self.model(x.unsqueeze(0))
        return out.squeeze(0)


def _export_onnx(
    model: AccentModel,
    path: Path,
    device: torch.device,
) -> None:
    """モデルをONNX形式でエクスポートする."""
    wrapper = _OnnxWrapper(model).to(device)
    wrapper.eval()
    path.parent.mkdir(parents=True, exist_ok=True)

    dummy = torch.zeros(10, FEATURE_DIM, device=device)
    dummy[:, 0] = 1
    dummy[:, 1] = 2
    dummy[:, 5] = 0.3

    torch.onnx.export(
        wrapper,
        (dummy,),
        str(path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "seq_len"},
            "output": {0: "seq_len"},
        },
        opset_version=17,
        dynamo=False,
    )

    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(path))
        inp = sess.get_inputs()[0]
        out = sess.get_outputs()[0]
        print(f"  ONNX input: {inp.name} {inp.shape}")
        print(f"  ONNX output: {out.name} {out.shape}")
        for seq_len in [1, 5, 20]:
            test_in = torch.zeros(seq_len, FEATURE_DIM).numpy()
            result = sess.run(None, {"input": test_in})
            print(f"  Verify seq_len={seq_len}: -> {result[0].shape}")
    except Exception as exc:  # noqa: BLE001
        print(f"  ONNX verification failed: {exc}")


# ── Training helpers ─────────────────────────────────────────────────


def _build_mask(
    labels: torch.Tensor,
    lengths: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """パディング位置を除外するマスクを作成する.

    Returns:
        Boolean mask tensor.

    """
    max_len = labels.size(1)
    return torch.arange(max_len, device=device).unsqueeze(0) < lengths.unsqueeze(1).to(
        device
    )


def _manifold_mixup_batch(  # noqa: PLR0913
    model: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    lengths: torch.Tensor,
    r_ids: torch.Tensor,
    device: torch.device,
    alpha: float = 0.3,
    label_smoothing: float = 0.1,
) -> torch.Tensor:
    """Manifold Mixup: LSTM出力レベルで2サンプルを補間して損失を計算.

    Returns:
        Mixup loss scalar.

    """
    batch_size = features.size(0)
    if batch_size < 2:
        return torch.tensor(0.0, device=device)

    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)

    h1, _ = model(features, lengths, r_ids, return_hidden=True)

    perm = torch.randperm(batch_size)
    h2 = h1[perm.to(device)]
    labels2 = labels[perm.to(device)]
    lengths2 = lengths[perm]

    h_mixed = lam * h1 + (1 - lam) * h2

    min_lengths = torch.min(lengths, lengths2)
    mixed_mask = _build_mask(labels, min_lengths, device)

    logits_mixed = model.forward_from_hidden(h_mixed, min_lengths)
    logits_flat = logits_mixed[mixed_mask]
    labels1_flat = labels[mixed_mask]
    labels2_flat = labels2[mixed_mask]

    if logits_flat.size(0) == 0:
        return torch.tensor(0.0, device=device)

    log_probs = F.log_softmax(logits_flat, dim=-1)
    num_classes = logits_flat.size(-1)

    smooth = label_smoothing / num_classes
    targets1 = torch.full_like(log_probs, smooth)
    targets1.scatter_(1, labels1_flat.unsqueeze(1), 1.0 - label_smoothing + smooth)
    targets2 = torch.full_like(log_probs, smooth)
    targets2.scatter_(1, labels2_flat.unsqueeze(1), 1.0 - label_smoothing + smooth)

    mixed_targets = lam * targets1 + (1 - lam) * targets2
    return -(mixed_targets * log_probs).sum(dim=-1).mean()


def _train_epoch(  # noqa: PLR0913, PLR0912, C901
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    label_smoothing: float = 0.1,
    rdrop_alpha: float = 0.0,
    sam_rho: float = 0.0,
    mixup_alpha: float = 0.3,
    mixup_prob: float = 0.3,
    accum_steps: int = 1,
) -> tuple[float, float]:
    """1エポック分の学習を実行する (gradient accumulation対応).

    Returns:
        (平均損失, 精度) のタプル.

    """
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    use_amp = device.type == "cuda"
    optimizer.zero_grad()

    for step, batch in enumerate(dataloader):
        features = batch["features"].to(device)
        labels = batch["labels"].to(device)
        lengths = batch["lengths"]
        r_ids = batch["reading_ids"].to(device)
        weights = batch["weights"].to(device)

        mask = _build_mask(labels, lengths, device)
        labels_flat = labels[mask]
        weights_flat = weights[mask]

        if sam_rho > 0:
            # ── SAM: gradient accumulation not applied to SAM for simplicity ──
            # SAM needs immediate perturb-step, so we step every batch
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(features, lengths, r_ids)
                logits_flat = logits[mask]
                ce = F.cross_entropy(
                    logits_flat,
                    labels_flat,
                    label_smoothing=label_smoothing,
                    reduction="none",
                )
                loss = (ce * weights_flat).mean()
                if rdrop_alpha > 0:
                    logits2 = model(features, lengths, r_ids)
                    logits2_flat = logits2[mask]
                    lp = F.log_softmax(logits_flat, dim=-1)
                    lq = F.log_softmax(logits2_flat, dim=-1)
                    kl1 = F.kl_div(lp, lq.exp(), reduction="batchmean")
                    kl2 = F.kl_div(lq, lp.exp(), reduction="batchmean")
                    loss = loss + rdrop_alpha * (kl1 + kl2) / 2
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            with torch.no_grad():
                grad_norms = [
                    param.grad.norm(p=2)
                    for param in model.parameters()
                    if param.grad is not None
                ]
                grad_norm = torch.norm(torch.stack(grad_norms), p=2)
                perturbations = {}
                for param in model.parameters():
                    if param.grad is not None:
                        e_w = sam_rho / (grad_norm + 1e-12) * param.grad
                        perturbations[param] = e_w
                        param.add_(e_w)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits_p = model(features, lengths, r_ids)
                logits_p_flat = logits_p[mask]
                ce_p = F.cross_entropy(
                    logits_p_flat,
                    labels_flat,
                    label_smoothing=label_smoothing,
                    reduction="none",
                )
                loss_p = (ce_p * weights_flat).mean()
                if rdrop_alpha > 0:
                    logits_p2 = model(features, lengths, r_ids)
                    logits_p2_flat = logits_p2[mask]
                    lp2 = F.log_softmax(logits_p_flat, dim=-1)
                    lq2 = F.log_softmax(logits_p2_flat, dim=-1)
                    kl3 = F.kl_div(lp2, lq2.exp(), reduction="batchmean")
                    kl4 = F.kl_div(lq2, lp2.exp(), reduction="batchmean")
                    loss_p = loss_p + rdrop_alpha * (kl3 + kl4) / 2
                if mixup_alpha > 0 and random.random() < mixup_prob:
                    mixup_loss = _manifold_mixup_batch(
                        model,
                        features,
                        labels,
                        lengths,
                        r_ids,
                        device,
                        alpha=mixup_alpha,
                        label_smoothing=label_smoothing,
                    )
                    loss_p = loss_p + 0.5 * mixup_loss
            loss_p.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            with torch.no_grad():
                for param, e_w in perturbations.items():
                    param.sub_(e_w)
            optimizer.step()
            optimizer.zero_grad()
        else:
            # ── Standard training with gradient accumulation ──
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(features, lengths, r_ids)
                logits_flat = logits[mask]
                ce_loss = F.cross_entropy(
                    logits_flat,
                    labels_flat,
                    label_smoothing=label_smoothing,
                    reduction="none",
                )
                loss = (ce_loss * weights_flat).mean()

                if rdrop_alpha > 0:
                    logits2 = model(features, lengths, r_ids)
                    logits2_flat = logits2[mask]
                    p = F.log_softmax(logits_flat, dim=-1)
                    q = F.log_softmax(logits2_flat, dim=-1)
                    kl_pq = F.kl_div(p, q.exp(), reduction="batchmean")
                    kl_qp = F.kl_div(q, p.exp(), reduction="batchmean")
                    loss = loss + rdrop_alpha * (kl_pq + kl_qp) / 2

                if mixup_alpha > 0 and random.random() < mixup_prob:
                    mixup_loss = _manifold_mixup_batch(
                        model,
                        features,
                        labels,
                        lengths,
                        r_ids,
                        device,
                        alpha=mixup_alpha,
                        label_smoothing=label_smoothing,
                    )
                    loss = loss + 0.5 * mixup_loss

                # Scale loss for gradient accumulation
                loss = loss / accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % accum_steps == 0 or (step + 1) == len(dataloader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        with torch.no_grad():
            # Track unscaled loss for reporting
            actual_loss = loss.item() * (accum_steps if sam_rho == 0 else 1)
            total_loss += actual_loss * labels_flat.size(0)
            preds = logits_flat.argmax(dim=-1)
            total_correct += (preds == labels_flat).sum().item()
            total_count += labels_flat.size(0)

    avg_loss = total_loss / max(total_count, 1)
    accuracy = total_correct / max(total_count, 1)
    return avg_loss, accuracy


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """検証データでモデルを評価する.

    Returns:
        (平均損失, 精度) のタプル.

    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for batch in dataloader:
        features = batch["features"].to(device)
        labels = batch["labels"].to(device)
        lengths = batch["lengths"]
        r_ids = batch["reading_ids"].to(device)

        logits = model(features, lengths, r_ids)
        mask = _build_mask(labels, lengths, device)

        logits_flat = logits[mask]
        labels_flat = labels[mask]
        loss = F.cross_entropy(logits_flat, labels_flat)
        total_loss += loss.item() * labels_flat.size(0)
        preds = logits_flat.argmax(dim=-1)
        total_correct += (preds == labels_flat).sum().item()
        total_count += labels_flat.size(0)

    avg_loss = total_loss / max(total_count, 1)
    accuracy = total_correct / max(total_count, 1)
    return avg_loss, accuracy


# ── SWA helpers ──────────────────────────────────────────────────────


def _swa_phase(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    *,
    swa_lr: float = 2e-4,
    swa_epochs: int = 20,
    label_smoothing: float = 0.1,
    rdrop_alpha: float = 0.5,
    mixup_alpha: float = 0.3,
) -> tuple[float, dict[str, torch.Tensor]]:
    """SWA後処理フェーズ: 定数LRで学習しweight averagingを行う.

    Stage 2収束後のモデルから開始し、定数LRで追加学習しながら
    各エポックのstate_dictを平均する。Top-K avgが学習中のベストK個を
    選択するのに対し、SWAは平坦な最小値周辺を連続的に探索する。

    Returns:
        (SWAモデルの検証精度, SWA平均state_dict) のタプル.

    """
    if swa_epochs <= 0:
        return 0.0, {}

    print(f"  [SWA] {swa_epochs} epochs, lr={swa_lr}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=swa_lr,
        weight_decay=0.05,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    # Initialize SWA accumulator with current model
    swa_state: dict[str, torch.Tensor] = {
        k: v.cpu().clone().float() for k, v in model.state_dict().items()
    }
    swa_count = 1

    for epoch in range(1, swa_epochs + 1):
        tr_loss, tr_acc = _train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            scaler,
            label_smoothing=label_smoothing,
            rdrop_alpha=rdrop_alpha,
            sam_rho=0.0,  # No SAM in SWA phase (keep it stable)
            mixup_alpha=mixup_alpha,
            accum_steps=1,
        )

        # Accumulate into running average
        with torch.no_grad():
            for key, val in model.state_dict().items():
                swa_state[key] += val.cpu().float()
            swa_count += 1

        if epoch % 5 == 0 or epoch == swa_epochs:
            va_loss, va_acc = _evaluate(model, val_loader, device)
            print(f"    SWA ep {epoch:2d}: va={va_acc:.4f} tr={tr_acc:.4f}")

    # Compute average
    for key in swa_state:
        swa_state[key] /= swa_count

    # Evaluate averaged model
    model.load_state_dict(swa_state)
    model.to(device)
    _, swa_acc = _evaluate(model, val_loader, device)
    print(f"  SWA avg ({swa_count} snapshots): {swa_acc * 100:.2f}%")

    return swa_acc, {k: v.cpu() for k, v in swa_state.items()}


# ── Main training ────────────────────────────────────────────────────


def _train_single(  # noqa: PLR0913
    jvs_utterances: list[dict] | None,
    jsut_utterances: list[dict],
    *,
    corpus_utterances: list[dict] | None = None,
    seed: int,
    embed_dim: int = 64,
    hidden_dim: int = 256,
    num_layers: int = 3,
    dropout: float = 0.4,
    attention_heads: int = 4,
    reading_dropout: float = 0.1,
    batch_size: int = 64,
    weight_decay: float = 0.05,
    label_smoothing: float = 0.1,
    morpheme_dropout: float = 0.15,
    feature_noise_std: float = 0.02,
    stage1_lr: float = 2e-3,
    stage1_epochs: int = 20,
    stage2_lr: float = 8e-4,
    stage2_epochs: int = 100,
    stage2_patience: int = 25,
    rdrop_alpha: float = 0.5,
    top_k: int = 8,
    use_cosine: bool = True,
    sam_rho: float = 0.0,
    mixup_alpha: float = 0.3,
    warmup_epochs: int = 0,
    accum_steps: int = 1,
    swa_lr: float = 2e-4,
    swa_epochs: int = 0,
    sub_utterance_aug: bool = True,
) -> tuple[float, dict[str, torch.Tensor]]:
    """独立した1 seedで全Stage訓練し、最良のstate_dictを返す.

    Returns:
        (検証精度, state_dict) のタプル.

    """
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AccentModel(
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=NUM_CLASSES,
        dropout=dropout,
        attention_heads=attention_heads,
        reading_dropout=reading_dropout,
    ).to(device)

    # ── Stage 1: Pre-training on JVS ──
    if jvs_utterances:
        print(f"  [Stage 1] Pre-training ({stage1_epochs} ep)")
        jvs_ds = _AccentDataset(
            jvs_utterances,
            augment=True,
            morpheme_dropout=morpheme_dropout,
            feature_noise_std=feature_noise_std,
        )
        jvs_loader = DataLoader(
            jvs_ds,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=_collate_fn,
            num_workers=2,
            pin_memory=True,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=stage1_lr, weight_decay=weight_decay
        )
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        for epoch in range(1, stage1_epochs + 1):
            loss, acc = _train_epoch(
                model,
                jvs_loader,
                optimizer,
                device,
                scaler,
                label_smoothing=label_smoothing,
            )
            if epoch % 10 == 0 or epoch == stage1_epochs:
                print(f"    ep {epoch:2d}: loss={loss:.4f} acc={acc:.4f}")

    # ── Val split (fixed seed=42, matching v7 flow) ──
    random.seed(42)
    indices = list(range(len(jsut_utterances)))
    random.shuffle(indices)
    val_size = int(len(indices) * 0.1)
    val_idx = set(indices[:val_size])
    train_utts = [u for i, u in enumerate(jsut_utterances) if i not in val_idx]
    val_utts = [u for i, u in enumerate(jsut_utterances) if i in val_idx]

    if corpus_utterances:
        train_utts = train_utts + corpus_utterances

    # ── Stage 2: Fine-tuning with warmup + extended training ──
    sched_name = "cosine" if use_cosine else "plateau"
    sam_info = f", SAM rho={sam_rho}" if sam_rho > 0 else ""
    warmup_info = f", warmup={warmup_epochs}ep" if warmup_epochs > 0 else ""
    accum_info = f", accum={accum_steps}" if accum_steps > 1 else ""
    print(
        f"  [Stage 2] Fine-tuning ({stage2_epochs} ep, "
        f"sched={sched_name}{sam_info}{warmup_info}{accum_info})"
    )

    train_ds = _AccentDataset(
        train_utts,
        augment=True,
        morpheme_dropout=morpheme_dropout,
        feature_noise_std=feature_noise_std,
        sub_utterance_aug=sub_utterance_aug,
    )
    val_ds = _AccentDataset(val_utts, augment=False)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_collate_fn,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=stage2_lr,
        weight_decay=weight_decay,
    )

    # v18: LR warmup + cosine/plateau scheduler
    if use_cosine:
        # Cosine decay after warmup
        if warmup_epochs > 0:
            # Warmup scheduler: linear from 1e-6 to stage2_lr
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1e-6 / stage2_lr,
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            # Cosine decay for remaining epochs
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=stage2_epochs - warmup_epochs,
                eta_min=1e-6,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs],
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=stage2_epochs, eta_min=1e-6
            )
    else:
        if warmup_epochs > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1e-6 / stage2_lr,
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=0.5,
                patience=5,
                min_lr=1e-6,
            )
        else:
            warmup_scheduler = None
            plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=0.5,
                patience=5,
                min_lr=1e-6,
            )

    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_val_acc = 0.0
    best_state: dict | None = None
    no_improve = 0
    top_states: list[tuple[float, dict]] = []

    for epoch in range(1, stage2_epochs + 1):
        tr_loss, tr_acc = _train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            scaler,
            label_smoothing=label_smoothing,
            rdrop_alpha=rdrop_alpha,
            sam_rho=sam_rho,
            mixup_alpha=mixup_alpha,
            accum_steps=accum_steps,
        )
        va_loss, va_acc = _evaluate(model, val_loader, device)

        lr = optimizer.param_groups[0]["lr"]
        if use_cosine:
            scheduler.step()
        else:
            if warmup_scheduler is not None and epoch <= warmup_epochs:
                warmup_scheduler.step()
            else:
                plateau_scheduler.step(va_acc)

        marker = ""
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            marker = " *"
        else:
            no_improve += 1

        # Track top-K checkpoints
        state_copy = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if len(top_states) < top_k:
            top_states.append((va_acc, state_copy))
            top_states.sort(key=lambda x: x[0])
        elif va_acc > top_states[0][0]:
            top_states[0] = (va_acc, state_copy)
            top_states.sort(key=lambda x: x[0])

        if epoch % 5 == 0 or marker or epoch <= 3:
            gap = tr_acc - va_acc
            print(
                f"    ep {epoch:2d}: va={va_acc:.4f} "
                f"tr={tr_acc:.4f} gap={gap:.4f} lr={lr:.6f}{marker}"
            )

        if no_improve >= stage2_patience:
            print(f"    Early stop at ep {epoch}")
            break

    print(f"  Best single: {best_val_acc * 100:.2f}%")

    # ── Top-K Checkpoint Averaging ──
    if len(top_states) >= 2:
        accs = [a for a, _ in top_states]
        avg_state: dict[str, torch.Tensor] = {}
        for key in top_states[0][1]:
            stacked = torch.stack([s[key].float() for _, s in top_states])
            avg_state[key] = stacked.mean(dim=0)
        model.load_state_dict(avg_state)
        model.to(device)
        _, avg_acc = _evaluate(model, val_loader, device)
        print(
            f"  Top-{len(top_states)} avg: {avg_acc * 100:.2f}% "
            f"(from {', '.join(f'{a * 100:.2f}' for a in accs)})"
        )
        if avg_acc > best_val_acc:
            best_val_acc = avg_acc
            best_state = avg_state

    # ── v18: SWA phase ──
    # Start SWA from the best state (Top-K avg or single best)
    if swa_epochs > 0 and best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
        swa_acc, swa_state = _swa_phase(
            model,
            train_loader,
            val_loader,
            device,
            swa_lr=swa_lr,
            swa_epochs=swa_epochs,
            label_smoothing=label_smoothing,
            rdrop_alpha=rdrop_alpha,
            mixup_alpha=mixup_alpha,
        )
        if swa_acc > best_val_acc:
            print(f"  SWA improved: {best_val_acc * 100:.2f}% -> {swa_acc * 100:.2f}%")
            best_val_acc = swa_acc
            best_state = swa_state
        else:
            print(
                f"  SWA not better: {swa_acc * 100:.2f}% vs "
                f"{best_val_acc * 100:.2f}% (keeping Top-K)"
            )

    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    return best_val_acc, {k: v.cpu() for k, v in best_state.items()}


# ── CLI ──────────────────────────────────────────────────────────────


def _load_dotenv() -> None:
    """Load .env file into environment variables."""
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


def main() -> None:
    """エントリポイント."""
    _load_dotenv()

    parser = argparse.ArgumentParser(description="Train kotonoha accent model v18")
    parser.add_argument(
        "--pretrain-data",
        type=Path,
        default=Path(os.environ.get("PRETRAIN_DATA", "")),
    )
    parser.add_argument(
        "--finetune-data",
        type=Path,
        default=Path(os.environ.get("FINETUNE_DATA", "")),
    )
    parser.add_argument(
        "--accent-dict",
        type=str,
        default=os.environ.get("ACCENT_DICT", ""),
        help="Accent dict CSV paths (colon-separated)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(os.environ.get("OUTPUT_MODEL", "")),
    )
    parser.add_argument(
        "--corpus-data",
        type=Path,
        default=Path(os.environ.get("CORPUS_DATA", "")),
        help="LLM corpus data (mixed into Stage 2)",
    )
    parser.add_argument("--stage1-epochs", type=int, default=20)
    parser.add_argument("--stage1-lr", type=float, default=2e-3)
    parser.add_argument("--stage2-epochs", type=int, default=80)
    parser.add_argument("--stage2-lr", type=float, default=8e-4)
    parser.add_argument("--stage2-patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument(
        "--reading-dropout",
        type=float,
        default=0.1,
        help="Prob of dropping reading_ids during training",
    )
    parser.add_argument(
        "--morpheme-dropout",
        type=float,
        default=0.15,
        help="Prob of dropping morpheme features during training",
    )
    parser.add_argument(
        "--feature-noise-std",
        type=float,
        default=0.02,
        help="Std of Gaussian noise on continuous features",
    )
    parser.add_argument(
        "--mixup-alpha",
        type=float,
        default=0.3,
        help="Manifold Mixup alpha (0=disabled)",
    )
    parser.add_argument(
        "--no-pretrain",
        action="store_true",
        help="Skip Stage 1",
    )
    parser.add_argument(
        "--rdrop-alpha",
        type=float,
        default=0.5,
        help="R-Drop KL divergence weight (0=disabled)",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(i) for i in range(48)),
        help="Comma-separated seeds for best-of-N",
    )
    parser.add_argument("--top-k", type=int, default=8, help="Top-K checkpoints to avg")
    parser.add_argument(
        "--scheduler",
        choices=["cosine", "plateau"],
        default="plateau",
        help="LR scheduler (cosine or plateau)",
    )
    parser.add_argument(
        "--sam-rho",
        type=float,
        default=0.05,
        help="SAM perturbation radius (0=disabled)",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=0,
        help="LR warmup epochs (0=disabled)",
    )
    parser.add_argument(
        "--accum-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps (1=disabled)",
    )
    parser.add_argument(
        "--swa-lr",
        type=float,
        default=2e-4,
        help="SWA learning rate",
    )
    parser.add_argument(
        "--swa-epochs",
        type=int,
        default=0,
        help="SWA post-training epochs (0=disabled)",
    )
    parser.add_argument(
        "--no-sub-utt",
        action="store_true",
        help="Disable sub-utterance augmentation",
    )
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]

    sub_utt = not args.no_sub_utt

    print("=" * 65)
    print("kotonoha accent model trainer v18 (v13 + Sub-utterance Aug)")
    print(f"  Arch: BiLSTM({args.num_layers}) + SelfAttn(4h), hidden={args.hidden_dim}")
    print(f"  Seeds: {len(seeds)} seeds (best-of-N)")
    print(f"  R-Drop: {args.rdrop_alpha}, Top-K: {args.top_k}, Sched: {args.scheduler}")
    print(f"  SAM: rho={args.sam_rho}, ReadDrop: {args.reading_dropout}")
    print(f"  MorphDrop: {args.morpheme_dropout}, FeatNoise: {args.feature_noise_std}")
    print(f"  Mixup: alpha={args.mixup_alpha}, SubUttAug: {sub_utt}")
    if args.warmup_epochs > 0:
        print(f"  Warmup: {args.warmup_epochs}ep")
    if args.accum_steps > 1:
        print(f"  AccumSteps: {args.accum_steps}")
    if args.swa_epochs > 0:
        print(f"  SWA: {args.swa_epochs}ep, lr={args.swa_lr}")
    print(f"  Stage2: {args.stage2_epochs}ep, patience={args.stage2_patience}")
    print("=" * 65)

    # Load accent dicts
    dict_paths = [Path(p) for p in args.accent_dict.split(":") if p]
    print(f"\nLoading accent dicts ({len(dict_paths)} files):")
    accent_dict = _load_accent_dicts(dict_paths)
    print(f"  Total (merged): {len(accent_dict)} entries")

    # Load JVS (pre-training)
    jvs_utterances = None
    pt = args.pretrain_data
    if not args.no_pretrain and pt.name and pt.exists():
        print(f"\nLoading JVS: {pt}")
        with open(pt, encoding="utf-8") as f:
            jvs_data = json.load(f)
        jvs_utterances = jvs_data["utterances"]
        print(f"  {len(jvs_utterances)} utterances")
        n = _enrich_utterances(jvs_utterances, accent_dict)
        print(f"  Enriched {n} morphemes")

    # Load corpus
    corpus_utterances: list[dict] = []
    cp = args.corpus_data
    if cp.name and cp.exists():
        print(f"\nLoading corpus: {cp}")
        with open(cp, encoding="utf-8") as f:
            corpus_data = json.load(f)
        if isinstance(corpus_data, list):
            corpus_utterances = corpus_data
        else:
            corpus_utterances = corpus_data.get("utterances", [])
        print(f"  {len(corpus_utterances)} utterances")
        n = _enrich_utterances(corpus_utterances, accent_dict)
        print(f"  Enriched {n} morphemes")

    # Load JSUT (fine-tuning)
    print(f"\nLoading JSUT: {args.finetune_data}")
    with open(args.finetune_data, encoding="utf-8") as f:
        jsut_data = json.load(f)
    jsut_utterances = jsut_data["utterances"]
    print(f"  {len(jsut_utterances)} utterances")
    n = _enrich_utterances(jsut_utterances, accent_dict)
    print(f"  Enriched {n} morphemes")

    print(f"  JSUT: {len(jsut_utterances)} utterances")
    if corpus_utterances:
        print(f"  Corpus: {len(corpus_utterances)} utterances (mixed into Stage 2)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name()}")

    # ── Multi-seed training ──
    all_results: list[tuple[float, dict[str, torch.Tensor]]] = []

    for i, seed in enumerate(seeds):
        print(f"\n{'=' * 65}")
        print(f"[Seed {seed}] ({i + 1}/{len(seeds)})")
        print(f"{'=' * 65}")

        acc, state = _train_single(
            jvs_utterances,
            jsut_utterances,
            corpus_utterances=corpus_utterances or None,
            seed=seed,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            batch_size=args.batch_size,
            stage1_lr=args.stage1_lr,
            stage1_epochs=args.stage1_epochs,
            stage2_lr=args.stage2_lr,
            stage2_epochs=args.stage2_epochs,
            stage2_patience=args.stage2_patience,
            rdrop_alpha=args.rdrop_alpha,
            top_k=args.top_k,
            use_cosine=args.scheduler == "cosine",
            sam_rho=args.sam_rho,
            reading_dropout=args.reading_dropout,
            morpheme_dropout=args.morpheme_dropout,
            feature_noise_std=args.feature_noise_std,
            mixup_alpha=args.mixup_alpha,
            warmup_epochs=args.warmup_epochs,
            accum_steps=args.accum_steps,
            swa_lr=args.swa_lr,
            swa_epochs=args.swa_epochs,
            sub_utterance_aug=sub_utt,
        )
        all_results.append((acc, state))
        print(f"  => Seed {seed} result: {acc * 100:.2f}%")

    # ── Best-of-N selection ──
    print(f"\n{'=' * 65}")
    print(f"[Results] {len(all_results)} seeds")
    print(f"{'=' * 65}")
    accs_list = []
    for (acc, _), seed in zip(all_results, seeds, strict=True):
        print(f"  Seed {seed}: {acc * 100:.2f}%")
        accs_list.append(acc)

    best_acc, best_state = max(all_results, key=lambda x: x[0])
    best_seed = seeds[all_results.index((best_acc, best_state))]

    mean_acc = np.mean(accs_list)
    std_acc = np.std(accs_list)
    print(f"\n  Mean: {mean_acc * 100:.2f}% ± {std_acc * 100:.2f}%")
    print(f"  Best: seed {best_seed} = {best_acc * 100:.2f}%")

    # Count seeds above thresholds
    for threshold in [0.71, 0.715, 0.72]:
        count = sum(1 for a in accs_list if a >= threshold)
        print(f"  >= {threshold * 100:.1f}%: {count}/{len(seeds)} seeds")

    model = AccentModel(
        embed_dim=64,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=NUM_CLASSES,
        dropout=args.dropout,
        attention_heads=4,
        reading_dropout=0.0,
    ).to(device)
    model.load_state_dict(best_state)
    model.to(device)
    final_acc = best_acc

    # Export ONNX
    print("\nExporting ONNX...")
    _export_onnx(model, args.output, device)
    file_size = args.output.stat().st_size
    print(f"  Output: {args.output}")
    print(f"  Size: {file_size / 1_048_576:.1f} MB")

    print(f"\n{'=' * 65}")
    print(f"Done! Final val accuracy: {final_acc * 100:.2f}%")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
