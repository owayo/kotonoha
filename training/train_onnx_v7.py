"""kotonoha ONNX accent model trainer (v7).

v6ベースに以下の改善を加えた学習パイプライン:
- R-Drop正則化（ドロップアウトマスク間のKL発散ペナルティ）
- Top-K checkpoint averaging（上位チェックポイントの重み平均）
- LLMコーパスデータのStage 2混合学習

v6アーキテクチャ（Embedding層 + カスタムSelf-Attention + 残差接続 + 2層MLP）を
継承した2段階転移学習パイプライン。

Stage 1: JVS データで事前学習（ノイズの多い複数話者データから基本パターン学習）
Stage 2: JSUT + コーパスデータで微調整（クリーンな単一話者データ + LLM生成コーパス）

特徴量は nn.rs の extract_features_v2 と完全に一致させること:
  [pos_id, pd1_id, pd2_id, ct_id, cf_id,
   mora_count, reading_hash, first_char_hash,
   last_char_hash, position, dict_accent_type]

Usage:
  cp .env.example .env  # edit paths
  cd training
  uv run python train_onnx_v7.py --seed 1 --rdrop-alpha 0.5
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
from pathlib import Path

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

    - 機能語（助詞・助動詞）: 常に1.0（句のアクセント型を共有するだけ）
    - 辞書エントリなし: 1.0（判断材料なし）
    - 辞書とラベル一致: 1.0
    - dict=0 かつ label!=0: 0.8（アクセント句結合の可能性）
    - dict!=0 かつ label!=dict: 0.5（ラベルエラーの疑い）

    Returns:
        信頼度重み [0.5, 1.0].

    """
    pos = morpheme.get("pos", "")
    if pos in _FUNCTION_POS:
        return 1.0

    dict_val = morpheme.get("dict_accent_type", "*")
    if dict_val in ("*", ""):
        return 1.0

    # 現状では辞書/ラベル不一致の多くが正しいラベルであるため、
    # 重み付けは精度向上に寄与しない。全て1.0を返す。
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
            # Skip header or comment rows
            if lemma in ("lemma", "#") or lemma.startswith("#"):
                continue
            lookup[(lemma, reading)] = accent
    return lookup


def _load_accent_dicts(
    paths: list[Path],
) -> dict[tuple[str, str], str]:
    """複数のアクセント辞書をマージして読み込む.

    後のファイルが優先される（上書き）。

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


class _AccentDataset(Dataset):
    """Accent prediction dataset with morpheme dropout."""

    def __init__(
        self,
        utterances: list[dict],
        *,
        augment: bool = False,
        morpheme_dropout: float = 0.1,
    ) -> None:
        """Build dataset from parsed utterance dicts."""
        self.augment = augment
        self.morpheme_dropout = morpheme_dropout
        self.samples: list[dict] = []

        for utt in utterances:
            morphemes = utt.get("morphemes", [])
            if not morphemes:
                continue

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

            if features:
                self.samples.append(
                    {
                        "features": features,
                        "labels": labels,
                        "reading_ids": r_ids,
                        "weights": weights,
                    }
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
        if self.augment and self.morpheme_dropout > 0:
            features = []
            r_ids = []
            for feat, rids in zip(
                sample["features"],
                sample["reading_ids"],
                strict=True,
            ):
                if random.random() < self.morpheme_dropout:
                    features.append([0.0] * FEATURE_DIM)
                    r_ids.append([0] * MAX_READING_LEN)
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
    """BiLSTM + Self-Attention accent predictor (v5 arch).

    Categorical features use Embedding layers. Readings use
    kana-level embedding with mean pooling. LSTM output has
    residual connection. Classifier is a 2-layer MLP with GELU.
    """

    def __init__(
        self,
        embed_dim: int = 64,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.4,
        attention_heads: int = 4,
    ) -> None:
        """Initialize the accent predictor model."""
        super().__init__()

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
    ) -> torch.Tensor:
        """Forward pass.

        Returns:
            logits: [batch, seq_len, num_classes].

        """
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

        if lengths is not None:
            max_len = h.size(1)
            arange = torch.arange(max_len, device=h.device)
            attn_mask = arange.unsqueeze(0) >= lengths.unsqueeze(1).to(h.device)
        else:
            attn_mask = None

        h = self.self_attn(h, mask=attn_mask)
        h = self.dropout(h)
        return self.classifier(h)


class _SinusoidalPE(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, dim: int, max_len: int = 512) -> None:
        """Initialize positional encoding buffer."""
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding.

        Returns:
            Input tensor with positional encoding added.

        """
        return x + self.pe[:, : x.size(1)]


class TransformerAccentModel(nn.Module):
    """Transformer Encoder accent predictor.

    Same embedding frontend and classifier as AccentModel,
    but replaces BiLSTM + custom Self-Attention with
    a standard Transformer Encoder.
    """

    def __init__(
        self,
        embed_dim: int = 64,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.3,
        attention_heads: int = 4,
        dim_feedforward: int = 512,
    ) -> None:
        """Initialize the Transformer accent predictor."""
        super().__init__()

        self.frontend = _EmbeddingFrontend(embed_dim, hidden_dim)
        self.pos_enc = _SinusoidalPE(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=attention_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
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
    ) -> torch.Tensor:
        """Forward pass.

        Returns:
            logits: [batch, seq_len, num_classes].

        """
        h = self.frontend(x, reading_ids)
        h = self.pos_enc(h)

        if lengths is not None:
            max_len = h.size(1)
            arange = torch.arange(max_len, device=h.device)
            pad_mask = arange.unsqueeze(0) >= lengths.unsqueeze(1).to(h.device)
        else:
            pad_mask = None

        h = self.transformer(h, src_key_padding_mask=pad_mask)
        h = self.dropout(h)
        return self.classifier(h)


# ── CRF layer ────────────────────────────────────────────────────────


class _CRF(nn.Module):
    """Linear-chain Conditional Random Field.

    Learns transition scores between labels to enforce
    sequence-level consistency (e.g. same accent_type within
    an accent phrase).
    """

    def __init__(self, num_tags: int) -> None:
        """Initialize CRF with transition parameters."""
        super().__init__()
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)

    def _compute_log_normalizer(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward algorithm to compute log partition function.

        Returns:
            Log normalizer [batch].

        """
        # emissions: [batch, seq_len, num_tags], mask: [batch, seq_len] bool
        seq_len = emissions.size(1)
        score = self.start_transitions + emissions[:, 0]  # [batch, num_tags]

        for i in range(1, seq_len):
            # score: [batch, num_tags, 1] + transitions: [num_tags, num_tags]
            # + emissions: [batch, 1, num_tags]
            broadcast_score = score.unsqueeze(2)
            broadcast_emissions = emissions[:, i].unsqueeze(1)
            next_score = broadcast_score + self.transitions + broadcast_emissions
            next_score = torch.logsumexp(next_score, dim=1)  # [batch, num_tags]
            # Only update where mask is True
            score = torch.where(mask[:, i].unsqueeze(1), next_score, score)

        score = score + self.end_transitions
        return torch.logsumexp(score, dim=1)  # [batch]

    def _compute_score(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute score of a given tag sequence.

        Returns:
            Scores [batch].

        """
        batch_size, seq_len = tags.shape
        score = (
            self.start_transitions[tags[:, 0]]
            + emissions[torch.arange(batch_size), 0, tags[:, 0]]
        )

        for i in range(1, seq_len):
            trans = self.transitions[tags[:, i - 1], tags[:, i]]
            batch_idx = torch.arange(batch_size, device=tags.device)
            emit = emissions[batch_idx, i, tags[:, i]]
            step_score = trans + emit
            score = score + step_score * mask[:, i].float()

        # End transition
        last_idx = mask.long().sum(dim=1) - 1
        last_tags = tags[torch.arange(batch_size, device=tags.device), last_idx]
        score = score + self.end_transitions[last_tags]
        return score

    def forward(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute negative log-likelihood.

        Returns:
            Scalar loss (mean over batch).

        """
        log_normalizer = self._compute_log_normalizer(emissions, mask)
        sequence_score = self._compute_score(emissions, tags, mask)
        return (log_normalizer - sequence_score).mean()

    @torch.no_grad()
    def decode(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor,
    ) -> list[list[int]]:
        """Viterbi decoding.

        Returns:
            List of tag sequences.

        """
        batch_size, seq_len = emissions.shape[:2]
        score = self.start_transitions + emissions[:, 0]
        history: list[torch.Tensor] = []

        for i in range(1, seq_len):
            broadcast_score = score.unsqueeze(2)
            broadcast_emissions = emissions[:, i].unsqueeze(1)
            next_score = broadcast_score + self.transitions + broadcast_emissions
            next_score, indices = next_score.max(dim=1)
            score = torch.where(mask[:, i].unsqueeze(1), next_score, score)
            history.append(indices)

        score = score + self.end_transitions

        results: list[list[int]] = []
        for b in range(batch_size):
            length = mask[b].long().sum().item()
            best_tags = [0] * length
            best_tags[-1] = score[b].argmax().item()
            for i in range(length - 2, -1, -1):
                best_tags[i] = history[i][b, best_tags[i + 1]].item()
            results.append(best_tags)
        return results


# ── ONNX export ──────────────────────────────────────────────────────


class _OnnxWrapper(nn.Module):
    """Batch dim removal for ONNX export.

    Input: [seq, 11] -> Output: [seq, 21].
    """

    def __init__(self, model: AccentModel | TransformerAccentModel) -> None:
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
    model: AccentModel | TransformerAccentModel,
    path: Path,
    device: torch.device,
) -> None:
    """モデルをONNX形式でエクスポートする."""
    wrapper = _OnnxWrapper(model).to(device)
    wrapper.eval()
    path.parent.mkdir(parents=True, exist_ok=True)

    dummy = torch.zeros(10, FEATURE_DIM, device=device)
    dummy[:, 0] = 1  # pos
    dummy[:, 1] = 2  # pd1
    dummy[:, 5] = 0.3  # mora

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

    # Verify with onnxruntime
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


def _train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    label_smoothing: float = 0.1,
    crf: _CRF | None = None,
    focal_gamma: float = 0.0,
    rdrop_alpha: float = 0.0,
) -> tuple[float, float]:
    """1エポック分の学習を実行する.

    Returns:
        (平均損失, 精度) のタプル.

    """
    model.train()
    if crf is not None:
        crf.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for batch in dataloader:
        features = batch["features"].to(device)
        labels = batch["labels"].to(device)
        lengths = batch["lengths"]
        r_ids = batch["reading_ids"].to(device)
        weights = batch["weights"].to(device)

        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(features, lengths, r_ids)
            mask = _build_mask(labels, lengths, device)

            if crf is not None:
                loss = crf(logits, labels, mask)
            else:
                logits_flat = logits[mask]
                labels_flat = labels[mask]
                weights_flat = weights[mask]
                ce_loss = F.cross_entropy(
                    logits_flat,
                    labels_flat,
                    label_smoothing=label_smoothing,
                    reduction="none",
                )
                if focal_gamma > 0:
                    probs = F.softmax(logits_flat, dim=-1)
                    pt = probs.gather(1, labels_flat.unsqueeze(1)).squeeze(1)
                    focal_weight = (1 - pt) ** focal_gamma
                    ce_loss = ce_loss * focal_weight
                loss = (ce_loss * weights_flat).mean()

                # R-Drop: KL divergence between two dropout passes
                if rdrop_alpha > 0:
                    logits2 = model(features, lengths, r_ids)
                    logits2_flat = logits2[mask]
                    p = F.log_softmax(logits_flat, dim=-1)
                    q = F.log_softmax(logits2_flat, dim=-1)
                    kl_pq = F.kl_div(p, q.exp(), reduction="batchmean")
                    kl_qp = F.kl_div(q, p.exp(), reduction="batchmean")
                    loss = loss + rdrop_alpha * (kl_pq + kl_qp) / 2

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        params = list(model.parameters())
        if crf is not None:
            params = params + list(crf.parameters())
        nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(optimizer)
        scaler.update()

        # Accuracy (always argmax for speed during training)
        with torch.no_grad():
            logits_flat = logits[mask]
            labels_flat = labels[mask]
            total_loss += loss.item() * labels_flat.size(0)
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
    crf: _CRF | None = None,
) -> tuple[float, float]:
    """検証データでモデルを評価する.

    Returns:
        (平均損失, 精度) のタプル.

    """
    model.eval()
    if crf is not None:
        crf.eval()
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

        if crf is not None:
            loss = crf(logits, labels, mask)
            # Viterbi decoding
            decoded = crf.decode(logits, mask)
            for seq_preds, seq_labels, length in zip(
                decoded,
                labels.cpu().tolist(),
                lengths.tolist(),
                strict=True,
            ):
                for p, g in zip(
                    seq_preds[:length],
                    seq_labels[:length],
                    strict=True,
                ):
                    total_correct += int(p == g)
                    total_count += 1
            total_loss += loss.item() * sum(lengths.tolist())
        else:
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


# ── Main training ────────────────────────────────────────────────────


def _train(  # noqa: PLR0913
    jvs_utterances: list[dict] | None,
    jsut_utterances: list[dict],
    output_path: Path,
    *,
    corpus_utterances: list[dict] | None = None,
    embed_dim: int = 64,
    hidden_dim: int = 256,
    num_layers: int = 3,
    dropout: float = 0.4,
    attention_heads: int = 4,
    batch_size: int = 64,
    weight_decay: float = 0.05,
    label_smoothing: float = 0.1,
    morpheme_dropout: float = 0.1,
    stage1_lr: float = 2e-3,
    stage1_epochs: int = 20,
    stage15_lr: float = 1e-3,
    stage15_epochs: int = 15,
    stage2_lr: float = 5e-4,
    stage2_epochs: int = 80,
    stage2_patience: int = 20,
    val_split: float = 0.1,
    use_transformer: bool = False,
    use_crf: bool = False,
    focal_gamma: float = 0.0,
    rdrop_alpha: float = 0.0,
    seed: int = 42,
) -> float:
    """3段階転移学習で訓練しONNX保存する.

    Returns:
        検証セットでの最良精度.

    """
    # Fix all random seeds for reproducibility
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name()}")

    # Build model
    print("\n[1/6] Building model...")
    if use_transformer:
        model: AccentModel | TransformerAccentModel = TransformerAccentModel(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=NUM_CLASSES,
            dropout=dropout,
            attention_heads=attention_heads,
        ).to(device)
        arch_name = "Transformer"
    else:
        model = AccentModel(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=NUM_CLASSES,
            dropout=dropout,
            attention_heads=attention_heads,
        ).to(device)
        arch_name = "BiLSTM+SelfAttn"

    crf: _CRF | None = None
    if use_crf:
        crf = _CRF(NUM_CLASSES).to(device)
        arch_name += "+CRF"

    param_count = sum(p.numel() for p in model.parameters())
    if crf is not None:
        crf_params = sum(p.numel() for p in crf.parameters())
        param_count += crf_params
    print(f"  Arch: {arch_name}")
    print(f"  Parameters: {param_count:,}")
    print(
        f"  embed={embed_dim}, hidden={hidden_dim}, "
        f"layers={num_layers}, dropout={dropout}"
    )

    # ── Stage 1: Pre-training on JVS ──
    if jvs_utterances:
        print(f"\n{'=' * 65}")
        print("[2/6] Stage 1: Pre-training on JVS")
        print(f"{'=' * 65}")

        jvs_ds = _AccentDataset(
            jvs_utterances,
            augment=True,
            morpheme_dropout=morpheme_dropout,
        )
        jvs_morph = sum(len(s["labels"]) for s in jvs_ds.samples)
        print(f"  {len(jvs_ds)} samples, {jvs_morph} morphemes")

        jvs_loader = DataLoader(
            jvs_ds,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=_collate_fn,
            num_workers=2,
            pin_memory=True,
        )

        opt_params = list(model.parameters())
        if crf is not None:
            opt_params = opt_params + list(crf.parameters())
        optimizer = torch.optim.AdamW(
            opt_params,
            lr=stage1_lr,
            weight_decay=weight_decay,
        )
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

        print(f"\n  {stage1_epochs} epochs, lr={stage1_lr}")
        hdr = f"  {'Epoch':>5} {'Loss':>10} {'Acc':>10} {'LR':>10}"
        print(hdr)
        print("  " + "-" * 40)

        for epoch in range(1, stage1_epochs + 1):
            loss, acc = _train_epoch(
                model,
                jvs_loader,
                optimizer,
                device,
                scaler,
                label_smoothing=label_smoothing,
                crf=crf,
                focal_gamma=focal_gamma,
                rdrop_alpha=rdrop_alpha,
            )
            lr = optimizer.param_groups[0]["lr"]
            print(f"  {epoch:5d} {loss:10.4f} {acc:10.4f} {lr:10.6f}")

        print("\n  Stage 1 complete.")
    else:
        print("\n[2/6] Stage 1: Skipped")

    # Stage 1.5 disabled — corpus is mixed into Stage 2 instead
    print("\n[3/6] Stage 1.5: Skipped (corpus mixed into Stage 2)")

    # ── Stage 2: Fine-tuning on JSUT (+ corpus) ──
    print(f"\n{'=' * 65}")
    print("[4/6] Stage 2: Fine-tuning on JSUT")
    print(f"{'=' * 65}")

    random.seed(42)
    indices = list(range(len(jsut_utterances)))
    random.shuffle(indices)
    val_size = int(len(indices) * val_split)
    val_idx = set(indices[:val_size])
    train_utts = [u for i, u in enumerate(jsut_utterances) if i not in val_idx]
    val_utts = [u for i, u in enumerate(jsut_utterances) if i in val_idx]

    # corpus データを訓練セットに混合
    if corpus_utterances:
        print(f"  JSUT train: {len(train_utts)}, Val: {len(val_utts)}")
        print(f"  Mixing corpus: {len(corpus_utterances)} utterances")
        train_utts = train_utts + corpus_utterances

    print(f"  Train: {len(train_utts)}, Val: {len(val_utts)}")

    train_ds = _AccentDataset(
        train_utts,
        augment=True,
        morpheme_dropout=morpheme_dropout,
    )
    val_ds = _AccentDataset(val_utts, augment=False)

    train_m = sum(len(s["labels"]) for s in train_ds.samples)
    val_m = sum(len(s["labels"]) for s in val_ds.samples)
    print(f"  Train: {len(train_ds)} ({train_m} morphemes)")
    print(f"  Val: {len(val_ds)} ({val_m} morphemes)")

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

    opt_params_s2 = list(model.parameters())
    if crf is not None:
        opt_params_s2 = opt_params_s2 + list(crf.parameters())
    optimizer = torch.optim.AdamW(
        opt_params_s2,
        lr=stage2_lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    # Pre-fine-tuning eval
    _, pre_acc = _evaluate(model, val_loader, device, crf=crf)
    print(f"\n  Pre-fine-tuning val acc: {pre_acc * 100:.2f}%")

    best_val_acc = 0.0
    best_state: dict | None = None
    no_improve = 0
    # Top-K checkpoint averaging: keep best 3 checkpoints
    top_k = 3
    top_states: list[tuple[float, dict]] = []

    print(
        f"\n  [5/6] Up to {stage2_epochs} epochs "
        f"(lr={stage2_lr}, patience={stage2_patience})"
    )
    hdr = (
        f"  {'Epoch':>5} {'TrLoss':>10} {'TrAcc':>10} "
        f"{'VaLoss':>10} {'VaAcc':>9} {'LR':>10}"
    )
    print(hdr)
    print("  " + "-" * 65)

    for epoch in range(1, stage2_epochs + 1):
        tr_loss, tr_acc = _train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            scaler,
            label_smoothing=label_smoothing,
            crf=crf,
            focal_gamma=focal_gamma,
            rdrop_alpha=rdrop_alpha,
        )
        va_loss, va_acc = _evaluate(model, val_loader, device, crf=crf)
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step(va_acc)

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

        print(
            f"  {epoch:5d} {tr_loss:10.4f} {tr_acc:10.4f} "
            f"{va_loss:10.4f} {va_acc:9.4f} "
            f"{lr:10.6f}{marker}"
        )

        if no_improve >= stage2_patience:
            print(
                f"\n  Early stopping at epoch {epoch} "
                f"(no improvement for "
                f"{stage2_patience} epochs)"
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    print(f"\n  Best val accuracy: {best_val_acc * 100:.2f}%")

    # ── Top-K Checkpoint Averaging ──
    if len(top_states) >= 2:
        accs = [a for a, _ in top_states]
        print(
            f"\n  [5.5/6] Top-{len(top_states)} checkpoint averaging "
            f"(accs: {', '.join(f'{a * 100:.2f}%' for a in accs)})"
        )
        avg_state: dict[str, torch.Tensor] = {}
        for key in top_states[0][1]:
            stacked = torch.stack([s[key].float() for _, s in top_states])
            avg_state[key] = stacked.mean(dim=0)
        model.load_state_dict(avg_state)
        model.to(device)
        avg_loss, avg_acc = _evaluate(model, val_loader, device)
        print(f"  Averaged val acc: {avg_acc * 100:.2f}%")

        if avg_acc > best_val_acc:
            print("  Averaging improved! Using averaged model.")
            best_val_acc = avg_acc
        else:
            print(
                f"  Averaging did not improve "
                f"({avg_acc * 100:.2f}% vs {best_val_acc * 100:.2f}%). "
                f"Keeping best single checkpoint."
            )
            if best_state is not None:
                model.load_state_dict(best_state)
                model.to(device)
    else:
        print("\n  [5.5/6] Top-K averaging: not enough checkpoints, skipping.")

    # Export ONNX
    print("\n[6/6] Exporting ONNX...")
    _export_onnx(model, output_path, device)
    file_size = output_path.stat().st_size
    print(f"  Output: {output_path}")
    print(f"  Size: {file_size / 1_048_576:.1f} MB")

    return best_val_acc


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

    parser = argparse.ArgumentParser(description="Train kotonoha accent model v7")
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
    parser.add_argument("--stage1-epochs", type=int, default=20)
    parser.add_argument("--stage1-lr", type=float, default=2e-3)
    parser.add_argument("--stage2-epochs", type=int, default=80)
    parser.add_argument("--stage2-lr", type=float, default=5e-4)
    parser.add_argument("--stage2-patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument(
        "--corpus-data",
        type=Path,
        default=Path(os.environ.get("CORPUS_DATA", "")),
        help="LLM corpus data (Stage 1.5)",
    )
    parser.add_argument("--stage15-lr", type=float, default=5e-4)
    parser.add_argument("--stage15-epochs", type=int, default=5)
    parser.add_argument(
        "--no-pretrain",
        action="store_true",
        help="Skip Stage 1",
    )
    parser.add_argument(
        "--transformer",
        action="store_true",
        help="Use Transformer Encoder instead of BiLSTM",
    )
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument(
        "--crf",
        action="store_true",
        help="Add CRF output layer for sequence-level consistency",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=0.0,
        help="Focal loss gamma (0=disabled)",
    )
    parser.add_argument(
        "--rdrop-alpha",
        type=float,
        default=0.0,
        help="R-Drop KL divergence weight (0=disabled)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    if args.transformer:
        arch_str = "Transformer Encoder + Emb + 2-MLP"
        default_layers = 4
        default_dropout = 0.3
    else:
        arch_str = "BiLSTM(3) + SelfAttn(4h) + Emb + 2-MLP"
        default_layers = 3
        default_dropout = 0.4

    num_layers = args.num_layers if args.num_layers is not None else default_layers
    dropout = args.dropout if args.dropout is not None else default_dropout

    print("=" * 65)
    print("kotonoha accent model trainer v7")
    print(f"  Arch: {arch_str}")
    print("  Strategy: JVS pretrain -> corpus -> JSUT finetune")
    print("=" * 65)

    # Load accent dict(s) — colon-separated paths
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
    elif args.no_pretrain:
        print("\nPre-training: skipped (--no-pretrain)")
    else:
        print(f"\nWarning: not found: {pt}")

    # Load corpus (Stage 1.5)
    corpus_utterances = None
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
    else:
        print("\nCorpus data: not found (Stage 1.5 will be skipped)")

    # Load JSUT (fine-tuning)
    print(f"\nLoading JSUT: {args.finetune_data}")
    with open(args.finetune_data, encoding="utf-8") as f:
        jsut_data = json.load(f)
    jsut_utterances = jsut_data["utterances"]
    print(f"  {len(jsut_utterances)} utterances")
    n = _enrich_utterances(jsut_utterances, accent_dict)
    print(f"  Enriched {n} morphemes")

    # Train
    best = _train(
        jvs_utterances,
        jsut_utterances,
        args.output,
        corpus_utterances=corpus_utterances,
        hidden_dim=args.hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        batch_size=args.batch_size,
        stage1_lr=args.stage1_lr,
        stage1_epochs=args.stage1_epochs,
        stage15_lr=args.stage15_lr,
        stage15_epochs=args.stage15_epochs,
        stage2_lr=args.stage2_lr,
        stage2_epochs=args.stage2_epochs,
        stage2_patience=args.stage2_patience,
        use_transformer=args.transformer,
        use_crf=args.crf,
        focal_gamma=args.focal_gamma,
        rdrop_alpha=args.rdrop_alpha,
        seed=args.seed,
    )

    print(f"\n{'=' * 65}")
    print(f"Done! Best val accuracy: {best * 100:.2f}%")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
