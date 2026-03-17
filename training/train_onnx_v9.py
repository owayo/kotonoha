"""kotonoha ONNX accent model trainer (v9).

v8ベースに以下の改善を加えた学習パイプライン:
- EMA (Exponential Moving Average) — Top-K checkpoint avgに加え、EMAで安定的に追跡
- Linear Warm-up + Cosine Annealing — 初期の学習不安定性を解消
- Focal Loss — クラス不均衡への対処（高頻度クラスの影響を抑制）
- Multi-seed 重み平均 — best-of-Nではなくtop seeds間で重み平均
- 2層Self-Attention — コンテキスト表現の強化
- Feature noise augmentation — 連続特徴量にガウスノイズ追加
- Gradient accumulation — 実効バッチサイズ拡大オプション
- SAM / R-Drop — v8から継承

Stage 1: JVS データで事前学習
Stage 2: JSUT + コーパスデータで微調整

特徴量は nn.rs の extract_features_v2 と完全に一致させること:
  [pos_id, pd1_id, pd2_id, ct_id, cf_id,
   mora_count, reading_hash, first_char_hash,
   last_char_hash, position, dict_accent_type]

Usage:
  cp .env.example .env  # edit paths
  cd training
  uv run python train_onnx_v9.py                          # default: 8 seeds
  uv run python train_onnx_v9.py --seeds 0,1,2            # seed数を指定
  uv run python train_onnx_v9.py --no-focal               # Focal Loss無効化
  uv run python train_onnx_v9.py --ensemble-topn 3        # top-3 seeds重み平均
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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
    """Must match nn.rs reading_hash."""
    h: int = 5381
    for c in reading:
        h = (h * 33 + ord(c)) & 0xFFFFFFFF
    return (h % 10000) / 10000.0


def _char_hash(ch: str) -> float:
    """Must match nn.rs char_hash."""
    h: int = 5381
    h = (h * 33 + ord(ch)) & 0xFFFFFFFF
    return (h % 10000) / 10000.0


_SMALL_KANA = set("ァィゥェォャュョヮ")
_FUNCTION_POS = {"助詞", "助動詞"}


def _count_mora(reading: str) -> int:
    """カタカナ読みからモーラ数を算出する."""
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
    """活用型をグループIDに変換する."""
    if conj_type == "*":
        return 0
    for prefix, idx in CONJ_TYPE_GROUPS.items():
        if prefix != "*" and conj_type.startswith(prefix):
            return idx
    return 0


def _get_conj_form_group(conj_form: str) -> int:
    """活用形をグループIDに変換する."""
    if conj_form == "*":
        return 0
    for prefix, idx in CONJ_FORM_GROUPS.items():
        if prefix != "*" and conj_form.startswith(prefix):
            return idx
    return 0


def _parse_dict_accent(val: str) -> float:
    """dict_accent_typeを数値に変換する."""
    if val in ("*", ""):
        return 0.0
    m = re.match(r'^"?(\d+)$', val)
    if m:
        return (int(m.group(1)) + 1) / 8.0
    return 0.0


def _encode_reading(reading: str) -> list[int]:
    """読みを仮名文字IDのリストに変換する."""
    ids = []
    for ch in reading[:MAX_READING_LEN]:
        ids.append(KANA_VOCAB.get(ch, 1))
    while len(ids) < MAX_READING_LEN:
        ids.append(0)
    return ids


def _extract_morpheme_features(
    morpheme: dict,
    position: float,
) -> list[float]:
    """形態素辞書から特徴量ベクトルを抽出する."""
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
    """単一のアクセント辞書CSVを読み込む."""
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
    """複数のアクセント辞書をマージして読み込む."""
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
    """dict_accent_typeが欠損している形態素を辞書で補完する."""
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
    """Accent prediction dataset with morpheme dropout and feature noise."""

    def __init__(
        self,
        utterances: list[dict],
        *,
        augment: bool = False,
        morpheme_dropout: float = 0.1,
        feature_noise_std: float = 0.02,
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

            n = len(morphemes)
            features = []
            labels = []
            r_ids = []

            for i, m in enumerate(morphemes):
                position = i / max(n - 1, 1)
                feat = _extract_morpheme_features(m, position)
                features.append(feat)
                accent = m.get("accent_type", 0)
                labels.append(min(accent, NUM_CLASSES - 1))
                reading = m.get("reading", "")
                r_ids.append(_encode_reading(reading))

            if features:
                self.samples.append(
                    {
                        "features": features,
                        "labels": labels,
                        "reading_ids": r_ids,
                    }
                )

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        """Return sample with optional augmentation."""
        sample = self.samples[idx]
        if not self.augment:
            return sample

        features = []
        r_ids = []
        for feat, rids in zip(
            sample["features"],
            sample["reading_ids"],
            strict=True,
        ):
            if random.random() < self.morpheme_dropout:
                # Morpheme dropout: zero out
                features.append([0.0] * FEATURE_DIM)
                r_ids.append([0] * MAX_READING_LEN)
            else:
                # v9: Add Gaussian noise to continuous features (indices 5-10)
                noisy_feat = list(feat)
                if self.feature_noise_std > 0:
                    for j in range(5, FEATURE_DIM):
                        noisy_feat[j] += random.gauss(0, self.feature_noise_std)
                features.append(noisy_feat)
                r_ids.append(rids)
        return {
            "features": features,
            "labels": sample["labels"],
            "reading_ids": r_ids,
        }


def _collate_fn(batch: list[dict]) -> dict:
    """可変長系列をパディングしてバッチ化する."""
    lengths = [len(item["features"]) for item in batch]
    max_len = max(lengths)

    padded_features = torch.zeros(len(batch), max_len, FEATURE_DIM)
    padded_labels = torch.zeros(len(batch), max_len, dtype=torch.long)
    padded_reading_ids = torch.zeros(
        len(batch), max_len, MAX_READING_LEN, dtype=torch.long
    )

    for i, item in enumerate(batch):
        seq_len = lengths[i]
        padded_features[i, :seq_len] = torch.tensor(
            item["features"], dtype=torch.float32
        )
        padded_labels[i, :seq_len] = torch.tensor(item["labels"], dtype=torch.long)
        padded_reading_ids[i, :seq_len] = torch.tensor(
            item["reading_ids"], dtype=torch.long
        )

    return {
        "features": padded_features,
        "labels": padded_labels,
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "reading_ids": padded_reading_ids,
    }


# ── EMA (Exponential Moving Average) ────────────────────────────────


class _EMA:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        """Initialize EMA with a copy of model parameters."""
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module) -> None:
        """Update EMA parameters."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay
                )

    def apply(self, model: nn.Module) -> dict[str, torch.Tensor]:
        """Apply EMA parameters to model, returning original state for restore."""
        backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
        return backup

    def restore(self, model: nn.Module, backup: dict[str, torch.Tensor]) -> None:
        """Restore original parameters from backup."""
        for name, param in model.named_parameters():
            if name in backup:
                param.data.copy_(backup[name])

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return EMA shadow parameters as a state dict."""
        return {k: v.clone() for k, v in self.shadow.items()}


# ── Focal Loss ───────────────────────────────────────────────────────


def _focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    label_smoothing: float = 0.1,
) -> torch.Tensor:
    """Focal loss with optional label smoothing.

    Reduces the contribution of easy (well-classified) examples,
    focusing training on hard examples.
    """
    ce = F.cross_entropy(
        logits, targets, label_smoothing=label_smoothing, reduction="none"
    )
    pt = torch.exp(-ce)
    # Focal weight: (1 - pt)^gamma
    focal_weight = (1.0 - pt) ** gamma
    return (focal_weight * ce).mean()


# ── Model (v9 architecture: 2-layer self-attention) ──────────────────


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
        """Self-attention with residual connection."""
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


class _FeedForwardLayer(nn.Module):
    """Feed-forward layer with residual + LayerNorm (for stacked attention)."""

    def __init__(self, dim: int, ff_dim: int = 512, dropout: float = 0.1) -> None:
        """Initialize feed-forward layer."""
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Feed-forward with residual."""
        return self.norm(x + self.ff(x))


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
        """Embed features and project to hidden_dim."""
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
    """BiLSTM + 2-layer Self-Attention accent predictor (v9 arch).

    v8からの変更点:
    - Self-Attentionを2層に増加（FFN付き）
    - LSTM残差接続はv8と同様
    """

    def __init__(
        self,
        embed_dim: int = 64,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.4,
        attention_heads: int = 4,
        num_attn_layers: int = 2,
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

        # v9: stacked self-attention layers (FFN only when >1 layer)
        self.attn_layers = nn.ModuleList()
        self.ff_layers = nn.ModuleList()
        for _ in range(num_attn_layers):
            self.attn_layers.append(
                _SelfAttentionLayer(
                    hidden_dim, num_heads=attention_heads, dropout=dropout
                )
            )
        if num_attn_layers > 1:
            for _ in range(num_attn_layers):
                self.ff_layers.append(
                    _FeedForwardLayer(
                        hidden_dim, ff_dim=hidden_dim * 2, dropout=dropout
                    )
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
        """Forward pass."""
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

        # v9: stacked attention (+ FFN when >1 layer)
        if self.ff_layers:
            for attn, ff in zip(self.attn_layers, self.ff_layers, strict=True):
                h = attn(h, mask=attn_mask)
                h = ff(h)
        else:
            for attn in self.attn_layers:
                h = attn(h, mask=attn_mask)

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
        """Map [seq_len, 11] to [seq_len, 21]."""
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
    """パディング位置を除外するマスクを作成する."""
    max_len = labels.size(1)
    return torch.arange(max_len, device=device).unsqueeze(0) < lengths.unsqueeze(1).to(
        device
    )


def _compute_loss(
    logits_flat: torch.Tensor,
    labels_flat: torch.Tensor,
    use_focal: bool,
    focal_gamma: float,
    label_smoothing: float,
) -> torch.Tensor:
    """損失を計算する（Focal Loss or Cross Entropy）."""
    if use_focal:
        return _focal_loss(
            logits_flat, labels_flat, gamma=focal_gamma, label_smoothing=label_smoothing
        )
    ce = F.cross_entropy(
        logits_flat, labels_flat, label_smoothing=label_smoothing, reduction="none"
    )
    return ce.mean()


def _rdrop_kl(logits1: torch.Tensor, logits2: torch.Tensor) -> torch.Tensor:
    """R-Drop KL divergence between two forward passes."""
    lp = F.log_softmax(logits1, dim=-1)
    lq = F.log_softmax(logits2, dim=-1)
    kl1 = F.kl_div(lp, lq.exp(), reduction="batchmean")
    kl2 = F.kl_div(lq, lp.exp(), reduction="batchmean")
    return (kl1 + kl2) / 2


def _train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    ema: _EMA | None = None,
    label_smoothing: float = 0.1,
    rdrop_alpha: float = 0.0,
    sam_rho: float = 0.0,
    use_focal: bool = False,
    focal_gamma: float = 2.0,
    grad_accum_steps: int = 1,
) -> tuple[float, float]:
    """1エポック分の学習を実行する."""
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

        mask = _build_mask(labels, lengths, device)
        labels_flat = labels[mask]

        if sam_rho > 0:
            # ── SAM: gradient at current → perturb → gradient at perturbed ──
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(features, lengths, r_ids)
                logits_flat = logits[mask]
                loss = _compute_loss(
                    logits_flat, labels_flat, use_focal, focal_gamma, label_smoothing
                )
                if rdrop_alpha > 0:
                    logits2 = model(features, lengths, r_ids)
                    logits2_flat = logits2[mask]
                    loss = loss + rdrop_alpha * _rdrop_kl(logits_flat, logits2_flat)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Perturb weights
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

            # Second forward-backward at perturbed point
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits_p = model(features, lengths, r_ids)
                logits_p_flat = logits_p[mask]
                loss_p = _compute_loss(
                    logits_p_flat, labels_flat, use_focal, focal_gamma, label_smoothing
                )
                if rdrop_alpha > 0:
                    logits_p2 = model(features, lengths, r_ids)
                    logits_p2_flat = logits_p2[mask]
                    kl = _rdrop_kl(logits_p_flat, logits_p2_flat)
                    loss_p = loss_p + rdrop_alpha * kl
            loss_p.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            with torch.no_grad():
                for param, e_w in perturbations.items():
                    param.sub_(e_w)
            optimizer.step()
            if ema is not None:
                ema.update(model)
        else:
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(features, lengths, r_ids)
                logits_flat = logits[mask]
                loss = _compute_loss(
                    logits_flat, labels_flat, use_focal, focal_gamma, label_smoothing
                )

                if rdrop_alpha > 0:
                    logits2 = model(features, lengths, r_ids)
                    logits2_flat = logits2[mask]
                    loss = loss + rdrop_alpha * _rdrop_kl(logits_flat, logits2_flat)

                # Gradient accumulation: scale loss
                loss = loss / grad_accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(dataloader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if ema is not None:
                    ema.update(model)

        with torch.no_grad():
            total_loss += loss.item() * grad_accum_steps * labels_flat.size(0)
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
    """検証データでモデルを評価する."""
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


# ── Warm-up + Cosine schedule ────────────────────────────────────────


class _WarmupCosineScheduler(torch.optim.lr_scheduler.LRScheduler):
    """Linear warm-up followed by cosine annealing."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        eta_min: float = 1e-6,
    ) -> None:
        """Initialize warmup + cosine scheduler."""
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.eta_min = eta_min
        super().__init__(optimizer)

    def get_lr(self) -> list[float]:
        """Compute learning rate for current step."""
        step = self.last_epoch
        if step < self.warmup_steps:
            # Linear warmup
            scale = (step + 1) / max(self.warmup_steps, 1)
            return [base_lr * scale for base_lr in self.base_lrs]
        # Cosine annealing
        progress = (step - self.warmup_steps) / max(
            self.total_steps - self.warmup_steps, 1
        )
        cosine_scale = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [
            self.eta_min + (base_lr - self.eta_min) * cosine_scale
            for base_lr in self.base_lrs
        ]


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
    num_attn_layers: int = 2,
    batch_size: int = 64,
    weight_decay: float = 0.05,
    label_smoothing: float = 0.1,
    morpheme_dropout: float = 0.1,
    feature_noise_std: float = 0.02,
    stage1_lr: float = 2e-3,
    stage1_epochs: int = 20,
    stage2_lr: float = 8e-4,
    stage2_epochs: int = 80,
    stage2_patience: int = 20,
    rdrop_alpha: float = 0.5,
    top_k: int = 3,
    use_cosine: bool = True,
    warmup_epochs: int = 5,
    sam_rho: float = 0.0,
    ema_decay: float = 0.999,
    use_focal: bool = True,
    focal_gamma: float = 2.0,
    grad_accum_steps: int = 1,
) -> tuple[float, dict[str, torch.Tensor]]:
    """独立した1 seedで全Stage訓練し、最良のstate_dictを返す."""
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
        num_attn_layers=num_attn_layers,
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

    # ── Stage 2: Fine-tuning ──
    sched_name = "warmup+cosine" if use_cosine else "plateau"
    extras = []
    if sam_rho > 0:
        extras.append(f"SAM={sam_rho}")
    if use_focal:
        extras.append(f"focal(γ={focal_gamma})")
    extras.append(f"EMA={ema_decay}")
    extra_str = ", ".join(extras)
    print(f"  [Stage 2] Fine-tuning ({stage2_epochs} ep, {sched_name}, {extra_str})")

    train_ds = _AccentDataset(
        train_utts,
        augment=True,
        morpheme_dropout=morpheme_dropout,
        feature_noise_std=feature_noise_std,
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

    # v9: warm-up + cosine as default
    total_steps = stage2_epochs * len(train_loader)
    warmup_steps = warmup_epochs * len(train_loader)

    if use_cosine:
        scheduler = _WarmupCosineScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            eta_min=1e-6,
        )
        step_per_batch = True
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
        )
        step_per_batch = False

    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    # v9: EMA tracking
    ema = _EMA(model, decay=ema_decay)

    best_val_acc = 0.0
    best_state: dict | None = None
    best_ema_acc = 0.0
    no_improve = 0
    top_states: list[tuple[float, dict]] = []

    for epoch in range(1, stage2_epochs + 1):
        tr_loss, tr_acc = _train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            scaler,
            ema=ema,
            label_smoothing=label_smoothing,
            rdrop_alpha=rdrop_alpha,
            sam_rho=sam_rho,
            use_focal=use_focal,
            focal_gamma=focal_gamma,
            grad_accum_steps=grad_accum_steps,
        )

        # Step scheduler
        if step_per_batch:
            # Already stepped per-batch via scheduler.step() in training loop?
            # No — we step per epoch for simplicity, multiply by loader length
            for _ in range(len(train_loader)):
                scheduler.step()

        # Evaluate with model weights
        va_loss, va_acc = _evaluate(model, val_loader, device)

        # Evaluate with EMA weights
        ema_backup = ema.apply(model)
        _, ema_acc = _evaluate(model, val_loader, device)
        ema.restore(model, ema_backup)

        lr = optimizer.param_groups[0]["lr"]
        if not step_per_batch:
            scheduler.step(va_acc)

        marker = ""
        effective_acc = max(va_acc, ema_acc)

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            marker = " *"
        elif ema_acc > best_val_acc:
            # EMA beats everything so far
            best_val_acc = ema_acc
            ema_backup_save = ema.apply(model)
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            ema.restore(model, ema_backup_save)
            no_improve = 0
            marker = " *ema"
        else:
            no_improve += 1

        if ema_acc > best_ema_acc:
            best_ema_acc = ema_acc
            ema_backup_save = ema.apply(model)
            ema.restore(model, ema_backup_save)

        # Track top-K checkpoints (use best of model/EMA)
        if effective_acc > 0:
            if ema_acc >= va_acc:
                ema_backup_save = ema.apply(model)
                state_copy = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                ema.restore(model, ema_backup_save)
            else:
                state_copy = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if len(top_states) < top_k:
                top_states.append((effective_acc, state_copy))
                top_states.sort(key=lambda x: x[0])
            elif effective_acc > top_states[0][0]:
                top_states[0] = (effective_acc, state_copy)
                top_states.sort(key=lambda x: x[0])

        if epoch % 5 == 0 or marker or epoch <= 3:
            ema_str = f" ema={ema_acc:.4f}" if ema_acc != va_acc else ""
            print(
                f"    ep {epoch:2d}: va={va_acc:.4f}{ema_str} "
                f"tr={tr_acc:.4f} lr={lr:.6f}{marker}"
            )

        if no_improve >= stage2_patience:
            print(f"    Early stop at ep {epoch}")
            break

    print(f"  Best single: {best_val_acc * 100:.2f}%")
    print(f"  Best EMA: {best_ema_acc * 100:.2f}%")

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

    parser = argparse.ArgumentParser(description="Train kotonoha accent model v9")
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
        default="0,1,2,3,4,5,6,7",
        help="Comma-separated seeds for best-of-N",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Top-K checkpoints to avg")
    parser.add_argument(
        "--scheduler",
        choices=["cosine", "plateau"],
        default="cosine",
        help="LR scheduler (warmup+cosine or plateau)",
    )
    parser.add_argument(
        "--sam-rho",
        type=float,
        default=0.05,
        help="SAM perturbation radius (0=disabled)",
    )
    # v9 new options
    parser.add_argument(
        "--num-attn-layers",
        type=int,
        default=2,
        help="Number of self-attention layers (v8=1, v9=2)",
    )
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.999,
        help="EMA decay rate (0=disabled)",
    )
    parser.add_argument(
        "--no-focal",
        action="store_true",
        help="Disable Focal Loss (use standard CE)",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
        help="Focal Loss gamma parameter",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=5,
        help="Number of warm-up epochs for cosine scheduler",
    )
    parser.add_argument(
        "--feature-noise",
        type=float,
        default=0.02,
        help="Gaussian noise std for continuous features (0=disabled)",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--ensemble-topn",
        type=int,
        default=0,
        help="Average top-N seeds' weights instead of best-of-1 (0=off)",
    )
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    use_focal = not args.no_focal

    print("=" * 65)
    print("kotonoha accent model trainer v9")
    n_lstm = args.num_layers
    n_attn = args.num_attn_layers
    print(f"  Arch: BiLSTM({n_lstm}) + {n_attn}x SelfAttn(4h) + Emb + 2-MLP")
    print(f"  Seeds: {seeds} (best-of-N)")
    print(f"  R-Drop: {args.rdrop_alpha}, Top-K: {args.top_k}, Sched: {args.scheduler}")
    print(f"  SAM: rho={args.sam_rho}" if args.sam_rho > 0 else "  SAM: disabled")
    print(f"  EMA: decay={args.ema_decay}")
    if use_focal:
        print(f"  Focal Loss: gamma={args.focal_gamma}")
    else:
        print("  Focal Loss: disabled")
    print(f"  Warm-up: {args.warmup_epochs} epochs")
    print(f"  Feature noise: std={args.feature_noise}")
    if args.grad_accum > 1:
        print(f"  Gradient accumulation: {args.grad_accum} steps")
    if args.ensemble_topn > 0:
        print(f"  Ensemble: top-{args.ensemble_topn} seeds weight averaging")
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
            warmup_epochs=args.warmup_epochs,
            sam_rho=args.sam_rho,
            num_attn_layers=args.num_attn_layers,
            ema_decay=args.ema_decay,
            use_focal=use_focal,
            focal_gamma=args.focal_gamma,
            feature_noise_std=args.feature_noise,
            grad_accum_steps=args.grad_accum,
        )
        all_results.append((acc, state))
        print(f"  => Seed {seed} result: {acc * 100:.2f}%")

    # ── Results ──
    print(f"\n{'=' * 65}")
    print(f"[Results] {len(all_results)} seeds")
    print(f"{'=' * 65}")
    for (acc, _), seed in zip(all_results, seeds, strict=True):
        print(f"  Seed {seed}: {acc * 100:.2f}%")

    # ── Best-of-N or Ensemble selection ──
    ensemble_n = args.ensemble_topn
    if ensemble_n > 1 and len(all_results) >= ensemble_n:
        # v9: Average top-N seeds' state dicts
        sorted_results = sorted(
            zip(all_results, seeds, strict=True),
            key=lambda x: x[0][0],
            reverse=True,
        )
        top_n = sorted_results[:ensemble_n]
        top_accs = [acc for (acc, _), _ in top_n]
        print(f"\n  Ensemble: averaging top-{ensemble_n} seeds")
        print(f"  Seeds: {[s for _, s in top_n]}")
        print(f"  Accs: {[f'{a * 100:.2f}%' for a in top_accs]}")

        # Average state dicts
        ref_state = top_n[0][0][1]
        avg_state: dict[str, torch.Tensor] = {}
        for key in ref_state:
            stacked = torch.stack([state[key].float() for (_, state), _ in top_n])
            avg_state[key] = stacked.mean(dim=0)

        model = AccentModel(
            embed_dim=64,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_classes=NUM_CLASSES,
            dropout=args.dropout,
            attention_heads=4,
            num_attn_layers=args.num_attn_layers,
        ).to(device)
        model.load_state_dict(avg_state)

        # Re-evaluate ensemble
        random.seed(42)
        indices = list(range(len(jsut_utterances)))
        random.shuffle(indices)
        val_size = int(len(indices) * 0.1)
        val_idx = set(indices[:val_size])
        val_utts = [u for i, u in enumerate(jsut_utterances) if i in val_idx]
        val_ds = _AccentDataset(val_utts, augment=False)
        val_loader = DataLoader(
            val_ds, batch_size=64, shuffle=False, collate_fn=_collate_fn
        )
        _, ensemble_acc = _evaluate(model, val_loader, device)
        print(f"  Ensemble accuracy: {ensemble_acc * 100:.2f}%")

        best_acc = max(max(top_accs), ensemble_acc)
        if ensemble_acc >= max(top_accs):
            best_state = avg_state
            final_acc = ensemble_acc
            print("  Using ensemble weights (better than any single seed)")
        else:
            best_acc_single, best_state = max(all_results, key=lambda x: x[0])
            final_acc = best_acc_single
            print("  Using best single seed (ensemble didn't help)")
    else:
        best_acc, best_state = max(all_results, key=lambda x: x[0])
        best_seed = seeds[all_results.index((best_acc, best_state))]
        print(f"\n  Best: seed {best_seed} = {best_acc * 100:.2f}%")
        final_acc = best_acc

    model = AccentModel(
        embed_dim=64,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=NUM_CLASSES,
        dropout=args.dropout,
        attention_heads=4,
        num_attn_layers=args.num_attn_layers,
    ).to(device)
    model.load_state_dict(best_state)
    model.to(device)

    # Export ONNX
    print("\nExporting ONNX...")
    _export_onnx(model, args.output, device)
    file_size = args.output.stat().st_size
    print(f"  Output: {args.output}")
    print(f"  Size: {file_size / 1_048_576:.1f} MB")

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    print(f"\n{'=' * 65}")
    print(f"Done! Final val accuracy: {final_acc * 100:.2f}%")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
