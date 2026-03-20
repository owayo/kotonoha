"""kotonoha ONNX accent model trainer (v10).

v8ベースに以下の改善を加えた学習パイプライン:
- Self-Distillation — multi-seedアンサンブルからの知識蒸留
- Greedy Checkpoint Soup — CosineAnnealingWarmRestartsで収集した
  チェックポイントを貪欲にマージ（Model Soups）

アーキテクチャはv8と同一（BiLSTM + 1-layer Self-Attention）。
v9で試みた容量増加（2層Attention, FFN等）はデータ量に対して過剰だったため、
v8アーキテクチャを維持する。

Training Flow:
  Phase 1: 複数seedで独立学習（v8と同等）→ teacher ensemble構築
  Phase 2: teacher ensembleからの知識蒸留で student model を学習
           CosineAnnealingWarmRestartsでチェックポイントを収集
  Phase 3: Greedy Checkpoint Soup でさらに精度向上

Stage 1: JVS データで事前学習
Stage 2: JSUT + コーパスデータで微調整（KD + WarmRestarts）

特徴量は nn.rs の extract_features_v2 と完全に一致させること:
  [pos_id, pd1_id, pd2_id, ct_id, cf_id,
   mora_count, reading_hash, first_char_hash,
   last_char_hash, position, dict_accent_type]

Usage:
  cp .env.example .env  # edit paths
  cd training
  uv run python train_onnx_v10.py                    # default: 8 seeds → KD
  uv run python train_onnx_v10.py --seeds 0,1,2      # 3 seeds
  uv run python train_onnx_v10.py --no-kd             # KDなし（v8相当+Soup）
  uv run python train_onnx_v10.py --kd-temp 5         # KD温度変更
  uv run python train_onnx_v10.py --teacher-topn 5    # top-5 teacherアンサンブル
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
        """Return number of samples.

        Returns:
            Sample count.

        """
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        """Return sample with optional augmentation.

        Returns:
            Dict with features, labels, reading_ids, and optional teacher_logits.

        """
        sample = self.samples[idx]
        result = {
            "labels": sample["labels"],
        }

        # Morpheme dropout augmentation
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
            result["features"] = features
            result["reading_ids"] = r_ids
        else:
            result["features"] = sample["features"]
            result["reading_ids"] = sample["reading_ids"]

        # Pass through teacher logits if available (pre-computed, no augmentation)
        if "teacher_logits" in sample:
            result["teacher_logits"] = sample["teacher_logits"]

        return result


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

    for i, item in enumerate(batch):
        seq_len = lengths[i]
        padded_features[i, :seq_len] = torch.tensor(
            item["features"], dtype=torch.float32
        )
        padded_labels[i, :seq_len] = torch.tensor(item["labels"], dtype=torch.long)
        padded_reading_ids[i, :seq_len] = torch.tensor(
            item["reading_ids"], dtype=torch.long
        )

    result = {
        "features": padded_features,
        "labels": padded_labels,
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "reading_ids": padded_reading_ids,
    }

    # v10: teacher logits (pre-computed, only present during KD phase)
    if "teacher_logits" in batch[0]:
        padded_teacher = torch.zeros(len(batch), max_len, NUM_CLASSES)
        for i, item in enumerate(batch):
            seq_len = lengths[i]
            padded_teacher[i, :seq_len] = torch.tensor(
                item["teacher_logits"], dtype=torch.float32
            )
        result["teacher_logits"] = padded_teacher

    return result


# ── Model (v8 architecture — proven optimal for this dataset size) ──


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
    """BiLSTM + Self-Attention accent predictor (v8 arch).

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


# ── ONNX export ──────────────────────────────────────────────────────


class _OnnxWrapper(nn.Module):
    """Batch dim removal for ONNX export.

    Input: [seq, 11] -> Output: [seq, 21].
    """

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


def _train_epoch(  # noqa: PLR0913
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    label_smoothing: float = 0.1,
    rdrop_alpha: float = 0.0,
    sam_rho: float = 0.0,
    kd_temp: float = 3.0,
    kd_alpha: float = 0.5,
) -> tuple[float, float]:
    """1エポック分の学習を実行する.

    teacher_logitsがバッチに含まれる場合、知識蒸留ロスを追加する。

    Returns:
        (平均損失, 精度) のタプル.

    """
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    use_amp = device.type == "cuda"

    for batch in dataloader:
        features = batch["features"].to(device)
        labels = batch["labels"].to(device)
        lengths = batch["lengths"]
        r_ids = batch["reading_ids"].to(device)
        has_teacher = "teacher_logits" in batch

        if has_teacher:
            teacher_logits = batch["teacher_logits"].to(device)

        mask = _build_mask(labels, lengths, device)
        labels_flat = labels[mask]

        if sam_rho > 0:
            # ── SAM: gradient at current → perturb → gradient at perturbed ──
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(features, lengths, r_ids)
                logits_flat = logits[mask]
                ce = F.cross_entropy(
                    logits_flat,
                    labels_flat,
                    label_smoothing=label_smoothing,
                    reduction="mean",
                )
                loss = ce

                # v10: KD loss
                if has_teacher:
                    teacher_flat = teacher_logits[mask]
                    soft_student = F.log_softmax(logits_flat / kd_temp, dim=-1)
                    soft_teacher = F.softmax(teacher_flat / kd_temp, dim=-1)
                    kd_loss = F.kl_div(
                        soft_student, soft_teacher, reduction="batchmean"
                    )
                    kd_loss = kd_temp * kd_temp * kd_loss
                    loss = kd_alpha * loss + (1.0 - kd_alpha) * kd_loss

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

            # Perturb weights along gradient direction
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
                ce_p = F.cross_entropy(
                    logits_p_flat,
                    labels_flat,
                    label_smoothing=label_smoothing,
                    reduction="mean",
                )
                loss_p = ce_p

                if has_teacher:
                    teacher_flat = teacher_logits[mask]
                    soft_sp = F.log_softmax(logits_p_flat / kd_temp, dim=-1)
                    soft_tp = F.softmax(teacher_flat / kd_temp, dim=-1)
                    kd_p = F.kl_div(soft_sp, soft_tp, reduction="batchmean")
                    kd_p = kd_temp * kd_temp * kd_p
                    loss_p = kd_alpha * loss_p + (1.0 - kd_alpha) * kd_p

                if rdrop_alpha > 0:
                    logits_p2 = model(features, lengths, r_ids)
                    logits_p2_flat = logits_p2[mask]
                    lp2 = F.log_softmax(logits_p_flat, dim=-1)
                    lq2 = F.log_softmax(logits_p2_flat, dim=-1)
                    kl3 = F.kl_div(lp2, lq2.exp(), reduction="batchmean")
                    kl4 = F.kl_div(lq2, lp2.exp(), reduction="batchmean")
                    loss_p = loss_p + rdrop_alpha * (kl3 + kl4) / 2
            loss_p.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Unperturb and step with gradients from perturbed point
            with torch.no_grad():
                for param, e_w in perturbations.items():
                    param.sub_(e_w)
            optimizer.step()
        else:
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(features, lengths, r_ids)
                logits_flat = logits[mask]
                ce_loss = F.cross_entropy(
                    logits_flat,
                    labels_flat,
                    label_smoothing=label_smoothing,
                    reduction="mean",
                )
                loss = ce_loss

                # v10: KD loss
                if has_teacher:
                    teacher_flat = teacher_logits[mask]
                    soft_student = F.log_softmax(logits_flat / kd_temp, dim=-1)
                    soft_teacher = F.softmax(teacher_flat / kd_temp, dim=-1)
                    kd_loss = F.kl_div(
                        soft_student, soft_teacher, reduction="batchmean"
                    )
                    kd_loss = kd_temp * kd_temp * kd_loss
                    loss = kd_alpha * loss + (1.0 - kd_alpha) * kd_loss

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
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

        with torch.no_grad():
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


# ── v10: Teacher ensemble logit pre-computation ──────────────────────


@torch.no_grad()
def _precompute_teacher_logits(
    teacher_states: list[dict[str, torch.Tensor]],
    dataset: _AccentDataset,
    model_args: dict,
    device: torch.device,
) -> None:
    """Teacher ensemble logitsを事前計算しdatasetの各sampleに付与する.

    各sampleの"teacher_logits"キーにlogitsリストを格納する。
    teacherは非拡張の入力に対してのみ推論する。

    """
    model = AccentModel(**model_args).to(device)
    model.eval()

    n_teachers = len(teacher_states)
    print(
        f"  Pre-computing teacher logits ({n_teachers} teachers, "
        f"{len(dataset)} samples)..."
    )

    for idx, sample in enumerate(dataset.samples):
        features = torch.tensor(
            [sample["features"]], dtype=torch.float32, device=device
        )
        r_ids = torch.tensor([sample["reading_ids"]], dtype=torch.long, device=device)

        logits_sum: torch.Tensor | None = None
        for state in teacher_states:
            model.load_state_dict(state)
            model.eval()
            logits = model(features, reading_ids=r_ids)
            if logits_sum is None:
                logits_sum = logits.clone()
            else:
                logits_sum += logits

        assert logits_sum is not None
        avg_logits = logits_sum / n_teachers
        sample["teacher_logits"] = avg_logits[0].cpu().tolist()

        if (idx + 1) % 1000 == 0:
            print(f"    {idx + 1}/{len(dataset)} samples done")

    print(f"  Done ({len(dataset)} samples)")


# ── v10: Greedy Checkpoint Soup ──────────────────────────────────────


def _greedy_soup(
    candidates: list[tuple[float, dict[str, torch.Tensor]]],
    model_args: dict,
    val_loader: DataLoader,
    device: torch.device,
) -> tuple[float, dict[str, torch.Tensor]]:
    """候補チェックポイントから貪欲にsoupを構築する.

    1. 最良の単一チェックポイントからスタート
    2. 残りの候補を精度順にイテレート
    3. soupに加えてval精度が改善すれば採用、そうでなければスキップ

    Returns:
        (最良精度, soup state_dict) のタプル.

    """
    model = AccentModel(**model_args).to(device)

    # Sort by val accuracy descending
    candidates_sorted = sorted(candidates, key=lambda x: x[0], reverse=True)

    best_acc = candidates_sorted[0][0]
    soup_states = [candidates_sorted[0][1]]
    soup_acc = best_acc

    print(f"  Greedy Soup: starting with best ckpt ({best_acc * 100:.2f}%)")

    for i, (acc, state) in enumerate(candidates_sorted[1:], 1):
        # Try adding this checkpoint to the soup
        trial_states = soup_states + [state]
        avg_state: dict[str, torch.Tensor] = {}
        for key in state:
            stacked = torch.stack([s[key].float() for s in trial_states])
            avg_state[key] = stacked.mean(dim=0)

        model.load_state_dict(avg_state)
        model.to(device)
        _, trial_acc = _evaluate(model, val_loader, device)

        if trial_acc > soup_acc:
            soup_states.append(state)
            soup_acc = trial_acc
            print(
                f"    + Added ckpt {i} ({acc * 100:.2f}%): "
                f"soup={soup_acc * 100:.2f}% ({len(soup_states)} ckpts)"
            )
        else:
            print(
                f"    - Skipped ckpt {i} ({acc * 100:.2f}%): "
                f"would be {trial_acc * 100:.2f}%"
            )

    # Build final soup state
    final_state: dict[str, torch.Tensor] = {}
    for key in soup_states[0]:
        stacked = torch.stack([s[key].float() for s in soup_states])
        final_state[key] = stacked.mean(dim=0)

    print(f"  Soup result: {soup_acc * 100:.2f}% ({len(soup_states)} ckpts)")
    return soup_acc, final_state


# ── Phase 1: Teacher training (same as v8 _train_single) ────────────


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
    batch_size: int = 64,
    weight_decay: float = 0.05,
    label_smoothing: float = 0.1,
    morpheme_dropout: float = 0.1,
    stage1_lr: float = 2e-3,
    stage1_epochs: int = 20,
    stage2_lr: float = 8e-4,
    stage2_epochs: int = 80,
    stage2_patience: int = 20,
    rdrop_alpha: float = 0.5,
    top_k: int = 3,
    use_cosine: bool = False,
    sam_rho: float = 0.0,
    swa_epochs: int = 10,
    swa_lr: float = 5e-5,
    pretrained_state: dict[str, torch.Tensor] | None = None,
) -> tuple[float, dict[str, torch.Tensor]]:
    """独立した1 seedで全Stage訓練し、最良のstate_dictを返す.

    pretrained_stateが提供された場合、Stage 1をスキップし
    共有事前学習済み重みから開始する。

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
    ).to(device)

    # ── Stage 1: Pre-training on JVS ──
    if pretrained_state is not None:
        # Use shared pre-trained weights (v10: enables Model Soups)
        model.load_state_dict(pretrained_state)
        model.to(device)
        print("  [Stage 1] Using shared pre-trained weights")
    elif jvs_utterances:
        print(f"  [Stage 1] Pre-training ({stage1_epochs} ep)")
        jvs_ds = _AccentDataset(
            jvs_utterances, augment=True, morpheme_dropout=morpheme_dropout
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
    sched_name = "cosine" if use_cosine else "plateau"
    sam_info = f", SAM rho={sam_rho}" if sam_rho > 0 else ""
    print(f"  [Stage 2] Fine-tuning ({stage2_epochs} ep, sched={sched_name}{sam_info})")

    train_ds = _AccentDataset(
        train_utts,
        augment=True,
        morpheme_dropout=morpheme_dropout,
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
    if use_cosine:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=stage2_epochs, eta_min=1e-6
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
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
        )
        va_loss, va_acc = _evaluate(model, val_loader, device)

        lr = optimizer.param_groups[0]["lr"]
        if use_cosine:
            scheduler.step()
        else:
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

        if epoch % 5 == 0 or marker or epoch <= 3:
            print(
                f"    ep {epoch:2d}: va={va_acc:.4f} "
                f"tr={tr_acc:.4f} lr={lr:.6f}{marker}"
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

    # ── v10: SWA Phase ──
    # Load the best checkpoint and continue training with constant low LR,
    # collecting checkpoints each epoch and averaging them.
    if best_state is not None and swa_epochs > 0:
        print(f"  [SWA] {swa_epochs} ep, LR={swa_lr}")
        model.load_state_dict(best_state)
        model.to(device)

        swa_optimizer = torch.optim.AdamW(
            model.parameters(), lr=swa_lr, weight_decay=weight_decay
        )
        swa_scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

        swa_states: list[dict[str, torch.Tensor]] = []
        for _ep in range(1, swa_epochs + 1):
            _train_epoch(
                model,
                train_loader,
                swa_optimizer,
                device,
                swa_scaler,
                label_smoothing=label_smoothing,
                rdrop_alpha=rdrop_alpha,
            )
            swa_states.append(
                {k: v.cpu().clone() for k, v in model.state_dict().items()}
            )

        # Average all SWA checkpoints
        swa_avg: dict[str, torch.Tensor] = {}
        for key in swa_states[0]:
            stacked = torch.stack([s[key].float() for s in swa_states])
            swa_avg[key] = stacked.mean(dim=0)

        model.load_state_dict(swa_avg)
        model.to(device)

        # Update BatchNorm stats with training data
        torch.optim.swa_utils.update_bn(train_loader, model, device=device)

        _, swa_acc = _evaluate(model, val_loader, device)
        print(f"  SWA avg ({len(swa_states)} ckpts): {swa_acc * 100:.2f}%")

        if swa_acc > best_val_acc:
            best_val_acc = swa_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    return best_val_acc, {k: v.cpu() for k, v in best_state.items()}


# ── Phase 2: Student training with KD + Warm Restarts ────────────────


def _train_student(  # noqa: PLR0913
    jvs_utterances: list[dict] | None,
    jsut_utterances: list[dict],
    teacher_states: list[dict[str, torch.Tensor]],
    *,
    corpus_utterances: list[dict] | None = None,
    seed: int,
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
    stage2_lr: float = 8e-4,
    stage2_epochs: int = 80,
    stage2_patience: int = 20,
    rdrop_alpha: float = 0.5,
    sam_rho: float = 0.05,
    kd_temp: float = 2.0,
    kd_alpha: float = 0.9,
    top_k: int = 3,
) -> tuple[float, dict[str, torch.Tensor], list[tuple[float, dict[str, torch.Tensor]]]]:
    """知識蒸留でstudent modelを学習する.

    v8と同じ学習ループ（ReduceLROnPlateau + SAM + Top-K avg）に
    KDを軽い正則化として追加。

    Returns:
        (最良精度, 最良state_dict, 全収集チェックポイントリスト) のタプル.

    """
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_args = {
        "embed_dim": embed_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "num_classes": NUM_CLASSES,
        "dropout": dropout,
        "attention_heads": attention_heads,
    }
    model = AccentModel(**model_args).to(device)

    # ── Stage 1: Pre-training on JVS (no KD — different domain) ──
    if jvs_utterances:
        print(f"  [Stage 1] Pre-training ({stage1_epochs} ep)")
        jvs_ds = _AccentDataset(
            jvs_utterances, augment=True, morpheme_dropout=morpheme_dropout
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

    # ── Val split (fixed seed=42) ──
    random.seed(42)
    indices = list(range(len(jsut_utterances)))
    random.shuffle(indices)
    val_size = int(len(indices) * 0.1)
    val_idx = set(indices[:val_size])
    train_utts = [u for i, u in enumerate(jsut_utterances) if i not in val_idx]
    val_utts = [u for i, u in enumerate(jsut_utterances) if i in val_idx]

    if corpus_utterances:
        train_utts = train_utts + corpus_utterances

    # ── Pre-compute teacher logits ──
    train_ds = _AccentDataset(
        train_utts,
        augment=True,
        morpheme_dropout=morpheme_dropout,
    )
    _precompute_teacher_logits(teacher_states, train_ds, model_args, device)

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

    # ── Stage 2: KD fine-tuning (v8-style loop + KD regularization) ──
    sam_info = f", SAM={sam_rho}" if sam_rho > 0 else ""
    print(
        f"  [Stage 2 - KD] Fine-tuning ({stage2_epochs} ep, T={kd_temp}, "
        f"α={kd_alpha}, plateau{sam_info})"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
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

    best_val_acc = 0.0
    best_state: dict | None = None
    no_improve = 0
    top_states: list[tuple[float, dict]] = []
    collected_ckpts: list[tuple[float, dict[str, torch.Tensor]]] = []

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
            kd_temp=kd_temp,
            kd_alpha=kd_alpha,
        )
        va_loss, va_acc = _evaluate(model, val_loader, device)

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

        # Track top-K checkpoints (same as v8)
        state_copy = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if len(top_states) < top_k:
            top_states.append((va_acc, state_copy))
            top_states.sort(key=lambda x: x[0])
        elif va_acc > top_states[0][0]:
            top_states[0] = (va_acc, state_copy)
            top_states.sort(key=lambda x: x[0])

        if epoch % 5 == 0 or marker or epoch <= 3:
            print(
                f"    ep {epoch:2d}: va={va_acc:.4f} "
                f"tr={tr_acc:.4f} lr={lr:.6f}{marker}"
            )

        if no_improve >= stage2_patience:
            print(f"    Early stop at ep {epoch}")
            break

    print(f"  Best single: {best_val_acc * 100:.2f}%")

    # ── Top-K Checkpoint Averaging (same as v8) ──
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
        collected_ckpts.append((avg_acc, avg_state))
        if avg_acc > best_val_acc:
            best_val_acc = avg_acc
            best_state = avg_state

    # Also collect the best state as a soup candidate
    if best_state is not None:
        collected_ckpts.append((best_val_acc, best_state))

    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    return best_val_acc, best_state, collected_ckpts


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

    parser = argparse.ArgumentParser(description="Train kotonoha accent model v10")
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
        default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15",
        help="Comma-separated seeds for multi-seed training",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Top-K checkpoints to avg")
    parser.add_argument(
        "--scheduler",
        choices=["cosine", "plateau"],
        default="plateau",
        help="Phase 1 LR scheduler (cosine or plateau)",
    )
    parser.add_argument(
        "--sam-rho",
        type=float,
        default=0.05,
        help="SAM perturbation radius (0=disabled)",
    )
    # v10 options
    parser.add_argument(
        "--stage0-seed", type=int, default=0, help="Shared Stage 0 seed"
    )
    parser.add_argument("--swa-epochs", type=int, default=10, help="SWA epochs (0=off)")
    parser.add_argument("--swa-lr", type=float, default=5e-5, help="SWA learning rate")
    # v10 KD options
    parser.add_argument(
        "--no-kd",
        action="store_true",
        help="Disable knowledge distillation (Phase 1 only + Soup)",
    )
    parser.add_argument(
        "--kd-temp",
        type=float,
        default=2.0,
        help="KD temperature for soft targets",
    )
    parser.add_argument(
        "--kd-alpha",
        type=float,
        default=0.9,
        help="Hard loss weight (1-alpha = soft loss weight)",
    )
    parser.add_argument(
        "--teacher-topn",
        type=int,
        default=3,
        help="Number of top seeds to use as teacher ensemble",
    )
    parser.add_argument(
        "--student-seed",
        type=int,
        default=42,
        help="Seed for Phase 2 student training",
    )
    parser.add_argument(
        "--student-patience",
        type=int,
        default=20,
        help="Early stopping patience for student",
    )
    parser.add_argument(
        "--student-sam-rho",
        type=float,
        default=0.05,
        help="SAM rho for student",
    )
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    use_kd = not args.no_kd

    print("=" * 65)
    print("kotonoha accent model trainer v10")
    print("  Arch: BiLSTM(3) + SelfAttn(4h) + Emb + 2-MLP (v8 arch)")
    swa_info = f", SWA={args.swa_epochs}ep" if args.swa_epochs > 0 else ""
    print(
        f"  Phase 1: {len(seeds)} seeds, sched={args.scheduler}, "
        f"SAM={args.sam_rho}{swa_info}"
    )
    if use_kd:
        print(
            f"  Phase 2: KD (T={args.kd_temp}, α={args.kd_alpha}, "
            f"teacher-top{args.teacher_topn}, SAM={args.student_sam_rho})"
        )
        print("  Phase 3: Greedy Checkpoint Soup")
    else:
        print("  KD: disabled (Phase 1 only + Greedy Soup)")
    print(f"  R-Drop: {args.rdrop_alpha}, Top-K: {args.top_k}")
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

    # ══════════════════════════════════════════════════════════════════
    # Stage 0: Shared pre-training (v10: all seeds share same init)
    # ══════════════════════════════════════════════════════════════════
    shared_state: dict[str, torch.Tensor] | None = None
    if jvs_utterances and not args.no_pretrain:
        print(f"\n{'=' * 65}")
        s0 = args.stage0_seed
        print(f"[Stage 0] Shared JVS pre-training (seed={s0})")
        print(f"{'=' * 65}")
        torch.manual_seed(s0)
        random.seed(s0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s0)

        pretrain_model = AccentModel(
            embed_dim=64,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_classes=NUM_CLASSES,
            dropout=args.dropout,
            attention_heads=4,
        ).to(device)

        jvs_ds = _AccentDataset(jvs_utterances, augment=True, morpheme_dropout=0.1)
        jvs_loader = DataLoader(
            jvs_ds,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=_collate_fn,
            num_workers=2,
            pin_memory=True,
        )
        pt_optimizer = torch.optim.AdamW(
            pretrain_model.parameters(),
            lr=args.stage1_lr,
            weight_decay=0.05,
        )
        pt_scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        for epoch in range(1, args.stage1_epochs + 1):
            loss, acc = _train_epoch(
                pretrain_model,
                jvs_loader,
                pt_optimizer,
                device,
                pt_scaler,
                label_smoothing=0.1,
            )
            if epoch % 10 == 0 or epoch == args.stage1_epochs:
                print(f"  ep {epoch:2d}: loss={loss:.4f} acc={acc:.4f}")

        shared_state = {
            k: v.cpu().clone() for k, v in pretrain_model.state_dict().items()
        }
        del pretrain_model
        print("  Shared pre-training done — all seeds will start from here")

    # ══════════════════════════════════════════════════════════════════
    # Phase 1: Multi-seed fine-tuning (from shared init)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 65}")
    print("[Phase 1] Multi-seed fine-tuning")
    print(f"{'=' * 65}")

    all_results: list[tuple[float, dict[str, torch.Tensor]]] = []

    for i, seed in enumerate(seeds):
        print(f"\n{'-' * 65}")
        print(f"[Seed {seed}] ({i + 1}/{len(seeds)})")
        print(f"{'-' * 65}")

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
            swa_epochs=args.swa_epochs,
            swa_lr=args.swa_lr,
            pretrained_state=shared_state,
        )
        all_results.append((acc, state))
        print(f"  => Seed {seed}: {acc * 100:.2f}%")

    # Phase 1 summary
    print(f"\n{'=' * 65}")
    print("[Phase 1 Results]")
    print(f"{'=' * 65}")
    for (acc, _), seed in zip(all_results, seeds, strict=True):
        print(f"  Seed {seed}: {acc * 100:.2f}%")

    phase1_best_acc, phase1_best_state = max(all_results, key=lambda x: x[0])
    phase1_best_seed = seeds[all_results.index((phase1_best_acc, phase1_best_state))]
    print(f"\n  Phase 1 best: seed {phase1_best_seed} = {phase1_best_acc * 100:.2f}%")

    if use_kd:
        # ══════════════════════════════════════════════════════════════
        # Phase 2: Knowledge Distillation with student model
        # ══════════════════════════════════════════════════════════════
        print(f"\n{'=' * 65}")
        print("[Phase 2] Knowledge Distillation")
        print(f"{'=' * 65}")

        # Select top-N teachers
        sorted_results = sorted(
            zip(all_results, seeds, strict=True),
            key=lambda x: x[0][0],
            reverse=True,
        )
        teacher_n = min(args.teacher_topn, len(sorted_results))
        teacher_entries = sorted_results[:teacher_n]
        teacher_states = [state for (_, state), _ in teacher_entries]
        teacher_accs = [acc for (acc, _), _ in teacher_entries]
        teacher_seeds = [s for _, s in teacher_entries]
        print(f"  Teacher ensemble: seeds {teacher_seeds}")
        print(f"  Teacher accs: {[f'{a * 100:.2f}%' for a in teacher_accs]}")

        print(f"\n{'-' * 65}")
        print(f"[Student seed {args.student_seed}]")
        print(f"{'-' * 65}")

        student_acc, student_state, student_ckpts = _train_student(
            jvs_utterances,
            jsut_utterances,
            teacher_states,
            corpus_utterances=corpus_utterances or None,
            seed=args.student_seed,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            batch_size=args.batch_size,
            stage1_lr=args.stage1_lr,
            stage1_epochs=args.stage1_epochs,
            stage2_lr=args.stage2_lr,
            stage2_epochs=args.stage2_epochs,
            stage2_patience=args.student_patience,
            rdrop_alpha=args.rdrop_alpha,
            sam_rho=args.student_sam_rho,
            kd_temp=args.kd_temp,
            kd_alpha=args.kd_alpha,
            top_k=args.top_k,
        )
        print(f"  => Student: {student_acc * 100:.2f}%")

        # ══════════════════════════════════════════════════════════════
        # Phase 3: Greedy Checkpoint Soup
        # ══════════════════════════════════════════════════════════════
        print(f"\n{'=' * 65}")
        print("[Phase 3] Greedy Checkpoint Soup")
        print(f"{'=' * 65}")

        model_args = {
            "embed_dim": 64,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "num_classes": NUM_CLASSES,
            "dropout": args.dropout,
            "attention_heads": 4,
        }

        # Build val loader for soup evaluation
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

        if len(student_ckpts) >= 2:
            soup_acc, soup_state = _greedy_soup(
                student_ckpts, model_args, val_loader, device
            )
        else:
            soup_acc = student_acc
            soup_state = student_state

        # Determine overall best
        final_candidates = [
            ("Phase 1 best", phase1_best_acc, phase1_best_state),
            ("Student best", student_acc, student_state),
            ("Soup", soup_acc, soup_state),
        ]
        best_name, final_acc, final_state = max(final_candidates, key=lambda x: x[1])
        print(f"\n  Phase 1 best: {phase1_best_acc * 100:.2f}%")
        print(f"  Student best: {student_acc * 100:.2f}%")
        print(f"  Soup:         {soup_acc * 100:.2f}%")
        print(f"  => Using: {best_name} ({final_acc * 100:.2f}%)")

    else:
        # No KD — apply Greedy Soup directly to Phase 1 checkpoints
        print(f"\n{'=' * 65}")
        print("[Greedy Checkpoint Soup (Phase 1 checkpoints)]")
        print(f"{'=' * 65}")

        model_args = {
            "embed_dim": 64,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "num_classes": NUM_CLASSES,
            "dropout": args.dropout,
            "attention_heads": 4,
        }

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

        if len(all_results) >= 2:
            soup_acc, soup_state = _greedy_soup(
                all_results, model_args, val_loader, device
            )
            if soup_acc > phase1_best_acc:
                final_acc = soup_acc
                final_state = soup_state
                print(f"  Using soup: {final_acc * 100:.2f}%")
            else:
                final_acc = phase1_best_acc
                final_state = phase1_best_state
                print(f"  Using best single: {final_acc * 100:.2f}%")
        else:
            final_acc = phase1_best_acc
            final_state = phase1_best_state

    # ══════════════════════════════════════════════════════════════════
    # Export ONNX
    # ══════════════════════════════════════════════════════════════════
    model = AccentModel(
        embed_dim=64,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=NUM_CLASSES,
        dropout=args.dropout,
        attention_heads=4,
    ).to(device)
    model.load_state_dict(final_state)
    model.to(device)

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
