"""kotonoha ONNX accent model trainer (v6).

BiLSTM + Self-Attention モデルを2段階転移学習で訓練し、ONNX形式で出力する。

Stage 1: JVS データで事前学習（ノイズの多い複数話者データから基本パターン学習）
Stage 2: JSUT データで微調整（クリーンな単一話者データで精度向上）

特徴量は nn.rs の extract_features_v2 と完全に一致させること:
  [pos_id, pd1_id, pd2_id, ct_id, cf_id,
   mora_count, reading_hash, first_char_hash,
   last_char_hash, position, dict_accent_type]

Usage:
  cd training
  uv run python main.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

# ── Constants (must match nn.rs) ──────────────────────────────────────────

FEATURE_DIM = 11
NUM_CLASSES = 21  # accent types 0-20

# ── Vocab mappings (must match nn.rs) ─────────────────────────────────────

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
    # Aliases
    "接頭詞": 11,
    "フィラー": 0,
}

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


def _conj_type_to_id(ctype: str) -> int:
    """活用型を数値IDに変換する.

    Returns:
        数値ID (0-8).

    """
    if ctype == "*":
        return 0
    if ctype.startswith("五段"):
        return 1
    if ctype.startswith("上一段"):
        return 2
    if ctype.startswith("下一段"):
        return 3
    if ctype.startswith("カ行変格"):
        return 4
    if ctype.startswith("サ行変格"):
        return 5
    if ctype.startswith("形容詞"):
        return 6
    if ctype.startswith("助動詞"):
        return 7
    if ctype.startswith("文語"):
        return 8
    return 0


def _conj_form_to_id(cform: str) -> int:
    """活用形を数値IDに変換する.

    Returns:
        数値ID (0-9).

    """
    if cform == "*":
        return 0
    if cform.startswith("未然形"):
        return 1
    if cform.startswith("連用形"):
        return 2
    if cform.startswith("終止形"):
        return 3
    if cform.startswith("連体形"):
        return 4
    if cform.startswith("仮定形"):
        return 5
    if cform.startswith("命令形"):
        return 6
    if cform.startswith("已然形"):
        return 7
    if cform.startswith("意志推量形"):
        return 8
    if cform.startswith("語幹"):
        return 9
    return 0


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


# ── Mora counting (must match mora.rs) ────────────────────────────────────

_SMALL_KANA = set("ァィゥェォャュョヮ")


def _count_mora(reading: str) -> int:
    """カタカナ読みからモーラ数を算出する.

    Returns:
        モーラ数.

    """
    count = 0
    for ch in reading:
        if ch in _SMALL_KANA:
            continue
        count += 1
    return count


# ── Data structures ──────────────────────────────────────────────────────


@dataclass
class _Token:
    """形態素トークン."""

    surface: str
    pos: str
    pos_detail1: str
    pos_detail2: str
    ctype: str
    cform: str
    reading: str
    pronunciation: str
    accent_type: int


@dataclass
class _Utterance:
    """発話単位."""

    tokens: list[_Token] = field(default_factory=list)


# ── Data loading ─────────────────────────────────────────────────────────


def _load_accent_dict(path: Path) -> dict[tuple[str, str], int]:
    """アクセント辞書を読み込む.

    Returns:
        (lemma, reading) をキーとしたアクセント型辞書.

    """
    d: dict[tuple[str, str], int] = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f)
        first = next(reader)
        # If first row looks like header, skip it
        if first[0] != "lemma" and len(first) >= 3:
            d[(first[0].strip(), first[1].strip())] = int(first[2].strip())
        for row in reader:
            if len(row) < 3:
                continue
            d[(row[0].strip(), row[1].strip())] = int(row[2].strip())
    return d


def _load_training_data_csv(path: Path) -> list[_Utterance]:
    """CRF training data v2 CSV を読み込む.

    Returns:
        発話リスト.

    """
    utterances: list[_Utterance] = []
    current = _Utterance()

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                continue
            if not line:
                if current.tokens:
                    utterances.append(current)
                    current = _Utterance()
                continue

            fields = line.split(",")
            if len(fields) >= 9:
                tok = _Token(
                    surface=fields[0],
                    pos=fields[1],
                    pos_detail1=fields[2],
                    pos_detail2=fields[3],
                    ctype=fields[4],
                    cform=fields[5],
                    reading=fields[6],
                    pronunciation=fields[7],
                    accent_type=int(fields[8]),
                )
            elif len(fields) >= 4:
                tok = _Token(
                    surface=fields[0],
                    pos=fields[1],
                    pos_detail1="*",
                    pos_detail2="*",
                    ctype="*",
                    cform="*",
                    reading=fields[2],
                    pronunciation=fields[2],
                    accent_type=int(fields[3]),
                )
            else:
                continue
            current.tokens.append(tok)

    if current.tokens:
        utterances.append(current)

    return utterances


def _load_training_data_json(path: Path) -> list[_Utterance]:
    """JSON形式の学習データを読み込む (JSUT/JVS).

    Returns:
        発話リスト.

    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    utterances: list[_Utterance] = []
    for utt_data in data["utterances"]:
        utt = _Utterance()
        for m in utt_data["morphemes"]:
            tok = _Token(
                surface=m["surface"],
                pos=m["pos"],
                pos_detail1=m.get("pos_detail1", "*"),
                pos_detail2=m.get("pos_detail2", "*"),
                ctype=m.get("conjugation_type", "*"),
                cform=m.get("conjugation_form", "*"),
                reading=m.get("reading", ""),
                pronunciation=m.get("pronunciation", ""),
                accent_type=int(m["accent_type"]),
            )
            utt.tokens.append(tok)
        if utt.tokens:
            utterances.append(utt)

    return utterances


def _load_training_data(path: Path) -> list[_Utterance]:
    """学習データを読み込む (CSV/JSON自動判別).

    Returns:
        発話リスト.

    """
    if path.suffix == ".json":
        return _load_training_data_json(path)
    return _load_training_data_csv(path)


# ── Feature extraction (must match nn.rs extract_features_v2) ────────────


def _extract_features(
    tok: _Token,
    position: float,
    accent_dict: dict[tuple[str, str], int],
) -> list[float]:
    """nn.rs extract_features_v2 と同一の特徴量を抽出する.

    Returns:
        11次元の特徴量リスト.

    """
    pos_id = float(POS_VOCAB.get(tok.pos, 0))
    pd1_id = float(PD1_VOCAB.get(tok.pos_detail1, 0))
    pd2_id = float(PD2_VOCAB.get(tok.pos_detail2, 0))
    ct_id = float(_conj_type_to_id(tok.ctype))
    cf_id = float(_conj_form_to_id(tok.cform))
    mora = _count_mora(tok.reading) / 10.0
    rh = _reading_hash(tok.reading)
    first_ch = _char_hash(tok.surface[0]) if tok.surface else 0.0
    last_ch = _char_hash(tok.surface[-1]) if tok.surface else 0.0

    # Dict accent type: lookup by (surface, reading)
    dict_at = accent_dict.get((tok.surface, tok.reading), 0)
    dict_feat = (dict_at + 1.0) / 8.0 if dict_at > 0 else 0.0

    return [
        pos_id,
        pd1_id,
        pd2_id,
        ct_id,
        cf_id,
        mora,
        rh,
        first_ch,
        last_ch,
        position,
        dict_feat,
    ]


def _utterance_to_tensors(
    utt: _Utterance,
    accent_dict: dict[tuple[str, str], int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """発話を (features [T, 11], labels [T]) テンソルに変換する.

    Returns:
        特徴量テンソルとラベルテンソルのタプル.

    """
    n = len(utt.tokens)
    feats = []
    labels = []
    for i, tok in enumerate(utt.tokens):
        pos = i / (n - 1) if n > 1 else 0.0
        feats.append(_extract_features(tok, pos, accent_dict))
        labels.append(min(tok.accent_type, NUM_CLASSES - 1))
    return (
        torch.tensor(feats, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long),
    )


# ── Model ────────────────────────────────────────────────────────────────


class AccentModel(nn.Module):
    """BiLSTM + Self-Attention accent predictor (v5/v6 architecture)."""

    def __init__(
        self,
        input_dim: int = FEATURE_DIM,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_heads: int = 4,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.4,
    ) -> None:
        """モデルを初期化する."""
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attn = nn.MultiheadAttention(
            hidden_dim * 2,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [batch, seq_len, input_dim]

        Returns:
            logits: [batch, seq_len, num_classes]

        """
        h = self.input_proj(x)  # [B, T, H]
        h, _ = self.lstm(h)  # [B, T, 2H]
        attn_out, _ = self.attn(h, h, h)  # [B, T, 2H]
        h = self.layer_norm(h + attn_out)
        h = self.dropout(h)
        return self.classifier(h)  # [B, T, C]


# ── ONNX export ──────────────────────────────────────────────────────────


def _export_onnx(model: AccentModel, path: Path, device: torch.device) -> None:
    """モデルをONNX形式でエクスポートする."""
    model.eval()
    dummy = torch.randn(1, 10, FEATURE_DIM, device=device)
    torch.onnx.export(
        model,
        dummy,
        str(path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch", 1: "seq_len"},
            "output": {0: "batch", 1: "seq_len"},
        },
        opset_version=17,
        dynamo=False,
    )


# ── Training ─────────────────────────────────────────────────────────────


def _prepare_data(
    utterances: list[_Utterance],
    accent_dict: dict[tuple[str, str], int],
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """発話リストをテンソルに変換する.

    Returns:
        (features, labels) タプルのリスト.

    """
    data = []
    for utt in utterances:
        if not utt.tokens:
            continue
        feat, lab = _utterance_to_tensors(utt, accent_dict)
        data.append((feat, lab))
    return data


def _compute_class_weights(
    train_data: list[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> torch.Tensor:
    """訓練データからクラス重みを計算する.

    Returns:
        クラス重みテンソル.

    """
    label_counts = [0] * NUM_CLASSES
    for _, labels in train_data:
        for label_id in labels.tolist():
            label_counts[label_id] += 1
    total_labels = sum(label_counts)
    print(f"  Total tokens: {total_labels}")
    print("  Label distribution:")
    for i, c in enumerate(label_counts):
        if c > 0:
            pct = c / total_labels * 100
            print(f"    Type {i}: {c} ({pct:.1f}%)")

    weights = torch.ones(NUM_CLASSES, device=device)
    for i in range(NUM_CLASSES):
        if label_counts[i] > 0:
            w = total_labels / (NUM_CLASSES * label_counts[i])
            weights[i] = min(w, 10.0)
    return weights


def _run_training_loop(
    model: AccentModel,
    train_data: list[tuple[torch.Tensor, torch.Tensor]],
    val_data: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    epochs: int,
    lr: float,
    batch_size: int,
    patience: int,
    device: torch.device,
    stage_name: str = "",
) -> float:
    """学習ループを実行する.

    Returns:
        検証セットでの最良精度.

    """
    weights = _compute_class_weights(train_data, device)
    print(f"  Class weights (top-5): {weights[:5].tolist()}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6
    )
    criterion = nn.CrossEntropyLoss(weight=weights)

    best_val_acc = 0.0
    best_state: dict | None = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        random.shuffle(train_data)

        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for batch_start in range(0, len(train_data), batch_size):
            batch = train_data[batch_start : batch_start + batch_size]
            feats = [f for f, _ in batch]
            labels = [lab for _, lab in batch]

            feats_padded = pad_sequence(feats, batch_first=True).to(device)
            labels_padded = pad_sequence(
                labels, batch_first=True, padding_value=-100
            ).to(device)

            logits = model(feats_padded)

            loss = criterion(logits.view(-1, NUM_CLASSES), labels_padded.view(-1))

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item() * len(batch)

            preds = logits.argmax(dim=-1)
            mask = labels_padded != -100
            epoch_correct += (preds[mask] == labels_padded[mask]).sum().item()
            epoch_total += mask.sum().item()

        train_acc = epoch_correct / epoch_total if epoch_total > 0 else 0.0
        avg_loss = epoch_loss / len(train_data)

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for feat, lab in val_data:
                feat_dev = feat.unsqueeze(0).to(device)
                logits = model(feat_dev)
                preds = logits.squeeze(0).argmax(dim=-1)
                lab_dev = lab.to(device)
                val_correct += (preds == lab_dev).sum().item()
                val_total += lab_dev.size(0)
        val_acc = val_correct / val_total if val_total > 0 else 0.0

        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0
        prefix = f"[{stage_name}] " if stage_name else ""
        print(
            f"  {prefix}Epoch {epoch:3d}/{epochs}: loss={avg_loss:.4f} "
            f"train_acc={train_acc * 100:.2f}% "
            f"val_acc={val_acc * 100:.2f}% "
            f"lr={current_lr:.6f} ({elapsed:.1f}s)"
        )

        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(
                    f"  Early stopping at epoch {epoch} "
                    f"(no improvement for {patience} epochs)"
                )
                break

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    return best_val_acc


def _train(
    pretrain_utterances: list[_Utterance] | None,
    finetune_utterances: list[_Utterance],
    accent_dict: dict[tuple[str, str], int],
    output_path: Path,
    *,
    pretrain_epochs: int = 20,
    pretrain_lr: float = 2e-3,
    finetune_epochs: int = 80,
    finetune_lr: float = 5e-4,
    batch_size: int = 32,
    val_ratio: float = 0.1,
    hidden_dim: int = 256,
    finetune_patience: int = 20,
) -> float:
    """2段階転移学習でモデルを訓練し、ONNX形式で保存する.

    Returns:
        検証セットでの最良精度.

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build model (v5 architecture: 3-layer BiLSTM, dropout=0.4)
    print("\n[1/5] Building model...")
    model = AccentModel(hidden_dim=hidden_dim, num_layers=3, dropout=0.4).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count:,}")

    # ── Stage 1: Pre-training on JVS ──
    if pretrain_utterances:
        print("\n[2/5] Stage 1: Pre-training on JVS...")
        pretrain_data = _prepare_data(pretrain_utterances, accent_dict)
        random.seed(42)
        random.shuffle(pretrain_data)
        # Use 10% for validation during pretraining
        pt_val_size = max(1, int(len(pretrain_data) * val_ratio))
        pt_val = pretrain_data[:pt_val_size]
        pt_train = pretrain_data[pt_val_size:]
        print(f"  Train: {len(pt_train)} utterances")
        print(f"  Val:   {len(pt_val)} utterances")

        pt_acc = _run_training_loop(
            model,
            pt_train,
            pt_val,
            epochs=pretrain_epochs,
            lr=pretrain_lr,
            batch_size=batch_size,
            patience=pretrain_epochs,  # no early stopping in pretraining
            device=device,
            stage_name="Stage1",
        )
        print(f"  Stage 1 best val accuracy: {pt_acc * 100:.2f}%")
    else:
        print("\n[2/5] Stage 1: Skipped (no pre-training data)")

    # ── Stage 2: Fine-tuning on JSUT ──
    print("\n[3/5] Stage 2: Preparing fine-tuning data (JSUT)...")
    finetune_data = _prepare_data(finetune_utterances, accent_dict)
    random.seed(42)
    random.shuffle(finetune_data)
    val_size = max(1, int(len(finetune_data) * val_ratio))
    val_data = finetune_data[:val_size]
    train_data = finetune_data[val_size:]
    print(f"  Train: {len(train_data)} utterances")
    print(f"  Val:   {len(val_data)} utterances")

    print("\n[4/5] Stage 2: Fine-tuning...")
    best_val_acc = _run_training_loop(
        model,
        train_data,
        val_data,
        epochs=finetune_epochs,
        lr=finetune_lr,
        batch_size=batch_size,
        patience=finetune_patience,
        device=device,
        stage_name="Stage2",
    )
    print(f"\n  Best val accuracy: {best_val_acc * 100:.2f}%")

    # Export ONNX
    print("\n[5/5] Exporting ONNX...")
    _export_onnx(model, output_path, device)
    file_size = output_path.stat().st_size
    print(f"  Output: {output_path}")
    print(f"  Size: {file_size / 1_048_576:.1f} MB")
    print(f"  Val accuracy: {best_val_acc * 100:.2f}%")

    return best_val_acc


# ── Main ─────────────────────────────────────────────────────────────────


def _load_dotenv() -> None:
    """カレントディレクトリの .env ファイルを読み込んで環境変数に設定する."""
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

    parser = argparse.ArgumentParser(description="Train kotonoha accent model v6")
    parser.add_argument(
        "--pretrain-data",
        type=Path,
        default=Path(os.environ.get("PRETRAIN_DATA", "")),
        help="Path to JVS pre-training data (JSON)",
    )
    parser.add_argument(
        "--finetune-data",
        type=Path,
        default=Path(os.environ.get("FINETUNE_DATA", "")),
        help="Path to JSUT fine-tuning data (JSON or CSV)",
    )
    parser.add_argument(
        "--accent-dict",
        type=Path,
        default=Path(os.environ.get("ACCENT_DICT", "")),
        help="Path to accent dictionary CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(os.environ.get("OUTPUT_MODEL", "")),
        help="Output ONNX model path",
    )
    parser.add_argument("--pretrain-epochs", type=int, default=20)
    parser.add_argument("--pretrain-lr", type=float, default=2e-3)
    parser.add_argument("--finetune-epochs", type=int, default=80)
    parser.add_argument("--finetune-lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--finetune-patience", type=int, default=20)
    parser.add_argument(
        "--no-pretrain",
        action="store_true",
        help="Skip Stage 1 pre-training",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("kotonoha accent model trainer v6")
    print("  Architecture: BiLSTM(3-layer) + Self-Attention(4-head)")
    print("  Strategy: 2-stage transfer learning (JVS → JSUT)")
    print("=" * 60)

    # Load accent dict
    print(f"\nLoading accent dict: {args.accent_dict}")
    accent_dict = _load_accent_dict(args.accent_dict)
    print(f"  Entries: {len(accent_dict)}")

    # Load pre-training data (JVS)
    pretrain_utterances = None
    if not args.no_pretrain and args.pretrain_data.exists():
        print(f"\nLoading pre-training data (JVS): {args.pretrain_data}")
        pretrain_utterances = _load_training_data(args.pretrain_data)
        total_pt = sum(len(u.tokens) for u in pretrain_utterances)
        print(f"  Utterances: {len(pretrain_utterances)}")
        print(f"  Tokens: {total_pt}")
    elif args.no_pretrain:
        print("\nPre-training: skipped (--no-pretrain)")
    else:
        print(f"\nWarning: Pre-training data not found: {args.pretrain_data}")

    # Load fine-tuning data (JSUT)
    print(f"\nLoading fine-tuning data (JSUT): {args.finetune_data}")
    finetune_utterances = _load_training_data(args.finetune_data)
    total_ft = sum(len(u.tokens) for u in finetune_utterances)
    print(f"  Utterances: {len(finetune_utterances)}")
    print(f"  Tokens: {total_ft}")

    # Train
    _train(
        pretrain_utterances,
        finetune_utterances,
        accent_dict,
        args.output,
        pretrain_epochs=args.pretrain_epochs,
        pretrain_lr=args.pretrain_lr,
        finetune_epochs=args.finetune_epochs,
        finetune_lr=args.finetune_lr,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        finetune_patience=args.finetune_patience,
    )

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
