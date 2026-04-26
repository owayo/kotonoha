"""v55: BERT fine-tuning end-to-end for accent type prediction.

Architecture:
  char-BERT-v3 (768 dim) → projection → BiLSTM (256) → SelfAttn → classifier (21)

Training:
  - val_split=0 (no leak)
  - end-to-end (BERT trainable with low LR, head with high LR)
  - mixed precision (fp16)
  - gradient checkpointing for BERT
  - batch_size 16 (small for memory)

Goal: > 85% val accuracy on val_split=0 held-out.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

NUM_CLASSES = 21


def _enrich_utterances(utts: list[dict], _ad: dict) -> int:
    """No-op enrich (not needed for BERT FT)."""
    return 0


class _BertAccentModel(nn.Module):
    """char-BERT + per-morpheme pooling + BiLSTM + classifier."""

    def __init__(
        self,
        bert_model: nn.Module,
        hidden_dim: int = 256,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.3,
        use_grad_ckpt: bool = True,
        freeze_lower_layers: int = 8,
    ) -> None:
        """Initialize.

        v58: freeze lower N layers of BERT (default 8 of 12) to prevent
        catastrophic forgetting and reduce overfitting on small data.
        """
        super().__init__()
        self.bert = bert_model
        if use_grad_ckpt:
            self.bert.gradient_checkpointing_enable()
        # v58: Freeze lower layers
        if freeze_lower_layers > 0:
            for p in self.bert.embeddings.parameters():
                p.requires_grad = False
            for i, layer in enumerate(self.bert.encoder.layer):
                if i < freeze_lower_layers:
                    for p in layer.parameters():
                        p.requires_grad = False
            print(
                f"  Frozen embeddings + lower {freeze_lower_layers} layers; "
                f"trainable upper {len(self.bert.encoder.layer) - freeze_lower_layers}"
            )
        bert_dim = bert_model.config.hidden_size
        self.proj = nn.Linear(bert_dim, hidden_dim)
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.lstm_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        morph_token_idx: torch.Tensor,
        morph_token_mask: torch.Tensor,
        morph_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Forward.

        Args:
            input_ids: [B, T] token ids.
            attention_mask: [B, T].
            morph_token_idx: [B, M, K] token indices for each morpheme (pad with 0).
            morph_token_mask: [B, M, K] mask (1 if token belongs to morpheme).
            morph_lengths: [B] number of morphemes per utt.

        Returns:
            logits [B, M, num_classes].

        """
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        h = bert_out.last_hidden_state  # [B, T, 768]
        # Gather per-morpheme tokens and mean pool
        b, m, k = morph_token_idx.shape
        idx = morph_token_idx.clamp(min=0)  # avoid -1
        gathered = torch.gather(
            h.unsqueeze(1).expand(-1, m, -1, -1),
            2,
            idx.unsqueeze(-1).expand(-1, -1, -1, h.size(-1)),
        )  # [B, M, K, D]
        mask = morph_token_mask.unsqueeze(-1).float()
        summed = (gathered * mask).sum(dim=2)  # [B, M, D]
        count = mask.sum(dim=2).clamp(min=1.0)  # [B, M, 1]
        morph_emb = summed / count  # [B, M, D]

        x = self.proj(morph_emb)  # [B, M, hidden]
        x_input = x
        if self.training:
            packed = pack_padded_sequence(
                x, morph_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.lstm(packed)
            x, _ = pad_packed_sequence(packed_out, batch_first=True)
        else:
            x, _ = self.lstm(x)
        x = self.lstm_proj(x)
        if x.size(1) < x_input.size(1):
            pad_len = x_input.size(1) - x.size(1)
            x = F.pad(x, (0, 0, 0, pad_len))
        x = self.norm(x + x_input)
        return self.classifier(x)


class _BertAccentDataset(Dataset):
    """Dataset for BERT FT: input_ids + per-morpheme token alignment."""

    def __init__(self, utterances: list[dict], tokenizer) -> None:
        """Build dataset."""
        self.samples: list[dict] = []
        for utt in utterances:
            ms = utt.get("morphemes", [])
            if not ms:
                continue
            surfaces = [m.get("surface", "") for m in ms]
            text = "".join(surfaces)
            if not text:
                continue
            morph_ranges: list[tuple[int, int]] = []
            pos = 0
            for s in surfaces:
                morph_ranges.append((pos, pos + len(s)))
                pos += len(s)

            enc = tokenizer(text, truncation=True, max_length=512)
            input_ids = enc["input_ids"]
            tokens = tokenizer.convert_ids_to_tokens(input_ids)
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
                if (
                    char_pos < len(text)
                    and text[char_pos : char_pos + len(clean)] == clean
                ):
                    token_to_char.append(char_pos)
                    char_pos += len(clean)
                else:
                    found = text.find(clean, char_pos) if clean else -1
                    if found >= 0:
                        token_to_char.append(found)
                        char_pos = found + len(clean)
                    else:
                        token_to_char.append(-1)
            morph_token_lists = []
            for m_start, m_end in morph_ranges:
                idxs = [
                    ti
                    for ti, c in enumerate(token_to_char)
                    if c >= 0 and m_start <= c < m_end
                ]
                if not idxs:
                    idxs = [0]  # fallback to CLS
                morph_token_lists.append(idxs)

            labels = [min(m.get("accent_type", 0), NUM_CLASSES - 1) for m in ms]
            self.samples.append(
                {
                    "input_ids": input_ids,
                    "morph_token_lists": morph_token_lists,
                    "labels": labels,
                }
            )

    def __len__(self) -> int:
        """Length."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        """Get item."""
        return self.samples[idx]


def _collate(batch: list[dict], pad_token_id: int = 0) -> dict:
    """Pad and stack."""
    max_t = max(len(s["input_ids"]) for s in batch)
    max_m = max(len(s["labels"]) for s in batch)
    max_k = max(max(len(idxs) for idxs in s["morph_token_lists"]) for s in batch)
    b = len(batch)

    input_ids = torch.full((b, max_t), pad_token_id, dtype=torch.long)
    attn_mask = torch.zeros(b, max_t, dtype=torch.long)
    morph_token_idx = torch.zeros(b, max_m, max_k, dtype=torch.long)
    morph_token_mask = torch.zeros(b, max_m, max_k, dtype=torch.long)
    labels = torch.full((b, max_m), -100, dtype=torch.long)
    morph_lengths = torch.zeros(b, dtype=torch.long)

    for i, s in enumerate(batch):
        t = len(s["input_ids"])
        m = len(s["labels"])
        input_ids[i, :t] = torch.tensor(s["input_ids"])
        attn_mask[i, :t] = 1
        morph_lengths[i] = m
        for j, idxs in enumerate(s["morph_token_lists"]):
            for kk, ti in enumerate(idxs):
                morph_token_idx[i, j, kk] = ti
                morph_token_mask[i, j, kk] = 1
        labels[i, :m] = torch.tensor(s["labels"])

    return {
        "input_ids": input_ids,
        "attention_mask": attn_mask,
        "morph_token_idx": morph_token_idx,
        "morph_token_mask": morph_token_mask,
        "morph_lengths": morph_lengths,
        "labels": labels,
    }


def main() -> None:
    """Train BERT FT model."""
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--finetune-data",
        default="/home/owayo/kotonoha-training/data/jsut_accent_data_v3.json",
    )
    ap.add_argument(
        "--corpus-data",
        default="/mnt/c/GitHub/kotonoha-training-data/train/corpus_converted.json",
    )
    ap.add_argument(
        "--extra-corpus",
        default="/mnt/c/GitHub/kotonoha/training/filtered_jvs_v24_t75.json",
    )
    ap.add_argument("--val-split-seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bert-lr", type=float, default=2e-5)
    ap.add_argument("--head-lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--bert-model", default="tohoku-nlp/bert-base-japanese-char-v3")
    ap.add_argument(
        "--output", default="/mnt/c/GitHub/kotonoha-models/accent_model_v58.pt"
    )
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    print(f"Loading BERT: {args.bert_model}")
    tk = AutoTokenizer.from_pretrained(args.bert_model)
    bert = AutoModel.from_pretrained(args.bert_model)

    model = _BertAccentModel(bert, hidden_dim=256, num_classes=NUM_CLASSES, dropout=0.3)
    model.to(device)

    # Load data
    with open(args.finetune_data, encoding="utf-8") as f:
        jsut = json.load(f)["utterances"]
    print(f"JSUT: {len(jsut)}")
    corpus_utts: list[dict] = []
    if Path(args.corpus_data).exists():
        with open(args.corpus_data, encoding="utf-8") as f:
            corpus_data = json.load(f)
        if isinstance(corpus_data, list):
            corpus_utts = corpus_data
        else:
            corpus_utts = corpus_data.get("utterances", [])
    if Path(args.extra_corpus).exists():
        with open(args.extra_corpus, encoding="utf-8") as f:
            extra = json.load(f)
        if isinstance(extra, list):
            corpus_utts = corpus_utts + extra
        else:
            corpus_utts = corpus_utts + extra.get("utterances", [])
    print(f"corpus: {len(corpus_utts)}")

    # Split
    random.seed(args.val_split_seed)
    indices = list(range(len(jsut)))
    random.shuffle(indices)
    val_size = int(len(indices) * 0.1)
    val_idx = set(indices[:val_size])
    train_utts = [u for i, u in enumerate(jsut) if i not in val_idx]
    val_utts = [u for i, u in enumerate(jsut) if i in val_idx]
    train_utts = train_utts + corpus_utts
    print(f"train: {len(train_utts)} val: {len(val_utts)}")

    train_ds = _BertAccentDataset(train_utts, tk)
    val_ds = _BertAccentDataset(val_utts, tk)
    print(f"train ds: {len(train_ds)} val ds: {len(val_ds)}")

    pad_id = tk.pad_token_id or 0

    def _coll(b: list[dict]) -> dict:
        return _collate(b, pad_token_id=pad_id)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=_coll,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=_coll,
        num_workers=2,
        pin_memory=True,
    )

    # Param groups: BERT lower lr, head higher lr
    bert_params = list(model.bert.parameters())
    head_params = [p for n, p in model.named_parameters() if not n.startswith("bert.")]
    optimizer = torch.optim.AdamW(
        [
            {"params": bert_params, "lr": args.bert_lr},
            {"params": head_params, "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[args.bert_lr * 2, args.head_lr * 2],
        total_steps=total_steps,
        pct_start=0.1,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_val_acc = 0.0
    best_state: dict | None = None
    no_improve = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        tot_loss = 0.0
        tot_correct = 0
        tot_count = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["morph_token_idx"],
                    batch["morph_token_mask"],
                    batch["morph_lengths"],
                )
                # mask labels=-100
                loss = F.cross_entropy(
                    logits.reshape(-1, NUM_CLASSES),
                    batch["labels"].reshape(-1),
                    ignore_index=-100,
                    label_smoothing=0.1,
                )
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                mask = batch["labels"] != -100
                tot_correct += ((preds == batch["labels"]) & mask).sum().item()
                tot_count += mask.sum().item()
                tot_loss += loss.item() * mask.sum().item()
        tr_loss = tot_loss / max(tot_count, 1)
        tr_acc = tot_correct / max(tot_count, 1)

        model.eval()
        v_correct = 0
        v_count = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    logits = model(
                        batch["input_ids"],
                        batch["attention_mask"],
                        batch["morph_token_idx"],
                        batch["morph_token_mask"],
                        batch["morph_lengths"],
                    )
                preds = logits.argmax(dim=-1)
                mask = batch["labels"] != -100
                v_correct += ((preds == batch["labels"]) & mask).sum().item()
                v_count += mask.sum().item()
        va_acc = v_correct / max(v_count, 1)

        marker = ""
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            marker = " *"
        else:
            no_improve += 1
        print(
            f"ep {epoch:2d}: tr_loss={tr_loss:.4f} tr={tr_acc:.4f} "
            f"va={va_acc:.4f}{marker}"
        )
        if no_improve >= args.patience:
            print(f"  Early stop at ep {epoch}")
            break

    print(f"\nBest val acc: {best_val_acc * 100:.2f}%")
    if best_state is not None:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"acc": best_val_acc, "state": best_state}, out_path)
        print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
