# kotonoha accent model training

BiLSTM + Self-Attention によるアクセント型予測モデルの学習パイプライン。

## モデル履歴

| Version | Val Accuracy | Architecture | Training Data | Notes |
|---------|-------------|--------------|---------------|-------|
| v1 | 27% | Single-layer LSTM | JSUT | 初版 |
| v2 | 62.76% | BiLSTM | JSUT | |
| v3 | 68.01% | BiLSTM + Self-Attention | JSUT | |
| v4 | 66.44% | BiLSTM + Self-Attention | JSUT+JVS混合 | JVSノイズで劣化 |
| v5 | 69.12% | BiLSTM + Self-Attention | JVS事前学習→JSUT微調整 | |
| v6 | 70.19% | 同上 | 同上+辞書補完+コーパス混合 | |
| v7 | 70.76% | 同上 | 同上 | R-Drop, Top-3 checkpoint avg |
| v8 | 71.29% | 同上 | 同上 | v7 + LR=8e-4, Multi-seed Best-of-N |
| v9 | 70.99% | BiLSTM + SelfAttn + FFN | 同上 | EMA, Focal Loss(opt), Warm-up(opt), Feature noise(opt) — v8未満 |

## アーキテクチャ詳細

- **パラメータ数**: 4,698,885
- **embed_dim**: 64
- **hidden_dim**: 256
- **LSTM層数**: 3
- **Attention heads**: 4
- **Dropout**: 0.4
- **NUM_CLASSES**: 21
- **FEATURE_DIM**: 11

### 特徴量 (11次元、`nn.rs extract_features_v2` と一致)

| Index | Feature | Encoding |
|-------|---------|----------|
| 0 | 品詞 ID | POS_VOCAB (15種) |
| 1 | 品詞細分類1 ID | PD1_VOCAB (21種) |
| 2 | 品詞細分類2 ID | PD2_VOCAB (11種) |
| 3 | 活用型 ID | グループ化 (9種) |
| 4 | 活用形 ID | グループ化 (10種) |
| 5 | モーラ数 | / 10.0 で正規化 |
| 6 | 読みハッシュ | DJB2, % 10000 / 10000.0 |
| 7 | 先頭文字ハッシュ | 同上 |
| 8 | 末尾文字ハッシュ | 同上 |
| 9 | 文内位置 | i / (n-1), [0, 1] |
| 10 | 辞書アクセント型 | (type+1)/8.0 if >0, else 0.0 |

## 学習設定

### Stage 1: JVS 事前学習

- **データ**: JVS corpus (3,039発話)
- **LR**: 2e-3
- **エポック**: 20 (固定、early stopping なし)
- **目的**: 複数話者のノイズの多いデータから基本パターンを学習

### Stage 2: JSUT 微調整

- **データ**: JSUT corpus (5,000発話) + LLMコーパス (420発話)
- **Train/Val分割**: 90% / 10% (seed=42)
- **LR**: 8e-4 (v8/v9)
- **最大エポック**: 80
- **LRスケジューラ**: ReduceLROnPlateau (v8) / Warm-up + CosineAnnealing (v9)
- **Early stopping patience**: 20
- **R-Drop**: alpha=0.5 (2回のdropout間のKLダイバージェンスペナルティ)
- **Top-K checkpoint averaging**: K=3 (上位3チェックポイントの重み平均)
- **Multi-seed Best-of-N**: 複数seedで独立学習し最良を選択
- **EMA**: decay=0.999 (v9, 学習中にパラメータの指数移動平均を追跡)
- **Focal Loss**: gamma=2.0 (v9, クラス不均衡への対処)
- **Feature noise**: std=0.02 (v9, 連続特徴量にガウスノイズ)
- **Ensemble avg**: top-N seeds重み平均 (v9, --ensemble-topn)

## 出力

- ONNX入力: `input` — `[seq_len, 11]` float32
- ONNX出力: `output` — `[seq_len, 21]` float32 (logits)

## セットアップ

`.env.example` をコピーして `.env` を作成し、各パスを環境に合わせて設定する:

```bash
cp .env.example .env
# .env を編集してデータパスを設定
```

`.env` の設定項目:

| 変数 | 説明 |
|------|------|
| `PRETRAIN_DATA` | JVS事前学習データ (JSON) |
| `FINETUNE_DATA` | JSUT微調整データ (JSON or CSV) |
| `CORPUS_DATA` | LLMコーパスデータ (JSON, Stage 2混合) |
| `ACCENT_DICT` | アクセント辞書CSV (コロン区切りで複数指定可) |
| `OUTPUT_MODEL` | 出力ONNXモデルパス |

コマンドライン引数で上書きも可能 (`--pretrain-data`, `--finetune-data`, `--accent-dict`, `--output`)。

## 実行方法

### v9 (実験的)

```bash
cd training
uv run python train_onnx_v9.py                          # デフォルト: 8 seeds, warmup+cosine
uv run python train_onnx_v9.py --seeds 0,1,2            # seed数を指定
uv run python train_onnx_v9.py --no-focal               # Focal Loss無効化
uv run python train_onnx_v9.py --ensemble-topn 3        # top-3 seeds重み平均
uv run python train_onnx_v9.py --num-attn-layers 1      # v8と同じ1層Attention
```

### v8

```bash
cd training
uv run python train_onnx_v8.py                          # デフォルト: 8 seeds, plateau, lr=8e-4
uv run python train_onnx_v8.py --seeds 0,1,2            # seed数を指定
uv run python train_onnx_v8.py --sam-rho 0.05           # SAM有効化
uv run python train_onnx_v8.py --scheduler cosine       # CosineAnnealingLR使用
```

### v7

```bash
cd training
uv run python train_onnx_v7.py --seed 1 --rdrop-alpha 0.5
```

### 依存関係

- Python 3.12+
- PyTorch >= 2.6.0 (CUDA 12.6)
- onnx, onnxruntime-gpu
