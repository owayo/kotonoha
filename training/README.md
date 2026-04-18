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
| v10 | 71.44% | BiLSTM + SelfAttn (v8同) | 同上 | 共有Stage0 + Greedy Soup, 16-seed Best-of-N |
| v11 | 71.51% | BiLSTM + SelfAttn (v8同) | 同上 | Top-5 avg, 24-seed Best-of-N |
| v12 | 67.24% | BiLSTM + SelfAttn (compact) | 同上 | hidden=192, 2層, embed=48 — アンダーフィッティング |
| v13 | 71.59% | BiLSTM + SelfAttn (v8同) | 同上 | v11 + Manifold Mixup, Feature noise, 24-seed Best-of-N |
| v14 | 70.93% | 同上 | 同上 | CosineWarmRestarts, R-Drop 0.7 — 安定化するが高ピーク低下 |
| v15 | 71.54% | 同上 | 同上 | v13 + 32-seed, corpus_v2(2098発話)はむしろ悪化 |
| v16 | 70.45% | BiLSTM + SelfAttn (13dim) | 同上 | 前後文脈POS追加 — LSTMが既に学習済みで冗長、過学習悪化 |
| v17 | 71.24% | BiLSTM + SelfAttn (v8同) | +filtered JVS | Self-Training: v13でJVSフィルタ、71%超seed率2倍だがpeak未更新 |
| v18 | 71.80% | BiLSTM + SelfAttn (v8同) | +recombined corpus (2000発話) | v13 + アクセント句リコンビネーション, 48-seed, mean=70.80%±0.43% |
| v19 | 71.82% | 同上 | 同上 | v18追加48-seed (seed 68), 合計96-seed, mean=70.80%±0.40% |
| v20 | 71.92% | 同上 | +combined corpus (3420発話) | LLM+recomb+replace複合コーパス, 48-seed, mean=71.00%±0.49% |
| v20-r2 | **72.20%** | 同上 | 同上 | **72%突破！** seeds 48-143 追加実行, seed 89 = 71.78% single, **Top-5 avg 72.20%** |
| v21 | 71.63% | 同上 | +combined corpus v2 (4920発話) | コーパス増量は逆効果、mean=70.95%±0.33% (80/96seeds時点で中止) |
| v22 | 71.56% | 同上 | +combined corpus (3420発話) | 2段階FT+rd schedule, mean=70.82% — v20未満で中止 |
| v24 | **72.33%** | 同上 | 同上 | 120ep cosine + Top-10 + greedy soup + EMA, 96-seed, mean=71.26%±0.33% |
| v22 | - | 同上 | +combined corpus (3420発話) | 2段階FT + rd schedule - mean 70.82%でv20未満につき中止 |
| v22 | 71.56% | 同上 | +combined corpus (3420発話) | 2-phase fine-tune + rd schedule、分散縮小(σ=0.29%)で逆効果 (14/96seeds時点で中止) |
| v23 | 70.42% | 同上 | +combined corpus v3 (3420発話) | morpheme_dropout 0.18 + label_smoothing 0.08 + corpus長文化、mean低下で中止 |

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
- **Self-Distillation**: teacher ensemble→student KD (v10)
- **Greedy Checkpoint Soup**: 相補的チェックポイントの貪欲選択 (v10)
- **CosineAnnealingWarmRestarts**: 周期的再始動でチェックポイント収集 (v10)
- **Manifold Mixup**: LSTM出力レベルで2サンプルを補間 (v13, alpha=0.3, prob=0.3)
- **Feature noise**: std=0.02 (v13, 連続特徴量にガウスノイズ — v9とは独立に再実装)
- **morpheme_dropout**: 0.15 (v13, v11の0.1から微増)

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

### v13

```bash
cd training
uv run python train_onnx_v13.py                          # デフォルト: 24 seeds, Manifold Mixup
uv run python train_onnx_v13.py --seeds 0,1,2,3          # seed数を指定
uv run python train_onnx_v13.py --mixup-alpha 0          # Manifold Mixup無効化
uv run python train_onnx_v13.py --feature-noise-std 0    # Feature noise無効化
```

### v11

```bash
cd training
uv run python train_onnx_v11.py                          # デフォルト: 24 seeds, Top-5 avg
uv run python train_onnx_v11.py --seeds 0,1,2,3          # seed数を指定
uv run python train_onnx_v11.py --reading-dropout 0.1    # reading dropout有効化
```

### v10

```bash
cd training
uv run python train_onnx_v10.py                          # デフォルト: 16 seeds, 共有Stage0
uv run python train_onnx_v10.py --seeds 0,1,2,3          # seed数を指定
uv run python train_onnx_v10.py --no-kd                  # KDなし（Best-of-N + Soup）
uv run python train_onnx_v10.py --stage0-seed 1          # Stage0のseed変更
uv run python train_onnx_v10.py --swa-epochs 0           # SWA無効化
```

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

## TODO: 今後の改善案

v10 実験で得た知見に基づく改善候補。現在の最大のボトルネックは
**データ量** (train 82% vs val 71% の過学習ギャップ) である。

### データ量の拡大 (最優先)

- [ ] LLM コーパスの増量 (420→2000-5000 発話)
  - v6 でコーパス混合が +1.07% 寄与した実績あり
  - kotonoha-training-data の LLM 生成ツールで追加生成可能
- [ ] 外部コーパスの活用 (Common Voice, JNAS 等からアクセントラベル自動抽出)

### 特徴量の改善 (nn.rs 変更が必要)

- [ ] train/inference の reading ミスマッチ解消
  - 訓練時は `kana_emb` (文字レベル埋め込み) を使用するが、
    ONNX 推論時は `reading_hash` にフォールバックする
  - 対策 A: 訓練時に一定確率で `reading_ids=None` にする (reading dropout)
  - 対策 B: ONNX 入力に読み文字列 ID 列を追加
- [ ] 前後文脈特徴の追加 (前/次形態素の品詞 ID)
  - アクセント型は前後の品詞に強く依存する
  - 現在は LSTM が暗黙的に学習するのみ
- [ ] 読みハッシュの改善
  - DJB2 % 10000 は衝突が多い
  - 複数ハッシュ (先頭2文字、末尾2文字、読み長) に分割して情報保持量を増加

### モデル容量の削減 (過学習対策)

- [ ] hidden_dim を 256→192 に縮小 (パラメータ数 ~4.7M→~2.5M)
- [ ] LSTM 層数を 3→2 に縮小
- [ ] Embedding dim を 64→32 に縮小 (品詞 15 種に対して 64 は過剰)
- [ ] これらの組み合わせで train/val ギャップの縮小を確認する

### 評価方法の改善

- [ ] k-fold 交差検証の導入 (現在は固定 seed=42 の 90/10 split)
  - seed 依存の分散が ±1% 観測されており、単一 split では信頼性が低い
- [ ] 発話レベル精度 (utterance-level exact match) の計測
  - 形態素単位の正解率に加え、発話全体が正しい割合も評価
- [ ] クラス別精度の分析 (どのアクセント型が間違いやすいか)

### 学習手法の改善 (小幅)

- [x] Manifold Mixup: 隠れ表現レベルで 2 サンプルを補間 (v13 で導入、過学習ギャップ 11%→4-7% に縮小)
- [x] Top-K を 3→5 に拡大 (v11 で導入)
- [ ] Confidence-based sample weighting の再検討
  - 辞書アクセント型とラベルが不一致のサンプルの低重み化
  - 現在はコメントアウトで全 1.0 (`_compute_confidence_weight`)

### v12-v15 実験で得た知見 (参考)

| 手法 | 結果 | 理由 |
|------|------|------|
| 容量削減 hidden=192, 2層 (v12) | 67.24% (アンダーフィッティング) | 容量削減+正則化強化の同時適用は過剰 |
| Manifold Mixup + Feature noise (v13) | **71.59%** (新記録) | 過学習ギャップ縮小に有効 |
| CosineWarmRestarts + R-Drop 0.7 (v14) | 70.93% (悪化) | 分散は下がるが高ピークを殺す |
| corpus_v2 (2098発話) 混合 (v15) | 70.68% (悪化) | LLM生成データの品質がJSUTより低く学習を希釈 |
| 高LR (1.2e-3) + cosine decay | 70.69% (変わらず) | 100epochのcosine decayではLR下降が遅すぎ |
| SAM rho=0.1 (2倍) | 71.06% (悪化) | 摂動が大きすぎ、rho=0.05が最適 |
| 32-seed / 64-seed | 71.54% / 71.17% | seed増加は収穫逓減、seed 14が安定して高い |
| GPU非決定性 (seed 14 × 5回) | 70.63-71.40% | 同一設定でも±0.5%のばらつき |
| LR warmup 5ep + LR=1e-3 | 70.95% (mean 70.39%) | 3-seed test では+0.14%だが24-seedではmean不変 |
| 前後文脈POS特徴量 (v16, 13dim) | 70.45% (mean 69.93%) | LSTMが既に文脈を学習、明示的特徴は冗長+過学習悪化 |
| JVS Self-Training (v17, 閾値80%) | 71.24% (71%超4/24=17%) | v13でJVSフィルタ→603発話追加、71%超出現率2倍だが最高値は未更新 |

### 現状のボトルネック分析

v12-v16 の網羅的実験により、**ハイパーパラメータ調整ではmean ~70.4% を超えられない**ことが判明。
v13 の 71.59% は GPU 非決定性による ~2.4σ の outlier (seed 14, 再現確率 ~1%)。

次の突破口には以下が必要:
- **特徴量の改善** (nn.rs 変更): 前後文脈POS追加、読みハッシュの改善
- **高品質な学習データ**: JSUT レベルの音声コーパス追加 (corpus_v2 は品質不足)
- **Train/Inference mismatch 解消**: ONNX入力に読み文字列IDを追加 (nn.rs + ONNX変更)

### v10 実験で効果がなかった手法 (参考)

| 手法 | 結果 | 理由 |
|------|------|------|
| Knowledge Distillation | Student 70.20% < Teacher 71.27% | 同アーキテクチャでは暗黙知が不足 |
| SWA | Top-K avg と同等 | 既に Top-3 avg が同等の効果を達成 |
| 2層 Self-Attention (v9) | 70.99% < v8 71.29% | データ量に対して容量過剰 |
| EMA (v9) | Top-K avg と競合 | 類似目的の手法が重複 |
| Focal Loss (v9) | 改善なし | 21 クラスの中程度の不均衡には過剰 |
| Cross-seed Greedy Soup (独立init) | 71.27%→27% | 異なる盆地の重み平均は破壊的 |
| Cross-seed Greedy Soup (共有init) | +0.01% | 同じ盆地内では改善幅が限定的 |
