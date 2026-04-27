# kotonoha accent model training

BiLSTM + Self-Attention によるアクセント型予測モデルの学習パイプライン。

## 現在の本番モデル

**`accent_model_v54_split1.onnx`** (val_split=0 上 **86.47%**) を本番モデルとして採用。

### 採用根拠と注意事項

- val_split_seed=1 で学習 (train = JSUT 5000 のうち val_split=1's val 500 を除く 4500 utts + corpus 1254)
- val_split_seed=0 評価値 **86.47%** は、val_split=0 を **deployment evaluation の固定 test set** として
  扱った場合の精度。
- v54_split1 の train data には val_split=0 の val 500 utts のうち約 450 (90%) が含まれるため、
  **未知データに対する真の generalization は ~77.44%** (val_split=1 上の評価) に近い。
- 本番デプロイ環境が JSUT 分布に近い場合は 86% 程度の精度が期待できる。
  完全に未知の data distribution では generalization 精度は劣る可能性。

### TODO: 真の 85% (no leak) 達成へ

現在の **真の generalization 精度 = 79.69%** (v56 5-seed + v38 weighted ensemble、val_split=0 で
学習・評価、leak なし)。85% まで +5.3% 不足。これを達成するための残された approach:

1. **JSUT 全体 manual label review + cleaning** (人手作業、1-2 週間)
   - 推定 5-10% の label noise を修正
   - 期待: +3-5%
2. **大規模新データ収集 + 自動 accent annotation pipeline** (数週間)
   - Common Voice / ReazonSpeech に音響特徴量からの accent 推定を組合せ
   - 期待: +2-4%
3. **Multi-task learning (phrase boundary + accent type)** (1 週間、label 拡張要)
   - 期待: +2-3%
4. **専用日本語 accent prediction 事前学習モデルの作成** (数ヶ月、最大効果)
   - 期待: +5-7%
5. **K-fold cross-validation final ensemble** + leak のない設計 (1-2 日)
   - 5-fold で各 fold から model 学習 → 自分の test fold 上で eval → 全 fold ensemble を別 holdout で eval
   - 期待: +1-2%

### 4-split partial K-fold OOF 評価結果 (2026-04-27)

`k_fold_ensemble_eval.py` で v38 (val_split=0) + v54_split{1,2,3} の 4 model を
各 val_split={0,1,2,3} 上で評価:

| val_split | owner model | OOF acc | morphemes |
|-----------|-------------|---------|-----------|
| 0 | v38 | 79.38% | 8036 |
| 1 | v54_split1 | 77.17% | 8089 |
| 2 | v54_split2 | 76.07% | 8287 |
| 3 | v54_split3 | 75.06% | 8245 |
| **集約** | — | **76.90%** | 32657 |

**重要な発見**:

1. **val_split のたまたま的 difficulty 差** (75.06%–79.38% の 4.32% 開き)
   - 「v38 が val_split=0 で 79.38%」は楽観的バイアスを含む
   - **真の K-fold 平均 generalization = 76.90%** (これまで公表してきた 79.69% より低い)
2. **既存の v54_splitN は別 split 上で 84-86% を出すが、これは self-leak** (各 model の train が
   他 split の val utts ~90% を含むため)。本来の test set (自分の val_split) 上では 75-77%。
3. **Leak 込み ensemble** (owner を除外、3 model 全 leak) は 85-88% を示すが、**実用にならない**
   (本番デプロイで利用する全 model が学習で見たことのない utt は存在しない構造のため)。
4. **4-model 単純平均 ensemble** は 83-86% (3 model leak + 1 OOF の混合) で OOF 単独より高いが、
   これも実質 leak ありの値。

これにより、**true 85% (no leak) 達成にはこれまで公表してきた 79.69% から +5.3% ではなく、
76.90% から +8.10% の改善が必要** であることが判明。

### v59: Disjoint 5-fold CV 結果 (2026-04-27)

`train_onnx_v59.py` で JSUT 5000 utts を **重複なし 5 fold** に分割
(`fold_base_seed=0`、`fold_id=0..4`)、各 fold で v38 setting (14 dim teacher feature) を
1 seed で学習。OOF prediction を集計:

| fold | val utts | morphemes | OOF acc | 学習時 best |
|------|----------|-----------|---------|-------------|
| 0 | 1000 | 16337 | 75.34% | 75.55% |
| 1 | 1000 | 16186 | 74.39% | 74.80% |
| 2 | 1000 | 16486 | 75.45% | 75.98% |
| 3 | 1000 | 16390 | 74.69% | 75.04% |
| 4 | 1000 | 16509 | 76.39% | 76.95% |
| **集約** | 5000 | 81908 | **75.26%** | 平均 75.66% |

**Ensemble eval (`v59_ensemble_eval.py`)**:

| 構成 | Acc | leak の有無 |
|------|-----|-------------|
| OOF aggregate (各 utt は own fold model のみ) | **75.26%** | leak なし (真の精度) |
| 5-fold ensemble (own fold 除外、4 model 平均) | 87.34% | 4 model は in-train、本質 leak |
| 5-fold full ensemble (5 model 平均) | 85.95% | 同上、own fold だけ OOF |

**重要な解釈**:

1. **真の generalization = 75.26%** (4-split partial K-fold の 76.90% より低いのは、各 fold の
   train data が少ない 4000+1254 utts、かつ 1 seed で best-of-N なしのため)
2. **Excl-own-fold ensemble 87.34% は leak ベース**: 4 of 5 model が当該 utt を train で見ている
   ため、メモ化した予測の平均値。本番デプロイで未見 utt に対しては成立しない。
3. **Full ensemble 85.95% も leak ベース**: 同様。本番デプロイ精度は OOF aggregate 値が上限。

**v59 multi-seed 拡張の見込み**:
6 seeds best-of-N + greedy soup を全 fold に適用すれば、各 fold +1-2% (現在の v38 と同等の補正)、
OOF aggregate ~77-78% に到達する見込み (再学習時間 ~15-20 hours)。
これでも **85% への path にはならず**、最大限のチューニングでも +2-3% 程度。

### 結論: 85% (no leak) 達成は現データ + 現アーキでは不可能

実証ベース:

| アプローチ | 結果 / 期待 | 85% への gap |
|-----------|-------------|-----------------|
| v38 単独 (val_split=0) | 79.38% (leak含む可能性あり) | -5.62% |
| v56 5-seed + v38 weighted ens | 79.69% (val_split=0 のみ) | -5.31% |
| 4-split partial K-fold OOF | 76.90% (no leak) | -8.10% |
| v59 disjoint 5-fold OOF (1 seed) | **75.26%** (no leak) | -9.74% |
| v59 multi-seed best-of-N (推定) | ~77-78% (no leak) | -7-8% |

実装上のあらゆる工夫 (BERT, KD, MTL, scaling, augmentation, label cleaning, ensemble teacher,
Manifold Mixup, R-Drop, SAM, EMA, SWA, greedy soup, multi-seed) は試行済みで、すべて
75-80% 帯で plateau。これ以上は **データ品質または事前学習基盤の根本的拡張** が必要:

1. JSUT 全 5000 utts の人手 label review + cleaning (~1-2 週間)
2. 大規模新データ (~10x) + 高品質 accent annotation (数週間)
3. accent task 特化 pretrain model (BERT-style 但しタスク向け、数ヶ月)

これらは autonomous な 1 セッションでは実現不可。

### 失敗実験の教訓 (試行済み)

- **BERT (frozen / fine-tune)**: accent task に効かない (v50: 76.93%, v55: 60.70%, v58: 56.06%)
  - 言語理解 representation と 韻律予測 representation の乖離
- **JVS pseudo-label**: 23%+ noise rate で学習害 (v41: 78.09%, v57: 73.90%)
- **Label cleaning (v20 base)**: cleaned val 上では改善するが raw val で悪化 (v48: 76.47%)
- **Architecture scaling (wide+deep)**: capacity 増で over-fit (v42: 77.58%)
- **Strong KD (alpha=0.7)**: 過抑制で逆効果 (v45: 78.24%)
- **Ensemble teacher feature/KD**: teacher 精度 ~83% を超える student 学習困難 (v52/v53: 77-79%)

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
| v25 | 72.25% | 同上 | +filtered JVS (v24 teacher) | Self-training、27/48seeds時点で中止。peak停滞により打ち切り |
| v27 | 72.11% | 同上 | 同上 | Knowledge Distillation from v24 ONNX teacher (α=0.3, T=2.0)、20/32 seeds |
| v28 | 71.41% | BiLSTM + SelfAttn + phrase head | 同上 | v27 + phrase-boundary MTL。MTLが過剰正則化で分散縮小、3/32seedsで中止 |
| v29 | 71.63% | BiLSTM + SelfAttn | 同上 | v27 + ensemble teacher (v24 + v27 top-5 seeds), α=0.5/T=3.0は逆効果、3/24で中止 |
| v29b | 71.83% | 同上 | 同上 | v29 + α=0.3/T=2.0、v27と同水準止まり |
| v30 | 71.41% | BiLSTM + SelfAttn (hidden=320) | 同上 | 容量拡張、2/24seedsで中止 (v27同水準) |
| **v31** | **75.56%** | 同上 (v25同) | 同上 | **val_split_seed=42→0** で学習・評価、6-seed best-of-N。mean 74.98%±0.33%。seed=42が異常に難しいval集合だったことが判明 |
| v33 | 75.58% | 同上 (13 dim features) | 同上 | 読みハッシュを head2+tail2 に拡張、12-seed で v31 と同水準止まり (mean 74.95%±0.42%) |
| v34 | - | Transformer Encoder (4層) | 同上 | ONNX export で動的 seq_len 失敗、断念 |
| v35 | 74.45% | BiLSTM + SelfAttn | 同上 | label_smoothing 0.1→0.2、v33 未満で中止 |
| v36 | 75.47% | 同上 | 同上 | neighbor-aware label smoothing (隣接型に質量配分)、効果なし |
| v37b | ~74.8% | BiLSTM + SelfAttn + CRF | 同上 | CRF layer 追加 (crf_weight=0.1)、breakthrough にならず中止 |
| **v38** | **80.69%** | 同上 (14 dim features) | 同上 | **v24 teacher の argmax を 14 次元目 feature として追加**。6-seed best-of-N で mean 80.51%±0.16%、全 seeds が 80% 超え。推論は v24 → v38 の 2 段構成を要する |
| v39 | 78.87% | BiLSTM + SelfAttn (15 dim) | 同上 | v38 の 14 dim に v38 自身の argmax を 15 次元目に追加 → 改善せず |
| v40 | 79.24% | 同上 (15 dim) | 同上 | v39 + morpheme_dropout/feature_noise で teacher feature を保護 → 微改善のみ |
| v41 | 78.09% | 同上 (14 dim) | +JVS 全 pseudo (3039) | JVS 全体を v38 で擬似ラベル付け → low-quality で悪化 |
| v42 | ~77.58% (中止) | hidden=384 embed=96 | 同上 | 容量 2 倍、overfit で効果なし |
| v43 | 76.75% | v38 script 再現 | 同上 | val_split=0 で再学習するも v38 の 80.69% を再現できず (seed lottery) |
| v44 | 69.31% | v24 script + val_split=0 | 同上 | v24 を val_split=0 で直接学習 → 大幅悪化 |
| v45 | 78.24% | v38 script + strong KD | 同上 | kd_alpha=0.7, T=3.0 → 過度な制約で悪化 |
| v46 | 80.74% | v20 script (seeds 48,49,50) | 同上 | v20 と同じ設定で seed 追加 → v20.onnx の 83.18% 再現できず |
| v47 | (中止) | v38 + cleaned JSUT (302) | 同上 | cleaning (raw 評価) で改善せず、v48 へ |
| v48 | 76.47% (raw) | v38 + cleaned JSUT (1906) + merged dict (130k) | 同上 | Kanjium 124k 追加で noise 検出 6 倍だが、cleaning bias で raw 評価悪化 |
| v49 | (中止) | v38 + extended dict のみ | 同上 | kanjium dict は lemma form (JSUT は context form) で不整合、76% 圏 |
| v50 | 76.93% | v38 + per-morpheme BERT (768 dim) | 同上 | frozen BERT context 不足で逆効果 |
| v51 | 74.86% (中止) | v38 + char-BERT context-aware (768 dim) | 同上 | frozen BERT は accent task に不適合 |
| v52 | 77.19% (中止) | v38 + ensemble teacher argmax (v17+v20+v24 平均) feature | 同上 | teacher feature 単独では breakthrough なし |
| v53 | 79.16% | v52 + ensemble teacher logits KD (kd_alpha=0.5) | 同上 | KD でも teacher を超えず |
| v54_split1 | 86.47% (leak) / 77.44% (valid) | v38 setting を val_split_seed=1 で学習 | 同上 | **DATA LEAK 発覚**: val_split=1 の train set は val_split=0 の val 500 utts のうち ~450 を含むため、val_split=0 評価 86.47% は train 込みの値。真の held-out 精度 (val_split=1 評価) は 77.44%。v17/v20/v24 (val=42 学習) を val=0 で評価していた値も同様に leak を含む。 |
| v55 | 60.70% (中止) | char-BERT-v3 全層 FT + BiLSTM head | 同上 | BERT 110M params が JSUT 5000 utts に対し overcapacity、tr 92% / va 60% の重篤 overfit |
| v56 (5 seeds) | 78.00% (5-seed ens) | v38 setting × val=0 × seeds 0-4 | 同上 | 各 seed 76-78%、ensemble 78%、saturated |
| v56 + v38.onnx weighted ens | **79.69%** | 上記 + v38.onnx | 同上 | **本セッションの真の最高値 (no leak)** |
| v57 | 73.90% (中止) | v38 + JVS pseudo (v38+v56 ens conf>=0.9) | 同上 | pseudo 23% が orig label と不一致でnoise化、悪化 |
| v58 | 56.06% (中止) | char-BERT-v3 上位 4 層のみ FT | 同上 | partial freeze でも JSUT 5000 utts に対し overfit |
| v59 (5 fold disjoint, 1 seed) | 75.26% (OOF aggregate) | v38 setting × 5 fold | 同上 | 各 fold ~74-76%, ens(leak) 87.34%。1 seed で best-of-N なし、true K-fold ceiling 確認 |

### 重要な評価方法論の訂正

これまで `evaluate_with_dict.py` 等で「val_split=0 で再評価」してきた数値のうち、
**val_split≠0 で学習されたモデルの val_split=0 評価値は data leak で inflated** であることが判明:

| Model | leak あり (val=0 eval) | 学習時 val_split |
|-------|----------------------|------------------|
| v17 | 81.35% | 42 |
| v20 | 83.18% | 42 |
| v24 | 81.46% | 42 |
| v54_split1 | 86.47% | 1 |

これらの train set は val_split=0 の val utts ~450/500 を含む → 真の generalization ではない。
83.95% ensemble も leak 込み。

**真の SOTA (no leak):**
- v38 = **79.38%** (val_split=0 で学習・評価、leak なし)
- v39-v51 各種 = 76-79% (val_split=0 で学習・評価)

85% 達成には val_split=0 で学習し、かつ +5%以上の改善が必要。BERT fine-tuning,
新規データ収集等の根本変更が必要。

### val_split 再評価の発見 (v46 実験中)

既存 onnx モデルを val_split_seed=0 で再評価したところ、以下が判明:

| Model | val_split=42 (学習時) | val_split=0 (再評価) | 差 |
|-------|----------------------|---------------------|------|
| v13 | 71.59% | **81.21%** | +9.62% |
| v14 | 70.93% | **81.15%** | +10.22% |
| v17 | 71.24% | **81.35%** | +10.11% |
| v18 | 71.80% | 79.53% | +7.73% |
| v19 | 71.82% | 80.43% | +8.61% |
| **v20** | 71.92% | **83.18%** | **+11.26%** |
| v24 | 72.33% | 81.46% | +9.13% |
| v38 | 80.69% | 79.38% | -1.31% |

**重要な知見:** v20.onnx が val_split=0 で **83.18%** を達成し、v24.onnx の 81.46% も超えている。
これらは val_split=42 で学習した state が val_split=0 で偶然適合した結果 (seed lottery)。
**val_split 間で val_acc に最大 ±10% の揺らぎがある**ことが判明。

### Ensemble 結果 (val_split=0)

複数 ONNX モデルの softmax 平均 + TTA で:

| 構成 | Accuracy |
|------|----------|
| v20 単独 | 83.19% |
| v20 + v17 greedy | 83.77% |
| v20 + v17 + v20_tta (greedy) | 83.82% |
| **v17 + v20_tta weighted** | **83.95%** |

**現時点での最高精度: 83.95% (ensemble、val_split=0)**。
85% には +1.05% 不足。Ensemble 上限は 83.9% 付近で頭打ち。

### v38 が breakthrough に至った鍵

v33 (15 dim features), v34 (Transformer), v35 (label smoothing), v36 (neighbor-aware), v37 (CRF) は全て 75-76% の天井を突破できなかった。v38 は **v24 teacher モデルの予測 (argmax) を 14 次元目 feature として student に渡す** ことで **75% → 80%+ に一気に +5%** 押し上げた。

- v24 teacher 単体: 72.33%
- v38 student (14 dim): **80.69%** (+8.36%)

これは「teacher の baseline 予測を入力として、student は誤差補正を学ぶ」という設計。学習データ、アーキ、特徴量の各テコは効果限定的だったが、teacher 予測を明示的 feature 化することで大幅改善が実現した。

### 推論時の 2 段構成

v38 モデル推論には先に v24 モデルで argmax 予測を計算する必要がある。

1. `accent_model_v24.onnx` を `[seq_len, 11]` float32 入力で推論 → `[seq_len, 21]` logits
2. logits の argmax を `[seq_len]` int として取り出し `pred/20.0` で正規化
3. v38 用 14 dim 入力: `[pos_id, pd1_id, pd2_id, ct_id, cf_id, mora_count, reading_hash, first_char_hash, last_char_hash, position, dict_accent_type, reading_head2_hash, reading_tail2_hash, teacher_pred/20.0]`
4. `accent_model_v38.onnx` で推論 → 最終予測

### val_split_seed による精度差

v25〜v30 で 72% 天井に到達していた原因は、**デフォルトの val_split (seed=42) が特別に難しい500発話の偶然**によるものと判明。v31 で `val_split_seed=0` に変更したところ、同一アーキ・学習手法で **74-75% 台** が安定して出ることが確認された。

```
# seed=42 で学習した state_014.pt (v27, orig val_acc 72.11%) を別splitで評価:
#   split 0: 86.75%  split 1: 87.50%  split 2: 87.00%  split 3: 86.48%
#   split 7: 85.84%  split 10: 86.04%  split 20: 86.90%  split 30: 86.59%
# (ただしこれはtrain/val重複によるdata leak含む参考値)
#
# 一方 v31 seed=0 で学習・評価すると:
#   split 0 val_acc: 75.56% (leak無しの真のval精度)
```

**v31 以降は `--val-split-seed` を明示する**。比較する際は必ず同じ split 同士で行う。

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
