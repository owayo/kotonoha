//! CRF（条件付き確率場）によるアクセント型予測モジュール
//!
//! 外部ML依存なしの純Rust実装。特徴ハッシングとビタビデコーディングを使い、
//! NjdNode列からアクセント型を予測する。

use crate::njd::{NjdNode, Pos};
use crate::nn::AccentPredictor;
use std::io::{Read, Write};
use std::path::Path;

/// 特徴量の次元数（ハッシュトリック用）
const FEATURE_DIM: usize = 50_000;

/// デフォルトのアクセント型ラベル数（0〜5+）
const DEFAULT_NUM_LABELS: usize = 6;

/// 品詞を数値IDにエンコードする
fn pos_to_id(pos: &Pos) -> usize {
    match pos {
        Pos::Meishi => 0,
        Pos::Doushi => 1,
        Pos::Keiyoushi => 2,
        Pos::Fukushi => 3,
        Pos::Joshi => 4,
        Pos::Jodoushi => 5,
        Pos::Rentaishi => 6,
        Pos::Setsuzokushi => 7,
        Pos::Kandoushi => 8,
        Pos::Settoushi => 9,
        Pos::Kigou => 10,
        Pos::Filler => 11,
        Pos::Sonota => 12,
    }
}

/// 文字列を特徴インデックスにハッシュする（ハッシュトリック）
fn feature_hash(prefix: &str, value: &str) -> usize {
    let mut hash: u32 = 2_166_136_261;
    for b in prefix.bytes() {
        hash ^= u32::from(b);
        hash = hash.wrapping_mul(16_777_619);
    }
    hash ^= b':' as u32;
    hash = hash.wrapping_mul(16_777_619);
    for b in value.bytes() {
        hash ^= u32::from(b);
        hash = hash.wrapping_mul(16_777_619);
    }
    (hash as usize) % FEATURE_DIM
}

/// 数値を特徴インデックスにハッシュする
fn feature_hash_num(prefix: &str, value: usize) -> usize {
    let mut hash: u32 = 2_166_136_261;
    for b in prefix.bytes() {
        hash ^= u32::from(b);
        hash = hash.wrapping_mul(16_777_619);
    }
    hash ^= b':' as u32;
    hash = hash.wrapping_mul(16_777_619);
    // Encode the number as bytes
    for b in value.to_string().bytes() {
        hash ^= u32::from(b);
        hash = hash.wrapping_mul(16_777_619);
    }
    (hash as usize) % FEATURE_DIM
}

/// 読みの先頭かな行を判定する（ア行、カ行、サ行…）
fn kana_group(reading: &str) -> &'static str {
    match reading.chars().next() {
        Some(c) => match c {
            'ア' | 'イ' | 'ウ' | 'エ' | 'オ' => "ア行",
            'カ' | 'キ' | 'ク' | 'ケ' | 'コ' => "カ行",
            'サ' | 'シ' | 'ス' | 'セ' | 'ソ' => "サ行",
            'タ' | 'チ' | 'ツ' | 'テ' | 'ト' => "タ行",
            'ナ' | 'ニ' | 'ヌ' | 'ネ' | 'ノ' => "ナ行",
            'ハ' | 'ヒ' | 'フ' | 'ヘ' | 'ホ' => "ハ行",
            'マ' | 'ミ' | 'ム' | 'メ' | 'モ' => "マ行",
            'ヤ' | 'ユ' | 'ヨ' => "ヤ行",
            'ラ' | 'リ' | 'ル' | 'レ' | 'ロ' => "ラ行",
            'ワ' | 'ヲ' | 'ン' => "ワ行",
            'ガ' | 'ギ' | 'グ' | 'ゲ' | 'ゴ' => "ガ行",
            'ザ' | 'ジ' | 'ズ' | 'ゼ' | 'ゾ' => "ザ行",
            'ダ' | 'ヂ' | 'ヅ' | 'デ' | 'ド' => "ダ行",
            'バ' | 'ビ' | 'ブ' | 'ベ' | 'ボ' => "バ行",
            'パ' | 'ピ' | 'プ' | 'ペ' | 'ポ' => "パ行",
            _ => "他",
        },
        None => "空",
    }
}

/// 文中の位置
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SentencePosition {
    Beginning,
    Middle,
    End,
}

impl SentencePosition {
    fn as_str(self) -> &'static str {
        match self {
            SentencePosition::Beginning => "BOS",
            SentencePosition::Middle => "MID",
            SentencePosition::End => "EOS",
        }
    }
}

/// 読みの末尾モーラ（最後の1〜2文字）を取得する
fn reading_suffix(reading: &str, n: usize) -> String {
    let chars: Vec<char> = reading.chars().collect();
    if chars.len() <= n {
        reading.to_string()
    } else {
        chars[chars.len() - n..].iter().collect()
    }
}

/// 位置からノードの特徴量をスパースベクトルとして抽出する
///
/// 返り値は `(特徴インデックス, 値)` のペア列。
pub fn extract_features(nodes: &[NjdNode], position: usize) -> Vec<(usize, f32)> {
    let mut features = Vec::with_capacity(60);
    let node = &nodes[position];
    let len = nodes.len();

    // 1. POS one-hot
    let pos_id = pos_to_id(&node.pos);
    features.push((feature_hash_num("pos", pos_id), 1.0));

    // 2. POS detail1 (hashed)
    features.push((feature_hash("pos_d1", &node.pos_detail1), 1.0));

    // 3. POS detail2 (hashed)
    if node.pos_detail2 != "*" && !node.pos_detail2.is_empty() {
        features.push((feature_hash("pos_d2", &node.pos_detail2), 1.0));
    }

    // 4. POS + detail1 conjunction (richer POS representation)
    let pos_d1_combo = format!("{}_{}", node.pos.to_label_str(), node.pos_detail1);
    features.push((feature_hash("pos_pd1", &pos_d1_combo), 1.0));

    // 5. POS + detail1 + detail2 conjunction
    if node.pos_detail2 != "*" && !node.pos_detail2.is_empty() {
        let pos_d12_combo = format!("{}_{}_{}", node.pos.to_label_str(), node.pos_detail1, node.pos_detail2);
        features.push((feature_hash("pos_pd12", &pos_d12_combo), 1.0));
    }

    // 6. Mora count (capped at 10)
    let mora = node.mora_count.min(10) as usize;
    features.push((feature_hash_num("mora", mora), 1.0));

    // 7. Reading length
    let reading_len = node.reading.chars().count().min(20);
    features.push((feature_hash_num("rlen", reading_len), 1.0));

    // 8. Kana group of first character
    features.push((feature_hash("kgrp", kana_group(&node.reading)), 1.0));

    // 9. Surface form (hashed) - for common words
    features.push((feature_hash("surf", &node.surface), 1.0));

    // 10. Previous word POS
    if position > 0 {
        let prev = &nodes[position - 1];
        let prev_pos = pos_to_id(&prev.pos);
        features.push((feature_hash_num("prev_pos", prev_pos), 1.0));
        // Previous word POS detail1
        features.push((feature_hash("prev_d1", &prev.pos_detail1), 1.0));
    } else {
        features.push((feature_hash("prev_pos", "NONE"), 1.0));
    }

    // 11. Next word POS
    if position + 1 < len {
        let next = &nodes[position + 1];
        let next_pos = pos_to_id(&next.pos);
        features.push((feature_hash_num("next_pos", next_pos), 1.0));
        // Next word POS detail1
        features.push((feature_hash("next_d1", &next.pos_detail1), 1.0));
    } else {
        features.push((feature_hash("next_pos", "NONE"), 1.0));
    }

    // 12. Position in sentence
    let sent_pos = if position == 0 {
        SentencePosition::Beginning
    } else if position == len - 1 {
        SentencePosition::End
    } else {
        SentencePosition::Middle
    };
    features.push((feature_hash("spos", sent_pos.as_str()), 1.0));

    // 13. Function word / content word
    if node.pos.is_function_word() {
        features.push((feature_hash("wtype", "func"), 1.0));
    } else if node.pos.is_content_word() {
        features.push((feature_hash("wtype", "cont"), 1.0));
    } else {
        features.push((feature_hash("wtype", "other"), 1.0));
    }

    // 14. Conjugation type (detailed)
    if node.ctype != "*" && !node.ctype.is_empty() {
        features.push((feature_hash("ctype", "present"), 1.0));
        features.push((feature_hash("ctype_v", &node.ctype), 1.0));
    } else {
        features.push((feature_hash("ctype", "absent"), 1.0));
    }

    // 15. Conjugation form (detailed)
    if node.cform != "*" && !node.cform.is_empty() {
        features.push((feature_hash("cform", "present"), 1.0));
        features.push((feature_hash("cform_v", &node.cform), 1.0));
    } else {
        features.push((feature_hash("cform", "absent"), 1.0));
    }

    // 16. Conjugation type + form conjunction
    if node.ctype != "*" && !node.ctype.is_empty() && node.cform != "*" && !node.cform.is_empty() {
        let ct_cf = format!("{}_{}", node.ctype, node.cform);
        features.push((feature_hash("ctcf", &ct_cf), 1.0));
    }

    // 17. Bigram: POS pair with previous
    if position > 0 {
        let prev_pos = pos_to_id(&nodes[position - 1].pos);
        let bigram = format!("{}_{}", prev_pos, pos_id);
        features.push((feature_hash("pos_bi", &bigram), 1.0));
    }

    // 18. Bigram: POS pair with next
    if position + 1 < len {
        let next_pos = pos_to_id(&nodes[position + 1].pos);
        let bigram = format!("{}_{}", pos_id, next_pos);
        features.push((feature_hash("pos_bi_n", &bigram), 1.0));
    }

    // 19. Kana group of last character of reading
    if let Some(last_ch) = node.reading.chars().last() {
        let last_str = last_ch.to_string();
        features.push((feature_hash("kgrp_end", kana_group(&last_str)), 1.0));
    }

    // 20. Pronunciation differs from reading (long vowel expansion marker)
    if node.pronunciation != node.reading && !node.pronunciation.is_empty() && node.pronunciation != "*" {
        features.push((feature_hash("pron_diff", "yes"), 1.0));
    }

    // === New features ===

    // 21. Trigram POS features (prev-current-next)
    if position > 0 && position + 1 < len {
        let prev_pos = pos_to_id(&nodes[position - 1].pos);
        let next_pos = pos_to_id(&nodes[position + 1].pos);
        let trigram = format!("{}_{}_{}", prev_pos, pos_id, next_pos);
        features.push((feature_hash("pos_tri", &trigram), 1.0));
    }

    // 22. Reading suffix features (last 1-2 mora strongly correlate with accent)
    if !node.reading.is_empty() {
        let suffix1 = reading_suffix(&node.reading, 1);
        features.push((feature_hash("rsuf1", &suffix1), 1.0));
        let suffix2 = reading_suffix(&node.reading, 2);
        features.push((feature_hash("rsuf2", &suffix2), 1.0));
    }

    // 23. Word length feature (number of characters in surface)
    let surface_len = node.surface.chars().count().min(15);
    features.push((feature_hash_num("slen", surface_len), 1.0));

    // 24. Morpheme boundary features (compound word detection)
    // A function word following a content word suggests a phrase boundary
    if position > 0 {
        let prev = &nodes[position - 1];
        if prev.pos.is_content_word() && node.pos.is_function_word() {
            features.push((feature_hash("bnd", "cont_func"), 1.0));
        }
        if prev.pos.is_content_word() && node.pos.is_content_word() {
            features.push((feature_hash("bnd", "cont_cont"), 1.0));
        }
        if prev.pos.is_function_word() && node.pos.is_content_word() {
            features.push((feature_hash("bnd", "func_cont"), 1.0));
        }
        // Prefix detection (接頭詞 followed by content word)
        if prev.pos == Pos::Settoushi {
            features.push((feature_hash("bnd", "prefix"), 1.0));
        }
    }
    if position + 1 < len {
        let next = &nodes[position + 1];
        // Suffix detection (content word followed by 接尾辞-like subcategory)
        if node.pos.is_content_word() && next.pos_detail1.contains("接尾") {
            features.push((feature_hash("bnd", "suffix_next"), 1.0));
        }
    }

    // 25. Reading prefix (first 2 chars of reading)
    {
        let chars: Vec<char> = node.reading.chars().collect();
        if chars.len() >= 2 {
            let prefix2: String = chars[..2].iter().collect();
            features.push((feature_hash("rpfx2", &prefix2), 1.0));
        }
    }

    // 26. POS + mora conjunction (accent type strongly depends on POS + length)
    {
        let pos_mora = format!("{}_{}", pos_id, mora);
        features.push((feature_hash("pos_mora", &pos_mora), 1.0));
    }

    // 27. Surface + POS conjunction for common short words
    if surface_len <= 3 {
        let sp = format!("{}_{}", node.surface, pos_id);
        features.push((feature_hash("sp", &sp), 1.0));
    }

    // 28. Previous word's reading suffix + current POS (cross-word accent interaction)
    if position > 0 {
        let prev = &nodes[position - 1];
        if !prev.reading.is_empty() {
            let prev_suf = reading_suffix(&prev.reading, 1);
            let combo = format!("{}_{}", prev_suf, pos_id);
            features.push((feature_hash("prs_cp", &combo), 1.0));
        }
    }

    // 29. Lemma feature (base form often determines accent)
    if !node.lemma.is_empty() && node.lemma != "*" {
        features.push((feature_hash("lemma", &node.lemma), 1.0));
    }

    // 30. Relative position in sentence (bucketed)
    if len > 1 {
        let rel_pos = (position * 4) / (len - 1); // 0..4
        features.push((feature_hash_num("rpos", rel_pos), 1.0));
    }

    features
}

/// CRFによるアクセント型予測器
pub struct CrfAccentPredictor {
    /// 特徴量重み: weights[label * FEATURE_DIM + feature_index]
    weights: Vec<f32>,
    /// 遷移スコア: transition[prev_label][cur_label]
    transition: Vec<Vec<f32>>,
    /// ラベル数
    num_labels: usize,
}

impl CrfAccentPredictor {
    /// バイナリ重みファイルからCRFモデルを読み込む
    pub fn new(weights_path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        CrfTrainer::load_weights(weights_path)
    }

    /// 特徴量とラベル列に対するスコアを計算する
    pub fn score_sequence(&self, features: &[Vec<(usize, f32)>], labels: &[usize]) -> f32 {
        let mut score = 0.0f32;
        for (t, feats) in features.iter().enumerate() {
            let label = labels[t];
            // Emission score
            for &(idx, val) in feats {
                score += self.weights[label * FEATURE_DIM + idx] * val;
            }
            // Transition score
            if t > 0 {
                score += self.transition[labels[t - 1]][label];
            }
        }
        score
    }

    /// ビタビデコーディングによる最適ラベル列の推定
    pub fn viterbi_decode(&self, emissions: &[Vec<f32>]) -> Vec<usize> {
        let seq_len = emissions.len();
        if seq_len == 0 {
            return Vec::new();
        }

        let num_labels = self.num_labels;

        // viterbi[t][l] = best score ending at time t with label l
        let mut viterbi = vec![vec![f32::NEG_INFINITY; num_labels]; seq_len];
        let mut backptr = vec![vec![0usize; num_labels]; seq_len];

        // Initialize t=0
        for l in 0..num_labels {
            viterbi[0][l] = emissions[0][l];
        }

        // Forward pass
        for t in 1..seq_len {
            for l in 0..num_labels {
                let mut best_score = f32::NEG_INFINITY;
                let mut best_prev = 0;
                for (p, &prev_score) in viterbi[t - 1].iter().enumerate() {
                    let s = prev_score + self.transition[p][l] + emissions[t][l];
                    if s > best_score {
                        best_score = s;
                        best_prev = p;
                    }
                }
                viterbi[t][l] = best_score;
                backptr[t][l] = best_prev;
            }
        }

        // Backtrace
        let mut path = vec![0usize; seq_len];
        let mut best_last = 0;
        let mut best_score = f32::NEG_INFINITY;
        for (l, &score) in viterbi[seq_len - 1].iter().enumerate() {
            if score > best_score {
                best_score = score;
                best_last = l;
            }
        }
        path[seq_len - 1] = best_last;
        for t in (0..seq_len - 1).rev() {
            path[t] = backptr[t + 1][path[t + 1]];
        }

        path
    }

    /// ノード列から各位置の emission スコアを計算する
    fn compute_emissions(&self, nodes: &[NjdNode]) -> Vec<Vec<f32>> {
        let mut emissions = Vec::with_capacity(nodes.len());
        for pos in 0..nodes.len() {
            let feats = extract_features(nodes, pos);
            let mut scores = vec![0.0f32; self.num_labels];
            for (label, score) in scores.iter_mut().enumerate() {
                for &(idx, val) in &feats {
                    *score += self.weights[label * FEATURE_DIM + idx] * val;
                }
            }
            emissions.push(scores);
        }
        emissions
    }
}

impl AccentPredictor for CrfAccentPredictor {
    fn predict(&self, nodes: &[NjdNode]) -> Vec<u8> {
        if nodes.is_empty() {
            return Vec::new();
        }
        let emissions = self.compute_emissions(nodes);
        let path = self.viterbi_decode(&emissions);
        path.into_iter().map(|l| l as u8).collect()
    }
}

/// 学習データの一例
pub struct TrainingExample {
    /// 入力ノード列
    pub nodes: Vec<NjdNode>,
    /// 正解アクセント型列
    pub labels: Vec<u8>,
}

/// CRFモデルの学習器
pub struct CrfTrainer {
    learning_rate: f32,
    num_epochs: usize,
    l2_reg: f32,
}

impl CrfTrainer {
    /// 学習器を構築する
    pub fn new(learning_rate: f32, num_epochs: usize) -> Self {
        Self {
            learning_rate,
            num_epochs,
            l2_reg: 0.01,
        }
    }

    /// L2正則化係数を設定する
    pub fn with_l2_reg(mut self, l2_reg: f32) -> Self {
        self.l2_reg = l2_reg;
        self
    }

    /// 学習データからCRFモデルを学習する
    ///
    /// 平均化構造パーセプトロンアルゴリズムで重みを更新する。
    /// 学習率減衰と重み平均化により汎化性能を向上させる。
    pub fn train(&self, data: &[TrainingExample]) -> CrfAccentPredictor {
        let num_labels = DEFAULT_NUM_LABELS;
        let weight_size = num_labels * FEATURE_DIM;
        let mut weights = vec![0.0f32; weight_size];
        let mut transition = vec![vec![0.0f32; num_labels]; num_labels];

        // Averaged perceptron: accumulate sum of all weight snapshots
        let mut weights_sum = vec![0.0f32; weight_size];
        let mut transition_sum = vec![vec![0.0f32; num_labels]; num_labels];
        let mut update_count: u64 = 0;

        // Pre-extract features for all examples (avoids redundant computation)
        let all_example_feats: Vec<Vec<Vec<(usize, f32)>>> = data
            .iter()
            .map(|example| {
                (0..example.nodes.len())
                    .map(|pos| extract_features(&example.nodes, pos))
                    .collect()
            })
            .collect();

        for epoch in 0..self.num_epochs {
            // Learning rate decay: lr * (1 - epoch / total_epochs)
            let decay = 1.0 - (epoch as f32 / self.num_epochs as f32);
            let lr = self.learning_rate * decay;

            let mut epoch_correct = 0usize;
            let mut epoch_total = 0usize;

            for (ex_idx, example) in data.iter().enumerate() {
                let gold_labels: Vec<usize> =
                    example.labels.iter().map(|&l| (l as usize).min(num_labels - 1)).collect();

                let all_feats = &all_example_feats[ex_idx];

                // Compute emissions inline (avoids cloning weights into a predictor)
                let mut emissions = Vec::with_capacity(example.nodes.len());
                for feats in all_feats.iter() {
                    let mut scores = vec![0.0f32; num_labels];
                    for (label, score) in scores.iter_mut().enumerate() {
                        for &(idx, val) in feats {
                            *score += weights[label * FEATURE_DIM + idx] * val;
                        }
                    }
                    emissions.push(scores);
                }

                // Inline viterbi decode (avoids allocating a predictor)
                let predicted = {
                    let seq_len = emissions.len();
                    if seq_len == 0 {
                        Vec::new()
                    } else {
                        let mut viterbi_scores = vec![vec![f32::NEG_INFINITY; num_labels]; seq_len];
                        let mut backptr = vec![vec![0usize; num_labels]; seq_len];

                        for l in 0..num_labels {
                            viterbi_scores[0][l] = emissions[0][l];
                        }
                        for t in 1..seq_len {
                            for l in 0..num_labels {
                                let mut best_score = f32::NEG_INFINITY;
                                let mut best_prev = 0;
                                for p in 0..num_labels {
                                    let s = viterbi_scores[t - 1][p] + transition[p][l] + emissions[t][l];
                                    if s > best_score {
                                        best_score = s;
                                        best_prev = p;
                                    }
                                }
                                viterbi_scores[t][l] = best_score;
                                backptr[t][l] = best_prev;
                            }
                        }

                        let mut path = vec![0usize; seq_len];
                        let mut best_last = 0;
                        let mut best_s = f32::NEG_INFINITY;
                        for (l, &score) in viterbi_scores[seq_len - 1].iter().enumerate() {
                            if score > best_s {
                                best_s = score;
                                best_last = l;
                            }
                        }
                        path[seq_len - 1] = best_last;
                        for t in (0..seq_len - 1).rev() {
                            path[t] = backptr[t + 1][path[t + 1]];
                        }
                        path
                    }
                };

                // Track accuracy
                for (t, &gold) in gold_labels.iter().enumerate() {
                    if predicted[t] == gold {
                        epoch_correct += 1;
                    }
                    epoch_total += 1;
                }

                // Update weights: promote gold, demote predicted
                let mut any_update = false;
                for (t, feats) in all_feats.iter().enumerate() {
                    let gold = gold_labels[t];
                    let pred = predicted[t];
                    if gold != pred {
                        any_update = true;
                        for &(idx, val) in feats {
                            weights[gold * FEATURE_DIM + idx] += lr * val;
                            weights[pred * FEATURE_DIM + idx] -= lr * val;
                        }
                    }
                    // Transition updates
                    if t > 0 {
                        let prev_gold = gold_labels[t - 1];
                        let prev_pred = predicted[t - 1];
                        if gold != pred || prev_gold != prev_pred {
                            transition[prev_gold][gold] += lr;
                            transition[prev_pred][pred] -= lr;
                        }
                    }
                }

                // Accumulate weights for averaging (after each example update)
                if any_update {
                    for (i, &w) in weights.iter().enumerate() {
                        weights_sum[i] += w;
                    }
                    for i in 0..num_labels {
                        for j in 0..num_labels {
                            transition_sum[i][j] += transition[i][j];
                        }
                    }
                    update_count += 1;
                } else {
                    // Even if no update, the current weights contribute to the average
                    for (i, &w) in weights.iter().enumerate() {
                        weights_sum[i] += w;
                    }
                    for i in 0..num_labels {
                        for j in 0..num_labels {
                            transition_sum[i][j] += transition[i][j];
                        }
                    }
                    update_count += 1;
                }
            }

            // L2 regularization (applied per-epoch with decayed rate)
            let reg_factor = 1.0 - lr * self.l2_reg;
            for w in &mut weights {
                *w *= reg_factor;
            }
            for row in &mut transition {
                for w in row.iter_mut() {
                    *w *= reg_factor;
                }
            }

            let acc = if epoch_total > 0 {
                epoch_correct as f64 / epoch_total as f64
            } else {
                0.0
            };
            eprintln!(
                "  Epoch {}/{}: accuracy={:.2}%, lr={:.4}",
                epoch + 1,
                self.num_epochs,
                acc * 100.0,
                lr
            );
        }

        // Use averaged weights for better generalization
        if update_count > 0 {
            let divisor = update_count as f32;
            for (i, w) in weights_sum.iter_mut().enumerate() {
                weights[i] = *w / divisor;
            }
            for i in 0..num_labels {
                for j in 0..num_labels {
                    transition[i][j] = transition_sum[i][j] / divisor;
                }
            }
        }

        CrfAccentPredictor {
            weights,
            transition,
            num_labels,
        }
    }

    /// モデルの重みをバイナリファイルに保存する
    ///
    /// フォーマット:
    /// - 4 bytes: num_features (u32, = FEATURE_DIM)
    /// - 4 bytes: num_labels (u32)
    /// - num_features * num_labels * 4 bytes: weights (f32, little-endian)
    /// - num_labels * num_labels * 4 bytes: transition (f32, little-endian)
    pub fn save_weights(
        predictor: &CrfAccentPredictor,
        path: &Path,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut file = std::fs::File::create(path)?;
        let num_features = FEATURE_DIM as u32;
        let num_labels = predictor.num_labels as u32;

        file.write_all(&num_features.to_le_bytes())?;
        file.write_all(&num_labels.to_le_bytes())?;

        for &w in &predictor.weights {
            file.write_all(&w.to_le_bytes())?;
        }
        for row in &predictor.transition {
            for &w in row {
                file.write_all(&w.to_le_bytes())?;
            }
        }

        Ok(())
    }

    /// バイナリファイルからモデルの重みを読み込む
    pub fn load_weights(path: &Path) -> Result<CrfAccentPredictor, Box<dyn std::error::Error>> {
        let mut file = std::fs::File::open(path)?;

        let mut buf4 = [0u8; 4];
        file.read_exact(&mut buf4)?;
        let num_features = u32::from_le_bytes(buf4) as usize;

        file.read_exact(&mut buf4)?;
        let num_labels = u32::from_le_bytes(buf4) as usize;

        if num_features != FEATURE_DIM {
            return Err(format!(
                "Feature dimension mismatch: expected {FEATURE_DIM}, got {num_features}"
            )
            .into());
        }

        let weight_count = num_features * num_labels;
        let mut weights = vec![0.0f32; weight_count];
        for w in &mut weights {
            file.read_exact(&mut buf4)?;
            *w = f32::from_le_bytes(buf4);
        }

        let mut transition = vec![vec![0.0f32; num_labels]; num_labels];
        for row in &mut transition {
            for w in row.iter_mut() {
                file.read_exact(&mut buf4)?;
                *w = f32::from_le_bytes(buf4);
            }
        }

        Ok(CrfAccentPredictor {
            weights,
            transition,
            num_labels,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::njd::InputToken;

    fn make_node(surface: &str, pos: &str, reading: &str) -> NjdNode {
        let token = InputToken::new(surface, pos, reading, reading);
        NjdNode::from_token(&token)
    }

    #[test]
    fn test_feature_extraction_basic() {
        let nodes = vec![
            make_node("猫", "名詞", "ネコ"),
            make_node("が", "助詞", "ガ"),
            make_node("走る", "動詞", "ハシル"),
        ];

        let feats0 = extract_features(&nodes, 0);
        let feats1 = extract_features(&nodes, 1);
        let feats2 = extract_features(&nodes, 2);

        // Each position should produce features
        assert!(!feats0.is_empty());
        assert!(!feats1.is_empty());
        assert!(!feats2.is_empty());

        // All feature indices should be within FEATURE_DIM
        for &(idx, _) in &feats0 {
            assert!(idx < FEATURE_DIM);
        }

        // Features at different positions should differ
        assert_ne!(feats0, feats1);
    }

    #[test]
    fn test_feature_extraction_position() {
        let nodes = vec![
            make_node("今日", "名詞", "キョウ"),
            make_node("は", "助詞", "ワ"),
            make_node("天気", "名詞", "テンキ"),
        ];

        // Beginning position
        let feats_begin = extract_features(&nodes, 0);
        // End position
        let feats_end = extract_features(&nodes, 2);

        // They should contain sentence position features that differ
        assert_ne!(feats_begin, feats_end);
    }

    #[test]
    fn test_viterbi_decode_known_weights() {
        let num_labels = 3;
        let weights = vec![0.0f32; num_labels * FEATURE_DIM];
        let transition = vec![vec![0.0f32; num_labels]; num_labels];

        let predictor = CrfAccentPredictor {
            weights,
            transition,
            num_labels,
        };

        // With uniform weights, viterbi should still produce a valid path.
        // Use explicit emissions for testing.
        let emissions = vec![
            vec![1.0, 0.5, 0.2],  // label 0 is best
            vec![0.1, 2.0, 0.3],  // label 1 is best
            vec![0.0, 0.1, 3.0],  // label 2 is best
        ];

        let path = predictor.viterbi_decode(&emissions);
        assert_eq!(path.len(), 3);
        assert_eq!(path[0], 0);
        assert_eq!(path[1], 1);
        assert_eq!(path[2], 2);
    }

    #[test]
    fn test_viterbi_decode_with_transitions() {
        let num_labels = 2;
        let weights = vec![0.0f32; num_labels * FEATURE_DIM];

        // Strong transition preference: 0->0 and 1->1 (self-loops)
        let mut transition = vec![vec![0.0f32; num_labels]; num_labels];
        transition[0][0] = 10.0;
        transition[1][1] = 10.0;
        transition[0][1] = -10.0;
        transition[1][0] = -10.0;

        let predictor = CrfAccentPredictor {
            weights,
            transition,
            num_labels,
        };

        // Emissions strongly favor label 0 at start, slightly favor label 1 later
        // With strong self-transitions, path should stay on label 0
        let emissions = vec![
            vec![100.0, 0.0],
            vec![0.5, 1.0],
            vec![0.5, 1.0],
        ];

        let path = predictor.viterbi_decode(&emissions);
        assert_eq!(path.len(), 3);
        // Initial strong emission for label 0 + self-transition bonus keeps it on 0
        assert_eq!(path[0], 0);
        assert_eq!(path[1], 0);
        assert_eq!(path[2], 0);
    }

    #[test]
    fn test_viterbi_decode_empty() {
        let predictor = CrfAccentPredictor {
            weights: vec![0.0; DEFAULT_NUM_LABELS * FEATURE_DIM],
            transition: vec![vec![0.0; DEFAULT_NUM_LABELS]; DEFAULT_NUM_LABELS],
            num_labels: DEFAULT_NUM_LABELS,
        };

        let path = predictor.viterbi_decode(&[]);
        assert!(path.is_empty());
    }

    #[test]
    fn test_score_computation() {
        let num_labels = 2;
        let mut weights = vec![0.0f32; num_labels * FEATURE_DIM];
        // Set a known weight
        weights[0 * FEATURE_DIM + 42] = 1.5; // label=0, feature=42
        weights[1 * FEATURE_DIM + 42] = -0.5; // label=1, feature=42

        let mut transition = vec![vec![0.0f32; num_labels]; num_labels];
        transition[0][1] = 2.0;

        let predictor = CrfAccentPredictor {
            weights,
            transition,
            num_labels,
        };

        let features = vec![
            vec![(42, 1.0f32)],
            vec![(42, 1.0f32)],
        ];

        // Score for labels [0, 1]: emission(0,42)*1.0 + emission(1,42)*1.0 + transition[0][1]
        // = 1.5 + (-0.5) + 2.0 = 3.0
        let score_01 = predictor.score_sequence(&features, &[0, 1]);
        assert!((score_01 - 3.0).abs() < 1e-6);

        // Score for labels [0, 0]: emission(0,42)*1.0 + emission(0,42)*1.0 + transition[0][0]
        // = 1.5 + 1.5 + 0.0 = 3.0
        let score_00 = predictor.score_sequence(&features, &[0, 0]);
        assert!((score_00 - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_weight_save_load_roundtrip() {
        let num_labels = 3;
        let mut weights = vec![0.0f32; num_labels * FEATURE_DIM];
        weights[0] = 1.0;
        weights[FEATURE_DIM] = -2.5;
        weights[2 * FEATURE_DIM + 999] = 0.123;

        let mut transition = vec![vec![0.0f32; num_labels]; num_labels];
        transition[0][1] = 0.5;
        transition[2][0] = -1.0;

        let predictor = CrfAccentPredictor {
            weights: weights.clone(),
            transition: transition.clone(),
            num_labels,
        };

        let dir = std::env::temp_dir();
        let path = dir.join("crf_test_weights.bin");

        CrfTrainer::save_weights(&predictor, &path).unwrap();
        let loaded = CrfTrainer::load_weights(&path).unwrap();

        assert_eq!(loaded.num_labels, num_labels);
        assert_eq!(loaded.weights.len(), weights.len());
        for (a, b) in loaded.weights.iter().zip(weights.iter()) {
            assert!((a - b).abs() < 1e-7);
        }
        for i in 0..num_labels {
            for j in 0..num_labels {
                assert!((loaded.transition[i][j] - transition[i][j]).abs() < 1e-7);
            }
        }

        // Cleanup
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_crf_as_accent_predictor() {
        // Create a zero-weight predictor (will predict label 0 for everything)
        let predictor = CrfAccentPredictor {
            weights: vec![0.0; DEFAULT_NUM_LABELS * FEATURE_DIM],
            transition: vec![vec![0.0; DEFAULT_NUM_LABELS]; DEFAULT_NUM_LABELS],
            num_labels: DEFAULT_NUM_LABELS,
        };

        let nodes = vec![
            make_node("猫", "名詞", "ネコ"),
            make_node("が", "助詞", "ガ"),
        ];

        let result = predictor.predict(&nodes);
        assert_eq!(result.len(), 2);
        // With all-zero weights, all emissions are 0, so label 0 wins (first checked)
        assert_eq!(result[0], 0);
        assert_eq!(result[1], 0);
    }

    #[test]
    fn test_crf_predict_empty() {
        let predictor = CrfAccentPredictor {
            weights: vec![0.0; DEFAULT_NUM_LABELS * FEATURE_DIM],
            transition: vec![vec![0.0; DEFAULT_NUM_LABELS]; DEFAULT_NUM_LABELS],
            num_labels: DEFAULT_NUM_LABELS,
        };

        let result = predictor.predict(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_trainer_basic() {
        let mut node1 = make_node("猫", "名詞", "ネコ");
        node1.accent_type = 1;
        let mut node2 = make_node("が", "助詞", "ガ");
        node2.accent_type = 0;

        let example = TrainingExample {
            nodes: vec![node1, node2],
            labels: vec![1, 0],
        };

        let trainer = CrfTrainer::new(0.1, 5);
        let predictor = trainer.train(&[example]);

        // The trained model should have non-zero weights
        let has_nonzero = predictor.weights.iter().any(|&w| w != 0.0);
        assert!(has_nonzero);
    }

    #[test]
    fn test_feature_hash_deterministic() {
        let h1 = feature_hash("pos", "名詞");
        let h2 = feature_hash("pos", "名詞");
        let h3 = feature_hash("pos", "動詞");
        assert_eq!(h1, h2);
        assert_ne!(h1, h3);
        assert!(h1 < FEATURE_DIM);
    }

    #[test]
    fn test_kana_group() {
        assert_eq!(kana_group("ネコ"), "ナ行");
        assert_eq!(kana_group("カメ"), "カ行");
        assert_eq!(kana_group("アリ"), "ア行");
        assert_eq!(kana_group(""), "空");
    }
}
