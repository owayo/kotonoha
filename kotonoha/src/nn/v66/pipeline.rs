//! v66 系の 12 ONNX 推論パイプライン
//!
//! 訓練側 `precompute_v66_stacker.py:99-164` と
//! `v66_exact_memory.py:227-253` を Rust に移植したもの。
//!
//! ## 段階
//!
//! 1. `feat13` を抽出 ([seq, 13])
//! 2. v24(`feat13[:11]`) → v24_log ([seq, 21])、v24_arg = argmax(-1)
//! 3. `feat14 = concat(feat13, v24_arg/20)` ([seq, 14])
//! 4. 9 student の softmax を計算 ([9, seq, 21])
//! 5. 各 token で `meta_9 = _meta_features(9 student の softmax stack)` ([5])
//! 6. `feat24 = concat(feat13, v24_arg/20, teacher_soft_stats(v24_log,dict_acc),
//!    meta_9)` ([seq, 24])
//! 7. v61, v63 の softmax ([seq, 21])
//! 8. 11 model の softmax stack [11, seq, 21] から
//!    `stacker84 = mean[21] + std[21] + vote_hist[21] + sm_v63[21]` ([seq, 84])
//! 9. `feat103 = concat(feat13, v24_arg/20, teacher_soft_stats, stacker84)`
//!    ([seq, 103])
//! 10. v66_split1(`feat103`) → argmax がアクセント類予測

use std::sync::{Arc, Mutex};

use ort::session::Session;
use ort::value::Tensor;

use super::bundle::V66Bundle;
use super::enrich::{resolve_dict_accents, resolve_one};
use super::features::{FEATURE13_DIM, MorphemeView, extract_feat13, morpheme_dict_accent};
use super::math::{argmax, meta_features, softmax, teacher_soft_stats};
use super::vocab::NUM_CLASSES;
use crate::nn::{ContextualAccentPredictor, FeatureMorpheme};

const FEATURE14_DIM: usize = 14;
const FEATURE24_DIM: usize = 24;
const FEATURE103_DIM: usize = 103;
const STACKER_DIM: usize = 84;
const NUM_STUDENTS: usize = 9;

/// v66 系推論パイプライン
///
/// `V66Bundle` を共有所有し、各 ONNX セッションへ排他アクセスして推論する。
/// 単一 instance を多スレッドから呼び出し可能 (`Send + Sync`)。
pub struct V66Pipeline {
    bundle: Arc<V66Bundle>,
}

impl V66Pipeline {
    /// バンドルから新しい推論器を作る
    pub fn new(bundle: V66Bundle) -> Self {
        Self {
            bundle: Arc::new(bundle),
        }
    }

    /// 推論を実行する (失敗時は `V66PredictError`)
    pub fn predict(
        &self,
        ctx: &[FeatureMorpheme<'_>],
    ) -> Result<Vec<u8>, V66PredictError> {
        let dict_accents: Vec<Option<u8>> = ctx
            .iter()
            .map(|m| {
                resolve_one(
                    &m.token.surface,
                    &m.token.lemma,
                    &m.token.reading,
                    &m.token.pronunciation,
                    &self.bundle.accent_dict,
                )
            })
            .collect();
        self.predict_with_dict_accents(ctx, &dict_accents)
    }

    /// 推論を実行する、`dict_accent_type` は事前計算値を使う
    ///
    /// 訓練データと完全一致するアクセント辞書値を渡したい場合 (テスト時) に使う。
    pub fn predict_with_dict_accents(
        &self,
        ctx: &[FeatureMorpheme<'_>],
        dict_accents: &[Option<u8>],
    ) -> Result<Vec<u8>, V66PredictError> {
        let seq_len = ctx.len();
        if seq_len == 0 {
            return Ok(Vec::new());
        }
        debug_assert_eq!(dict_accents.len(), seq_len);

        let mut feat13 = vec![0f32; seq_len * FEATURE13_DIM];
        for (i, m) in ctx.iter().enumerate() {
            let position = if seq_len > 1 {
                i as f32 / (seq_len - 1) as f32
            } else {
                0.0
            };
            let view = MorphemeView {
                surface: &m.token.surface,
                pos: &m.token.pos,
                pos_detail1: &m.token.pos_detail1,
                pos_detail2: &m.token.pos_detail2,
                conjugation_type: &m.token.ctype,
                conjugation_form: &m.token.cform,
                reading: &m.token.reading,
                dict_accent_type: dict_accents[i],
            };
            let row = extract_feat13(&view, position);
            feat13[i * FEATURE13_DIM..(i + 1) * FEATURE13_DIM].copy_from_slice(&row);
        }

        // ── 2) v24 推論 (feat13[:, :11]) ────────────────────────────────────
        let mut feat11 = vec![0f32; seq_len * 11];
        for i in 0..seq_len {
            feat11[i * 11..(i + 1) * 11]
                .copy_from_slice(&feat13[i * FEATURE13_DIM..i * FEATURE13_DIM + 11]);
        }
        let v24_logits = run_session(&self.bundle.models.v24, feat11, seq_len, 11)?;
        debug_assert_eq!(v24_logits.len(), seq_len * NUM_CLASSES);

        // teacher_soft_stats と v24_arg を per-token で用意
        let mut v24_arg_norm = vec![0f32; seq_len];
        let mut teacher_stats = vec![0f32; seq_len * 5];
        for i in 0..seq_len {
            let row = &v24_logits[i * NUM_CLASSES..(i + 1) * NUM_CLASSES];
            let arg = argmax(row);
            v24_arg_norm[i] = arg as f32 / 20.0;
            let dict_acc = morpheme_dict_accent(dict_accents[i]);
            let stats = teacher_soft_stats(row, dict_acc);
            teacher_stats[i * 5..(i + 1) * 5].copy_from_slice(&stats);
        }

        // ── 3) feat14 を組み立て、9 student の softmax を計算 ───────────────
        let mut feat14 = vec![0f32; seq_len * FEATURE14_DIM];
        for i in 0..seq_len {
            let dst = &mut feat14[i * FEATURE14_DIM..(i + 1) * FEATURE14_DIM];
            dst[..FEATURE13_DIM]
                .copy_from_slice(&feat13[i * FEATURE13_DIM..(i + 1) * FEATURE13_DIM]);
            dst[FEATURE13_DIM] = v24_arg_norm[i];
        }
        // 9 student の softmax を [9, seq, NUM_CLASSES] で保持
        let mut student_sm = vec![0f32; NUM_STUDENTS * seq_len * NUM_CLASSES];
        for (s_idx, sess) in self.bundle.models.students.iter().enumerate() {
            let logits = run_session(sess, feat14.clone(), seq_len, FEATURE14_DIM)?;
            for i in 0..seq_len {
                let src = &logits[i * NUM_CLASSES..(i + 1) * NUM_CLASSES];
                let sm = softmax(src);
                let dst_off =
                    s_idx * seq_len * NUM_CLASSES + i * NUM_CLASSES;
                student_sm[dst_off..dst_off + NUM_CLASSES].copy_from_slice(&sm);
            }
        }

        // meta_9 を per-token で計算
        let mut meta9 = vec![0f32; seq_len * 5];
        for i in 0..seq_len {
            let mut stack = [[0f32; NUM_CLASSES]; NUM_STUDENTS];
            for (s_idx, slot) in stack.iter_mut().enumerate() {
                let off = s_idx * seq_len * NUM_CLASSES + i * NUM_CLASSES;
                slot.copy_from_slice(&student_sm[off..off + NUM_CLASSES]);
            }
            let m = meta_features(&stack);
            meta9[i * 5..(i + 1) * 5].copy_from_slice(&m);
        }

        // ── 4) feat24 を組み立て、v61/v63 の softmax を計算 ─────────────────
        let mut feat24 = vec![0f32; seq_len * FEATURE24_DIM];
        for i in 0..seq_len {
            let dst = &mut feat24[i * FEATURE24_DIM..(i + 1) * FEATURE24_DIM];
            dst[..FEATURE13_DIM]
                .copy_from_slice(&feat13[i * FEATURE13_DIM..(i + 1) * FEATURE13_DIM]);
            dst[13] = v24_arg_norm[i];
            dst[14..19].copy_from_slice(&teacher_stats[i * 5..(i + 1) * 5]);
            dst[19..24].copy_from_slice(&meta9[i * 5..(i + 1) * 5]);
        }
        let v61_logits =
            run_session(&self.bundle.models.v61, feat24.clone(), seq_len, FEATURE24_DIM)?;
        let v63_logits = run_session(&self.bundle.models.v63, feat24, seq_len, FEATURE24_DIM)?;
        let mut v61_sm = vec![0f32; seq_len * NUM_CLASSES];
        let mut v63_sm = vec![0f32; seq_len * NUM_CLASSES];
        for i in 0..seq_len {
            let off = i * NUM_CLASSES;
            v61_sm[off..off + NUM_CLASSES]
                .copy_from_slice(&softmax(&v61_logits[off..off + NUM_CLASSES]));
            v63_sm[off..off + NUM_CLASSES]
                .copy_from_slice(&softmax(&v63_logits[off..off + NUM_CLASSES]));
        }

        // ── 5) 11 model の softmax stack から stacker84 を構築 ──────────────
        let mut stacker = vec![0f32; seq_len * STACKER_DIM];
        for i in 0..seq_len {
            // 11 softmax をまとめる
            let mut all_sm = [[0f32; NUM_CLASSES]; 11];
            for (s_idx, slot) in all_sm.iter_mut().take(NUM_STUDENTS).enumerate() {
                let off = s_idx * seq_len * NUM_CLASSES + i * NUM_CLASSES;
                slot.copy_from_slice(&student_sm[off..off + NUM_CLASSES]);
            }
            let off_i = i * NUM_CLASSES;
            all_sm[9].copy_from_slice(&v61_sm[off_i..off_i + NUM_CLASSES]);
            all_sm[10].copy_from_slice(&v63_sm[off_i..off_i + NUM_CLASSES]);

            // mean[21]
            let mut mean = [0f32; NUM_CLASSES];
            for sm in &all_sm {
                for (c, &p) in sm.iter().enumerate() {
                    mean[c] += p;
                }
            }
            for v in &mut mean {
                *v /= 11.0;
            }
            // std[21] (ddof=0)
            let mut std = [0f32; NUM_CLASSES];
            for sm in &all_sm {
                for (c, &p) in sm.iter().enumerate() {
                    let d = p - mean[c];
                    std[c] += d * d;
                }
            }
            for v in &mut std {
                *v = (*v / 11.0).sqrt();
            }
            // vote_hist[21] = bincount(argmax over 11 models) / 11
            let mut votes = [0f32; NUM_CLASSES];
            for sm in &all_sm {
                votes[argmax(sm)] += 1.0;
            }
            for v in &mut votes {
                *v /= 11.0;
            }
            // v63 prob (= all_sm[10])
            let v63_prob = all_sm[10];

            let row = &mut stacker[i * STACKER_DIM..(i + 1) * STACKER_DIM];
            row[0..21].copy_from_slice(&mean);
            row[21..42].copy_from_slice(&std);
            row[42..63].copy_from_slice(&votes);
            row[63..84].copy_from_slice(&v63_prob);
        }

        // ── 6) feat103 を組み立てて v66_split1 を実行 ───────────────────────
        let mut feat103 = vec![0f32; seq_len * FEATURE103_DIM];
        for i in 0..seq_len {
            let dst = &mut feat103[i * FEATURE103_DIM..(i + 1) * FEATURE103_DIM];
            dst[..FEATURE13_DIM]
                .copy_from_slice(&feat13[i * FEATURE13_DIM..(i + 1) * FEATURE13_DIM]);
            dst[13] = v24_arg_norm[i];
            dst[14..19].copy_from_slice(&teacher_stats[i * 5..(i + 1) * 5]);
            dst[19..103].copy_from_slice(&stacker[i * STACKER_DIM..(i + 1) * STACKER_DIM]);
        }
        let v66_logits = run_session(
            &self.bundle.models.v66_split1,
            feat103,
            seq_len,
            FEATURE103_DIM,
        )?;
        let mut accent_types = Vec::with_capacity(seq_len);
        for i in 0..seq_len {
            let row = &v66_logits[i * NUM_CLASSES..(i + 1) * NUM_CLASSES];
            accent_types.push(argmax(row) as u8);
        }
        Ok(accent_types)
    }
}

impl ContextualAccentPredictor for V66Pipeline {
    fn predict_with_context(&self, ctx: &[FeatureMorpheme<'_>]) -> Vec<u8> {
        match self.predict(ctx) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("V66Pipeline: predict failed: {e}; falling back to zeros");
                vec![0u8; ctx.len()]
            }
        }
    }
}

/// 単一 ONNX セッションを `[seq_len, dim]` 入力で実行し、`[seq_len, NUM_CLASSES]`
/// の出力を行優先 `Vec<f32>` で返す。
fn run_session(
    session: &Mutex<Session>,
    input_data: Vec<f32>,
    seq_len: usize,
    dim: usize,
) -> Result<Vec<f32>, V66PredictError> {
    debug_assert_eq!(input_data.len(), seq_len * dim);
    let tensor = Tensor::from_array((vec![seq_len as i64, dim as i64], input_data))
        .map_err(|e| V66PredictError::Tensor(e.to_string()))?;
    let mut sess = session
        .lock()
        .map_err(|e| V66PredictError::Mutex(e.to_string()))?;
    let outputs = sess
        .run(ort::inputs!["input" => tensor])
        .map_err(|e| V66PredictError::Run(e.to_string()))?;
    let (_shape, data) = outputs[0]
        .try_extract_tensor::<f32>()
        .map_err(|e| V66PredictError::Extract(e.to_string()))?;
    Ok(data.to_vec())
}

/// 推論時のエラー
#[derive(Debug, thiserror::Error)]
pub enum V66PredictError {
    /// 入力 tensor 構築失敗
    #[error("tensor construction failed: {0}")]
    Tensor(String),
    /// セッション mutex のロック失敗
    #[error("session mutex poisoned: {0}")]
    Mutex(String),
    /// session.run 失敗
    #[error("session.run failed: {0}")]
    Run(String),
    /// 出力 tensor 抽出失敗
    #[error("output tensor extraction failed: {0}")]
    Extract(String),
}

// resolve_dict_accents は外部モジュールでも使うので import を pull する関数を残す
#[doc(hidden)]
#[allow(dead_code)]
pub fn _ensure_resolve_dict_accents_used(
    tokens: &[crate::njd::InputToken],
    accent_dict: &crate::accent_dict::AccentDict,
) -> Vec<Option<u8>> {
    resolve_dict_accents(tokens, accent_dict)
}
