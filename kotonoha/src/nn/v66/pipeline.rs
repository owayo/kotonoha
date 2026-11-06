//! v66 系の 13 KNM 推論パイプライン
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

use std::ops::Range;
use std::sync::Arc;

use crate::nn::native::NativeModel;
use rayon::prelude::*;

use super::bundle::{STUDENT_FILENAMES, V66Bundle};
use super::enrich::resolve_one;
use super::features::{FEATURE13_DIM, MorphemeView, extract_feat13, morpheme_dict_accent};
use super::math::{argmax, meta_features, softmax, teacher_soft_stats};
use super::vocab::NUM_CLASSES;
use crate::nn::{ContextualAccentPredictor, FeatureMorpheme};

const FEATURE14_DIM: usize = 14;
const FEATURE24_DIM: usize = 24;
const FEATURE103_DIM: usize = 103;
const STACKER_DIM: usize = 84;
const NUM_STUDENTS: usize = STUDENT_FILENAMES.len();
/// teacher (v24) が読む feat13 の先頭の列数
const TEACHER_INPUT_DIM: usize = 11;
/// `teacher_soft_stats`・`meta_features` が返す値の数
const STATS_DIM: usize = 5;
/// feat14 以降で teacher の argmax/20 を置く列
const TEACHER_ARG_COLUMN: usize = FEATURE13_DIM;
/// feat24・feat103 で `teacher_soft_stats` を置く列
const TEACHER_STATS_COLUMNS: Range<usize> = FEATURE14_DIM..FEATURE14_DIM + STATS_DIM;
/// feat24 の meta_9、feat103 の stacker84 を置く先頭の列
const TAIL_COLUMN: usize = TEACHER_STATS_COLUMNS.end;

// 訓練側の concat 順と各特徴の次元の対応をコンパイル時に固定する。
const _: () = assert!(TEACHER_ARG_COLUMN + 1 == FEATURE14_DIM);
const _: () = assert!(TAIL_COLUMN + STATS_DIM == FEATURE24_DIM);
const _: () = assert!(TAIL_COLUMN + STACKER_DIM == FEATURE103_DIM);
const _: () = assert!(STACKER_DIM == 4 * NUM_CLASSES);

/// 1 形態素分のクラス確率 (softmax 後)
type Probabilities = [f32; NUM_CLASSES];

/// v66 系推論パイプライン
///
/// `V66Bundle` を共有所有し、不変の専用モデルを並列に呼んで推論する。
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
    pub fn predict(&self, ctx: &[FeatureMorpheme<'_>]) -> Result<Vec<u8>, V66PredictError> {
        let dict_accents: Vec<Option<u8>> = ctx
            .iter()
            .map(|m| resolve_one(&m.token.lemma, &m.token.reading, &self.bundle.accent_dict))
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
        let models = &self.bundle.models;
        validate_request(seq_len, dict_accents.len(), models.students.len())?;
        if seq_len == 0 {
            return Ok(Vec::new());
        }

        // ── 1) feat13 と 2) teacher (v24, feat13[:, :11]) ──────────────────
        let feat13 = morpheme_features(ctx, dict_accents);
        let teacher_input = feature_matrix(seq_len, TEACHER_INPUT_DIM, |i, row| {
            row.copy_from_slice(&feat13[i][..TEACHER_INPUT_DIM]);
        });
        let teacher = TeacherColumns::new(
            &run_session(&models.v24, &teacher_input, seq_len, TEACHER_INPUT_DIM)?,
            dict_accents,
        );

        // ── 3) feat14 = [feat13, v24_arg] で 9 student の softmax ─────────
        let feat14 = feature_matrix(seq_len, FEATURE14_DIM, |i, row| {
            teacher.write_prefix(i, &feat13[i], row);
        });
        let students = student_probabilities(&models.students, &feat14, seq_len)?;

        // ── 4) feat24 = [feat13, v24_arg, teacher_stats, meta_9] で v61/v63 ──
        let feat24 = feature_matrix(seq_len, FEATURE24_DIM, |i, row| {
            teacher.write_prefix_with_stats(i, &feat13[i], row);
            row[TAIL_COLUMN..].copy_from_slice(&meta_features(&students[i]));
        });
        let (v61, v63) = rayon::join(
            || run_session(&models.v61, &feat24, seq_len, FEATURE24_DIM),
            || run_session(&models.v63, &feat24, seq_len, FEATURE24_DIM),
        );
        let (v61, v63) = (probabilities(&v61?), probabilities(&v63?));

        // ── 5) feat103 = [feat13, v24_arg, teacher_stats, stacker84] で v66 ──
        let feat103 = feature_matrix(seq_len, FEATURE103_DIM, |i, row| {
            teacher.write_prefix_with_stats(i, &feat13[i], row);
            row[TAIL_COLUMN..].copy_from_slice(&stacker_features(&students[i], &v61[i], &v63[i]));
        });
        let logits = run_session(&models.v66_split1, &feat103, seq_len, FEATURE103_DIM)?;
        Ok(logits
            .as_chunks::<NUM_CLASSES>()
            .0
            .iter()
            .map(|row| argmax(row) as u8)
            .collect())
    }
}

/// 形態素ごとの 13 次元特徴。位置は先頭を 0、末尾を 1 に正規化する (1 語なら 0)。
fn morpheme_features(
    ctx: &[FeatureMorpheme<'_>],
    dict_accents: &[Option<u8>],
) -> Vec<[f32; FEATURE13_DIM]> {
    let last = ctx.len().saturating_sub(1);
    ctx.iter()
        .zip(dict_accents)
        .enumerate()
        .map(|(i, (m, &dict_accent_type))| {
            let position = if last > 0 {
                i as f32 / last as f32
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
                dict_accent_type,
            };
            extract_feat13(&view, position)
        })
        .collect()
}

/// 行ごとに値を書き込み、`[seq_len, dim]` の行優先の特徴行列を作る。
fn feature_matrix(seq_len: usize, dim: usize, mut fill: impl FnMut(usize, &mut [f32])) -> Vec<f32> {
    let mut matrix = vec![0f32; seq_len * dim];
    for (i, row) in matrix.chunks_exact_mut(dim).enumerate() {
        fill(i, row);
    }
    matrix
}

/// teacher (v24) の出力から作る、feat14 以降に共通する列。
struct TeacherColumns {
    /// argmax を 20 で割った値
    argmax_norm: Vec<f32>,
    /// `teacher_soft_stats` の 5 値
    stats: Vec<[f32; STATS_DIM]>,
}

impl TeacherColumns {
    fn new(logits: &[f32], dict_accents: &[Option<u8>]) -> Self {
        let (argmax_norm, stats) = logits
            .as_chunks::<NUM_CLASSES>()
            .0
            .iter()
            .zip(dict_accents)
            .map(|(row, &dict_accent)| {
                (
                    argmax(row) as f32 / 20.0,
                    teacher_soft_stats(row, morpheme_dict_accent(dict_accent)),
                )
            })
            .unzip();
        Self { argmax_norm, stats }
    }

    /// `[feat13, v24_arg]` を行の先頭へ書く。
    fn write_prefix(&self, i: usize, feat13: &[f32; FEATURE13_DIM], row: &mut [f32]) {
        row[..FEATURE13_DIM].copy_from_slice(feat13);
        row[TEACHER_ARG_COLUMN] = self.argmax_norm[i];
    }

    /// `[feat13, v24_arg, teacher_stats]` を行の先頭へ書く。
    fn write_prefix_with_stats(&self, i: usize, feat13: &[f32; FEATURE13_DIM], row: &mut [f32]) {
        self.write_prefix(i, feat13, row);
        row[TEACHER_STATS_COLUMNS].copy_from_slice(&self.stats[i]);
    }
}

/// 9 student を並列に実行し、softmax を `[形態素][student]` の順に並べる。
///
/// indexed な並列イテレータはモデルの順序を保つ。student 数は `validate_request` で検証済み。
fn student_probabilities(
    students: &[NativeModel],
    feat14: &[f32],
    seq_len: usize,
) -> Result<Vec<[Probabilities; NUM_STUDENTS]>, V66PredictError> {
    let logits = students
        .par_iter()
        .map(|model| run_session(model, feat14, seq_len, FEATURE14_DIM))
        .collect::<Result<Vec<_>, _>>()?;
    let mut probabilities = vec![[[0f32; NUM_CLASSES]; NUM_STUDENTS]; seq_len];
    for (student, logits) in logits.iter().enumerate() {
        for (token, row) in probabilities
            .iter_mut()
            .zip(logits.as_chunks::<NUM_CLASSES>().0)
        {
            token[student] = softmax(row);
        }
    }
    Ok(probabilities)
}

/// 1 モデル分のロジットを形態素ごとの softmax に変換する。
fn probabilities(logits: &[f32]) -> Vec<Probabilities> {
    logits
        .as_chunks::<NUM_CLASSES>()
        .0
        .iter()
        .map(|row| softmax(row))
        .collect()
}

/// 9 student・v61・v63 の softmax から
/// `mean[21] + std[21] (ddof=0) + vote_hist[21] + sm_v63[21]` を作る。
///
/// 和は student 1〜9、v61、v63 の順に取り、訓練側で stack した順と同じ丸めにする。
fn stacker_features(
    students: &[Probabilities; NUM_STUDENTS],
    v61: &Probabilities,
    v63: &Probabilities,
) -> [f32; STACKER_DIM] {
    let models = || students.iter().chain([v61, v63]);
    let count = (NUM_STUDENTS + 2) as f32;

    let mut mean = [0f32; NUM_CLASSES];
    for probabilities in models() {
        for (sum, &p) in mean.iter_mut().zip(probabilities) {
            *sum += p;
        }
    }
    for value in &mut mean {
        *value /= count;
    }

    let mut std = [0f32; NUM_CLASSES];
    for probabilities in models() {
        for ((sum, &p), &m) in std.iter_mut().zip(probabilities).zip(&mean) {
            let d = p - m;
            *sum += d * d;
        }
    }
    for value in &mut std {
        *value = (*value / count).sqrt();
    }

    // vote_hist = bincount(argmax over 11 models) / 11
    let mut votes = [0f32; NUM_CLASSES];
    for probabilities in models() {
        votes[argmax(probabilities)] += 1.0;
    }
    for value in &mut votes {
        *value /= count;
    }

    let mut row = [0f32; STACKER_DIM];
    for (part, values) in row
        .as_chunks_mut::<NUM_CLASSES>()
        .0
        .iter_mut()
        .zip([mean, std, votes, *v63])
    {
        *part = values;
    }
    row
}

/// 空入力を含め、特徴量の組み立てに必要な長さを推論前に検証する。
fn validate_request(
    seq_len: usize,
    dict_accent_count: usize,
    student_count: usize,
) -> Result<(), V66PredictError> {
    if dict_accent_count != seq_len {
        return Err(V66PredictError::Shape(format!(
            "dict_accents.len ({dict_accent_count}) != ctx.len ({seq_len})"
        )));
    }
    if student_count != NUM_STUDENTS {
        return Err(V66PredictError::Shape(format!(
            "students.len ({student_count}) != NUM_STUDENTS ({NUM_STUDENTS})"
        )));
    }
    Ok(())
}

/// 要素数が同じでも、転置や余分な軸を持つ出力を受け入れない。
fn validate_output_shape(
    shape: &[i64],
    output_len: usize,
    seq_len: usize,
) -> Result<(), V66PredictError> {
    let expected_seq = i64::try_from(seq_len)
        .map_err(|_| V66PredictError::Shape("seq_len exceeds i64 range".to_string()))?;
    let expected_len = seq_len
        .checked_mul(NUM_CLASSES)
        .ok_or_else(|| V66PredictError::Shape("output size overflow".to_string()))?;
    if shape != [expected_seq, NUM_CLASSES as i64] || output_len != expected_len {
        return Err(V66PredictError::Shape(format!(
            "expected output [{seq_len}, {NUM_CLASSES}] ({expected_len} values), \
             got {shape:?} ({output_len} values)"
        )));
    }
    Ok(())
}

impl ContextualAccentPredictor for V66Pipeline {
    fn predict_with_context(
        &self,
        ctx: &[FeatureMorpheme<'_>],
    ) -> Result<Vec<u8>, crate::nn::ContextualAccentError> {
        self.predict(ctx)
            .map_err(|e| Box::new(e) as crate::nn::ContextualAccentError)
    }
}

/// 単一専用モデルを `[seq_len, dim]` 入力で実行し、`[seq_len, NUM_CLASSES]`
/// の出力を行優先 `Vec<f32>` で返す。
///
/// 入力データはスライス借用なので clone 不要。同じ入力を複数セッションへ流す
/// (9 student) ケースでアロケーションが発生しない。
/// 入力長 / 出力形状を release でも検証し、不整合時は `V66PredictError::Shape` を返す。
fn run_session(
    session: &NativeModel,
    input_data: &[f32],
    seq_len: usize,
    dim: usize,
) -> Result<Vec<f32>, V66PredictError> {
    let expected_input_len = seq_len
        .checked_mul(dim)
        .ok_or_else(|| V66PredictError::Shape("input size overflow".to_string()))?;
    if input_data.len() != expected_input_len {
        return Err(V66PredictError::Shape(format!(
            "input_data.len ({}) != seq_len ({}) * dim ({}) = {}",
            input_data.len(),
            seq_len,
            dim,
            expected_input_len
        )));
    }
    if dim != session.input_dim() {
        return Err(V66PredictError::Shape(
            "model feature dimension mismatch".into(),
        ));
    }
    let output = session
        .predict(input_data, seq_len)
        .map_err(|e| V66PredictError::Run(e.to_string()))?;
    validate_output_shape(&[seq_len as i64, NUM_CLASSES as i64], output.len(), seq_len)?;
    Ok(output)
}

/// 推論時のエラー
#[derive(Debug, thiserror::Error)]
pub enum V66PredictError {
    /// 入出力の長さ / 形状が想定と合わない
    #[error("shape mismatch: {0}")]
    Shape(String),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_rejects_misaligned_dictionary_and_incomplete_ensembles() {
        assert!(validate_request(2, 2, NUM_STUDENTS).is_ok());
        assert!(validate_request(0, 0, NUM_STUDENTS).is_ok());
        for (seq, dict, students) in [
            (2, 1, NUM_STUDENTS),
            (0, 1, NUM_STUDENTS),
            (2, 2, NUM_STUDENTS - 1),
            (2, 2, NUM_STUDENTS + 1),
        ] {
            assert!(matches!(
                validate_request(seq, dict, students),
                Err(V66PredictError::Shape(_))
            ));
        }
    }

    #[test]
    fn output_shape_rejects_same_size_transposed_and_extra_axes() {
        assert!(validate_output_shape(&[2, 21], 42, 2).is_ok());
        assert!(validate_output_shape(&[1, 21], 21, 1).is_ok());
        for shape in [&[21, 2][..], &[42][..], &[1, 2, 21][..], &[2, 20][..]] {
            assert!(matches!(
                validate_output_shape(shape, 42, 2),
                Err(V66PredictError::Shape(_))
            ));
        }
        assert!(validate_output_shape(&[2, 21], 41, 2).is_err());
    }

    fn one_hot(class: usize) -> Probabilities {
        let mut row = [0.0; NUM_CLASSES];
        row[class] = 1.0;
        row
    }

    #[test]
    fn stacker_counts_every_model_once_and_appends_v63() {
        // student 5 個が 0、4 個が 1、v61 が 2、v63 が 1 を選ぶ。
        let students: [Probabilities; NUM_STUDENTS] =
            std::array::from_fn(|i| one_hot(usize::from(i >= 5)));
        let (v61, v63) = (one_hot(2), one_hot(1));
        let row = stacker_features(&students, &v61, &v63);
        let (mean, rest) = row.split_at(NUM_CLASSES);
        let (std, rest) = rest.split_at(NUM_CLASSES);
        let (votes, tail) = rest.split_at(NUM_CLASSES);
        let count = (NUM_STUDENTS + 2) as f32;
        for (class, models) in [(0, 5.0), (1, 5.0), (2, 1.0)] {
            let share = models / count;
            assert_eq!(mean[class], share, "mean {class}");
            assert_eq!(votes[class], share, "votes {class}");
            // one-hot の分散は p(1-p)。
            assert!((std[class] - (share * (1.0 - share)).sqrt()).abs() < 1e-6);
        }
        assert!(
            mean[3..]
                .iter()
                .chain(&std[3..])
                .chain(&votes[3..])
                .all(|v| *v == 0.0)
        );
        assert_eq!(tail, v63);
    }

    #[test]
    fn stacker_of_identical_models_has_no_spread() {
        let mut probabilities = [0.01; NUM_CLASSES];
        probabilities[7] = 1.0 - 0.01 * (NUM_CLASSES - 1) as f32;
        let students = [probabilities; NUM_STUDENTS];
        let row = stacker_features(&students, &probabilities, &probabilities);
        for (class, &p) in probabilities.iter().enumerate() {
            assert!((row[class] - p).abs() < 1e-6, "mean {class}");
            assert!(row[NUM_CLASSES + class].abs() < 1e-6, "std {class}");
        }
        assert_eq!(row[2 * NUM_CLASSES + 7], 1.0);
        assert_eq!(&row[3 * NUM_CLASSES..], &probabilities);
    }

    #[test]
    fn teacher_columns_scale_argmax_and_place_shared_prefix() {
        let mut logits = vec![0.0; 2 * NUM_CLASSES];
        logits[10] = 9.0;
        logits[NUM_CLASSES + 20] = 9.0;
        let teacher = TeacherColumns::new(&logits, &[Some(10), None]);
        assert_eq!(teacher.argmax_norm, [0.5, 1.0]);
        // 辞書のアクセント類の確率は辞書にない語で 0。
        assert!(teacher.stats[0][4] > 0.9);
        assert_eq!(teacher.stats[1][4], 0.0);

        let feat13: [f32; FEATURE13_DIM] = std::array::from_fn(|i| i as f32);
        let mut row = [f32::NAN; FEATURE24_DIM];
        teacher.write_prefix_with_stats(1, &feat13, &mut row);
        assert_eq!(&row[..FEATURE13_DIM], &feat13);
        assert_eq!(row[TEACHER_ARG_COLUMN], 1.0);
        assert_eq!(&row[TEACHER_STATS_COLUMNS], &teacher.stats[1]);
        // 後段の列は呼び出し側が書くので触れない。
        assert!(row[TAIL_COLUMN..].iter().all(|v| v.is_nan()));
    }
}
