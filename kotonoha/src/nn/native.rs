//! 固定構造の BiLSTM + Attention 推論。グラフ解釈、ORT、Python、共有ライブラリは不要。
//!
//! 全時刻の入力項を一括 GEMM し、再帰項は出力 32 個ごとの専用 GEMV で計算する。
//! 重みは不変であり、同じモデルを複数スレッドから同時に呼べる。

use crate::model::knm::{KnmError, Weights};
use std::path::Path;

const HIDDEN: usize = 256;
const GATES: usize = 4 * HIDDEN;

/// KNM の検証済み FP32 モデル。
pub struct NativeModel {
    weights: Weights,
}

impl NativeModel {
    /// KNM をロードする。テンソルの配置は変換時に確定している。
    pub fn load(path: &Path) -> Result<Self, KnmError> {
        Ok(Self {
            weights: Weights::load(path)?,
        })
    }

    /// このモデルの特徴量の次元。
    pub fn input_dim(&self) -> usize {
        self.weights.input_dim
    }

    fn weight(&self, name: &str) -> &[f32] {
        &self.weights.tensors[name]
    }

    fn linear(&self, input: &[f32], rows: usize, name: &str) -> Vec<f32> {
        let bias = self.weight(&format!("{name}.bias"));
        let cols = bias.len();
        let mut out = matmul(
            input,
            self.weight(&format!("{name}.weight")),
            rows,
            input.len() / rows,
            cols,
        );
        for row in out.chunks_exact_mut(cols) {
            for (value, bias) in row.iter_mut().zip(bias) {
                *value += bias;
            }
        }
        out
    }

    fn normalize(&self, input: &mut [f32], name: &str) {
        let scale = self.weight(&format!("{name}.weight"));
        let bias = self.weight(&format!("{name}.bias"));
        for row in input.chunks_exact_mut(scale.len()) {
            let mean = row.iter().sum::<f32>() / row.len() as f32;
            let variance =
                row.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / row.len() as f32;
            let inv_std = (variance + 1e-5).sqrt().recip();
            for ((value, scale), bias) in row.iter_mut().zip(scale).zip(bias) {
                *value = (*value - mean) * inv_std * scale + bias;
            }
        }
    }

    fn frontend(&self, input: &[f32], seq: usize) -> Vec<f32> {
        let dim = self.input_dim();
        let mut kana = vec![0.; seq * 16];
        let mut continuous = Vec::with_capacity(seq * (dim - 5));
        for (i, row) in input.chunks_exact(dim).enumerate() {
            kana[i * 16..(i + 1) * 16].fill(row[6]);
            continuous.extend_from_slice(&row[5..]);
        }
        let kana = self.linear(&kana, seq, "frontend.kana_proj");
        let continuous = self.linear(&continuous, seq, "frontend.cont_proj");
        let mut concatenated = vec![0.; seq * 88];
        for (i, row) in input.chunks_exact(dim).enumerate() {
            let dest = &mut concatenated[i * 88..(i + 1) * 88];
            let mut offset = 0;
            for (j, (name, rows, width)) in [
                ("pos", 15, 16),
                ("pd1", 21, 16),
                ("pd2", 11, 8),
                ("ct", 9, 8),
                ("cf", 10, 8),
            ]
            .iter()
            .enumerate()
            {
                let id = (row[j] as i64).clamp(0, rows - 1) as usize;
                let values = self.weight(&format!("frontend.{name}_emb.weight"));
                dest[offset..offset + width].copy_from_slice(&values[id * width..(id + 1) * width]);
                offset += width;
            }
            dest[56..72].copy_from_slice(&kana[i * 16..(i + 1) * 16]);
            dest[72..88].copy_from_slice(&continuous[i * 16..(i + 1) * 16]);
        }
        self.normalize(&mut concatenated, "frontend.input_norm");
        self.linear(&concatenated, seq, "frontend.input_proj")
    }

    fn lstm(&self, input: &[f32], seq: usize, layer: usize) -> Vec<f32> {
        let width = input.len() / seq;
        let w = self.weight(&format!("lstm.{layer}.input"));
        let r = self.weight(&format!("lstm.{layer}.recurrent"));
        let b = self.weight(&format!("lstm.{layer}.bias"));
        let mut output = vec![0.; seq * HIDDEN * 2];
        for direction in 0..2 {
            let mut projected = packed_projection(
                input,
                &w[direction * width * GATES..(direction + 1) * width * GATES],
                seq,
                width,
            );
            let bias = &b[direction * GATES * 2..(direction + 1) * GATES * 2];
            for row in projected.as_chunks_mut::<GATES>().0.iter_mut() {
                for j in 0..GATES {
                    row[j] += bias[j] + bias[GATES + j];
                }
            }
            let recurrence = &r[direction * GATES * HIDDEN..(direction + 1) * GATES * HIDDEN];
            let mut hidden = [0.; HIDDEN];
            let mut cell = [0.; HIDDEN];
            let mut gates = [0.; GATES];
            for step in 0..seq {
                let t = if direction == 0 { step } else { seq - 1 - step };
                recurrent_gemv(&hidden, recurrence, &mut gates);
                let x = &projected[t * GATES..(t + 1) * GATES];
                for j in (0..HIDDEN).step_by(8) {
                    use wide::f32x8;
                    let load = |a: &[f32], start| {
                        f32x8::from(<[f32; 8]>::try_from(&a[start..start + 8]).unwrap())
                    };
                    let i = sigmoid8(load(x, j) + load(&gates, j));
                    let o = sigmoid8(load(x, HIDDEN + j) + load(&gates, HIDDEN + j));
                    let f = sigmoid8(load(x, 2 * HIDDEN + j) + load(&gates, 2 * HIDDEN + j));
                    let g = tanh8(load(x, 3 * HIDDEN + j) + load(&gates, 3 * HIDDEN + j));
                    let c = f * load(&cell, j) + i * g;
                    cell[j..j + 8].copy_from_slice(&c.to_array());
                    hidden[j..j + 8].copy_from_slice(&(o * tanh8(c)).to_array());
                }
                output[t * HIDDEN * 2 + direction * HIDDEN
                    ..t * HIDDEN * 2 + (direction + 1) * HIDDEN]
                    .copy_from_slice(&hidden);
            }
        }
        output
    }

    fn attention(&self, input: &[f32], seq: usize) -> Vec<f32> {
        let qkv = self.linear(input, seq, "self_attn.qkv_proj");
        let mut context = vec![0.; seq * HIDDEN];
        let mut scores = vec![0.; seq];
        // 1 query 分ずつ softmax するので、Attention の作業メモリは O(seq)。
        for t in 0..seq {
            for head in 0..4 {
                let start = head * 64;
                let q = &qkv[t * 768 + start..t * 768 + start + 64];
                for (s, score) in scores.iter_mut().enumerate() {
                    let k = &qkv[s * 768 + 256 + start..s * 768 + 256 + start + 64];
                    *score = q.iter().zip(k).map(|(q, k)| q * k).sum::<f32>() * 0.125;
                }
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.;
                for score in &mut scores {
                    *score = (*score - max).exp();
                    sum += *score;
                }
                for (s, score) in scores.iter().enumerate() {
                    let v = &qkv[s * 768 + 512 + start..s * 768 + 512 + start + 64];
                    let probability = score / sum;
                    for j in 0..64 {
                        context[t * HIDDEN + start + j] += probability * v[j];
                    }
                }
            }
        }
        let mut projected = self.linear(&context, seq, "self_attn.out_proj");
        for (value, residual) in projected.iter_mut().zip(input) {
            *value += residual;
        }
        self.normalize(&mut projected, "self_attn.norm");
        projected
    }

    /// `[seq, input_dim]` → `[seq, 21]`。空入力は空出力。
    pub fn predict(&self, input: &[f32], seq: usize) -> Result<Vec<f32>, KnmError> {
        if seq.checked_mul(self.input_dim()) != Some(input.len())
            || !input.iter().all(|x| x.is_finite())
        {
            return Err(KnmError::Invalid(
                "input shape or non-finite features".into(),
            ));
        }
        // 内部の最大作業行列 seq×1024 が usize/isize を超えないことを先に確認。
        if seq > (isize::MAX as usize) / (GATES * 4) {
            return Err(KnmError::Invalid("sequence too long".into()));
        }
        if seq == 0 {
            return Ok(Vec::new());
        }
        let frontend = self.frontend(input, seq);
        let mut h = self.lstm(&frontend, seq, 0);
        for layer in 1..3 {
            h = self.lstm(&h, seq, layer);
        }
        let mut h = self.linear(&h, seq, "lstm_proj");
        for (value, residual) in h.iter_mut().zip(&frontend) {
            *value += residual;
        }
        self.normalize(&mut h, "lstm_norm");
        let h = self.attention(&h, seq);
        let mut h = self.linear(&h, seq, "classifier.0");
        for value in &mut h {
            *value *= 0.5 * (1. + libm::erff(*value * std::f32::consts::FRAC_1_SQRT_2));
        }
        let logits = self.linear(&h, seq, "classifier.3");
        if !logits.iter().all(|v| v.is_finite()) {
            return Err(KnmError::Invalid("non-finite logits".into()));
        }
        Ok(logits)
    }
}

fn sigmoid8(value: wide::f32x8) -> wide::f32x8 {
    wide::f32x8::ONE
        / (wide::f32x8::ONE
            + (-value)
                .max(wide::f32x8::splat(-80.))
                .min(wide::f32x8::splat(80.))
                .exp())
}
fn tanh8(value: wide::f32x8) -> wide::f32x8 {
    wide::f32x8::splat(2.) * sigmoid8(wide::f32x8::splat(2.) * value) - wide::f32x8::ONE
}

fn matmul(a: &[f32], b: &[f32], rows: usize, inner: usize, cols: usize) -> Vec<f32> {
    assert_eq!(a.len(), rows * inner);
    assert_eq!(b.len(), inner * cols);
    let mut output = vec![0.; rows * cols];
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        let mut row = 0;
        while row + 6 <= rows {
            // SAFETY: CPU feature と全行列の長さを上で検証した。6 行が範囲内。
            unsafe {
                dense_avx2::<6>(&a[row * inner..], b, &mut output[row * cols..], inner, cols);
            }
            row += 6;
        }
        // SAFETY: 端数もまとめて計算し、M 行分だけを範囲内で読み書きする。
        unsafe {
            match rows - row {
                1 => dense_avx2::<1>(&a[row * inner..], b, &mut output[row * cols..], inner, cols),
                2 => dense_avx2::<2>(&a[row * inner..], b, &mut output[row * cols..], inner, cols),
                3 => dense_avx2::<3>(&a[row * inner..], b, &mut output[row * cols..], inner, cols),
                4 => dense_avx2::<4>(&a[row * inner..], b, &mut output[row * cols..], inner, cols),
                5 => dense_avx2::<5>(&a[row * inner..], b, &mut output[row * cols..], inner, cols),
                _ => (),
            }
        }
        for row in 0..rows {
            for col in (cols / 16 * 16)..cols {
                output[row * cols + col] = (0..inner)
                    .map(|k| a[row * inner + k] * b[k * cols + col])
                    .sum();
            }
        }
        return output;
    }
    // SAFETY: 長さを上で検証した独立した連続配列。出力と入力は重ならない。
    unsafe {
        matrixmultiply::sgemm(
            rows,
            inner,
            cols,
            1.,
            a.as_ptr(),
            inner as isize,
            1,
            b.as_ptr(),
            cols as isize,
            1,
            0.,
            output.as_mut_ptr(),
            cols as isize,
            1,
        );
    }
    output
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dense_avx2<const M: usize>(
    input: &[f32],
    weights: &[f32],
    output: &mut [f32],
    inner: usize,
    cols: usize,
) {
    use std::arch::x86_64::*;
    for col in (0..cols / 16 * 16).step_by(16) {
        let mut accum = [[_mm256_setzero_ps(); 2]; M];
        for k in 0..inner {
            let offset = k * cols + col;
            // SAFETY: col は末尾の 16 要素まで。各 load は検証済み行内に収まる。
            let (w0, w1) = unsafe {
                (
                    _mm256_loadu_ps(weights.as_ptr().add(offset)),
                    _mm256_loadu_ps(weights.as_ptr().add(offset + 8)),
                )
            };
            for row in 0..M {
                let h = _mm256_set1_ps(input[row * inner + k]);
                accum[row][0] = _mm256_fmadd_ps(h, w0, accum[row][0]);
                accum[row][1] = _mm256_fmadd_ps(h, w1, accum[row][1]);
            }
        }
        for (row, acc) in accum.into_iter().enumerate() {
            // SAFETY: 呼出し側で M 行分の空間を検証済み。col..col+16 は範囲内。
            unsafe {
                _mm256_storeu_ps(output.as_mut_ptr().add(row * cols + col), acc[0]);
                _mm256_storeu_ps(output.as_mut_ptr().add(row * cols + col + 8), acc[1]);
            }
        }
    }
}

fn recurrent_gemv(hidden: &[f32; HIDDEN], weights: &[f32], output: &mut [f32; GATES]) {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        // SAFETY: CPU feature を検出済み。検証済みの形状で、全読み書きが配列内。
        unsafe {
            recurrent_avx2(hidden, weights, output);
        }
        return;
    }
    recurrent_scalar(hidden, weights, output);
}

fn packed_projection(input: &[f32], weights: &[f32], rows: usize, inner: usize) -> Vec<f32> {
    let mut output = vec![0.; rows * GATES];
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        let mut row = 0;
        while row + 6 <= rows {
            // SAFETY: CPU feature とテンソル形状を検証済み。6 行分が範囲内。
            unsafe {
                projection_avx2::<6>(
                    &input[row * inner..],
                    weights,
                    &mut output[row * GATES..],
                    inner,
                );
            }
            row += 6;
        }
        // SAFETY: 上と同じ。端数の M 行分が範囲内に存在する。
        unsafe {
            match rows - row {
                1 => projection_avx2::<1>(
                    &input[row * inner..],
                    weights,
                    &mut output[row * GATES..],
                    inner,
                ),
                2 => projection_avx2::<2>(
                    &input[row * inner..],
                    weights,
                    &mut output[row * GATES..],
                    inner,
                ),
                3 => projection_avx2::<3>(
                    &input[row * inner..],
                    weights,
                    &mut output[row * GATES..],
                    inner,
                ),
                4 => projection_avx2::<4>(
                    &input[row * inner..],
                    weights,
                    &mut output[row * GATES..],
                    inner,
                ),
                5 => projection_avx2::<5>(
                    &input[row * inner..],
                    weights,
                    &mut output[row * GATES..],
                    inner,
                ),
                _ => (),
            }
        }
        return output;
    }
    for row in 0..rows {
        for (j, value) in output[row * GATES..(row + 1) * GATES]
            .iter_mut()
            .enumerate()
        {
            *value = (0..inner)
                .map(|k| input[row * inner + k] * weights[(j / 32) * inner * 32 + k * 32 + j % 32])
                .sum();
        }
    }
    output
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn projection_avx2<const M: usize>(
    input: &[f32],
    weights: &[f32],
    output: &mut [f32],
    inner: usize,
) {
    use std::arch::x86_64::*;
    for block in 0..GATES / 32 {
        for half in [0, 16] {
            let mut accum = [[_mm256_setzero_ps(); 2]; M];
            for k in 0..inner {
                let offset = block * inner * 32 + k * 32 + half;
                // SAFETY: half=0/16、それぞれ 16 要素を読み、検証済みタイル内に収まる。
                let (w0, w1) = unsafe {
                    (
                        _mm256_loadu_ps(weights.as_ptr().add(offset)),
                        _mm256_loadu_ps(weights.as_ptr().add(offset + 8)),
                    )
                };
                for row in 0..M {
                    let h = _mm256_set1_ps(input[row * inner + k]);
                    accum[row][0] = _mm256_fmadd_ps(h, w0, accum[row][0]);
                    accum[row][1] = _mm256_fmadd_ps(h, w1, accum[row][1]);
                }
            }
            for (row, acc) in accum.into_iter().enumerate() {
                let offset = row * GATES + block * 32 + half;
                // SAFETY: M 行が存在することは呼び出し側が検証済み。16 要素の連続 store。
                unsafe {
                    _mm256_storeu_ps(output.as_mut_ptr().add(offset), acc[0]);
                    _mm256_storeu_ps(output.as_mut_ptr().add(offset + 8), acc[1]);
                }
            }
        }
    }
}

fn recurrent_scalar(hidden: &[f32; HIDDEN], weights: &[f32], output: &mut [f32; GATES]) {
    for (block, dest) in weights
        .as_chunks::<{ HIDDEN * 32 }>()
        .0
        .iter()
        .zip(output.as_chunks_mut::<32>().0.iter_mut())
    {
        dest.fill(0.);
        for (row, &h) in block.as_chunks::<32>().0.iter().zip(hidden) {
            for (value, w) in dest.iter_mut().zip(row) {
                *value += h * w;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn recurrent_avx2(hidden: &[f32; HIDDEN], weights: &[f32], output: &mut [f32; GATES]) {
    use std::arch::x86_64::*;
    for (block, dest) in weights
        .as_chunks::<{ HIDDEN * 32 }>()
        .0
        .iter()
        .zip(output.as_chunks_mut::<32>().0.iter_mut())
    {
        let mut a = [_mm256_setzero_ps(); 4];
        for (row, &h) in block.as_chunks::<32>().0.iter().zip(hidden) {
            let h = _mm256_set1_ps(h);
            for (i, acc) in a.iter_mut().enumerate() {
                // SAFETY: 各 row は 32 要素。i=0..4 の各 load は 8 要素。
                unsafe {
                    *acc = _mm256_fmadd_ps(h, _mm256_loadu_ps(row.as_ptr().add(i * 8)), *acc);
                }
            }
        }
        for (i, acc) in a.into_iter().enumerate() {
            // SAFETY: dest は 32 要素。各 store は重ならない 8 要素。
            unsafe {
                _mm256_storeu_ps(dest.as_mut_ptr().add(i * 8), acc);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiled_recurrence_matches_scalar_and_plain_matrix_product() {
        let hidden = std::array::from_fn(|i| (i as f32 * 0.3).sin());
        let weights: Vec<_> = (0..GATES * HIDDEN)
            .map(|i| (i as f32 * 0.01).cos() * 0.01)
            .collect();
        let mut expected = [0.; GATES];
        for (j, value) in expected.iter_mut().enumerate() {
            *value = (0..HIDDEN)
                .map(|k| hidden[k] * weights[(j / 32) * HIDDEN * 32 + k * 32 + j % 32])
                .sum();
        }
        let mut actual = [0.; GATES];
        recurrent_gemv(&hidden, &weights, &mut actual);
        assert!(
            actual
                .iter()
                .zip(expected)
                .all(|(a, b)| (a - b).abs() < 1e-5)
        );
    }

    #[test]
    fn packed_projection_handles_simd_blocks_and_tail_rows() {
        let inner = 7;
        let weights: Vec<f32> = (0..GATES * inner)
            .map(|i| (i as f32 * 0.02).sin())
            .collect();
        for rows in [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13] {
            let input: Vec<f32> = (0..rows * inner).map(|i| i as f32 * 0.01).collect();
            let actual = packed_projection(&input, &weights, rows, inner);
            for row in 0..rows {
                for col in 0..GATES {
                    let expected: f32 = (0..inner)
                        .map(|k| {
                            input[row * inner + k]
                                * weights[(col / 32) * inner * 32 + k * 32 + col % 32]
                        })
                        .sum();
                    assert!((actual[row * GATES + col] - expected).abs() < 1e-5);
                }
            }
        }
    }

    #[test]
    fn dense_projection_handles_short_outputs_and_simd_tail_columns() {
        for rows in [1, 3, 4, 5, 6, 7, 8, 11, 12, 13] {
            for cols in [1, 7, 16, 21, 32, 45] {
                let inner = 7;
                let a: Vec<f32> = (0..rows * inner).map(|i| (i as f32 * 0.1).sin()).collect();
                let b: Vec<f32> = (0..inner * cols).map(|i| (i as f32 * 0.2).cos()).collect();
                let output = matmul(&a, &b, rows, inner, cols);
                for row in 0..rows {
                    for col in 0..cols {
                        let expected: f32 = (0..inner)
                            .map(|k| a[row * inner + k] * b[k * cols + col])
                            .sum();
                        assert!((output[row * cols + col] - expected).abs() < 1e-5);
                    }
                }
            }
        }
    }

    #[test]
    fn requests_validate_shape_and_empty_before_computation() {
        let model = NativeModel {
            weights: Weights {
                input_dim: 11,
                source_sha256: String::new(),
                tensors: Default::default(),
            },
        };
        assert!(model.predict(&[], 0).unwrap().is_empty());
        assert!(model.predict(&[0.; 10], 1).is_err());
        assert!(model.predict(&[], usize::MAX).is_err());
        assert!(model.predict(&[f32::NAN; 11], 1).is_err());
    }
}
