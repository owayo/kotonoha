//! v66 系で使う数値ヘルパ
//!
//! `train_onnx_v60.py::_softmax_1d` / `_teacher_soft_stats` /
//! `precompute_v61_meta.py::_meta_features` と数値完全一致させる。

use super::vocab::NUM_CLASSES;

/// 数値安定 softmax (`np.exp(x - max(x)) / sum`)
///
/// `train_onnx_v60.py::_softmax_1d` (357-366) と等価。
/// 入力は長さ `NUM_CLASSES` のスライス、出力は同じ長さの確率分布。
pub fn softmax(logits: &[f32]) -> [f32; NUM_CLASSES] {
    debug_assert_eq!(logits.len(), NUM_CLASSES);
    let mut max = f32::NEG_INFINITY;
    for &v in logits {
        if v > max {
            max = v;
        }
    }
    let mut out = [0f32; NUM_CLASSES];
    let mut sum = 0f32;
    for (i, &v) in logits.iter().enumerate() {
        let e = (v - max).exp();
        out[i] = e;
        sum += e;
    }
    if sum > 0.0 {
        for v in &mut out {
            *v /= sum;
        }
    }
    out
}

/// argmax (`numpy.argmax(logits)` の f32 版)
pub fn argmax(logits: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best
}

/// `_teacher_soft_stats` の 5-tuple
///
/// `train_onnx_v60.py::_teacher_soft_stats` (373-396) と等価:
/// - `E[y]/20`           — クラス期待値を 20 で割った値
/// - `max(p)`             — 最大確率
/// - `top1 - top2 margin` — ソートして上 2 位の差
/// - `entropy / log(NUM_CLASSES)` — 正規化エントロピー
/// - `p[dict_acc_type]`   — 辞書アクセント類の確率 (範囲外なら 0)
pub fn teacher_soft_stats(logits: &[f32], dict_acc_type: i32) -> [f32; 5] {
    let p = softmax(logits);
    let mut exp_y = 0f32;
    for (i, &pi) in p.iter().enumerate() {
        exp_y += pi * (i as f32);
    }
    exp_y /= 20.0;

    let mut pmax = 0f32;
    for &pi in &p {
        if pi > pmax {
            pmax = pi;
        }
    }

    // top1 - top2 margin: numpy の `np.partition(p, -2)[-2:]` は上位 2 つの値
    // (順不同) を返すため、`max - second_max` ではなく単純に「top1 - top2」と
    // 同義になる。
    let mut top1 = 0f32;
    let mut top2 = 0f32;
    for &pi in &p {
        if pi >= top1 {
            top2 = top1;
            top1 = pi;
        } else if pi > top2 {
            top2 = pi;
        }
    }
    let margin = top1 - top2;

    let mut entropy = 0f32;
    for &pi in &p {
        entropy -= pi * (pi + 1e-9).ln();
    }
    entropy /= (NUM_CLASSES as f32).ln();

    let p_dict = if dict_acc_type >= 0 && (dict_acc_type as usize) < NUM_CLASSES {
        p[dict_acc_type as usize]
    } else {
        0.0
    };

    [exp_y, pmax, margin, entropy, p_dict]
}

/// `_meta_features` の 5-tuple
///
/// `precompute_v61_meta.py::_meta_features` (37-60) と等価。
/// 入力は [M (model 数), NUM_CLASSES] の softmax stack を行優先で渡す。
///
/// 戻り値: [mean_exp, std_exp (ddof=0), agreement, mean_entropy_normalized,
/// max_p_consensus]。
pub fn meta_features(softmax_stack: &[[f32; NUM_CLASSES]]) -> [f32; 5] {
    let m = softmax_stack.len();
    debug_assert!(m > 0);

    // exp_per[i] = sum(softmax[i] * class_idx) / 20
    let mut exp_per = vec![0f32; m];
    for (i, sm) in softmax_stack.iter().enumerate() {
        let mut s = 0f32;
        for (c, &p) in sm.iter().enumerate() {
            s += p * (c as f32);
        }
        exp_per[i] = s / 20.0;
    }
    // mean
    let mean_exp = exp_per.iter().copied().sum::<f32>() / m as f32;
    // std with ddof=0 (numpy default for ndarray.std())
    let mut var = 0f32;
    for &x in &exp_per {
        let d = x - mean_exp;
        var += d * d;
    }
    let std_exp = (var / m as f32).sqrt();

    // consensus = argmax of bincount over per-model argmaxes
    let mut argmaxes = vec![0usize; m];
    for (i, sm) in softmax_stack.iter().enumerate() {
        argmaxes[i] = argmax(sm);
    }
    let mut counts = [0u32; NUM_CLASSES];
    for &a in &argmaxes {
        counts[a] += 1;
    }
    let mut consensus = 0usize;
    let mut consensus_count = 0u32;
    for (i, &c) in counts.iter().enumerate() {
        if c > consensus_count {
            consensus_count = c;
            consensus = i;
        }
    }
    let agreement = consensus_count as f32 / m as f32;

    // mean entropy / log(NUM_CLASSES)
    let mut mean_entropy = 0f32;
    for sm in softmax_stack {
        let mut h = 0f32;
        for &p in sm {
            h -= p * (p + 1e-9).ln();
        }
        mean_entropy += h;
    }
    mean_entropy /= m as f32 * (NUM_CLASSES as f32).ln();

    // max_p_consensus = max over m of softmax[m, consensus]
    let mut max_p_consensus = 0f32;
    for sm in softmax_stack {
        if sm[consensus] > max_p_consensus {
            max_p_consensus = sm[consensus];
        }
    }

    [mean_exp, std_exp, agreement, mean_entropy, max_p_consensus]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn softmax_uniform() {
        let logits = [0.0f32; NUM_CLASSES];
        let p = softmax(&logits);
        let expected = 1.0 / NUM_CLASSES as f32;
        for &pi in &p {
            assert!(approx(pi, expected, 1e-6));
        }
    }

    #[test]
    fn argmax_basic() {
        let mut logits = [0.0f32; NUM_CLASSES];
        logits[7] = 5.0;
        assert_eq!(argmax(&logits), 7);
    }

    #[test]
    fn teacher_stats_uniform() {
        // uniform softmax over NUM_CLASSES (=21) classes
        let logits = [0.0f32; NUM_CLASSES];
        let stats = teacher_soft_stats(&logits, -1);
        // E[y] = (sum 0..21) / 21 / 20 = 210 / 21 / 20 = 0.5
        assert!(approx(stats[0], 0.5, 1e-6));
        // max p = 1/21
        assert!(approx(stats[1], 1.0 / 21.0, 1e-6));
        // margin = 0 (uniform)
        assert!(approx(stats[2], 0.0, 1e-6));
        // entropy normalized = log(21) / log(21) = 1.0
        assert!(approx(stats[3], 1.0, 1e-5));
        // p_dict = 0 since dict_acc_type = -1
        assert_eq!(stats[4], 0.0);
    }

    #[test]
    fn meta_features_unanimous_models() {
        // 3 models all return softmax with class 5 dominant
        let mut sm = [[0.0f32; NUM_CLASSES]; 3];
        for row in &mut sm {
            row[5] = 0.9;
            for (i, p) in row.iter_mut().enumerate() {
                if i != 5 {
                    *p = 0.1 / 20.0;
                }
            }
        }
        let m = meta_features(&sm);
        // agreement = 1.0 (all picked class 5)
        assert!(approx(m[2], 1.0, 1e-6));
        // std_exp = 0 (all identical)
        assert!(approx(m[1], 0.0, 1e-6));
        // max_p_consensus = 0.9
        assert!(approx(m[4], 0.9, 1e-6));
    }
}
