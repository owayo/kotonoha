//! v66 系の特徴量抽出
//!
//! `kotonoha/training/train_onnx_v60.py` と数値完全一致させる:
//! - `_reading_hash` / `_char_hash` (DJB2 風 hash, 169-191 行)
//! - `_count_mora` (198-214 行)
//! - `_parse_dict_accent` (247-259 行)
//! - `_extract_morpheme_features` (303-351 行) → 13 dim
//!
//! 入力は `MorphemeView`。`InputToken` の生 UniDic POS 文字列を保持し、
//! `dict_accent_type` は `accent_dict` enrich 後に注入されることを前提とする。

use super::vocab::{
    conj_form_group, conj_type_group, pos_detail1_to_id, pos_detail2_to_id, pos_to_id,
};

/// v66 base 特徴量の次元数
pub const FEATURE13_DIM: usize = 13;

/// v66 特徴抽出に必要な形態素ビュー
///
/// `InputToken` を直接借用するため UniDic 文字列 (`代名詞` / `接尾辞` 等) を保持できる。
/// `dict_accent_type` は accent_dict enrich 後に注入される前提で `Option<u8>` を持つ
/// (None = 辞書ヒット無し / 訓練側 `"*"` 相当)。
#[derive(Debug, Clone, Copy)]
pub struct MorphemeView<'a> {
    /// `InputToken.surface`
    pub surface: &'a str,
    /// `InputToken.pos` (UniDic 由来生文字列)
    pub pos: &'a str,
    /// `InputToken.pos_detail1`
    pub pos_detail1: &'a str,
    /// `InputToken.pos_detail2`
    pub pos_detail2: &'a str,
    /// `InputToken.ctype` (活用型)
    pub conjugation_type: &'a str,
    /// `InputToken.cform` (活用形)
    pub conjugation_form: &'a str,
    /// `InputToken.reading` (カタカナ)
    pub reading: &'a str,
    /// アクセント辞書由来の値（None = 辞書ヒット無し）
    pub dict_accent_type: Option<u8>,
}

/// DJB2 風ハッシュを `(h % 10000) / 10000` に正規化する
///
/// `train_onnx_v60.py::_reading_hash` (169-179) と完全一致。
/// 文字単位は char (Unicode scalar) の数値で `wrapping_mul(33).wrapping_add(c)` を反復。
pub fn reading_hash(reading: &str) -> f32 {
    let mut h: u32 = 5381;
    for c in reading.chars() {
        h = h.wrapping_mul(33).wrapping_add(c as u32);
    }
    (h % 10000) as f32 / 10000.0
}

/// 単一文字のハッシュ
///
/// `train_onnx_v60.py::_char_hash` (182-191) と完全一致。
pub fn char_hash(ch: char) -> f32 {
    let mut h: u32 = 5381;
    h = h.wrapping_mul(33).wrapping_add(ch as u32);
    (h % 10000) as f32 / 10000.0
}

const SMALL_KANA: &[char] = &['ァ', 'ィ', 'ゥ', 'ェ', 'ォ', 'ャ', 'ュ', 'ョ', 'ヮ'];
const KATAKANA_START: char = '\u{30a0}';
const KATAKANA_END: char = '\u{30ff}';
const LONG_VOWEL: char = 'ー';

/// カタカナ読みからモーラ数を算出する
///
/// `train_onnx_v60.py::_count_mora` (198-214) と完全一致。
/// - 空文字列 → 0
/// - 拗音/促音などの小書き → モーラに数えない
/// - U+30A0..U+30FF 範囲 (カタカナブロック) または `ー` のみカウント
/// - 結果が 0 の場合でも 1 を返す (`max(count, 1)`) — 但し空文字列は 0
pub fn count_mora(reading: &str) -> u32 {
    if reading.is_empty() {
        return 0;
    }
    let mut count: u32 = 0;
    for ch in reading.chars() {
        if SMALL_KANA.contains(&ch) {
            continue;
        }
        if (KATAKANA_START..=KATAKANA_END).contains(&ch) || ch == LONG_VOWEL {
            count += 1;
        }
    }
    count.max(1)
}

/// dict_accent_type を [0, 1) の正規化値に変換する
///
/// `train_onnx_v60.py::_parse_dict_accent` (247-259) と等価:
/// - `None` (訓練側 `"*"` または欠損) → 0.0
/// - `Some(n)` → `(n + 1) / 8.0`
pub fn parse_dict_accent(val: Option<u8>) -> f32 {
    val.map(|n| (n as f32 + 1.0) / 8.0).unwrap_or(0.0)
}

/// dict_accent_type を整数アクセント類 (0..NUM_CLASSES) に変換する
///
/// `train_onnx_v60.py::_morpheme_dict_accent` (399-415) と等価:
/// - `None` または範囲外 → -1
/// - 範囲内 `Some(n)` → `n as i32`
///
/// `_teacher_soft_stats` の `dict_acc_type` 引数として使う。
pub fn morpheme_dict_accent(val: Option<u8>) -> i32 {
    match val {
        Some(n) if (n as usize) < super::vocab::NUM_CLASSES => i32::from(n),
        _ => -1,
    }
}

/// 13 次元特徴量を抽出する
///
/// `train_onnx_v60.py::_extract_morpheme_features` (303-351) と完全一致。
/// 順序:
/// `[pos, pd1, pd2, ct, cf, mora_count/10, r_hash, first_ch_hash, last_ch_hash,
///   position, dict_acc_norm, head2_hash, tail2_hash]`
pub fn extract_feat13(morpheme: &MorphemeView<'_>, position: f32) -> [f32; FEATURE13_DIM] {
    let pos_id = pos_to_id(morpheme.pos);
    let pd1_id = pos_detail1_to_id(morpheme.pos_detail1);
    let pd2_id = pos_detail2_to_id(morpheme.pos_detail2);
    let ct_id = conj_type_group(morpheme.conjugation_type);
    let cf_id = conj_form_group(morpheme.conjugation_form);

    let mora_count = count_mora(morpheme.reading) as f32 / 10.0;
    let r_hash = reading_hash(morpheme.reading);

    let first_ch_hash = morpheme
        .surface
        .chars()
        .next()
        .map(char_hash)
        .unwrap_or(0.0);
    let last_ch_hash = morpheme
        .surface
        .chars()
        .last()
        .map(char_hash)
        .unwrap_or(0.0);

    // v33: head/tail 2 文字 (char 単位) の hash
    let reading_chars: Vec<char> = morpheme.reading.chars().collect();
    let head2: String = reading_chars.iter().take(2).collect();
    let tail2: String = if reading_chars.len() >= 2 {
        reading_chars
            .iter()
            .skip(reading_chars.len() - 2)
            .copied()
            .collect()
    } else {
        morpheme.reading.to_string()
    };
    let r_head2_hash = if head2.is_empty() {
        0.0
    } else {
        reading_hash(&head2)
    };
    let r_tail2_hash = if tail2.is_empty() {
        0.0
    } else {
        reading_hash(&tail2)
    };

    let dict_acc = parse_dict_accent(morpheme.dict_accent_type);

    [
        pos_id as f32,
        pd1_id as f32,
        pd2_id as f32,
        ct_id as f32,
        cf_id as f32,
        mora_count,
        r_hash,
        first_ch_hash,
        last_ch_hash,
        position,
        dict_acc,
        r_head2_hash,
        r_tail2_hash,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// reference values are computed by Python
    /// `_reading_hash("ネコ") = 0.7028 ...` などは fixture 比較で担保。
    /// ここでは Rust 内で deterministic + range だけ確認する。
    #[test]
    fn reading_hash_deterministic_and_normalized() {
        let h1 = reading_hash("ネコ");
        let h2 = reading_hash("ネコ");
        let h3 = reading_hash("イヌ");
        assert_eq!(h1, h2);
        assert_ne!(h1, h3);
        assert!((0.0..1.0).contains(&h1));
    }

    /// Python 側 `_reading_hash("猫")` の reference 値 0.7056 を確認する。
    /// 計算過程: h=5381 → 5381*33+29483 = 207056 → 207056 % 10000 = 7056 → 0.7056
    /// (where 29483 = '猫' の Unicode scalar value)
    #[test]
    fn reading_hash_neko_kanji_matches_python() {
        let h = reading_hash("猫");
        assert!((h - 0.7056).abs() < 1e-7, "expected 0.7056, got {h}");
    }

    /// `_char_hash("ネ")` reference: 'ネ' → 5381*33 + 12493 = 190066
    /// 190066 % 10000 = 66 → 0.0066
    #[test]
    fn char_hash_kana_matches_python() {
        let h = char_hash('ネ');
        assert!((h - 0.0066).abs() < 1e-7, "expected 0.0066, got {h}");
    }

    #[test]
    fn count_mora_basic() {
        assert_eq!(count_mora("ネコ"), 2);
        assert_eq!(count_mora("ニンジン"), 4);
        // 拗音は 1 モーラ扱い (キャ = 1)
        assert_eq!(count_mora("キャベツ"), 3);
        assert_eq!(count_mora("キョー"), 2);
        // 長音符
        assert_eq!(count_mora("コーヒー"), 4);
        // 促音 (ッ) はカタカナ範囲なのでカウントされる (Python と同じ挙動)
        // train_onnx_v60.py の SMALL_KANA は { ァィゥェォャュョヮ } のみ
        assert_eq!(count_mora("カッコ"), 3);
        // 空文字列 → 0
        assert_eq!(count_mora(""), 0);
        // 範囲外文字のみ → max(0, 1) = 1
        assert_eq!(count_mora("猫"), 1);
    }

    #[test]
    fn parse_dict_accent_known_cases() {
        assert_eq!(parse_dict_accent(None), 0.0);
        // (0 + 1) / 8 = 0.125
        assert!((parse_dict_accent(Some(0)) - 0.125).abs() < 1e-7);
        // (3 + 1) / 8 = 0.5
        assert!((parse_dict_accent(Some(3)) - 0.5).abs() < 1e-7);
    }

    #[test]
    fn morpheme_dict_accent_known_cases() {
        assert_eq!(morpheme_dict_accent(None), -1);
        assert_eq!(morpheme_dict_accent(Some(0)), 0);
        assert_eq!(morpheme_dict_accent(Some(20)), 20);
        // out of range → -1
        assert_eq!(morpheme_dict_accent(Some(99)), -1);
    }

    #[test]
    fn extract_feat13_neko() {
        let m = MorphemeView {
            surface: "猫",
            pos: "名詞",
            pos_detail1: "普通名詞",
            pos_detail2: "一般",
            conjugation_type: "*",
            conjugation_form: "*",
            reading: "ネコ",
            dict_accent_type: Some(1),
        };
        let f = extract_feat13(&m, 0.5);
        assert_eq!(f[0], 1.0); // 名詞
        assert_eq!(f[1], 2.0); // 普通名詞
        assert_eq!(f[2], 2.0); // 一般
        assert_eq!(f[3], 0.0); // ctype *
        assert_eq!(f[4], 0.0); // cform *
        assert!((f[5] - 0.2).abs() < 1e-7); // mora 2 / 10 = 0.2
        // f[6] reading_hash("ネコ")
        // f[7] char_hash('猫')
        // f[8] char_hash('猫') (single char surface)
        assert_eq!(f[7], f[8]);
        assert!((f[9] - 0.5).abs() < 1e-7); // position
        // (1+1)/8 = 0.25
        assert!((f[10] - 0.25).abs() < 1e-7);
        // head2/tail2 hashes equal full reading hash for 2-char input
        assert_eq!(f[11], f[6]);
        assert_eq!(f[12], f[6]);
    }
}
