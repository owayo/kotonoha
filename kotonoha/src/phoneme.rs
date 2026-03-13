//! 音素定義モジュール
//! HTS Full-Context Labelで使用される音素セットを定義する

/// 音素の種類
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PhonemeType {
    Vowel,
    Consonant,
    Special,
}

/// 音素情報
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Phoneme {
    pub symbol: &'static str,
    pub phoneme_type: PhonemeType,
}

// 母音
pub const A: Phoneme = Phoneme { symbol: "a", phoneme_type: PhonemeType::Vowel };
pub const I: Phoneme = Phoneme { symbol: "i", phoneme_type: PhonemeType::Vowel };
pub const U: Phoneme = Phoneme { symbol: "u", phoneme_type: PhonemeType::Vowel };
pub const E: Phoneme = Phoneme { symbol: "e", phoneme_type: PhonemeType::Vowel };
pub const O: Phoneme = Phoneme { symbol: "o", phoneme_type: PhonemeType::Vowel };

// 無声母音
pub const VOICELESS_A: Phoneme = Phoneme { symbol: "A", phoneme_type: PhonemeType::Vowel };
pub const VOICELESS_I: Phoneme = Phoneme { symbol: "I", phoneme_type: PhonemeType::Vowel };
pub const VOICELESS_U: Phoneme = Phoneme { symbol: "U", phoneme_type: PhonemeType::Vowel };
pub const VOICELESS_E: Phoneme = Phoneme { symbol: "E", phoneme_type: PhonemeType::Vowel };
pub const VOICELESS_O: Phoneme = Phoneme { symbol: "O", phoneme_type: PhonemeType::Vowel };

// 特殊記号
pub const SIL: Phoneme = Phoneme { symbol: "sil", phoneme_type: PhonemeType::Special };
pub const PAU: Phoneme = Phoneme { symbol: "pau", phoneme_type: PhonemeType::Special };
pub const CL: Phoneme = Phoneme { symbol: "cl", phoneme_type: PhonemeType::Special };  // 促音

// 子音
pub const CONSONANTS: &[&str] = &[
    "k", "ky", "g", "gy",
    "s", "sh", "z", "j",
    "t", "ts", "ch", "d", "dy",
    "n", "ny",
    "h", "hy", "f",
    "b", "by", "p", "py",
    "m", "my",
    "r", "ry",
    "w", "y",
    "v",
];

pub const VOWELS: &[&str] = &["a", "i", "u", "e", "o", "A", "I", "U", "E", "O"];

pub const SPECIAL_SYMBOLS: &[&str] = &["sil", "pau", "cl", "N"];

/// 全音素リスト
pub const ALL_PHONEMES: &[&str] = &[
    "a", "i", "u", "e", "o",
    "A", "I", "U", "E", "O",
    "N", "cl", "pau", "sil",
    "k", "ky", "g", "gy",
    "s", "sh", "z", "j",
    "t", "ts", "ch", "d", "dy",
    "n", "ny",
    "h", "hy", "f",
    "b", "by", "p", "py",
    "m", "my",
    "r", "ry",
    "w", "y",
    "v",
];

/// 音素が母音かどうか判定
pub fn is_vowel(symbol: &str) -> bool {
    matches!(symbol, "a" | "i" | "u" | "e" | "o" | "A" | "I" | "U" | "E" | "O")
}

/// 音素が無声母音かどうか判定
pub fn is_voiceless_vowel(symbol: &str) -> bool {
    matches!(symbol, "A" | "I" | "U" | "E" | "O")
}

/// 音素が子音かどうか判定
pub fn is_consonant(symbol: &str) -> bool {
    CONSONANTS.contains(&symbol)
}

/// 音素が特殊記号かどうか判定
pub fn is_special(symbol: &str) -> bool {
    SPECIAL_SYMBOLS.contains(&symbol)
}

/// カタカナ表記から音素列に変換する
/// 例: "カ" → ["k", "a"], "キャ" → ["ky", "a"]
pub fn katakana_to_phonemes(kana: &str) -> Vec<String> {
    let mut phonemes = Vec::new();
    let chars: Vec<char> = kana.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        let ch = chars[i];
        // 次の文字が小文字（拗音）かチェック
        let next = if i + 1 < chars.len() { Some(chars[i + 1]) } else { None };
        let is_small_next = next.is_some_and(|c| matches!(c, 'ァ' | 'ィ' | 'ゥ' | 'ェ' | 'ォ' | 'ャ' | 'ュ' | 'ョ'));

        match ch {
            'ア' => phonemes.push("a".to_string()),
            'イ' => phonemes.push("i".to_string()),
            'ウ' => {
                if is_small_next {
                    match next.unwrap() {
                        'ィ' => { phonemes.extend(["w".to_string(), "i".to_string()]); i += 1; }
                        'ェ' => { phonemes.extend(["w".to_string(), "e".to_string()]); i += 1; }
                        'ォ' => { phonemes.extend(["w".to_string(), "o".to_string()]); i += 1; }
                        _ => phonemes.push("u".to_string()),
                    }
                } else {
                    phonemes.push("u".to_string());
                }
            }
            'エ' => phonemes.push("e".to_string()),
            'オ' => phonemes.push("o".to_string()),
            'カ' => phonemes.extend(["k".to_string(), "a".to_string()]),
            'キ' => {
                if is_small_next && next == Some('ャ') {
                    phonemes.extend(["ky".to_string(), "a".to_string()]); i += 1;
                } else if is_small_next && next == Some('ュ') {
                    phonemes.extend(["ky".to_string(), "u".to_string()]); i += 1;
                } else if is_small_next && next == Some('ョ') {
                    phonemes.extend(["ky".to_string(), "o".to_string()]); i += 1;
                } else if is_small_next && next == Some('ェ') {
                    phonemes.extend(["ky".to_string(), "e".to_string()]); i += 1;
                } else {
                    phonemes.extend(["k".to_string(), "i".to_string()]);
                }
            }
            'ク' => phonemes.extend(["k".to_string(), "u".to_string()]),
            'ケ' => phonemes.extend(["k".to_string(), "e".to_string()]),
            'コ' => phonemes.extend(["k".to_string(), "o".to_string()]),
            'サ' => phonemes.extend(["s".to_string(), "a".to_string()]),
            'シ' => {
                if is_small_next && next == Some('ャ') {
                    phonemes.extend(["sh".to_string(), "a".to_string()]); i += 1;
                } else if is_small_next && next == Some('ュ') {
                    phonemes.extend(["sh".to_string(), "u".to_string()]); i += 1;
                } else if is_small_next && next == Some('ョ') {
                    phonemes.extend(["sh".to_string(), "o".to_string()]); i += 1;
                } else if is_small_next && next == Some('ェ') {
                    phonemes.extend(["sh".to_string(), "e".to_string()]); i += 1;
                } else {
                    phonemes.extend(["sh".to_string(), "i".to_string()]);
                }
            }
            'ス' => phonemes.extend(["s".to_string(), "u".to_string()]),
            'セ' => phonemes.extend(["s".to_string(), "e".to_string()]),
            'ソ' => phonemes.extend(["s".to_string(), "o".to_string()]),
            'タ' => phonemes.extend(["t".to_string(), "a".to_string()]),
            'チ' => {
                if is_small_next && next == Some('ャ') {
                    phonemes.extend(["ch".to_string(), "a".to_string()]); i += 1;
                } else if is_small_next && next == Some('ュ') {
                    phonemes.extend(["ch".to_string(), "u".to_string()]); i += 1;
                } else if is_small_next && next == Some('ョ') {
                    phonemes.extend(["ch".to_string(), "o".to_string()]); i += 1;
                } else if is_small_next && next == Some('ェ') {
                    phonemes.extend(["ch".to_string(), "e".to_string()]); i += 1;
                } else {
                    phonemes.extend(["ch".to_string(), "i".to_string()]);
                }
            }
            'ツ' => {
                if is_small_next && next == Some('ァ') {
                    phonemes.extend(["ts".to_string(), "a".to_string()]); i += 1;
                } else if is_small_next && next == Some('ィ') {
                    phonemes.extend(["ts".to_string(), "i".to_string()]); i += 1;
                } else if is_small_next && next == Some('ェ') {
                    phonemes.extend(["ts".to_string(), "e".to_string()]); i += 1;
                } else if is_small_next && next == Some('ォ') {
                    phonemes.extend(["ts".to_string(), "o".to_string()]); i += 1;
                } else {
                    phonemes.extend(["ts".to_string(), "u".to_string()]);
                }
            }
            'テ' => {
                if is_small_next && next == Some('ィ') {
                    phonemes.extend(["t".to_string(), "i".to_string()]); i += 1;
                } else if is_small_next && next == Some('ュ') {
                    phonemes.extend(["ty".to_string(), "u".to_string()]); i += 1;
                } else {
                    phonemes.extend(["t".to_string(), "e".to_string()]);
                }
            }
            'ト' => phonemes.extend(["t".to_string(), "o".to_string()]),
            'ナ' => phonemes.extend(["n".to_string(), "a".to_string()]),
            'ニ' => {
                if is_small_next && next == Some('ャ') {
                    phonemes.extend(["ny".to_string(), "a".to_string()]); i += 1;
                } else if is_small_next && next == Some('ュ') {
                    phonemes.extend(["ny".to_string(), "u".to_string()]); i += 1;
                } else if is_small_next && next == Some('ョ') {
                    phonemes.extend(["ny".to_string(), "o".to_string()]); i += 1;
                } else {
                    phonemes.extend(["n".to_string(), "i".to_string()]);
                }
            }
            'ヌ' => phonemes.extend(["n".to_string(), "u".to_string()]),
            'ネ' => phonemes.extend(["n".to_string(), "e".to_string()]),
            'ノ' => phonemes.extend(["n".to_string(), "o".to_string()]),
            'ハ' => phonemes.extend(["h".to_string(), "a".to_string()]),
            'ヒ' => {
                if is_small_next && next == Some('ャ') {
                    phonemes.extend(["hy".to_string(), "a".to_string()]); i += 1;
                } else if is_small_next && next == Some('ュ') {
                    phonemes.extend(["hy".to_string(), "u".to_string()]); i += 1;
                } else if is_small_next && next == Some('ョ') {
                    phonemes.extend(["hy".to_string(), "o".to_string()]); i += 1;
                } else {
                    phonemes.extend(["h".to_string(), "i".to_string()]);
                }
            }
            'フ' => {
                if is_small_next && next == Some('ァ') {
                    phonemes.extend(["f".to_string(), "a".to_string()]); i += 1;
                } else if is_small_next && next == Some('ィ') {
                    phonemes.extend(["f".to_string(), "i".to_string()]); i += 1;
                } else if is_small_next && next == Some('ェ') {
                    phonemes.extend(["f".to_string(), "e".to_string()]); i += 1;
                } else if is_small_next && next == Some('ォ') {
                    phonemes.extend(["f".to_string(), "o".to_string()]); i += 1;
                } else {
                    phonemes.extend(["f".to_string(), "u".to_string()]);
                }
            }
            'ヘ' => phonemes.extend(["h".to_string(), "e".to_string()]),
            'ホ' => phonemes.extend(["h".to_string(), "o".to_string()]),
            'マ' => phonemes.extend(["m".to_string(), "a".to_string()]),
            'ミ' => {
                if is_small_next && next == Some('ャ') {
                    phonemes.extend(["my".to_string(), "a".to_string()]); i += 1;
                } else if is_small_next && next == Some('ュ') {
                    phonemes.extend(["my".to_string(), "u".to_string()]); i += 1;
                } else if is_small_next && next == Some('ョ') {
                    phonemes.extend(["my".to_string(), "o".to_string()]); i += 1;
                } else {
                    phonemes.extend(["m".to_string(), "i".to_string()]);
                }
            }
            'ム' => phonemes.extend(["m".to_string(), "u".to_string()]),
            'メ' => phonemes.extend(["m".to_string(), "e".to_string()]),
            'モ' => phonemes.extend(["m".to_string(), "o".to_string()]),
            'ヤ' => phonemes.extend(["y".to_string(), "a".to_string()]),
            'ユ' => phonemes.extend(["y".to_string(), "u".to_string()]),
            'ヨ' => phonemes.extend(["y".to_string(), "o".to_string()]),
            'ラ' => phonemes.extend(["r".to_string(), "a".to_string()]),
            'リ' => {
                if is_small_next && next == Some('ャ') {
                    phonemes.extend(["ry".to_string(), "a".to_string()]); i += 1;
                } else if is_small_next && next == Some('ュ') {
                    phonemes.extend(["ry".to_string(), "u".to_string()]); i += 1;
                } else if is_small_next && next == Some('ョ') {
                    phonemes.extend(["ry".to_string(), "o".to_string()]); i += 1;
                } else {
                    phonemes.extend(["r".to_string(), "i".to_string()]);
                }
            }
            'ル' => phonemes.extend(["r".to_string(), "u".to_string()]),
            'レ' => phonemes.extend(["r".to_string(), "e".to_string()]),
            'ロ' => phonemes.extend(["r".to_string(), "o".to_string()]),
            'ワ' => phonemes.extend(["w".to_string(), "a".to_string()]),
            'ヲ' => phonemes.extend(["o".to_string()]),
            'ン' => phonemes.push("N".to_string()),
            'ガ' => phonemes.extend(["g".to_string(), "a".to_string()]),
            'ギ' => {
                if is_small_next && next == Some('ャ') {
                    phonemes.extend(["gy".to_string(), "a".to_string()]); i += 1;
                } else if is_small_next && next == Some('ュ') {
                    phonemes.extend(["gy".to_string(), "u".to_string()]); i += 1;
                } else if is_small_next && next == Some('ョ') {
                    phonemes.extend(["gy".to_string(), "o".to_string()]); i += 1;
                } else {
                    phonemes.extend(["g".to_string(), "i".to_string()]);
                }
            }
            'グ' => phonemes.extend(["g".to_string(), "u".to_string()]),
            'ゲ' => phonemes.extend(["g".to_string(), "e".to_string()]),
            'ゴ' => phonemes.extend(["g".to_string(), "o".to_string()]),
            'ザ' => phonemes.extend(["z".to_string(), "a".to_string()]),
            'ジ' => {
                if is_small_next && next == Some('ャ') {
                    phonemes.extend(["j".to_string(), "a".to_string()]); i += 1;
                } else if is_small_next && next == Some('ュ') {
                    phonemes.extend(["j".to_string(), "u".to_string()]); i += 1;
                } else if is_small_next && next == Some('ョ') {
                    phonemes.extend(["j".to_string(), "o".to_string()]); i += 1;
                } else {
                    phonemes.extend(["j".to_string(), "i".to_string()]);
                }
            }
            'ズ' => phonemes.extend(["z".to_string(), "u".to_string()]),
            'ゼ' => phonemes.extend(["z".to_string(), "e".to_string()]),
            'ゾ' => phonemes.extend(["z".to_string(), "o".to_string()]),
            'ダ' => phonemes.extend(["d".to_string(), "a".to_string()]),
            'ヂ' => phonemes.extend(["j".to_string(), "i".to_string()]),
            'ヅ' => phonemes.extend(["z".to_string(), "u".to_string()]),
            'デ' => {
                if is_small_next && next == Some('ィ') {
                    phonemes.extend(["dy".to_string(), "i".to_string()]); i += 1;
                } else if is_small_next && next == Some('ュ') {
                    phonemes.extend(["dy".to_string(), "u".to_string()]); i += 1;
                } else {
                    phonemes.extend(["d".to_string(), "e".to_string()]);
                }
            }
            'ド' => {
                if is_small_next && next == Some('ゥ') {
                    phonemes.extend(["d".to_string(), "u".to_string()]); i += 1;
                } else {
                    phonemes.extend(["d".to_string(), "o".to_string()]);
                }
            }
            'バ' => phonemes.extend(["b".to_string(), "a".to_string()]),
            'ビ' => {
                if is_small_next && next == Some('ャ') {
                    phonemes.extend(["by".to_string(), "a".to_string()]); i += 1;
                } else if is_small_next && next == Some('ュ') {
                    phonemes.extend(["by".to_string(), "u".to_string()]); i += 1;
                } else if is_small_next && next == Some('ョ') {
                    phonemes.extend(["by".to_string(), "o".to_string()]); i += 1;
                } else {
                    phonemes.extend(["b".to_string(), "i".to_string()]);
                }
            }
            'ブ' => phonemes.extend(["b".to_string(), "u".to_string()]),
            'ベ' => phonemes.extend(["b".to_string(), "e".to_string()]),
            'ボ' => phonemes.extend(["b".to_string(), "o".to_string()]),
            'パ' => phonemes.extend(["p".to_string(), "a".to_string()]),
            'ピ' => {
                if is_small_next && next == Some('ャ') {
                    phonemes.extend(["py".to_string(), "a".to_string()]); i += 1;
                } else if is_small_next && next == Some('ュ') {
                    phonemes.extend(["py".to_string(), "u".to_string()]); i += 1;
                } else if is_small_next && next == Some('ョ') {
                    phonemes.extend(["py".to_string(), "o".to_string()]); i += 1;
                } else {
                    phonemes.extend(["p".to_string(), "i".to_string()]);
                }
            }
            'プ' => phonemes.extend(["p".to_string(), "u".to_string()]),
            'ペ' => phonemes.extend(["p".to_string(), "e".to_string()]),
            'ポ' => phonemes.extend(["p".to_string(), "o".to_string()]),
            'ヴ' => {
                if is_small_next && next == Some('ァ') {
                    phonemes.extend(["v".to_string(), "a".to_string()]); i += 1;
                } else if is_small_next && next == Some('ィ') {
                    phonemes.extend(["v".to_string(), "i".to_string()]); i += 1;
                } else if is_small_next && next == Some('ェ') {
                    phonemes.extend(["v".to_string(), "e".to_string()]); i += 1;
                } else if is_small_next && next == Some('ォ') {
                    phonemes.extend(["v".to_string(), "o".to_string()]); i += 1;
                } else {
                    phonemes.extend(["v".to_string(), "u".to_string()]);
                }
            }
            // 促音
            'ッ' => phonemes.push("cl".to_string()),
            // 長音
            'ー' => {
                // 直前の母音を繰り返す
                if let Some(last) = phonemes.last().cloned()
                    && is_vowel(&last)
                {
                    phonemes.push(last);
                }
            }
            // 小文字単体（直前にマッチしなかった場合）
            'ァ' => phonemes.push("a".to_string()),
            'ィ' => phonemes.push("i".to_string()),
            'ゥ' => phonemes.push("u".to_string()),
            'ェ' => phonemes.push("e".to_string()),
            'ォ' => phonemes.push("o".to_string()),
            'ャ' => phonemes.extend(["y".to_string(), "a".to_string()]),
            'ュ' => phonemes.extend(["y".to_string(), "u".to_string()]),
            'ョ' => phonemes.extend(["y".to_string(), "o".to_string()]),
            _ => {} // 未知の文字はスキップ
        }
        i += 1;
    }
    phonemes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_katakana_to_phonemes() {
        assert_eq!(katakana_to_phonemes("カ"), vec!["k", "a"]);
        assert_eq!(katakana_to_phonemes("キャ"), vec!["ky", "a"]);
        assert_eq!(katakana_to_phonemes("シ"), vec!["sh", "i"]);
        assert_eq!(katakana_to_phonemes("チ"), vec!["ch", "i"]);
        assert_eq!(katakana_to_phonemes("ツ"), vec!["ts", "u"]);
        assert_eq!(katakana_to_phonemes("ン"), vec!["N"]);
        assert_eq!(katakana_to_phonemes("ッ"), vec!["cl"]);
    }

    #[test]
    fn test_katakana_word() {
        assert_eq!(
            katakana_to_phonemes("コンニチワ"),
            vec!["k", "o", "N", "n", "i", "ch", "i", "w", "a"]
        );
    }

    #[test]
    fn test_foreign_sounds() {
        assert_eq!(katakana_to_phonemes("ウォ"), vec!["w", "o"]);
        assert_eq!(katakana_to_phonemes("ウィ"), vec!["w", "i"]);
        assert_eq!(katakana_to_phonemes("ウェ"), vec!["w", "e"]);
    }

    #[test]
    fn test_long_vowel() {
        assert_eq!(
            katakana_to_phonemes("コーヒー"),
            vec!["k", "o", "o", "h", "i", "i"]
        );
    }

    #[test]
    fn test_is_vowel() {
        assert!(is_vowel("a"));
        assert!(is_vowel("A"));
        assert!(!is_vowel("k"));
        assert!(!is_vowel("N"));
    }
}
