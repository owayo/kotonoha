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
pub const A: Phoneme = Phoneme {
    symbol: "a",
    phoneme_type: PhonemeType::Vowel,
};
pub const I: Phoneme = Phoneme {
    symbol: "i",
    phoneme_type: PhonemeType::Vowel,
};
pub const U: Phoneme = Phoneme {
    symbol: "u",
    phoneme_type: PhonemeType::Vowel,
};
pub const E: Phoneme = Phoneme {
    symbol: "e",
    phoneme_type: PhonemeType::Vowel,
};
pub const O: Phoneme = Phoneme {
    symbol: "o",
    phoneme_type: PhonemeType::Vowel,
};

// 無声母音
pub const VOICELESS_A: Phoneme = Phoneme {
    symbol: "A",
    phoneme_type: PhonemeType::Vowel,
};
pub const VOICELESS_I: Phoneme = Phoneme {
    symbol: "I",
    phoneme_type: PhonemeType::Vowel,
};
pub const VOICELESS_U: Phoneme = Phoneme {
    symbol: "U",
    phoneme_type: PhonemeType::Vowel,
};
pub const VOICELESS_E: Phoneme = Phoneme {
    symbol: "E",
    phoneme_type: PhonemeType::Vowel,
};
pub const VOICELESS_O: Phoneme = Phoneme {
    symbol: "O",
    phoneme_type: PhonemeType::Vowel,
};

// 特殊記号
pub const SIL: Phoneme = Phoneme {
    symbol: "sil",
    phoneme_type: PhonemeType::Special,
};
pub const PAU: Phoneme = Phoneme {
    symbol: "pau",
    phoneme_type: PhonemeType::Special,
};
pub const CL: Phoneme = Phoneme {
    symbol: "cl",
    phoneme_type: PhonemeType::Special,
}; // 促音

// 子音
pub const CONSONANTS: &[&str] = &[
    "k", "ky", "g", "gy", "s", "sh", "z", "j", "t", "ts", "ch", "ty", "d", "dy", "n", "ny", "h",
    "hy", "f", "b", "by", "p", "py", "m", "my", "r", "ry", "w", "y", "v",
];

pub const VOWELS: &[&str] = &["a", "i", "u", "e", "o", "A", "I", "U", "E", "O"];

pub const SPECIAL_SYMBOLS: &[&str] = &["sil", "pau", "cl", "N"];

/// 全音素リスト
pub const ALL_PHONEMES: &[&str] = &[
    "a", "i", "u", "e", "o", "A", "I", "U", "E", "O", "N", "cl", "pau", "sil", "k", "ky", "g",
    "gy", "s", "sh", "z", "j", "t", "ts", "ch", "ty", "d", "dy", "n", "ny", "h", "hy", "f", "b",
    "by", "p", "py", "m", "my", "r", "ry", "w", "y", "v",
];

/// 音素が母音かどうか判定
pub fn is_vowel(symbol: &str) -> bool {
    matches!(
        symbol,
        "a" | "i" | "u" | "e" | "o" | "A" | "I" | "U" | "E" | "O"
    )
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
    let mut phonemes: Vec<String> = Vec::new();
    let mut chars = kana.chars().peekable();

    while let Some(ch) = chars.next() {
        if let Some(pair) = chars
            .peek()
            .and_then(|&next| combined_kana_phonemes(ch, next))
        {
            phonemes.extend(pair.map(str::to_owned));
            chars.next();
            continue;
        }

        if ch == 'ー' {
            if let Some(last) = phonemes.last()
                && is_vowel(last)
            {
                phonemes.push(last.clone());
            }
            continue;
        }

        // 明示した組み合わせ以外の小書き母音も、直前が子音と母音なら母音を差し替える。
        // 「ビィ」から存在しない子音 "bi" が作られることを防ぐ。
        if matches!(ch, 'ァ' | 'ィ' | 'ゥ' | 'ェ' | 'ォ')
            && phonemes.len() >= 2
            && is_vowel(&phonemes[phonemes.len() - 1])
            && is_consonant(&phonemes[phonemes.len() - 2])
        {
            phonemes.pop();
        }
        phonemes.extend(
            single_kana_phonemes(ch)
                .iter()
                .map(|&phone| phone.to_owned()),
        );
    }
    phonemes
}

/// 2文字で1モーラになる組み合わせの子音と母音を返す。
/// 対応しない組み合わせは文字を消費せず、単独の仮名として処理する。
pub(crate) fn combined_kana_phonemes(first: char, second: char) -> Option<[&'static str; 2]> {
    let consonant = match (first, second) {
        ('イ', 'ェ') => "y",
        ('ウ', 'ィ' | 'ェ' | 'ォ') => "w",
        ('キ', 'ャ' | 'ュ' | 'ョ' | 'ェ') => "ky",
        ('ギ', 'ャ' | 'ュ' | 'ョ' | 'ェ') => "gy",
        ('シ', 'ャ' | 'ュ' | 'ョ' | 'ェ') => "sh",
        ('ジ', 'ャ' | 'ュ' | 'ョ' | 'ェ') => "j",
        ('ス', 'ィ') => "s",
        ('ズ', 'ィ') => "z",
        ('チ', 'ャ' | 'ュ' | 'ョ' | 'ェ') => "ch",
        ('ツ', 'ァ' | 'ィ' | 'ェ' | 'ォ') => "ts",
        ('テ', 'ャ' | 'ュ' | 'ョ') => "ty",
        ('デ', 'ャ' | 'ュ' | 'ョ') => "dy",
        ('テ', 'ィ') | ('ト', 'ゥ') => "t",
        ('デ', 'ィ') | ('ド', 'ゥ') => "d",
        ('ニ', 'ャ' | 'ュ' | 'ョ' | 'ェ') => "ny",
        ('ヒ', 'ャ' | 'ュ' | 'ョ' | 'ェ') => "hy",
        ('ビ', 'ャ' | 'ュ' | 'ョ' | 'ェ') | ('ヴ', 'ャ' | 'ュ' | 'ョ') => "by",
        ('ピ', 'ャ' | 'ュ' | 'ョ' | 'ェ') => "py",
        ('フ', 'ァ' | 'ィ' | 'ェ' | 'ォ') => "f",
        ('ミ', 'ャ' | 'ュ' | 'ョ' | 'ェ') => "my",
        ('リ', 'ャ' | 'ュ' | 'ョ' | 'ェ') => "ry",
        ('ヴ', 'ァ' | 'ィ' | 'ェ' | 'ォ') => "v",
        _ => return None,
    };
    let vowel = match second {
        'ァ' | 'ャ' => "a",
        'ィ' => "i",
        'ゥ' | 'ュ' => "u",
        'ェ' => "e",
        'ォ' | 'ョ' => "o",
        _ => return None,
    };
    Some([consonant, vowel])
}

/// 単独の仮名を音素へ写す。長音と小書き母音の文脈依存処理は走査側で行う。
pub(crate) fn single_kana_phonemes(kana: char) -> &'static [&'static str] {
    match kana {
        'ア' | 'ァ' => &["a"],
        'イ' | 'ィ' | 'ヰ' => &["i"],
        'ウ' | 'ゥ' => &["u"],
        'エ' | 'ェ' | 'ヱ' => &["e"],
        'オ' | 'ォ' | 'ヲ' => &["o"],
        'カ' | 'ヵ' => &["k", "a"],
        'キ' => &["k", "i"],
        'ク' => &["k", "u"],
        'ケ' | 'ヶ' => &["k", "e"],
        'コ' => &["k", "o"],
        'ガ' => &["g", "a"],
        'ギ' => &["g", "i"],
        'グ' => &["g", "u"],
        'ゲ' => &["g", "e"],
        'ゴ' => &["g", "o"],
        'サ' => &["s", "a"],
        'シ' => &["sh", "i"],
        'ス' => &["s", "u"],
        'セ' => &["s", "e"],
        'ソ' => &["s", "o"],
        'ザ' => &["z", "a"],
        'ジ' | 'ヂ' => &["j", "i"],
        'ズ' | 'ヅ' => &["z", "u"],
        'ゼ' => &["z", "e"],
        'ゾ' => &["z", "o"],
        'タ' => &["t", "a"],
        'チ' => &["ch", "i"],
        'ツ' => &["ts", "u"],
        'テ' => &["t", "e"],
        'ト' => &["t", "o"],
        'ダ' => &["d", "a"],
        'デ' => &["d", "e"],
        'ド' => &["d", "o"],
        'ナ' => &["n", "a"],
        'ニ' => &["n", "i"],
        'ヌ' => &["n", "u"],
        'ネ' => &["n", "e"],
        'ノ' => &["n", "o"],
        'ハ' => &["h", "a"],
        'ヒ' => &["h", "i"],
        'フ' => &["f", "u"],
        'ヘ' => &["h", "e"],
        'ホ' => &["h", "o"],
        'バ' => &["b", "a"],
        'ビ' => &["b", "i"],
        'ブ' => &["b", "u"],
        'ベ' => &["b", "e"],
        'ボ' => &["b", "o"],
        'パ' => &["p", "a"],
        'ピ' => &["p", "i"],
        'プ' => &["p", "u"],
        'ペ' => &["p", "e"],
        'ポ' => &["p", "o"],
        'マ' => &["m", "a"],
        'ミ' => &["m", "i"],
        'ム' => &["m", "u"],
        'メ' => &["m", "e"],
        'モ' => &["m", "o"],
        'ヤ' | 'ャ' => &["y", "a"],
        'ユ' | 'ュ' => &["y", "u"],
        'ヨ' | 'ョ' => &["y", "o"],
        'ラ' => &["r", "a"],
        'リ' => &["r", "i"],
        'ル' => &["r", "u"],
        'レ' => &["r", "e"],
        'ロ' => &["r", "o"],
        'ワ' | 'ヮ' => &["w", "a"],
        'ヴ' => &["v", "u"],
        'ン' => &["N"],
        'ッ' => &["cl"],
        _ => &[], // 未知の文字はスキップする。
    }
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
