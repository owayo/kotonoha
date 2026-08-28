//! モーラ定義モジュール
//! カタカナ読みからモーラ数を計算する

use crate::phoneme;

/// モーラ情報
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mora {
    pub text: String,              // カタカナ表記（例: "キャ"）
    pub consonant: Option<String>, // 子音（例: "ky"）
    pub vowel: String,             // 母音（例: "a"）
}

/// カタカナ読みからモーラ数を計算する
/// 拗音は1モーラ、促音・撥音も各1モーラとカウント
pub fn count_mora(reading: &str) -> u8 {
    let chars: Vec<char> = reading.chars().collect();
    let mut count: u8 = 0;
    let mut i = 0;

    while i < chars.len() {
        let ch = chars[i];
        let next = if i + 1 < chars.len() {
            Some(chars[i + 1])
        } else {
            None
        };

        // 小文字カナ（拗音記号）は直前の文字とセットで1モーラ
        let is_small_next = next
            .is_some_and(|c| matches!(c, 'ァ' | 'ィ' | 'ゥ' | 'ェ' | 'ォ' | 'ャ' | 'ュ' | 'ョ'));

        // 語頭の長音記号は伸ばす母音が無いので音にならない。
        // 「リーター」が「リータ」+「ー」に分かれるようなケースで、
        // 音素の無いモーラを数えないようにする（parse_mora と数を揃える）
        if ch == 'ー' && count == 0 {
            i += 1;
            continue;
        }

        if is_katakana(ch) || ch == 'ー' {
            count += 1;
            if is_small_next && !is_small_kana(ch) {
                i += 1; // 拗音の小文字をスキップ
            }
        }
        i += 1;
    }
    count
}

/// カタカナ読みからモーラ列を構築する
pub fn parse_mora(reading: &str) -> Vec<Mora> {
    let chars: Vec<char> = reading.chars().collect();
    let mut moras = Vec::new();
    let mut i = 0;

    while i < chars.len() {
        let ch = chars[i];
        let next = if i + 1 < chars.len() {
            Some(chars[i + 1])
        } else {
            None
        };

        let is_small_next = next
            .is_some_and(|c| matches!(c, 'ァ' | 'ィ' | 'ゥ' | 'ェ' | 'ォ' | 'ャ' | 'ュ' | 'ョ'));

        if !is_katakana(ch) && ch != 'ー' {
            i += 1;
            continue;
        }

        let mora_text;
        if is_small_next && !is_small_kana(ch) {
            mora_text = format!("{}{}", ch, next.unwrap());
            i += 2;
        } else {
            mora_text = ch.to_string();
            i += 1;
        }

        if mora_text == "ー" {
            // 長音記号は直前のモーラの母音を伸ばす。直前が無ければ伸ばす母音が
            // 無く音にならないので、母音が空のモーラを作らずに読み飛ばす
            // （空のモーラを作ると下流の音素列に空文字が混ざる）
            if let Some(vowel) = moras.last().map(|m: &Mora| m.vowel.clone()) {
                moras.push(Mora {
                    text: mora_text,
                    consonant: None,
                    vowel,
                });
            }
            continue;
        }

        let phonemes = phoneme::katakana_to_phonemes(&mora_text);
        if phonemes.is_empty() {
            continue;
        }
        let (consonant, vowel) = split_cv(&phonemes);

        moras.push(Mora {
            text: mora_text,
            consonant,
            vowel,
        });
    }
    moras
}

/// 音素列を子音と母音に分離する
fn split_cv(phonemes: &[String]) -> (Option<String>, String) {
    match phonemes.len() {
        0 => (None, String::new()),
        1 => {
            let p = &phonemes[0];
            if phoneme::is_vowel(p) || p == "N" || p == "cl" {
                (None, p.clone())
            } else {
                (Some(p.clone()), String::new())
            }
        }
        _ => {
            let last = &phonemes[phonemes.len() - 1];
            if phoneme::is_vowel(last) {
                let consonant = phonemes[..phonemes.len() - 1].join("");
                (
                    if consonant.is_empty() {
                        None
                    } else {
                        Some(consonant)
                    },
                    last.clone(),
                )
            } else {
                (None, phonemes.join(""))
            }
        }
    }
}

/// カタカナ文字かどうか判定（長音含む）
///
/// 中黒「・」とダブルハイフン「゠」はカタカナブロックにあるが音を持たない。
/// モーラとして数えると音素の無いモーラができ、音素列に空文字が混ざるため除外する。
/// 辞書には「ヒューレット・パッカード」のように発音フィールドへ中黒が入った
/// エントリがある。
fn is_katakana(c: char) -> bool {
    ('\u{30A0}'..='\u{30FF}').contains(&c) && !matches!(c, '・' | '゠')
}

/// 小文字カタカナかどうか判定
fn is_small_kana(c: char) -> bool {
    matches!(
        c,
        'ァ' | 'ィ' | 'ゥ' | 'ェ' | 'ォ' | 'ャ' | 'ュ' | 'ョ' | 'ッ'
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_count_mora() {
        assert_eq!(count_mora("ア"), 1);
        assert_eq!(count_mora("カキクケコ"), 5);
        assert_eq!(count_mora("キャ"), 1);
        assert_eq!(count_mora("シャ"), 1);
        assert_eq!(count_mora("ッ"), 1);
        assert_eq!(count_mora("ン"), 1);
        assert_eq!(count_mora("コンニチワ"), 5);
        assert_eq!(count_mora("ガッコー"), 4); // ガ・ッ・コ・ー
        assert_eq!(count_mora("トーキョー"), 4); // ト・ー・キョ・ー
    }

    #[test]
    fn test_parse_mora() {
        let moras = parse_mora("カキ");
        assert_eq!(moras.len(), 2);
        assert_eq!(moras[0].consonant, Some("k".to_string()));
        assert_eq!(moras[0].vowel, "a");
        assert_eq!(moras[1].consonant, Some("k".to_string()));
        assert_eq!(moras[1].vowel, "i");
    }

    #[test]
    fn test_parse_mora_youon() {
        let moras = parse_mora("キャ");
        assert_eq!(moras.len(), 1);
        assert_eq!(moras[0].consonant, Some("ky".to_string()));
        assert_eq!(moras[0].vowel, "a");
    }

    #[test]
    fn test_parse_mora_special() {
        let moras = parse_mora("ンッ");
        assert_eq!(moras.len(), 2);
        assert_eq!(moras[0].consonant, None);
        assert_eq!(moras[0].vowel, "N");
        assert_eq!(moras[1].consonant, None);
        assert_eq!(moras[1].vowel, "cl");
    }
}
