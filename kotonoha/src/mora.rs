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
pub fn count_mora(reading: &str) -> usize {
    MoraIter::new(reading).count()
}

/// カタカナ読みからモーラ列を構築する。
/// 音素化できる組だけをまとめ、数え上げと同じ走査を使う。
pub fn parse_mora(reading: &str) -> Vec<Mora> {
    MoraIter::new(reading)
        .map(|m| Mora {
            text: if m.has_gap {
                m.text
                    .chars()
                    .filter(|&ch| ch == 'ー' || !phoneme::single_kana_phonemes(ch).is_empty())
                    .collect()
            } else {
                m.text.to_owned()
            },
            consonant: m.consonant.map(str::to_owned),
            vowel: m.vowel.to_owned(),
        })
        .collect()
}

/// 音素表を借用して走査する。数え上げでは文字列も配列も確保しない。
struct MoraIter<'a> {
    reading: &'a str,
    chars: std::iter::Peekable<std::str::CharIndices<'a>>,
    last_vowel: Option<&'static str>,
}

struct MoraRef<'a> {
    text: &'a str,
    consonant: Option<&'static str>,
    vowel: &'static str,
    has_gap: bool,
}

impl<'a> MoraIter<'a> {
    fn new(reading: &'a str) -> Self {
        Self {
            reading,
            chars: reading.char_indices().peekable(),
            last_vowel: None,
        }
    }
}

impl<'a> Iterator for MoraIter<'a> {
    type Item = MoraRef<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some((start, ch)) = self.chars.next() {
            let mut end = start + ch.len_utf8();
            if ch == 'ー' {
                if let Some(vowel) = self.last_vowel {
                    return Some(MoraRef {
                        text: &self.reading[start..end],
                        consonant: None,
                        vowel,
                        has_gap: false,
                    });
                }
                continue;
            }
            let single = phoneme::single_kana_phonemes(ch);
            let (mut consonant, mut vowel) = match *single {
                [vowel] => (None, vowel),
                [consonant, vowel] => (Some(consonant), vowel),
                _ => continue,
            };
            if let Some(&(next_start, next)) = self.chars.peek()
                && let Some([c, v]) = phoneme::combined_kana_phonemes(ch, next)
            {
                self.chars.next();
                end = next_start + next.len_utf8();
                consonant = Some(c);
                vowel = v;
            }
            // 音素変換と同じ小書き母音の差し替え。「キャィ」は ky+i の1モーラ。
            // 無音の文字を挟んでも差し替えを保ち、「キ?ィ」を ki+i に増やさない。
            let mut has_gap = false;
            if consonant.is_some() {
                while let Some(&(next_start, next)) = self.chars.peek() {
                    if matches!(next, 'ァ' | 'ィ' | 'ゥ' | 'ェ' | 'ォ') {
                        self.chars.next();
                        vowel = phoneme::single_kana_phonemes(next)[0];
                        end = next_start + next.len_utf8();
                    } else if next != 'ー' && phoneme::single_kana_phonemes(next).is_empty() {
                        self.chars.next();
                        has_gap = true;
                    } else {
                        break;
                    }
                }
            }
            self.last_vowel = phoneme::is_vowel(vowel).then_some(vowel);
            return Some(MoraRef {
                text: &self.reading[start..end],
                consonant,
                vowel,
                has_gap,
            });
        }
        None
    }
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
        assert_eq!(count_mora(&"カ".repeat(256)), 256);
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
