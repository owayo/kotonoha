//! 韻律抽出モジュール
//! NjdNodeとアクセント句からphone+toneペアおよび韻律記号を抽出する

use crate::accent::AccentPhrase;
use crate::mora;
use crate::njd::NjdNode;
use serde::{Deserialize, Serialize};

/// 音素とトーン（高低）のペア
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhoneTone {
    pub phone: String,
    pub tone: u8, // 0=低, 1=高
}

/// 韻律記号の定義
pub const SYMBOL_PHRASE_START: &str = "^"; // 文頭
pub const SYMBOL_PHRASE_END: &str = "$"; // 文末
pub const SYMBOL_QUESTION: &str = "?"; // 疑問
pub const SYMBOL_PAUSE: &str = "_"; // ポーズ（アクセント句境界）
pub const SYMBOL_BREATH: &str = "#"; // 呼気段落境界
pub const SYMBOL_ACCENT_UP: &str = "["; // アクセント上昇
pub const SYMBOL_ACCENT_DOWN: &str = "]"; // アクセント下降

/// NjdNodeとアクセント句からPhoneTone列を抽出する
pub fn extract_phone_tones(nodes: &[NjdNode], phrases: &[AccentPhrase]) -> Vec<PhoneTone> {
    collect_phone_tones(nodes, phrases, false)
}

/// NjdNodeとアクセント句からPhoneTone列を抽出する（句読点を保持）
///
/// `extract_phone_tones` と異なり、モーラを持たないノードは
/// 表層形の各文字を phone として tone=0 で出力する。
/// 長音記号は直前の母音を、直前の出力と同じ高低で繰り返す。
pub fn extract_phone_tones_with_punct(
    nodes: &[NjdNode],
    phrases: &[AccentPhrase],
) -> Vec<PhoneTone> {
    collect_phone_tones(nodes, phrases, true)
}

/// 音素列と高低を共通の走査で構築し、必要に応じてモーラのない表層形を補う。
fn collect_phone_tones(
    nodes: &[NjdNode],
    phrases: &[AccentPhrase],
    preserve_punctuation: bool,
) -> Vec<PhoneTone> {
    let mut result = Vec::new();
    let mut last_vowel: Option<String> = None;

    result.push(PhoneTone {
        phone: "sil".to_string(),
        tone: 0,
    });

    for phrase in phrases {
        let mut mora_idx: usize = 0;

        for &node_idx in &phrase.nodes {
            let node = &nodes[node_idx];
            let moras = mora::parse_mora(&node.pronunciation);

            // モーラがないノード（記号・空pronunciation等）は表層形を句読点として出力
            // 半角カンマ等が辞書で名詞扱いになる場合でも脱落しない
            if preserve_punctuation && moras.is_empty() {
                append_punctuation(&mut result, &node.surface, &mut last_vowel);
            }

            for m in moras {
                let tone = compute_tone(mora_idx, phrase.accent_type);

                if let Some(consonant) = m.consonant {
                    result.push(PhoneTone {
                        phone: consonant,
                        tone,
                    });
                }

                if preserve_punctuation && matches!(m.vowel.as_str(), "a" | "i" | "u" | "e" | "o") {
                    last_vowel = Some(m.vowel.clone());
                }
                result.push(PhoneTone {
                    phone: m.vowel,
                    tone,
                });

                mora_idx += 1;
            }
        }
    }

    result.push(PhoneTone {
        phone: "sil".to_string(),
        tone: 0,
    });

    result
}

/// モーラのない表層形を出力する。長音は最後の母音と直前の出力の高低を引き継ぐ。
fn append_punctuation(result: &mut Vec<PhoneTone>, surface: &str, last_vowel: &mut Option<String>) {
    for ch in surface.chars() {
        if ch == 'ー' {
            if let Some(vowel) = last_vowel {
                let tone = result.last().map_or(0, |pt| pt.tone);
                result.push(PhoneTone {
                    phone: vowel.clone(),
                    tone,
                });
            }
        } else {
            let phone = ch.to_string();
            if matches!(phone.as_str(), "a" | "i" | "u" | "e" | "o") {
                *last_vowel = Some(phone.clone());
            }
            result.push(PhoneTone { phone, tone: 0 });
        }
    }
}

/// モーラ位置とアクセント型からトーン（高低）を計算する
///
/// 日本語のアクセント規則:
/// - 0型（平板）: 1モーラ目=低, 2モーラ目以降=高
/// - 1型（頭高）: 1モーラ目=高, 2モーラ目以降=低
/// - n型（中高/尾高）: 1モーラ目=低, 2〜nモーラ目=高, n+1以降=低
fn compute_tone(mora_idx: usize, accent_type: usize) -> u8 {
    match accent_type {
        0 => u8::from(mora_idx > 0),
        1 => u8::from(mora_idx == 0),
        n => u8::from(mora_idx > 0 && mora_idx < n),
    }
}

/// NjdNodeとアクセント句から韻律記号列を抽出する
pub fn extract_prosody_symbols(nodes: &[NjdNode], phrases: &[AccentPhrase]) -> Vec<String> {
    let mut result = Vec::new();

    result.push(SYMBOL_PHRASE_START.to_string());

    for (phrase_idx, phrase) in phrases.iter().enumerate() {
        let mut mora_idx: usize = 0;

        for &node_idx in &phrase.nodes {
            let node = &nodes[node_idx];
            let moras = mora::parse_mora(&node.pronunciation);

            for m in &moras {
                let tone = compute_tone(mora_idx, phrase.accent_type);
                let prev_tone = if mora_idx > 0 {
                    compute_tone(mora_idx - 1, phrase.accent_type)
                } else {
                    0
                };

                // アクセント上昇の検出
                if mora_idx > 0 && prev_tone == 0 && tone == 1 {
                    result.push(SYMBOL_ACCENT_UP.to_string());
                }

                // モーラ表記を追加
                result.push(m.text.clone());

                // アクセント下降の検出
                let next_tone = if mora_idx + 1 < phrase.mora_count {
                    compute_tone(mora_idx + 1, phrase.accent_type)
                } else {
                    0
                };

                if tone == 1 && next_tone == 0 {
                    result.push(SYMBOL_ACCENT_DOWN.to_string());
                }

                mora_idx += 1;
            }
        }

        // アクセント句境界
        if phrase_idx < phrases.len() - 1 {
            result.push(SYMBOL_PAUSE.to_string());
        }
    }

    // 文末記号
    if phrases.last().is_some_and(|p| p.is_interrogative) {
        result.push(SYMBOL_QUESTION.to_string());
    } else {
        result.push(SYMBOL_PHRASE_END.to_string());
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_tone_flat() {
        // 平板型 (0型): 低高高高...
        assert_eq!(compute_tone(0, 0), 0);
        assert_eq!(compute_tone(1, 0), 1);
        assert_eq!(compute_tone(2, 0), 1);
        assert_eq!(compute_tone(3, 0), 1);
    }

    #[test]
    fn test_compute_tone_atamadaka() {
        // 頭高型 (1型): 高低低低...
        assert_eq!(compute_tone(0, 1), 1);
        assert_eq!(compute_tone(1, 1), 0);
        assert_eq!(compute_tone(2, 1), 0);
    }

    #[test]
    fn test_compute_tone_nakadaka() {
        // 中高型 (2型): 低高低低...
        assert_eq!(compute_tone(0, 2), 0);
        assert_eq!(compute_tone(1, 2), 1);
        assert_eq!(compute_tone(2, 2), 0);
        assert_eq!(compute_tone(3, 2), 0);
    }

    #[test]
    fn test_extract_phone_tones() {
        use crate::njd::InputToken;

        let tokens = [InputToken::new("猫", "名詞", "ネコ", "ネコ")];
        let nodes: Vec<NjdNode> = tokens.iter().map(NjdNode::from_token).collect();
        let phrases = vec![AccentPhrase {
            nodes: vec![0],
            accent_type: 1, // 頭高型
            mora_count: 2,
            is_interrogative: false,
        }];

        let phone_tones = extract_phone_tones(&nodes, &phrases);

        // sil, n, e, k, o, sil
        assert_eq!(phone_tones.first().unwrap().phone, "sil");
        assert_eq!(phone_tones.last().unwrap().phone, "sil");
        assert!(phone_tones.len() >= 4); // sil + ne + ko + sil

        // 頭高型: ネ=高(1), コ=低(0)
        let ne_vowel = &phone_tones[2]; // n=1, e=2
        assert_eq!(ne_vowel.tone, 1);
        let ko_vowel = &phone_tones[4]; // k=3, o=4
        assert_eq!(ko_vowel.tone, 0);
    }

    #[test]
    fn test_extract_prosody_symbols() {
        use crate::njd::InputToken;

        let tokens = [InputToken::new("猫", "名詞", "ネコ", "ネコ")];
        let nodes: Vec<NjdNode> = tokens.iter().map(NjdNode::from_token).collect();
        let phrases = vec![AccentPhrase {
            nodes: vec![0],
            accent_type: 1, // 頭高型: ネ↓コ
            mora_count: 2,
            is_interrogative: false,
        }];

        let symbols = extract_prosody_symbols(&nodes, &phrases);
        assert_eq!(symbols[0], "^"); // 文頭
        assert_eq!(symbols[1], "ネ"); // 高（頭高の1モーラ目）
        assert_eq!(symbols[2], "]"); // 下降
        assert_eq!(symbols[3], "コ"); // 低
        assert_eq!(symbols[4], "$"); // 文末
    }
}
