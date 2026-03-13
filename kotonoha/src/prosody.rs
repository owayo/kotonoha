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
pub const SYMBOL_PHRASE_END: &str = "$";   // 文末
pub const SYMBOL_QUESTION: &str = "?";      // 疑問
pub const SYMBOL_PAUSE: &str = "_";         // ポーズ（アクセント句境界）
pub const SYMBOL_BREATH: &str = "#";        // 呼気段落境界
pub const SYMBOL_ACCENT_UP: &str = "[";     // アクセント上昇
pub const SYMBOL_ACCENT_DOWN: &str = "]";   // アクセント下降

/// NjdNodeとアクセント句からPhoneTone列を抽出する
pub fn extract_phone_tones(nodes: &[NjdNode], phrases: &[AccentPhrase]) -> Vec<PhoneTone> {
    let mut result = Vec::new();

    // 先頭の無音
    result.push(PhoneTone {
        phone: "sil".to_string(),
        tone: 0,
    });

    for phrase in phrases {
        let mut mora_idx: u8 = 0;

        for &node_idx in &phrase.nodes {
            let node = &nodes[node_idx];
            let moras = mora::parse_mora(&node.pronunciation);

            for m in &moras {
                let tone = compute_tone(mora_idx, phrase.accent_type, phrase.mora_count);

                // 子音
                if let Some(ref consonant) = m.consonant {
                    result.push(PhoneTone {
                        phone: consonant.clone(),
                        tone,
                    });
                }

                // 母音
                result.push(PhoneTone {
                    phone: m.vowel.clone(),
                    tone,
                });

                mora_idx += 1;
            }
        }

        // アクセント句間のポーズ（最後以外）
        // 注: 実際にはポーズは文脈に依存するが、簡易実装ではスキップ
    }

    // 末尾の無音
    result.push(PhoneTone {
        phone: "sil".to_string(),
        tone: 0,
    });

    result
}

/// NjdNodeとアクセント句からPhoneTone列を抽出する（句読点を保持）
///
/// `extract_phone_tones` と異なり、Pos::Kigou ノードのうちモーラを持たないものは
/// 表層形の各文字を phone として tone=0 で出力する。
pub fn extract_phone_tones_with_punct(
    nodes: &[NjdNode],
    phrases: &[AccentPhrase],
) -> Vec<PhoneTone> {
    let mut result = Vec::new();

    result.push(PhoneTone {
        phone: "sil".to_string(),
        tone: 0,
    });

    for phrase in phrases {
        let mut mora_idx: u8 = 0;

        for &node_idx in &phrase.nodes {
            let node = &nodes[node_idx];
            let moras = mora::parse_mora(&node.pronunciation);

            // モーラがないノード（記号・空pronunciation等）は表層形を句読点として出力
            // 半角カンマ等が辞書で名詞扱いになる場合でも脱落しない
            if moras.is_empty() {
                for ch in node.surface.chars() {
                    if ch == 'ー' {
                        // 長音記号: 直前の母音を繰り返す
                        let prev_vowel = result.iter().rev().find_map(|pt| {
                            match pt.phone.as_str() {
                                "a" | "i" | "u" | "e" | "o" => Some(pt.phone.clone()),
                                _ => None,
                            }
                        });
                        if let Some(vowel) = prev_vowel {
                            let tone = result.last().map_or(0, |pt| pt.tone);
                            result.push(PhoneTone { phone: vowel, tone });
                        }
                    } else {
                        result.push(PhoneTone {
                            phone: ch.to_string(),
                            tone: 0,
                        });
                    }
                }
                continue;
            }

            for m in &moras {
                let tone = compute_tone(mora_idx, phrase.accent_type, phrase.mora_count);

                if let Some(ref consonant) = m.consonant {
                    result.push(PhoneTone {
                        phone: consonant.clone(),
                        tone,
                    });
                }

                result.push(PhoneTone {
                    phone: m.vowel.clone(),
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

/// モーラ位置とアクセント型からトーン（高低）を計算する
///
/// 日本語のアクセント規則:
/// - 0型（平板）: 1モーラ目=低, 2モーラ目以降=高
/// - 1型（頭高）: 1モーラ目=高, 2モーラ目以降=低
/// - n型（中高/尾高）: 1モーラ目=低, 2〜nモーラ目=高, n+1以降=低
fn compute_tone(mora_idx: u8, accent_type: u8, _mora_count: u8) -> u8 {
    if accent_type == 0 {
        // 平板型: 1モーラ目は低、2モーラ目以降は高
        if mora_idx == 0 { 0 } else { 1 }
    } else if accent_type == 1 {
        // 頭高型: 1モーラ目は高、2モーラ目以降は低
        if mora_idx == 0 { 1 } else { 0 }
    } else {
        // n型: 1モーラ目は低、2〜nモーラ目は高、n+1以降は低
        let n = accent_type;
        if mora_idx == 0 {
            0
        } else if mora_idx < n {
            1
        } else {
            0
        }
    }
}

/// NjdNodeとアクセント句から韻律記号列を抽出する
pub fn extract_prosody_symbols(nodes: &[NjdNode], phrases: &[AccentPhrase]) -> Vec<String> {
    let mut result = Vec::new();

    result.push(SYMBOL_PHRASE_START.to_string());

    for (phrase_idx, phrase) in phrases.iter().enumerate() {
        let mut mora_idx: u8 = 0;

        for &node_idx in &phrase.nodes {
            let node = &nodes[node_idx];
            let moras = mora::parse_mora(&node.pronunciation);

            for m in &moras {
                let tone = compute_tone(mora_idx, phrase.accent_type, phrase.mora_count);
                let prev_tone = if mora_idx > 0 {
                    compute_tone(mora_idx - 1, phrase.accent_type, phrase.mora_count)
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
                    compute_tone(mora_idx + 1, phrase.accent_type, phrase.mora_count)
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
        assert_eq!(compute_tone(0, 0, 4), 0);
        assert_eq!(compute_tone(1, 0, 4), 1);
        assert_eq!(compute_tone(2, 0, 4), 1);
        assert_eq!(compute_tone(3, 0, 4), 1);
    }

    #[test]
    fn test_compute_tone_atamadaka() {
        // 頭高型 (1型): 高低低低...
        assert_eq!(compute_tone(0, 1, 4), 1);
        assert_eq!(compute_tone(1, 1, 4), 0);
        assert_eq!(compute_tone(2, 1, 4), 0);
    }

    #[test]
    fn test_compute_tone_nakadaka() {
        // 中高型 (2型): 低高低低...
        assert_eq!(compute_tone(0, 2, 4), 0);
        assert_eq!(compute_tone(1, 2, 4), 1);
        assert_eq!(compute_tone(2, 2, 4), 0);
        assert_eq!(compute_tone(3, 2, 4), 0);
    }

    #[test]
    fn test_extract_phone_tones() {
        use crate::njd::InputToken;

        let tokens = vec![InputToken::new("猫", "名詞", "ネコ", "ネコ")];
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

        let tokens = vec![InputToken::new("猫", "名詞", "ネコ", "ネコ")];
        let nodes: Vec<NjdNode> = tokens.iter().map(NjdNode::from_token).collect();
        let phrases = vec![AccentPhrase {
            nodes: vec![0],
            accent_type: 1, // 頭高型: ネ↓コ
            mora_count: 2,
            is_interrogative: false,
        }];

        let symbols = extract_prosody_symbols(&nodes, &phrases);
        assert_eq!(symbols[0], "^");  // 文頭
        assert_eq!(symbols[1], "ネ"); // 高（頭高の1モーラ目）
        assert_eq!(symbols[2], "]");  // 下降
        assert_eq!(symbols[3], "コ"); // 低
        assert_eq!(symbols[4], "$");  // 文末
    }
}
