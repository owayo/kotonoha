//! HTS Full-Context Label 生成モジュール
//! NjdNodeとアクセント句情報からHTS形式のラベル文字列を生成する

use crate::accent::AccentPhrase;
use crate::mora;
use crate::njd::NjdNode;

/// 呼気段落（breath group）情報
#[derive(Debug, Clone)]
struct BreathGroup {
    /// 呼気段落内のアクセント句数
    accent_phrase_count: usize,
    /// 呼気段落内の総モーラ数
    mora_count: usize,
    /// 発話内の呼気段落位置（前から、1始まり）
    position_forward: usize,
    /// 発話内の呼気段落位置（後ろから、1始まり）
    position_backward: usize,
    /// この呼気段落の最初のアクセント句の発話内位置（前から、1始まり）
    utt_ap_start_forward: usize,
    /// この呼気段落の最後のアクセント句の発話内位置（後ろから、1始まり）
    utt_ap_end_backward: usize,
    /// この呼気段落の最初のモーラの発話内位置（前から、1始まり）
    utt_mora_start_forward: usize,
    /// この呼気段落の最後のモーラの発話内位置（後ろから、1始まり）
    utt_mora_end_backward: usize,
}

/// 発話（utterance）情報
#[derive(Debug, Clone)]
struct UtteranceInfo {
    /// 呼気段落数
    breath_group_count: usize,
    /// アクセント句数
    accent_phrase_count: usize,
    /// 総モーラ数
    mora_count: usize,
}

/// ラベル生成コンテキスト
#[derive(Debug, Clone)]
struct LabelContext {
    /// 音素列
    phonemes: Vec<PhonemeInfo>,
    /// アクセント句情報
    phrases: Vec<PhraseInfo>,
    /// 呼気段落情報（現在は1つのみ）
    breath_groups: Vec<BreathGroup>,
    /// 発話情報
    utterance: UtteranceInfo,
}

/// 音素情報
#[derive(Debug, Clone)]
struct PhonemeInfo {
    symbol: String,
    /// 所属モーラのインデックス
    mora_index: usize,
    /// 所属アクセント句のインデックス
    phrase_index: usize,
}

/// アクセント句ラベル情報
#[derive(Debug, Clone)]
struct PhraseInfo {
    mora_count: u8,
    accent_type: u8,
    is_interrogative: bool,
    /// 呼気段落内のアクセント句位置（前から、1始まり）
    bg_position_forward: usize,
    /// 呼気段落内のアクセント句位置（後ろから、1始まり）
    bg_position_backward: usize,
    /// 呼気段落内のモーラ位置（前から、1始まり）
    bg_mora_position_forward: usize,
    /// 呼気段落内のモーラ位置（後ろから、1始まり）
    bg_mora_position_backward: usize,
    /// 所属する呼気段落のインデックス
    breath_group_index: usize,
}

/// NjdNodeとアクセント句からHTS Full-Context Labelを生成する
pub fn generate_labels(nodes: &[NjdNode], phrases: &[AccentPhrase]) -> Vec<String> {
    if nodes.is_empty() || phrases.is_empty() {
        return vec![format_sil_label(None)];
    }

    // 音素列を構築
    let context = build_label_context(nodes, phrases);

    // ラベル文字列を生成
    let mut labels = Vec::new();

    // 先頭 sil
    labels.push(format_sil_label(Some(&context.utterance)));

    for (i, pinfo) in context.phonemes.iter().enumerate() {
        let p_prev2 = if i >= 2 {
            &context.phonemes[i - 2].symbol
        } else {
            "xx"
        };
        let p_prev = if i >= 1 {
            &context.phonemes[i - 1].symbol
        } else {
            "xx"
        };
        let p_curr = &pinfo.symbol;
        let p_next = context
            .phonemes
            .get(i + 1)
            .map(|p| p.symbol.as_str())
            .unwrap_or("xx");
        let p_next2 = context
            .phonemes
            .get(i + 2)
            .map(|p| p.symbol.as_str())
            .unwrap_or("xx");

        let phrase = &context.phrases[pinfo.phrase_index.min(context.phrases.len() - 1)];
        let prev_phrase = if pinfo.phrase_index > 0 {
            Some(&context.phrases[pinfo.phrase_index - 1])
        } else {
            None
        };
        let next_phrase = context.phrases.get(pinfo.phrase_index + 1);

        let bg_idx = phrase.breath_group_index;
        let curr_bg = &context.breath_groups[bg_idx];

        let label = format_full_context_label(&LabelFormatInput {
            phonemes: [p_prev2, p_prev, p_curr, p_next, p_next2],
            phrase,
            prev_phrase,
            next_phrase,
            mora_pos: pinfo.mora_index,
            curr_breath_group: curr_bg,
            utterance: &context.utterance,
        });
        labels.push(label);
    }

    // 末尾 sil
    labels.push(format_sil_label(Some(&context.utterance)));

    labels
}

/// ラベルコンテキストを構築する
fn build_label_context(nodes: &[NjdNode], phrases: &[AccentPhrase]) -> LabelContext {
    let mut phonemes = Vec::new();
    let mut phrase_infos = Vec::new();

    // 現在は全体を1つの呼気段落として扱う
    let total_mora_count: usize = phrases.iter().map(|p| p.mora_count as usize).sum();
    let breath_group = BreathGroup {
        accent_phrase_count: phrases.len(),
        mora_count: total_mora_count,
        position_forward: 1,
        position_backward: 1,
        utt_ap_start_forward: 1,
        utt_ap_end_backward: phrases.len(),
        utt_mora_start_forward: 1,
        utt_mora_end_backward: total_mora_count,
    };

    let utterance = UtteranceInfo {
        breath_group_count: 1,
        accent_phrase_count: phrases.len(),
        mora_count: total_mora_count,
    };

    // モーラの累計位置を追跡（呼気段落内のモーラ位置計算用）
    let mut mora_offset: usize = 0;

    for (phrase_idx, phrase) in phrases.iter().enumerate() {
        let bg_position_forward = phrase_idx + 1;
        let bg_position_backward = phrases.len() - phrase_idx;

        let bg_mora_position_forward = mora_offset + 1;
        let bg_mora_position_backward = total_mora_count - mora_offset;

        phrase_infos.push(PhraseInfo {
            mora_count: phrase.mora_count,
            accent_type: phrase.accent_type,
            is_interrogative: phrase.is_interrogative,
            bg_position_forward,
            bg_position_backward,
            bg_mora_position_forward,
            bg_mora_position_backward,
            breath_group_index: 0,
        });

        mora_offset += phrase.mora_count as usize;

        // フレーズ内の全モーラを先に収集する（無声化判定にはフレーズ全体のモーラ列が必要）
        // 各ノードのモーラ数を記録して1モーラ語を検出する
        let mut phrase_moras: Vec<mora::Mora> = Vec::new();
        let mut node_mora_counts: Vec<u8> = Vec::new();
        for &node_idx in &phrase.nodes {
            let node = &nodes[node_idx];
            let moras = mora::parse_mora(&node.reading);
            node_mora_counts.push(moras.len() as u8);
            phrase_moras.extend(moras);
        }

        let is_last_phrase = phrase_idx == phrases.len() - 1;

        // 各モーラがどのノードに属するかを計算し、1モーラ語かどうかを判定する
        let mut mora_to_single_mora_word: Vec<bool> = Vec::new();
        for &count in &node_mora_counts {
            let is_single = count == 1 && phrase.nodes.len() == 1;
            for _ in 0..count {
                mora_to_single_mora_word.push(is_single);
            }
        }

        for (mora_idx_in_phrase, m) in phrase_moras.iter().enumerate() {
            // 子音がある場合
            if let Some(ref consonant) = m.consonant {
                // アクセント核位置の判定:
                // accent_type > 0 のとき、mora_idx_in_phrase + 1 == accent_type がアクセント核
                let is_accent_nucleus = phrase.accent_type > 0
                    && (mora_idx_in_phrase + 1) == phrase.accent_type as usize;

                let is_single_mora_word = mora_to_single_mora_word
                    .get(mora_idx_in_phrase)
                    .copied()
                    .unwrap_or(false);

                let voiceless_ctx = VoicelessContext {
                    is_accent_nucleus,
                    is_single_mora_word,
                };

                // 無声化判定
                let vowel_symbol = determine_voiceless(
                    &m.vowel,
                    &phrase_moras,
                    mora_idx_in_phrase,
                    is_last_phrase,
                    &voiceless_ctx,
                );

                phonemes.push(PhonemeInfo {
                    symbol: consonant.clone(),
                    mora_index: mora_idx_in_phrase,
                    phrase_index: phrase_idx,
                });
                phonemes.push(PhonemeInfo {
                    symbol: vowel_symbol,
                    mora_index: mora_idx_in_phrase,
                    phrase_index: phrase_idx,
                });
            } else {
                phonemes.push(PhonemeInfo {
                    symbol: m.vowel.clone(),
                    mora_index: mora_idx_in_phrase,
                    phrase_index: phrase_idx,
                });
            }
        }
    }

    LabelContext {
        phonemes,
        phrases: phrase_infos,
        breath_groups: vec![breath_group],
        utterance,
    }
}

/// 無声子音かどうか判定
fn is_voiceless_consonant(consonant: &str) -> bool {
    matches!(
        consonant,
        "k" | "ky" | "s" | "sh" | "t" | "ts" | "ch" | "h" | "hy" | "f" | "p" | "py"
    )
}

/// 無声化判定の追加コンテキスト
struct VoicelessContext {
    /// アクセント核の位置にあるか（accent_type と一致するモーラ位置）
    is_accent_nucleus: bool,
    /// 1モーラ語かどうか（単独の語が1モーラのみ）
    is_single_mora_word: bool,
}

/// 無声母音の判定
/// OpenJTalkのnjd_set_unvoiced_vowelに相当
///
/// 無声化規則:
/// 1. 母音 "i" / "u" が無声子音に挟まれている場合 → 無声化
/// 2. 母音 "i" / "u" が発話末（フレーズ末）で直前が無声子音の場合 → 無声化
/// 3. 連続する無声化候補モーラでは、先頭のみ無声化する（交互規則）
/// 4. アクセント核位置の母音は無声化しない
/// 5. 1モーラ語の母音は無声化しない
/// 6. 長母音（次のモーラが同じ母音）の場合は無声化しない
fn determine_voiceless(
    vowel: &str,
    moras: &[mora::Mora],
    mora_idx: usize,
    is_last_phrase: bool,
    ctx: &VoicelessContext,
) -> String {
    // 無声化対象は "i" と "u" のみ
    if vowel != "i" && vowel != "u" {
        return vowel.to_string();
    }

    let current = &moras[mora_idx];

    // 現在のモーラに無声子音がなければ無声化しない
    if !current
        .consonant
        .as_ref()
        .is_some_and(|c| is_voiceless_consonant(c))
    {
        return vowel.to_string();
    }

    // 1モーラ語は無声化しない（「き」「し」「す」等）
    if ctx.is_single_mora_word {
        return vowel.to_string();
    }

    // アクセント核位置の母音は無声化しない
    if ctx.is_accent_nucleus {
        return vowel.to_string();
    }

    // 長母音規則: 次のモーラが子音なしで同じ母音なら無声化しない
    if mora_idx + 1 < moras.len() {
        let next = &moras[mora_idx + 1];
        if next.consonant.is_none() && next.vowel == vowel {
            return vowel.to_string();
        }
    }

    // 次のモーラの子音が無声子音かどうか、または発話末かどうかを判定
    let next_is_voiceless_or_end = if mora_idx + 1 < moras.len() {
        // 次のモーラがある場合: 次のモーラの子音が無声子音であること
        moras[mora_idx + 1]
            .consonant
            .as_ref()
            .is_some_and(|c| is_voiceless_consonant(c))
    } else {
        // フレーズ末尾: 最終フレーズなら発話末として無声化
        is_last_phrase
    };

    if !next_is_voiceless_or_end {
        return vowel.to_string();
    }

    // 交互規則: 直前のモーラも無声化候補（i/u + 無声子音）だった場合、
    // 連続無声化を避けるため、現在のモーラは無声化しない
    if mora_idx > 0 {
        let prev = &moras[mora_idx - 1];
        if (prev.vowel == "i" || prev.vowel == "u")
            && prev
                .consonant
                .as_ref()
                .is_some_and(|c| is_voiceless_consonant(c))
        {
            // 前のモーラも無声化候補 → さらにその前を確認
            // 前のモーラが実際に無声化される（=その前が無声化候補でない）なら、
            // 現在のモーラは無声化しない（交互規則）
            let prev_prev_is_candidate = mora_idx >= 2 && {
                let pp = &moras[mora_idx - 2];
                (pp.vowel == "i" || pp.vowel == "u")
                    && pp
                        .consonant
                        .as_ref()
                        .is_some_and(|c| is_voiceless_consonant(c))
            };
            // 前のモーラの前が候補でなければ、前のモーラが無声化される → 現在は無声化しない
            if !prev_prev_is_candidate {
                return vowel.to_string();
            }
            // 前のモーラの前も候補なら、前のモーラは無声化されない → 現在は無声化してよい
        }
    }

    // 無声化する: 大文字に変換
    vowel.to_uppercase()
}

/// sil ラベルを生成
fn format_sil_label(utterance: Option<&UtteranceInfo>) -> String {
    let k_field = match utterance {
        Some(u) => format!(
            "{}+{}-{}",
            u.breath_group_count, u.accent_phrase_count, u.mora_count
        ),
        None => "xx+xx-xx".to_string(),
    };
    format!(
        "xx^xx-sil+xx=xx/A:xx+xx+xx/B:xx-xx_xx/C:xx_xx+xx/D:xx+xx_xx\
         /E:xx_xx!xx_xx-xx/F:xx_xx#xx_xx@xx_xx|xx_xx\
         /G:xx_xx%xx_xx_xx/H:xx_xx\
         /I:xx-xx@xx+xx&xx-xx|xx+xx/J:xx_xx/K:{k_field}"
    )
}

/// Full-Context Label生成に必要なコンテキスト
struct LabelFormatInput<'a> {
    phonemes: [&'a str; 5],
    phrase: &'a PhraseInfo,
    prev_phrase: Option<&'a PhraseInfo>,
    next_phrase: Option<&'a PhraseInfo>,
    mora_pos: usize,
    curr_breath_group: &'a BreathGroup,
    utterance: &'a UtteranceInfo,
}

/// Full-Context Label文字列を組み立てる
fn format_full_context_label(input: &LabelFormatInput<'_>) -> String {
    let [p1, p2, p3, p4, p5] = input.phonemes;
    let phrase = input.phrase;
    let prev_phrase = input.prev_phrase;
    let next_phrase = input.next_phrase;
    let mora_pos = input.mora_pos;

    // A: アクセント句のアクセント型情報
    let a1 = if phrase.accent_type == 0 {
        0i8
    } else {
        phrase.accent_type as i8 - (mora_pos as i8 + 1)
    };
    let a2 = mora_pos + 1;
    let a3 = phrase.mora_count as usize - mora_pos;

    // B: 前のアクセント句情報
    let (b1, b2, b3) = match prev_phrase {
        Some(p) => (
            format!("{}", p.mora_count),
            format!("{}", p.accent_type),
            if p.is_interrogative { "1" } else { "0" }.to_string(),
        ),
        None => ("xx".to_string(), "xx".to_string(), "xx".to_string()),
    };

    // C: 現在のアクセント句情報
    let c1 = phrase.mora_count;
    let c2 = phrase.accent_type;
    let c3 = if phrase.is_interrogative { 1 } else { 0 };

    // D: 次のアクセント句情報
    let (d1, d2, d3) = match next_phrase {
        Some(p) => (
            format!("{}", p.mora_count),
            format!("{}", p.accent_type),
            if p.is_interrogative { "1" } else { "0" }.to_string(),
        ),
        None => ("xx".to_string(), "xx".to_string(), "xx".to_string()),
    };

    // E: 前のアクセント句情報
    let e_field = match prev_phrase {
        Some(p) => format!(
            "{}_{}\
             !0_xx-0",
            p.mora_count, p.accent_type
        ),
        None => "xx_xx!xx_xx-xx".to_string(),
    };

    // F: 現在のアクセント句情報（呼気段落内での位置を含む）
    // f1=mora_count, f2=accent_type, f3=0(placeholder), f4=xx(placeholder),
    // f5=AP position in BG (fwd), f6=AP position in BG (bwd),
    // f7=mora position in utterance (fwd), f8=mora position in utterance (bwd)
    let f_field = format!(
        "{}_{}#0_xx@{}_{}|{}_{}",
        phrase.mora_count,
        phrase.accent_type,
        phrase.bg_position_forward,
        phrase.bg_position_backward,
        phrase.bg_mora_position_forward,
        phrase.bg_mora_position_backward,
    );

    // G: 次のアクセント句情報
    let g_field = match next_phrase {
        Some(p) => format!(
            "{}_{}\
             %0_xx_0",
            p.mora_count, p.accent_type
        ),
        None => "xx_xx%xx_xx_xx".to_string(),
    };

    // H: 前の発話情報（単一発話のためxx）
    let h_field = "xx_xx";

    // I: 現在の呼気段落情報
    // i1=AP count in BG, i2=mora count in BG,
    // i3=BG position in utterance (fwd), i4=BG position in utterance (bwd),
    // i5=cumulative AP position from start, i6=from end,
    // i7=cumulative mora position from start, i8=from end
    let bg = input.curr_breath_group;
    let utt = input.utterance;
    let i_field = format!(
        "{}-{}@{}+{}&{}-{}|{}+{}",
        bg.accent_phrase_count,
        bg.mora_count,
        bg.position_forward,
        bg.position_backward,
        bg.utt_ap_start_forward,
        bg.utt_ap_end_backward,
        bg.utt_mora_start_forward,
        bg.utt_mora_end_backward,
    );

    // J: 次の発話情報（単一発話のためxx）
    let j_field = "xx_xx";

    // K: 全体情報
    let k_field = format!(
        "{}+{}-{}",
        utt.breath_group_count, utt.accent_phrase_count, utt.mora_count
    );

    format!(
        "{p1}^{p2}-{p3}+{p4}={p5}\
         /A:{a1}+{a2}+{a3}\
         /B:{b1}-{b2}_{b3}\
         /C:{c1}_{c2}+{c3}\
         /D:{d1}+{d2}_{d3}\
         /E:{e_field}\
         /F:{f_field}\
         /G:{g_field}\
         /H:{h_field}\
         /I:{i_field}\
         /J:{j_field}\
         /K:{k_field}"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::njd::InputToken;

    #[test]
    fn test_empty_labels() {
        let labels = generate_labels(&[], &[]);
        assert_eq!(labels.len(), 1); // sil only
        assert!(labels[0].contains("sil"));
        // 空入力ではK:xx+xx-xx
        assert!(labels[0].contains("/K:xx+xx-xx"));
    }

    #[test]
    fn test_sil_label_format() {
        let label = format_sil_label(None);
        assert!(label.contains("sil"));
        assert!(label.contains("/A:"));
        assert!(label.contains("/K:xx+xx-xx"));
    }

    #[test]
    fn test_sil_label_with_utterance() {
        let utt = UtteranceInfo {
            breath_group_count: 1,
            accent_phrase_count: 2,
            mora_count: 5,
        };
        let label = format_sil_label(Some(&utt));
        assert!(label.contains("/K:1+2-5"));
    }

    #[test]
    fn test_basic_label_generation() {
        let tokens = vec![InputToken::new("猫", "名詞", "ネコ", "ネコ")];
        let mut nodes: Vec<NjdNode> = tokens.iter().map(NjdNode::from_token).collect();
        nodes[0].accent_type = 1;

        let phrases = vec![AccentPhrase {
            nodes: vec![0],
            accent_type: 1,
            mora_count: 2,
            is_interrogative: false,
        }];

        let labels = generate_labels(&nodes, &phrases);
        // sil + phonemes + sil
        assert!(labels.len() >= 3);
        assert!(labels.first().unwrap().contains("sil"));
        assert!(labels.last().unwrap().contains("sil"));

        // silラベルにはK:1+1-2が含まれる
        assert!(labels.first().unwrap().contains("/K:1+1-2"));
        assert!(labels.last().unwrap().contains("/K:1+1-2"));

        // 内部ラベルにもK:1+1-2が含まれる
        let inner = &labels[1];
        assert!(inner.contains("/K:1+1-2"));
        // F: アクセント句情報（mora_count=2, accent_type=1, AP pos 1/1, mora pos 1/2）
        assert!(inner.contains("/F:2_1#0_xx@1_1|1_2"));
        // E: 前のアクセント句なし
        assert!(inner.contains("/E:xx_xx!xx_xx-xx"));
        // G: 次のアクセント句なし
        assert!(inner.contains("/G:xx_xx%xx_xx_xx"));
    }

    #[test]
    fn test_multi_phrase_ef_fields() {
        let tokens = vec![
            InputToken::new("猫", "名詞", "ネコ", "ネコ"),
            InputToken::new("犬", "名詞", "イヌ", "イヌ"),
        ];
        let mut nodes: Vec<NjdNode> = tokens.iter().map(NjdNode::from_token).collect();
        nodes[0].accent_type = 1;
        nodes[1].accent_type = 0;

        let phrases = vec![
            AccentPhrase {
                nodes: vec![0],
                accent_type: 1,
                mora_count: 2,
                is_interrogative: false,
            },
            AccentPhrase {
                nodes: vec![1],
                accent_type: 0,
                mora_count: 2,
                is_interrogative: false,
            },
        ];

        let labels = generate_labels(&nodes, &phrases);

        // 全ラベルにK:1+2-4（1呼気段落、2アクセント句、4モーラ）
        for label in &labels {
            assert!(label.contains("/K:1+2-4"), "Missing K field in: {label}");
        }

        // 最初のアクセント句の音素: F:2_1#0_xx@1_2|1_4
        // (mora_count=2, accent_type=1, AP pos 1/2, mora pos 1/4)
        let first_inner = &labels[1];
        assert!(
            first_inner.contains("/F:2_1#0_xx@1_2|1_4"),
            "Unexpected F field in first phrase label: {first_inner}"
        );

        // E: 前のアクセント句なし → xx
        assert!(
            first_inner.contains("/E:xx_xx!xx_xx-xx"),
            "Unexpected E field in first phrase label: {first_inner}"
        );

        // G: 次のアクセント句（mora_count=2, accent_type=0）
        assert!(
            first_inner.contains("/G:2_0%0_xx_0"),
            "Unexpected G field in first phrase label: {first_inner}"
        );

        // 2番目のアクセント句の音素を見つける
        // 「ネコ」= n, e, k, o → 4音素 → labels[1..5]
        // 「イヌ」= i, n, u → 3音素 → labels[5..8]
        let second_inner = &labels[5];
        assert!(
            second_inner.contains("/F:2_0#0_xx@2_1|3_2"),
            "Unexpected F field in second phrase label: {second_inner}"
        );

        // E: 前のアクセント句（mora_count=2, accent_type=1）
        assert!(
            second_inner.contains("/E:2_1!0_xx-0"),
            "Unexpected E field in second phrase label: {second_inner}"
        );

        // G: 次のアクセント句なし → xx
        assert!(
            second_inner.contains("/G:xx_xx%xx_xx_xx"),
            "Unexpected G field in second phrase label: {second_inner}"
        );
    }
}
