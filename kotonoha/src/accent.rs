//! アクセント推定モジュール
//! アクセント句の境界を検出し、アクセント型を結合規則に基づいて推定する
//!
//! OpenJTalkのnjd_set_accent_typeに相当する処理を行う。
//! 数詞+助数詞、形容動詞、複合動詞、連体詞+名詞の特殊処理を含む。

use crate::accent_rule::{AccentRuleTable, AccentRuleType};
use crate::njd::{NjdNode, Pos};

/// アクセント句
#[derive(Debug, Clone)]
pub struct AccentPhrase {
    pub nodes: Vec<usize>, // NjdNodeのインデックス
    pub accent_type: u8,   // アクセント型 (0=平板)
    pub mora_count: u8,    // 総モーラ数
    pub is_interrogative: bool, // 疑問文末か
}

/// アクセント句境界を検出し、アクセント型を推定する
pub fn estimate_accent(nodes: &mut [NjdNode], rule_table: &AccentRuleTable) -> Vec<AccentPhrase> {
    if nodes.is_empty() {
        return Vec::new();
    }

    // Phase 1: chain_flagの設定（接続判定）
    set_chain_flags(nodes);

    // Phase 2: アクセント句の構築
    let mut phrases = build_accent_phrases(nodes);

    // Phase 3: アクセント型の結合
    combine_accent_types(&mut phrases, nodes, rule_table);

    // Phase 4: 特殊規則の適用
    apply_special_rules(&mut phrases, nodes);

    phrases
}

/// 接続フラグを設定する
fn set_chain_flags(nodes: &mut [NjdNode]) {
    if nodes.is_empty() {
        return;
    }

    // 最初のノードは接続しない
    nodes[0].chain_flag = 0;

    for i in 1..nodes.len() {
        // 安全に前後のノード情報を取得するためにインデックスで参照
        let prev_pos = nodes[i - 1].pos.clone();
        let prev_detail1 = nodes[i - 1].pos_detail1.clone();
        let curr_pos = nodes[i].pos.clone();
        let curr_detail1 = nodes[i].pos_detail1.clone();

        nodes[i].chain_flag =
            if should_chain(&prev_pos, &prev_detail1, &curr_pos, &curr_detail1) {
                1
            } else {
                0
            };
    }
}

/// 2つのノードが同一アクセント句に接続するかどうか判定
///
/// 訓練データ(JSUT v2)の分析に基づく接続判定:
/// - 機能語（助詞・助動詞）→前に接続（93-97%）
/// - 接尾辞→前に接続
/// - 複合動詞（動詞+動詞）→接続（82%）
/// - 接頭辞+内容語→接続（87-95%）
/// - 連体詞+名詞→接続しない（37%しかchainしない）
fn should_chain(prev_pos: &Pos, prev_detail1: &str, curr_pos: &Pos, curr_detail1: &str) -> bool {
    // 記号は接続しない
    if *curr_pos == Pos::Kigou {
        return false;
    }

    // フィラーは接続しない
    if *curr_pos == Pos::Filler {
        return false;
    }

    // 機能語（助詞・助動詞）は前の語に接続
    // 訓練データ: 名詞+助詞 93.8%, 動詞+助動詞 96.4%, 動詞+助詞 94.5%
    if curr_pos.is_function_word() {
        return true;
    }

    // 接頭詞は次の語に接続（次のノードの判断で処理）
    // 訓練データ: 接頭詞+名詞 86.9%, 接頭詞+動詞 94.7%
    if *prev_pos == Pos::Settoushi {
        return true;
    }

    // 名詞の接尾は前に接続
    if *curr_pos == Pos::Meishi && is_suffix(curr_detail1) {
        return true;
    }

    // 数詞（名詞,数）の後の助数詞（名詞,接尾,助数詞）は接続
    if *prev_pos == Pos::Meishi
        && is_numeral(prev_detail1)
        && *curr_pos == Pos::Meishi
        && is_counter_suffix(curr_detail1)
    {
        return true;
    }

    // 形容動詞語幹の後の助動詞（「だ」「な」等）は接続
    if *prev_pos == Pos::Meishi
        && is_keiyoudoushi_gokan(prev_detail1)
        && curr_pos.is_function_word()
    {
        return true;
    }

    // 複合動詞: 動詞+動詞（「食べ始める」等）は接続
    // 訓練データ: 81.9%
    if *prev_pos == Pos::Doushi && *curr_pos == Pos::Doushi {
        return true;
    }

    // 動詞,非自立（「いる」「ある」「おく」等の補助動詞）は前に接続
    // 訓練データ: 助詞,接続助詞 + 動詞,非自立 → 87-96% chain
    if *curr_pos == Pos::Doushi && is_non_independent(curr_detail1) {
        return true;
    }

    // 形容詞,非自立（「ない」「ほしい」等）は前に接続
    if *curr_pos == Pos::Keiyoushi && is_non_independent(curr_detail1) {
        return true;
    }

    // 注意: 連体詞+名詞は接続しない（訓練データで37%しかchainしない）
    // OpenJTalkでは接続していたが、実データに基づき独立アクセント句とする

    // 内容語は新しいアクセント句を開始
    false
}

/// 非自立語かどうか判定（動詞,非自立 / 形容詞,非自立）
fn is_non_independent(pos_detail: &str) -> bool {
    pos_detail.contains("非自立")
}

/// 接尾辞かどうか判定
fn is_suffix(pos_detail: &str) -> bool {
    pos_detail.contains("接尾")
}

/// 数詞かどうか判定
fn is_numeral(pos_detail: &str) -> bool {
    pos_detail.contains("数")
}

/// 助数詞かどうか判定
fn is_counter_suffix(pos_detail: &str) -> bool {
    pos_detail.contains("助数詞")
}

/// 形容動詞語幹かどうか判定
fn is_keiyoudoushi_gokan(pos_detail: &str) -> bool {
    pos_detail.contains("形容動詞語幹")
}

/// アクセント句を構築する
fn build_accent_phrases(nodes: &[NjdNode]) -> Vec<AccentPhrase> {
    let mut phrases = Vec::new();
    let mut current_nodes = Vec::new();
    let mut current_mora_count: u8 = 0;

    for (i, node) in nodes.iter().enumerate() {
        if node.chain_flag == 0 && !current_nodes.is_empty() {
            // 新しいアクセント句を開始
            phrases.push(AccentPhrase {
                nodes: current_nodes.clone(),
                accent_type: 0,
                mora_count: current_mora_count,
                is_interrogative: false,
            });
            current_nodes.clear();
            current_mora_count = 0;
        }
        current_nodes.push(i);
        current_mora_count = current_mora_count.saturating_add(node.mora_count);
    }

    // 最後のアクセント句
    if !current_nodes.is_empty() {
        phrases.push(AccentPhrase {
            nodes: current_nodes,
            accent_type: 0,
            mora_count: current_mora_count,
            is_interrogative: false,
        });
    }

    phrases
}

/// 品詞の完全な文字列を構築する（階層マッチ用）
fn build_full_pos_str(node: &NjdNode) -> String {
    let base = node.pos.to_label_str();
    if node.pos_detail1 == "*" || node.pos_detail1.is_empty() {
        return base.to_string();
    }
    let mut full = format!("{},{}", base, node.pos_detail1);
    if node.pos_detail2 != "*" && !node.pos_detail2.is_empty() {
        full = format!("{},{}", full, node.pos_detail2);
    }
    full
}

/// アクセント型を結合規則に基づいて計算する
fn combine_accent_types(
    phrases: &mut [AccentPhrase],
    nodes: &[NjdNode],
    rule_table: &AccentRuleTable,
) {
    for phrase in phrases.iter_mut() {
        if phrase.nodes.is_empty() {
            continue;
        }

        // 最初のノードのアクセント型を初期値とする
        let first_idx = phrase.nodes[0];
        let mut combined_accent = nodes[first_idx].accent_type;
        let mut combined_mora: u8 = nodes[first_idx].mora_count;

        // 2番目以降のノードとアクセントを結合
        for window_idx in 1..phrase.nodes.len() {
            let node_idx = phrase.nodes[window_idx];
            let prev_idx = phrase.nodes[window_idx - 1];
            let node = &nodes[node_idx];
            let prev_node = &nodes[prev_idx];

            let left_pos = build_full_pos_str(prev_node);
            let right_pos = build_full_pos_str(node);

            if let Some(rule) = rule_table.find_rule(&left_pos, &right_pos) {
                combined_accent = apply_accent_rule(
                    &rule.rule_type,
                    combined_accent,
                    combined_mora,
                    node.accent_type,
                    node.mora_count,
                );
            }

            combined_mora = combined_mora.saturating_add(node.mora_count);
        }

        phrase.accent_type = combined_accent;
        phrase.mora_count = combined_mora;
    }
}

/// アクセント結合規則を適用する
fn apply_accent_rule(
    rule_type: &AccentRuleType,
    left_accent: u8,
    left_mora: u8,
    right_accent: u8,
    right_mora: u8,
) -> u8 {
    match rule_type {
        AccentRuleType::KeepLeft => left_accent,
        AccentRuleType::KeepRight => {
            if right_accent == 0 {
                0
            } else {
                left_mora.saturating_add(right_accent)
            }
        }
        AccentRuleType::Fixed(n) => *n,
        AccentRuleType::LeftMoraCount => left_mora,
        AccentRuleType::LeftMoraCountPlus(offset) => {
            (left_mora as i8 + offset).max(0) as u8
        }
        AccentRuleType::RightMoraCount => {
            // 後部のモーラ数がアクセント位置（前部モーラ数を加算）
            left_mora.saturating_add(right_mora)
        }
        AccentRuleType::Flat => 0,
    }
}

/// 特殊規則の適用
fn apply_special_rules(phrases: &mut [AccentPhrase], nodes: &[NjdNode]) {
    // 疑問文末の検出
    if let Some(last_phrase) = phrases.last_mut()
        && let Some(&last_node_idx) = last_phrase.nodes.last()
    {
        let last_node = &nodes[last_node_idx];
        if last_node.surface == "？" || last_node.surface == "?" || last_node.surface == "か" {
            last_phrase.is_interrogative = true;
        }
    }

    // フィラー（「えー」「あの」等）は平板型
    for phrase in phrases.iter_mut() {
        if let Some(&first_idx) = phrase.nodes.first()
            && nodes[first_idx].pos == Pos::Filler
        {
            phrase.accent_type = 0;
        }
    }

    // 数詞のみのフレーズでアクセント型が0の場合、1型に
    for phrase in phrases.iter_mut() {
        if phrase.accent_type == 0 && phrase.mora_count > 0 {
            let all_numerals = phrase
                .nodes
                .iter()
                .all(|&idx| nodes[idx].pos == Pos::Meishi && is_numeral(&nodes[idx].pos_detail1));
            if all_numerals {
                phrase.accent_type = 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::njd::InputToken;

    fn make_token(surface: &str, pos: &str, reading: &str) -> InputToken {
        InputToken::new(surface, pos, reading, reading)
    }

    fn make_detailed_token(
        surface: &str,
        pos: &str,
        detail1: &str,
        reading: &str,
    ) -> InputToken {
        let mut token = InputToken::new(surface, pos, reading, reading);
        token.pos_detail1 = detail1.to_string();
        token
    }

    #[test]
    fn test_basic_accent_phrase() {
        let tokens = vec![
            make_token("東京", "名詞", "トウキョウ"),
            make_token("に", "助詞", "ニ"),
        ];
        let mut nodes: Vec<NjdNode> = tokens.iter().map(NjdNode::from_token).collect();
        let table = AccentRuleTable::default_rules();
        let phrases = estimate_accent(&mut nodes, &table);

        // 「東京に」は1つのアクセント句
        assert_eq!(phrases.len(), 1);
        assert_eq!(phrases[0].nodes.len(), 2);
    }

    #[test]
    fn test_content_word_boundary() {
        let tokens = vec![
            make_token("猫", "名詞", "ネコ"),
            make_token("が", "助詞", "ガ"),
            make_token("走る", "動詞", "ハシル"),
        ];
        let mut nodes: Vec<NjdNode> = tokens.iter().map(NjdNode::from_token).collect();
        let table = AccentRuleTable::default_rules();
        let phrases = estimate_accent(&mut nodes, &table);

        // 「猫が」「走る」の2アクセント句
        assert_eq!(phrases.len(), 2);
    }

    #[test]
    fn test_compound_verb_chains() {
        let tokens = vec![
            make_token("食べ", "動詞", "タベ"),
            make_token("始める", "動詞", "ハジメル"),
        ];
        let mut nodes: Vec<NjdNode> = tokens.iter().map(NjdNode::from_token).collect();
        let table = AccentRuleTable::default_rules();
        let phrases = estimate_accent(&mut nodes, &table);

        // 複合動詞は1つのアクセント句
        assert_eq!(phrases.len(), 1);
        assert_eq!(phrases[0].nodes.len(), 2);
    }

    #[test]
    fn test_numeral_counter_chains() {
        let tokens = vec![
            make_detailed_token("三", "名詞", "数", "サン"),
            make_detailed_token("個", "名詞", "接尾,助数詞", "コ"),
        ];
        let mut nodes: Vec<NjdNode> = tokens.iter().map(NjdNode::from_token).collect();
        let table = AccentRuleTable::default_rules();
        let phrases = estimate_accent(&mut nodes, &table);

        // 数詞+助数詞は1つのアクセント句
        assert_eq!(phrases.len(), 1);
        assert_eq!(phrases[0].nodes.len(), 2);
    }

    #[test]
    fn test_keiyoudoushi_chains() {
        let tokens = vec![
            make_detailed_token("静か", "名詞", "形容動詞語幹", "シズカ"),
            make_token("だ", "助動詞", "ダ"),
        ];
        let mut nodes: Vec<NjdNode> = tokens.iter().map(NjdNode::from_token).collect();
        let table = AccentRuleTable::default_rules();
        let phrases = estimate_accent(&mut nodes, &table);

        // 形容動詞語幹+助動詞は1つのアクセント句
        assert_eq!(phrases.len(), 1);
        assert_eq!(phrases[0].nodes.len(), 2);
    }

    #[test]
    fn test_rentaishi_noun_separate() {
        let tokens = vec![
            make_token("この", "連体詞", "コノ"),
            make_token("本", "名詞", "ホン"),
        ];
        let mut nodes: Vec<NjdNode> = tokens.iter().map(NjdNode::from_token).collect();
        let table = AccentRuleTable::default_rules();
        let phrases = estimate_accent(&mut nodes, &table);

        // 訓練データでは連体詞+名詞は37%しかchainしないため、独立アクセント句
        assert_eq!(phrases.len(), 2);
    }

    #[test]
    fn test_suffix_chains() {
        let tokens = vec![
            make_token("田中", "名詞", "タナカ"),
            make_detailed_token("さん", "名詞", "接尾", "サン"),
        ];
        let mut nodes: Vec<NjdNode> = tokens.iter().map(NjdNode::from_token).collect();
        let table = AccentRuleTable::default_rules();
        let phrases = estimate_accent(&mut nodes, &table);

        // 名詞+接尾辞は1つのアクセント句
        assert_eq!(phrases.len(), 1);
        assert_eq!(phrases[0].nodes.len(), 2);
    }

    #[test]
    fn test_prefix_chains() {
        let tokens = vec![
            make_token("お", "接頭詞", "オ"),
            make_token("茶", "名詞", "チャ"),
        ];
        let mut nodes: Vec<NjdNode> = tokens.iter().map(NjdNode::from_token).collect();
        let table = AccentRuleTable::default_rules();
        let phrases = estimate_accent(&mut nodes, &table);

        // 接頭辞+名詞は1つのアクセント句
        assert_eq!(phrases.len(), 1);
        assert_eq!(phrases[0].nodes.len(), 2);
    }

    #[test]
    fn test_apply_accent_rule_keep_left() {
        let result = apply_accent_rule(&AccentRuleType::KeepLeft, 1, 3, 0, 2);
        assert_eq!(result, 1);
    }

    #[test]
    fn test_apply_accent_rule_keep_right() {
        let result = apply_accent_rule(&AccentRuleType::KeepRight, 1, 3, 2, 2);
        assert_eq!(result, 5); // 3 + 2
    }

    #[test]
    fn test_apply_accent_rule_flat() {
        let result = apply_accent_rule(&AccentRuleType::Flat, 1, 3, 2, 2);
        assert_eq!(result, 0);
    }

    #[test]
    fn test_apply_accent_rule_right_mora() {
        let result = apply_accent_rule(&AccentRuleType::RightMoraCount, 1, 3, 2, 2);
        assert_eq!(result, 5); // 3 + 2 (left_mora + right_mora)
    }

    #[test]
    fn test_non_independent_verb_chains() {
        // 動詞 + 助詞,接続助詞(て) + 動詞,非自立(いる) → 1アクセント句
        let tokens = vec![
            make_token("食べ", "動詞", "タベ"),
            make_detailed_token("て", "助詞", "接続助詞", "テ"),
            make_detailed_token("いる", "動詞", "非自立", "イル"),
        ];
        let mut nodes: Vec<NjdNode> = tokens.iter().map(NjdNode::from_token).collect();
        let table = AccentRuleTable::default_rules();
        let phrases = estimate_accent(&mut nodes, &table);

        // 「食べている」は1つのアクセント句
        assert_eq!(phrases.len(), 1);
        assert_eq!(phrases[0].nodes.len(), 3);
    }

    #[test]
    fn test_verb_auxiliary_chains() {
        // 動詞 + 助動詞（「食べました」）
        let tokens = vec![
            make_token("食べ", "動詞", "タベ"),
            make_token("まし", "助動詞", "マシ"),
            make_token("た", "助動詞", "タ"),
        ];
        let mut nodes: Vec<NjdNode> = tokens.iter().map(NjdNode::from_token).collect();
        let table = AccentRuleTable::default_rules();
        let phrases = estimate_accent(&mut nodes, &table);

        // 動詞+助動詞+助動詞 は1つのアクセント句
        assert_eq!(phrases.len(), 1);
        assert_eq!(phrases[0].nodes.len(), 3);
    }

    #[test]
    fn test_adjective_auxiliary_chains() {
        // 形容詞 + 助動詞（「美しいです」）
        let tokens = vec![
            make_token("美しい", "形容詞", "ウツクシイ"),
            make_token("です", "助動詞", "デス"),
        ];
        let mut nodes: Vec<NjdNode> = tokens.iter().map(NjdNode::from_token).collect();
        let table = AccentRuleTable::default_rules();
        let phrases = estimate_accent(&mut nodes, &table);

        // 形容詞+助動詞 は1つのアクセント句
        assert_eq!(phrases.len(), 1);
        assert_eq!(phrases[0].nodes.len(), 2);
    }

    #[test]
    fn test_noun_particle_chains() {
        // 名詞 + 助詞（「東京に」）は1アクセント句
        let tokens = vec![
            make_token("東京", "名詞", "トウキョウ"),
            make_token("に", "助詞", "ニ"),
        ];
        let mut nodes: Vec<NjdNode> = tokens.iter().map(NjdNode::from_token).collect();
        let table = AccentRuleTable::default_rules();
        let phrases = estimate_accent(&mut nodes, &table);

        assert_eq!(phrases.len(), 1);
        assert_eq!(phrases[0].nodes.len(), 2);
    }
}
