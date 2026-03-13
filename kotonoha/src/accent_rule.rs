//! アクセント結合規則モジュール
//! OpenJTalkのnjd_set_accent_typeに相当する規則テーブルを管理する
//!
//! 品詞の階層的マッチングをサポートし、より具体的な規則が優先される。
//! 例: "名詞,接尾,助数詞" > "名詞,接尾" > "名詞" > "*"

use std::collections::HashMap;
use std::path::Path;

/// アクセント結合規則
#[derive(Debug, Clone)]
pub struct AccentRule {
    /// 前接語の品詞パターン（カンマ区切りの階層マッチ）
    pub left_pos: String,
    /// 後接語の品詞パターン（カンマ区切りの階層マッチ）
    pub right_pos: String,
    /// 結合後のアクセント型の計算方法
    pub rule_type: AccentRuleType,
    /// 規則の優先度（品詞部分の具体性に基づく。高いほど優先）
    pub priority: u8,
}

/// アクセント型の計算方法
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AccentRuleType {
    /// 前部のアクセントを保持
    KeepLeft,
    /// 後部のアクセントを保持（前部のモーラ数を加算）
    KeepRight,
    /// 固定値を設定
    Fixed(u8),
    /// 前部のモーラ数をアクセント位置とする
    LeftMoraCount,
    /// 前部モーラ数 + オフセット
    LeftMoraCountPlus(i8),
    /// 後部のモーラ数をアクセント位置とする（前部モーラ数を加算）
    RightMoraCount,
    /// 結合してフラット（0型）にする
    Flat,
}

/// アクセント規則テーブル
#[derive(Debug, Clone)]
pub struct AccentRuleTable {
    rules: Vec<AccentRule>,
    /// 品詞ペアによる高速引き用インデックス
    index: HashMap<(String, String), Vec<usize>>,
}

impl AccentRuleTable {
    /// デフォルトの組み込み規則を生成
    pub fn default_rules() -> Self {
        let rules = build_default_rules();
        let mut table = Self {
            rules: Vec::new(),
            index: HashMap::new(),
        };
        for rule in rules {
            table.add_rule(rule);
        }
        table
    }

    /// CSVファイルから規則を読み込む
    pub fn from_csv(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let mut table = Self {
            rules: Vec::new(),
            index: HashMap::new(),
        };

        let mut reader = csv::ReaderBuilder::new()
            .has_headers(true)
            .comment(Some(b'#'))
            .flexible(true)
            .from_path(path)?;

        for result in reader.records() {
            let record = result?;
            if record.len() < 3 {
                continue;
            }

            let fields: Vec<&str> = record.iter().collect();
            let (left_pos, right_pos, rule_str) = parse_csv_fields(&fields);

            let rule_type = parse_rule_type(&rule_str);
            let priority = compute_priority(&left_pos, &right_pos);

            table.add_rule(AccentRule {
                left_pos,
                right_pos,
                rule_type,
                priority,
            });
        }

        Ok(table)
    }

    fn add_rule(&mut self, rule: AccentRule) {
        let idx = self.rules.len();
        let key = (rule.left_pos.clone(), rule.right_pos.clone());
        self.index.entry(key).or_default().push(idx);
        self.rules.push(rule);
    }

    /// 品詞ペアに対応する規則を検索する（階層マッチ対応）
    ///
    /// マッチ戦略:
    /// 1. 完全一致を試みる
    /// 2. 品詞の詳細度を下げて階層的にフォールバック
    /// 3. ワイルドカードにフォールバック
    /// 4. 最も優先度の高い規則を返す
    pub fn find_rule(&self, left_pos: &str, right_pos: &str) -> Option<&AccentRule> {
        let left_variants = generate_pos_variants(left_pos);
        let right_variants = generate_pos_variants(right_pos);

        let mut best: Option<&AccentRule> = None;

        for left in &left_variants {
            for right in &right_variants {
                if let Some(indices) = self.index.get(&(left.clone(), right.clone())) {
                    for &idx in indices {
                        let rule = &self.rules[idx];
                        if best.is_none() || rule.priority > best.unwrap().priority {
                            best = Some(rule);
                        }
                    }
                }
            }
        }

        best
    }

    /// pos_detail付きの品詞文字列で規則を検索する
    pub fn find_rule_with_detail(
        &self,
        left_pos: &str,
        left_detail: &str,
        right_pos: &str,
        right_detail: &str,
    ) -> Option<&AccentRule> {
        let left_full = build_full_pos(left_pos, left_detail);
        let right_full = build_full_pos(right_pos, right_detail);
        self.find_rule(&left_full, &right_full)
    }
}

/// 品詞の完全文字列を組み立てる
fn build_full_pos(pos: &str, detail: &str) -> String {
    if detail.is_empty() || detail == "*" {
        pos.to_string()
    } else {
        format!("{},{}", pos, detail)
    }
}

/// 品詞文字列から階層的なバリアント列を生成する
/// "名詞,接尾,助数詞" → ["名詞,接尾,助数詞", "名詞,接尾", "名詞", "*"]
fn generate_pos_variants(pos: &str) -> Vec<String> {
    let mut variants = Vec::new();

    if pos == "*" {
        variants.push("*".to_string());
        return variants;
    }

    // 完全な文字列
    variants.push(pos.to_string());

    // カンマ区切りの各レベル
    let parts: Vec<&str> = pos.split(',').collect();
    for i in (1..parts.len()).rev() {
        variants.push(parts[..i].join(","));
    }

    // ワイルドカード
    variants.push("*".to_string());

    variants
}

/// CSVフィールドから左品詞、右品詞、規則文字列を抽出する
///
/// フォーマット:
/// - 3フィールド: left_pos, right_pos, rule_type
/// - 4フィールド: left_pos,detail, right_pos, rule_type → "left_pos,detail" + right_pos
/// - 5フィールド: left_pos,detail, right_pos,detail, rule_type
///
/// 規則文字列は最後のフィールドで、parse_rule_typeで認識される値を持つ。
fn parse_csv_fields(fields: &[&str]) -> (String, String, String) {
    let n = fields.len();
    if n < 3 {
        return (String::new(), String::new(), String::new());
    }

    // 最後のフィールドが規則文字列
    let rule_str = fields[n - 1].trim().to_string();

    // 残りのフィールドを左右品詞に分割する
    // 規則文字列を除いた残りのフィールドを見て、左と右に分割
    let pos_fields = &fields[..n - 1];

    match pos_fields.len() {
        2 => {
            // left_pos, right_pos
            (
                pos_fields[0].trim().to_string(),
                pos_fields[1].trim().to_string(),
                rule_str,
            )
        }
        3 => {
            // left_pos,detail1, right_pos OR left_pos, right_pos,detail1
            let f0 = pos_fields[0].trim();
            let f1 = pos_fields[1].trim();
            let f2 = pos_fields[2].trim();

            if is_known_pos(f2) {
                (format!("{},{}", f0, f1), f2.to_string(), rule_str)
            } else {
                (f0.to_string(), format!("{},{}", f1, f2), rule_str)
            }
        }
        4 => {
            // left_pos,detail, right_pos,detail
            let f0 = pos_fields[0].trim();
            let f1 = pos_fields[1].trim();
            let f2 = pos_fields[2].trim();
            let f3 = pos_fields[3].trim();

            if is_known_pos(f2) {
                (format!("{},{}", f0, f1), format!("{},{}", f2, f3), rule_str)
            } else if is_known_pos(f1) {
                (f0.to_string(), format!("{},{},{}", f1, f2, f3), rule_str)
            } else {
                (format!("{},{},{}", f0, f1, f2), f3.to_string(), rule_str)
            }
        }
        5 => {
            let f0 = pos_fields[0].trim();
            let f1 = pos_fields[1].trim();
            let f2 = pos_fields[2].trim();
            let f3 = pos_fields[3].trim();
            let f4 = pos_fields[4].trim();

            if is_known_pos(f2) {
                (format!("{},{}", f0, f1), format!("{},{},{}", f2, f3, f4), rule_str)
            } else if is_known_pos(f3) {
                (
                    format!("{},{},{}", f0, f1, f2),
                    format!("{},{}", f3, f4),
                    rule_str,
                )
            } else {
                (format!("{},{}", f0, f1), format!("{},{},{}", f2, f3, f4), rule_str)
            }
        }
        _ => {
            // Fallback: first half = left, second half = right
            let mid = pos_fields.len() / 2;
            let left: Vec<&str> = pos_fields[..mid].iter().map(|s| s.trim()).collect();
            let right: Vec<&str> = pos_fields[mid..].iter().map(|s| s.trim()).collect();
            (left.join(","), right.join(","), rule_str)
        }
    }
}

/// 既知の主品詞かどうか判定（CSV分割点の推測に使用）
fn is_known_pos(s: &str) -> bool {
    matches!(
        s,
        "名詞"
            | "動詞"
            | "形容詞"
            | "副詞"
            | "助詞"
            | "助動詞"
            | "連体詞"
            | "接続詞"
            | "感動詞"
            | "接頭詞"
            | "接頭辞"
            | "記号"
            | "フィラー"
            | "その他"
            | "*"
    )
}

/// 品詞パターンの具体性から優先度を計算する
fn compute_priority(left: &str, right: &str) -> u8 {
    let left_score = if left == "*" {
        0
    } else {
        left.split(',').count() as u8
    };
    let right_score = if right == "*" {
        0
    } else {
        right.split(',').count() as u8
    };
    left_score + right_score
}

fn parse_rule_type(s: &str) -> AccentRuleType {
    match s.trim() {
        "keep_left" => AccentRuleType::KeepLeft,
        "keep_right" => AccentRuleType::KeepRight,
        "flat" => AccentRuleType::Flat,
        "left_mora" => AccentRuleType::LeftMoraCount,
        "right_mora" => AccentRuleType::RightMoraCount,
        s if s.starts_with("fixed:") => {
            let n = s[6..].parse().unwrap_or(0);
            AccentRuleType::Fixed(n)
        }
        s if s.starts_with("left_mora+") => {
            let n = s[10..].parse().unwrap_or(0);
            AccentRuleType::LeftMoraCountPlus(n)
        }
        _ => AccentRuleType::KeepLeft,
    }
}

/// デフォルト規則の構築（OpenJTalk互換）
fn build_default_rules() -> Vec<AccentRule> {
    vec![
        // ========================================
        // 名詞 + 機能語（助詞・助動詞）
        // ========================================
        AccentRule {
            left_pos: "名詞".to_string(),
            right_pos: "助詞".to_string(),
            rule_type: AccentRuleType::KeepLeft,
            priority: 2,
        },
        AccentRule {
            left_pos: "名詞".to_string(),
            right_pos: "助動詞".to_string(),
            rule_type: AccentRuleType::KeepLeft,
            priority: 2,
        },
        // ========================================
        // 動詞 + 機能語
        // ========================================
        AccentRule {
            left_pos: "動詞".to_string(),
            right_pos: "助詞".to_string(),
            rule_type: AccentRuleType::KeepLeft,
            priority: 2,
        },
        AccentRule {
            left_pos: "動詞".to_string(),
            right_pos: "助動詞".to_string(),
            rule_type: AccentRuleType::KeepLeft,
            priority: 2,
        },
        // ========================================
        // 形容詞 + 機能語
        // ========================================
        AccentRule {
            left_pos: "形容詞".to_string(),
            right_pos: "助詞".to_string(),
            rule_type: AccentRuleType::KeepLeft,
            priority: 2,
        },
        AccentRule {
            left_pos: "形容詞".to_string(),
            right_pos: "助動詞".to_string(),
            rule_type: AccentRuleType::KeepLeft,
            priority: 2,
        },
        // ========================================
        // 副詞 + 機能語
        // ========================================
        AccentRule {
            left_pos: "副詞".to_string(),
            right_pos: "助詞".to_string(),
            rule_type: AccentRuleType::KeepLeft,
            priority: 2,
        },
        AccentRule {
            left_pos: "副詞".to_string(),
            right_pos: "助動詞".to_string(),
            rule_type: AccentRuleType::KeepLeft,
            priority: 2,
        },
        // ========================================
        // 連体詞 + 名詞
        // ========================================
        AccentRule {
            left_pos: "連体詞".to_string(),
            right_pos: "名詞".to_string(),
            rule_type: AccentRuleType::KeepLeft,
            priority: 2,
        },
        // ========================================
        // 接続詞
        // ========================================
        AccentRule {
            left_pos: "接続詞".to_string(),
            right_pos: "名詞".to_string(),
            rule_type: AccentRuleType::KeepLeft,
            priority: 2,
        },
        // ========================================
        // 名詞 + 接尾辞
        // ========================================
        AccentRule {
            left_pos: "名詞".to_string(),
            right_pos: "名詞,接尾".to_string(),
            rule_type: AccentRuleType::KeepRight,
            priority: 3,
        },
        AccentRule {
            left_pos: "名詞".to_string(),
            right_pos: "名詞,接尾,助数詞".to_string(),
            rule_type: AccentRuleType::RightMoraCount,
            priority: 4,
        },
        // ========================================
        // 接頭辞 + 名詞
        // ========================================
        AccentRule {
            left_pos: "接頭詞".to_string(),
            right_pos: "名詞".to_string(),
            rule_type: AccentRuleType::KeepRight,
            priority: 2,
        },
        AccentRule {
            left_pos: "接頭詞".to_string(),
            right_pos: "動詞".to_string(),
            rule_type: AccentRuleType::KeepRight,
            priority: 2,
        },
        AccentRule {
            left_pos: "接頭詞".to_string(),
            right_pos: "形容詞".to_string(),
            rule_type: AccentRuleType::KeepRight,
            priority: 2,
        },
        // ========================================
        // 名詞 + 名詞（複合語）
        // ========================================
        AccentRule {
            left_pos: "名詞".to_string(),
            right_pos: "名詞".to_string(),
            rule_type: AccentRuleType::LeftMoraCount,
            priority: 2,
        },
        // ========================================
        // 数詞 + 助数詞
        // ========================================
        AccentRule {
            left_pos: "名詞,数".to_string(),
            right_pos: "名詞,接尾,助数詞".to_string(),
            rule_type: AccentRuleType::RightMoraCount,
            priority: 5,
        },
        // ========================================
        // 動詞 + 動詞（複合動詞）
        // ========================================
        AccentRule {
            left_pos: "動詞".to_string(),
            right_pos: "動詞".to_string(),
            rule_type: AccentRuleType::KeepLeft,
            priority: 2,
        },
        // ========================================
        // 形容詞 + 動詞
        // ========================================
        AccentRule {
            left_pos: "形容詞".to_string(),
            right_pos: "動詞".to_string(),
            rule_type: AccentRuleType::KeepLeft,
            priority: 2,
        },
        // ========================================
        // 副詞 + 動詞
        // ========================================
        AccentRule {
            left_pos: "副詞".to_string(),
            right_pos: "動詞".to_string(),
            rule_type: AccentRuleType::KeepLeft,
            priority: 2,
        },
        // ========================================
        // 形容動詞語幹 + 助動詞
        // ========================================
        AccentRule {
            left_pos: "名詞,形容動詞語幹".to_string(),
            right_pos: "助動詞".to_string(),
            rule_type: AccentRuleType::KeepLeft,
            priority: 3,
        },
        // ========================================
        // 形容詞 + 名詞（く/さ形）
        // ========================================
        AccentRule {
            left_pos: "形容詞".to_string(),
            right_pos: "名詞,接尾".to_string(),
            rule_type: AccentRuleType::KeepLeft,
            priority: 3,
        },
        // ========================================
        // 感動詞
        // ========================================
        AccentRule {
            left_pos: "感動詞".to_string(),
            right_pos: "助詞".to_string(),
            rule_type: AccentRuleType::KeepLeft,
            priority: 2,
        },
        // ========================================
        // 助詞 + 助詞（95.2% chain in training data）
        // ========================================
        AccentRule {
            left_pos: "助詞".to_string(),
            right_pos: "助詞".to_string(),
            rule_type: AccentRuleType::KeepLeft,
            priority: 2,
        },
        // ========================================
        // 助動詞 + 助動詞（97.5% chain）
        // ========================================
        AccentRule {
            left_pos: "助動詞".to_string(),
            right_pos: "助動詞".to_string(),
            rule_type: AccentRuleType::KeepLeft,
            priority: 2,
        },
        // ========================================
        // 助動詞 + 助詞（96.5% chain）
        // ========================================
        AccentRule {
            left_pos: "助動詞".to_string(),
            right_pos: "助詞".to_string(),
            rule_type: AccentRuleType::KeepLeft,
            priority: 2,
        },
        // ========================================
        // 助詞 + 助動詞（94.9% chain）
        // ========================================
        AccentRule {
            left_pos: "助詞".to_string(),
            right_pos: "助動詞".to_string(),
            rule_type: AccentRuleType::KeepLeft,
            priority: 2,
        },
        // ========================================
        // 動詞,非自立 への接続規則
        // ========================================
        AccentRule {
            left_pos: "動詞".to_string(),
            right_pos: "動詞,非自立".to_string(),
            rule_type: AccentRuleType::KeepLeft,
            priority: 3,
        },
        AccentRule {
            left_pos: "助詞".to_string(),
            right_pos: "動詞,非自立".to_string(),
            rule_type: AccentRuleType::KeepLeft,
            priority: 3,
        },
        AccentRule {
            left_pos: "助動詞".to_string(),
            right_pos: "動詞,非自立".to_string(),
            rule_type: AccentRuleType::KeepLeft,
            priority: 3,
        },
        // ========================================
        // 形容詞,非自立 への接続規則
        // ========================================
        AccentRule {
            left_pos: "動詞".to_string(),
            right_pos: "形容詞,非自立".to_string(),
            rule_type: AccentRuleType::KeepLeft,
            priority: 3,
        },
        AccentRule {
            left_pos: "助動詞".to_string(),
            right_pos: "形容詞,非自立".to_string(),
            rule_type: AccentRuleType::KeepLeft,
            priority: 3,
        },
        // ========================================
        // フィラー
        // ========================================
        AccentRule {
            left_pos: "フィラー".to_string(),
            right_pos: "*".to_string(),
            rule_type: AccentRuleType::Flat,
            priority: 1,
        },
        // ========================================
        // デフォルト: 接続時は前部保持
        // ========================================
        AccentRule {
            left_pos: "*".to_string(),
            right_pos: "*".to_string(),
            rule_type: AccentRuleType::KeepLeft,
            priority: 0,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_rules() {
        let table = AccentRuleTable::default_rules();
        let rule = table.find_rule("名詞", "助詞");
        assert!(rule.is_some());
        assert_eq!(rule.unwrap().rule_type, AccentRuleType::KeepLeft);
    }

    #[test]
    fn test_wildcard_rule() {
        let table = AccentRuleTable::default_rules();
        // "*"/"*" のワイルドカードルールにマッチ
        let rule = table.find_rule("フィラー", "記号");
        assert!(rule.is_some());
    }

    #[test]
    fn test_hierarchical_match() {
        let table = AccentRuleTable::default_rules();

        // "名詞,接尾" は "名詞,接尾" 規則にマッチ
        let rule = table.find_rule("名詞", "名詞,接尾");
        assert!(rule.is_some());
        assert_eq!(rule.unwrap().rule_type, AccentRuleType::KeepRight);

        // "名詞,接尾,助数詞" はより具体的な規則にマッチ
        let rule = table.find_rule("名詞", "名詞,接尾,助数詞");
        assert!(rule.is_some());
        assert_eq!(rule.unwrap().rule_type, AccentRuleType::RightMoraCount);
    }

    #[test]
    fn test_hierarchical_fallback() {
        let table = AccentRuleTable::default_rules();

        // "名詞,一般" → "名詞" 規則にフォールバック
        let rule = table.find_rule("名詞,一般", "助詞");
        assert!(rule.is_some());
        assert_eq!(rule.unwrap().rule_type, AccentRuleType::KeepLeft);
    }

    #[test]
    fn test_priority_ordering() {
        let table = AccentRuleTable::default_rules();

        // 数詞+助数詞は最も優先度が高い
        let rule = table.find_rule("名詞,数", "名詞,接尾,助数詞");
        assert!(rule.is_some());
        assert_eq!(rule.unwrap().rule_type, AccentRuleType::RightMoraCount);
        assert_eq!(rule.unwrap().priority, 5);
    }

    #[test]
    fn test_generate_pos_variants() {
        let variants = generate_pos_variants("名詞,接尾,助数詞");
        assert_eq!(
            variants,
            vec![
                "名詞,接尾,助数詞".to_string(),
                "名詞,接尾".to_string(),
                "名詞".to_string(),
                "*".to_string(),
            ]
        );
    }

    #[test]
    fn test_generate_pos_variants_wildcard() {
        let variants = generate_pos_variants("*");
        assert_eq!(variants, vec!["*".to_string()]);
    }

    #[test]
    fn test_parse_rule_type_right_mora() {
        assert_eq!(parse_rule_type("right_mora"), AccentRuleType::RightMoraCount);
    }

    #[test]
    fn test_compute_priority() {
        assert_eq!(compute_priority("*", "*"), 0);
        assert_eq!(compute_priority("名詞", "助詞"), 2);
        assert_eq!(compute_priority("名詞,数", "名詞,接尾,助数詞"), 5);
    }

    #[test]
    fn test_find_rule_with_detail() {
        let table = AccentRuleTable::default_rules();
        let rule = table.find_rule_with_detail("名詞", "数", "名詞", "接尾,助数詞");
        assert!(rule.is_some());
        assert_eq!(rule.unwrap().rule_type, AccentRuleType::RightMoraCount);
    }
}
