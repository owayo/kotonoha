//! アクセント推定テスト
//! アクセント句境界、アクセント型、結合規則の網羅的テスト

use kotonoha::accent::AccentPhrase;
use kotonoha::accent_rule::{AccentRuleTable, AccentRuleType};
use kotonoha::njd::{InputToken, NjdNode, Pos};
use kotonoha::Engine;

// ============================================================
// helpers
// ============================================================

fn tok(surface: &str, pos: &str, reading: &str) -> InputToken {
    InputToken::new(surface, pos, reading, reading)
}

fn tok_suffix(surface: &str, reading: &str) -> InputToken {
    let mut t = InputToken::new(surface, "名詞", reading, reading);
    t.pos_detail1 = "接尾".to_string();
    t
}

// ============================================================
// Pos::parse tests
// ============================================================

#[test]
fn test_pos_parse_meishi() {
    assert_eq!(Pos::parse("名詞"), Pos::Meishi);
}

#[test]
fn test_pos_parse_doushi() {
    assert_eq!(Pos::parse("動詞"), Pos::Doushi);
}

#[test]
fn test_pos_parse_keiyoushi() {
    assert_eq!(Pos::parse("形容詞"), Pos::Keiyoushi);
}

#[test]
fn test_pos_parse_fukushi() {
    assert_eq!(Pos::parse("副詞"), Pos::Fukushi);
}

#[test]
fn test_pos_parse_joshi() {
    assert_eq!(Pos::parse("助詞"), Pos::Joshi);
}

#[test]
fn test_pos_parse_jodoushi() {
    assert_eq!(Pos::parse("助動詞"), Pos::Jodoushi);
}

#[test]
fn test_pos_parse_rentaishi() {
    assert_eq!(Pos::parse("連体詞"), Pos::Rentaishi);
}

#[test]
fn test_pos_parse_setsuzokushi() {
    assert_eq!(Pos::parse("接続詞"), Pos::Setsuzokushi);
}

#[test]
fn test_pos_parse_kandoushi() {
    assert_eq!(Pos::parse("感動詞"), Pos::Kandoushi);
}

#[test]
fn test_pos_parse_settoushi() {
    assert_eq!(Pos::parse("接頭詞"), Pos::Settoushi);
}

#[test]
fn test_pos_parse_settoushi_alt() {
    assert_eq!(Pos::parse("接頭辞"), Pos::Settoushi);
}

#[test]
fn test_pos_parse_kigou() {
    assert_eq!(Pos::parse("記号"), Pos::Kigou);
}

#[test]
fn test_pos_parse_filler() {
    assert_eq!(Pos::parse("フィラー"), Pos::Filler);
}

#[test]
fn test_pos_parse_unknown() {
    assert_eq!(Pos::parse("未知語"), Pos::Sonota);
}

// ============================================================
// is_content_word / is_function_word
// ============================================================

#[test]
fn test_content_words() {
    let content = [
        Pos::Meishi,
        Pos::Doushi,
        Pos::Keiyoushi,
        Pos::Fukushi,
        Pos::Rentaishi,
        Pos::Setsuzokushi,
        Pos::Kandoushi,
    ];
    for pos in &content {
        assert!(pos.is_content_word(), "{:?} should be content word", pos);
    }
}

#[test]
fn test_function_words() {
    assert!(Pos::Joshi.is_function_word());
    assert!(Pos::Jodoushi.is_function_word());
}

#[test]
fn test_non_content_words() {
    let non_content = [
        Pos::Joshi,
        Pos::Jodoushi,
        Pos::Settoushi,
        Pos::Kigou,
        Pos::Filler,
        Pos::Sonota,
    ];
    for pos in &non_content {
        assert!(!pos.is_content_word(), "{:?} should not be content word", pos);
    }
}

#[test]
fn test_non_function_words() {
    let non_func = [
        Pos::Meishi, Pos::Doushi, Pos::Keiyoushi, Pos::Kigou,
    ];
    for pos in &non_func {
        assert!(!pos.is_function_word(), "{:?} should not be function word", pos);
    }
}

// ============================================================
// to_label_str
// ============================================================

#[test]
fn test_to_label_str() {
    assert_eq!(Pos::Meishi.to_label_str(), "名詞");
    assert_eq!(Pos::Doushi.to_label_str(), "動詞");
    assert_eq!(Pos::Joshi.to_label_str(), "助詞");
    assert_eq!(Pos::Jodoushi.to_label_str(), "助動詞");
    assert_eq!(Pos::Keiyoushi.to_label_str(), "形容詞");
    assert_eq!(Pos::Fukushi.to_label_str(), "副詞");
    assert_eq!(Pos::Rentaishi.to_label_str(), "連体詞");
    assert_eq!(Pos::Setsuzokushi.to_label_str(), "接続詞");
    assert_eq!(Pos::Kandoushi.to_label_str(), "感動詞");
    assert_eq!(Pos::Settoushi.to_label_str(), "接頭詞");
    assert_eq!(Pos::Kigou.to_label_str(), "記号");
    assert_eq!(Pos::Filler.to_label_str(), "フィラー");
    assert_eq!(Pos::Sonota.to_label_str(), "その他");
}

// ============================================================
// Accent phrase boundary: noun + particle (same phrase)
// ============================================================

#[test]
fn test_boundary_meishi_joshi() {
    let engine = Engine::default();
    let tokens = vec![tok("猫", "名詞", "ネコ"), tok("が", "助詞", "ガ")];
    let mut nodes = engine.analyze(&tokens);
    let phrases = engine.estimate_accent(&mut nodes);
    assert_eq!(phrases.len(), 1);
    assert_eq!(phrases[0].nodes.len(), 2);
}

#[test]
fn test_boundary_meishi_jodoushi() {
    let engine = Engine::default();
    let tokens = vec![tok("猫", "名詞", "ネコ"), tok("だ", "助動詞", "ダ")];
    let mut nodes = engine.analyze(&tokens);
    let phrases = engine.estimate_accent(&mut nodes);
    assert_eq!(phrases.len(), 1);
    assert_eq!(phrases[0].nodes.len(), 2);
}

#[test]
fn test_boundary_doushi_joshi() {
    let engine = Engine::default();
    let tokens = vec![tok("走る", "動詞", "ハシル"), tok("の", "助詞", "ノ")];
    let mut nodes = engine.analyze(&tokens);
    let phrases = engine.estimate_accent(&mut nodes);
    assert_eq!(phrases.len(), 1);
    assert_eq!(phrases[0].nodes.len(), 2);
}

#[test]
fn test_boundary_doushi_jodoushi() {
    let engine = Engine::default();
    let tokens = vec![tok("食べ", "動詞", "タベ"), tok("ます", "助動詞", "マス")];
    let mut nodes = engine.analyze(&tokens);
    let phrases = engine.estimate_accent(&mut nodes);
    assert_eq!(phrases.len(), 1);
}

#[test]
fn test_boundary_keiyoushi_joshi() {
    let engine = Engine::default();
    let tokens = vec![tok("美しい", "形容詞", "ウツクシイ"), tok("の", "助詞", "ノ")];
    let mut nodes = engine.analyze(&tokens);
    let phrases = engine.estimate_accent(&mut nodes);
    assert_eq!(phrases.len(), 1);
}

#[test]
fn test_boundary_keiyoushi_jodoushi() {
    let engine = Engine::default();
    let tokens = vec![tok("美しい", "形容詞", "ウツクシイ"), tok("です", "助動詞", "デス")];
    let mut nodes = engine.analyze(&tokens);
    let phrases = engine.estimate_accent(&mut nodes);
    assert_eq!(phrases.len(), 1);
}

// ============================================================
// Accent phrase boundary: content + content (separate)
// ============================================================

#[test]
fn test_boundary_meishi_meishi() {
    let engine = Engine::default();
    let tokens = vec![tok("猫", "名詞", "ネコ"), tok("犬", "名詞", "イヌ")];
    let mut nodes = engine.analyze(&tokens);
    let phrases = engine.estimate_accent(&mut nodes);
    assert_eq!(phrases.len(), 2);
}

#[test]
fn test_boundary_meishi_doushi() {
    let engine = Engine::default();
    let tokens = vec![tok("猫", "名詞", "ネコ"), tok("走る", "動詞", "ハシル")];
    let mut nodes = engine.analyze(&tokens);
    let phrases = engine.estimate_accent(&mut nodes);
    assert_eq!(phrases.len(), 2);
}

#[test]
fn test_boundary_doushi_meishi() {
    let engine = Engine::default();
    let tokens = vec![tok("走る", "動詞", "ハシル"), tok("猫", "名詞", "ネコ")];
    let mut nodes = engine.analyze(&tokens);
    let phrases = engine.estimate_accent(&mut nodes);
    assert_eq!(phrases.len(), 2);
}

#[test]
fn test_boundary_doushi_doushi() {
    let engine = Engine::default();
    let tokens = vec![tok("走る", "動詞", "ハシル"), tok("食べる", "動詞", "タベル")];
    let mut nodes = engine.analyze(&tokens);
    let phrases = engine.estimate_accent(&mut nodes);
    // 複合動詞は1つのアクセント句に接続する
    assert_eq!(phrases.len(), 1);
}

#[test]
fn test_boundary_keiyoushi_meishi() {
    let engine = Engine::default();
    let tokens = vec![tok("良い", "形容詞", "ヨイ"), tok("天気", "名詞", "テンキ")];
    let mut nodes = engine.analyze(&tokens);
    let phrases = engine.estimate_accent(&mut nodes);
    assert_eq!(phrases.len(), 2);
}

#[test]
fn test_boundary_fukushi_doushi() {
    let engine = Engine::default();
    let tokens = vec![tok("とても", "副詞", "トテモ"), tok("走る", "動詞", "ハシル")];
    let mut nodes = engine.analyze(&tokens);
    let phrases = engine.estimate_accent(&mut nodes);
    assert_eq!(phrases.len(), 2);
}

// ============================================================
// Accent phrase boundary: prefix / suffix chaining
// ============================================================

#[test]
fn test_boundary_settoushi_meishi() {
    let engine = Engine::default();
    let tokens = vec![tok("お", "接頭詞", "オ"), tok("茶", "名詞", "チャ")];
    let mut nodes = engine.analyze(&tokens);
    let phrases = engine.estimate_accent(&mut nodes);
    assert_eq!(phrases.len(), 1);
    assert_eq!(phrases[0].nodes.len(), 2);
}

#[test]
fn test_boundary_meishi_suffix() {
    let engine = Engine::default();
    let tokens = vec![tok("東京", "名詞", "トウキョウ"), tok_suffix("駅", "エキ")];
    let mut nodes = engine.analyze(&tokens);
    let phrases = engine.estimate_accent(&mut nodes);
    assert_eq!(phrases.len(), 1);
    assert_eq!(phrases[0].nodes.len(), 2);
}

// ============================================================
// Accent phrase boundary: multi-particle chain
// ============================================================

#[test]
fn test_boundary_particle_chain() {
    let engine = Engine::default();
    let tokens = vec![
        tok("学校", "名詞", "ガッコウ"),
        tok("に", "助詞", "ニ"),
        tok("は", "助詞", "ワ"),
    ];
    let mut nodes = engine.analyze(&tokens);
    let phrases = engine.estimate_accent(&mut nodes);
    assert_eq!(phrases.len(), 1);
    assert_eq!(phrases[0].nodes.len(), 3);
}

#[test]
fn test_boundary_particle_chain_three() {
    let engine = Engine::default();
    let tokens = vec![
        tok("学校", "名詞", "ガッコウ"),
        tok("に", "助詞", "ニ"),
        tok("は", "助詞", "ワ"),
        tok("も", "助詞", "モ"),
    ];
    let mut nodes = engine.analyze(&tokens);
    let phrases = engine.estimate_accent(&mut nodes);
    assert_eq!(phrases.len(), 1);
    assert_eq!(phrases[0].nodes.len(), 4);
}

// ============================================================
// Accent phrase boundary: symbol breaks phrase
// ============================================================

#[test]
fn test_boundary_kigou_breaks() {
    let engine = Engine::default();
    let tokens = vec![
        tok("猫", "名詞", "ネコ"),
        tok("、", "記号", "、"),
        tok("犬", "名詞", "イヌ"),
    ];
    let mut nodes = engine.analyze(&tokens);
    let phrases = engine.estimate_accent(&mut nodes);
    assert!(phrases.len() >= 2);
}

// ============================================================
// Accent phrase boundary: filler
// ============================================================

#[test]
fn test_boundary_filler_separate() {
    let engine = Engine::default();
    let tokens = vec![
        tok("えー", "フィラー", "エー"),
        tok("猫", "名詞", "ネコ"),
    ];
    let mut nodes = engine.analyze(&tokens);
    let phrases = engine.estimate_accent(&mut nodes);
    assert_eq!(phrases.len(), 2);
}

#[test]
fn test_filler_always_flat() {
    let engine = Engine::default();
    let tokens = vec![tok("あの", "フィラー", "アノ")];
    let mut nodes = engine.analyze(&tokens);
    nodes[0].accent_type = 2;
    let phrases = engine.estimate_accent(&mut nodes);
    assert_eq!(phrases[0].accent_type, 0);
}

// ============================================================
// Accent phrase boundary: setsuzokushi
// ============================================================

#[test]
fn test_boundary_setsuzokushi_separate() {
    let engine = Engine::default();
    let tokens = vec![
        tok("しかし", "接続詞", "シカシ"),
        tok("猫", "名詞", "ネコ"),
    ];
    let mut nodes = engine.analyze(&tokens);
    let phrases = engine.estimate_accent(&mut nodes);
    assert_eq!(phrases.len(), 2);
}

// ============================================================
// chain_flag behavior
// ============================================================

#[test]
fn test_chain_flag_first_node_always_zero() {
    let engine = Engine::default();
    let tokens = vec![tok("猫", "名詞", "ネコ")];
    let mut nodes = engine.analyze(&tokens);
    engine.estimate_accent(&mut nodes);
    assert_eq!(nodes[0].chain_flag, 0);
}

#[test]
fn test_chain_flag_particle_is_one() {
    let engine = Engine::default();
    let tokens = vec![tok("猫", "名詞", "ネコ"), tok("が", "助詞", "ガ")];
    let mut nodes = engine.analyze(&tokens);
    engine.estimate_accent(&mut nodes);
    assert_eq!(nodes[1].chain_flag, 1);
}

#[test]
fn test_chain_flag_content_is_zero() {
    let engine = Engine::default();
    let tokens = vec![tok("猫", "名詞", "ネコ"), tok("犬", "名詞", "イヌ")];
    let mut nodes = engine.analyze(&tokens);
    engine.estimate_accent(&mut nodes);
    assert_eq!(nodes[1].chain_flag, 0);
}

#[test]
fn test_chain_flag_kigou_is_zero() {
    let engine = Engine::default();
    let tokens = vec![tok("猫", "名詞", "ネコ"), tok("、", "記号", "、")];
    let mut nodes = engine.analyze(&tokens);
    engine.estimate_accent(&mut nodes);
    assert_eq!(nodes[1].chain_flag, 0);
}

// ============================================================
// Accent type tests (0-5 with 1-6 mora words)
// ============================================================

#[test]
fn test_accent_type0_1mora() {
    let engine = Engine::default();
    let tokens = vec![tok("木", "名詞", "キ")];
    let mut nodes = engine.analyze(&tokens);
    nodes[0].accent_type = 0;
    let phrases = vec![AccentPhrase {
        nodes: vec![0],
        accent_type: 0,
        mora_count: 1,
        is_interrogative: false,
    }];
    let pts = engine.extract_phone_tones(&nodes, &phrases);
    // 1 mora flat: low
    let inner: Vec<_> = pts.iter().filter(|p| p.phone != "sil").collect();
    assert_eq!(inner[0].tone, 0); // k
    assert_eq!(inner[1].tone, 0); // i
}

#[test]
fn test_accent_type0_3mora() {
    let engine = Engine::default();
    let tokens = vec![tok("桜", "名詞", "サクラ")];
    let mut nodes = engine.analyze(&tokens);
    nodes[0].accent_type = 0;
    let phrases = vec![AccentPhrase {
        nodes: vec![0],
        accent_type: 0,
        mora_count: 3,
        is_interrogative: false,
    }];
    let pts = engine.extract_phone_tones(&nodes, &phrases);
    let inner: Vec<_> = pts.iter().filter(|p| p.phone != "sil").collect();
    // sa(0) ku(1) ra(1)
    assert_eq!(inner[0].tone, 0); // s
    assert_eq!(inner[1].tone, 0); // a
    assert_eq!(inner[2].tone, 1); // k
    assert_eq!(inner[3].tone, 1); // u
    assert_eq!(inner[4].tone, 1); // r
    assert_eq!(inner[5].tone, 1); // a
}

#[test]
fn test_accent_type1_2mora() {
    let engine = Engine::default();
    let tokens = vec![tok("猫", "名詞", "ネコ")];
    let mut nodes = engine.analyze(&tokens);
    nodes[0].accent_type = 1;
    let phrases = vec![AccentPhrase {
        nodes: vec![0],
        accent_type: 1,
        mora_count: 2,
        is_interrogative: false,
    }];
    let pts = engine.extract_phone_tones(&nodes, &phrases);
    let inner: Vec<_> = pts.iter().filter(|p| p.phone != "sil").collect();
    // ne(1) ko(0)
    assert_eq!(inner[0].tone, 1);
    assert_eq!(inner[1].tone, 1);
    assert_eq!(inner[2].tone, 0);
    assert_eq!(inner[3].tone, 0);
}

#[test]
fn test_accent_type1_4mora() {
    let engine = Engine::default();
    let tokens = vec![tok("アタマ", "名詞", "アタマカ")]; // 4 moras
    let mut nodes = engine.analyze(&tokens);
    nodes[0].accent_type = 1;
    let phrases = vec![AccentPhrase {
        nodes: vec![0],
        accent_type: 1,
        mora_count: 4,
        is_interrogative: false,
    }];
    let pts = engine.extract_phone_tones(&nodes, &phrases);
    let inner: Vec<_> = pts.iter().filter(|p| p.phone != "sil").collect();
    // First mora high, rest low
    assert_eq!(inner[0].tone, 1); // a (mora 0)
    assert_eq!(inner[1].tone, 0); // t (mora 1)
}

#[test]
fn test_accent_type2_3mora() {
    let engine = Engine::default();
    let tokens = vec![tok("心", "名詞", "ココロ")];
    let mut nodes = engine.analyze(&tokens);
    nodes[0].accent_type = 2;
    let phrases = vec![AccentPhrase {
        nodes: vec![0],
        accent_type: 2,
        mora_count: 3,
        is_interrogative: false,
    }];
    let pts = engine.extract_phone_tones(&nodes, &phrases);
    let inner: Vec<_> = pts.iter().filter(|p| p.phone != "sil").collect();
    // ko(0) ko(1) ro(0)
    assert_eq!(inner[0].tone, 0);
    assert_eq!(inner[1].tone, 0);
    assert_eq!(inner[2].tone, 1);
    assert_eq!(inner[3].tone, 1);
    assert_eq!(inner[4].tone, 0);
    assert_eq!(inner[5].tone, 0);
}

#[test]
fn test_accent_type2_5mora() {
    let engine = Engine::default();
    let tokens = vec![tok("test", "名詞", "カキクケコ")];
    let mut nodes = engine.analyze(&tokens);
    nodes[0].accent_type = 2;
    let phrases = vec![AccentPhrase {
        nodes: vec![0],
        accent_type: 2,
        mora_count: 5,
        is_interrogative: false,
    }];
    let pts = engine.extract_phone_tones(&nodes, &phrases);
    let inner: Vec<_> = pts.iter().filter(|p| p.phone != "sil").collect();
    // mora0(0) mora1(1) mora2(0) mora3(0) mora4(0)
    assert_eq!(inner[0].tone, 0); // k (mora 0)
    assert_eq!(inner[1].tone, 0); // a
    assert_eq!(inner[2].tone, 1); // k (mora 1)
    assert_eq!(inner[3].tone, 1); // i
    assert_eq!(inner[4].tone, 0); // k (mora 2)
    assert_eq!(inner[5].tone, 0); // u
}

#[test]
fn test_accent_type3_4mora() {
    let engine = Engine::default();
    let tokens = vec![tok("test", "名詞", "タマゴヤ")]; // 4 moras
    let mut nodes = engine.analyze(&tokens);
    nodes[0].accent_type = 3;
    let phrases = vec![AccentPhrase {
        nodes: vec![0],
        accent_type: 3,
        mora_count: 4,
        is_interrogative: false,
    }];
    let pts = engine.extract_phone_tones(&nodes, &phrases);
    let inner: Vec<_> = pts.iter().filter(|p| p.phone != "sil").collect();
    // mora0(0) mora1(1) mora2(1) mora3(0)
    assert_eq!(inner[0].tone, 0); // t
    assert_eq!(inner[1].tone, 0); // a
    assert_eq!(inner[2].tone, 1); // m
    assert_eq!(inner[3].tone, 1); // a
    assert_eq!(inner[4].tone, 1); // g
    assert_eq!(inner[5].tone, 1); // o
    assert_eq!(inner[6].tone, 0); // y
    assert_eq!(inner[7].tone, 0); // a
}

#[test]
fn test_accent_type4_4mora_odaka() {
    let engine = Engine::default();
    let tokens = vec![tok("妹", "名詞", "イモウト")];
    let mut nodes = engine.analyze(&tokens);
    nodes[0].accent_type = 4;
    let phrases = vec![AccentPhrase {
        nodes: vec![0],
        accent_type: 4,
        mora_count: 4,
        is_interrogative: false,
    }];
    let pts = engine.extract_phone_tones(&nodes, &phrases);
    let inner: Vec<_> = pts.iter().filter(|p| p.phone != "sil").collect();
    // mora0(0) mora1(1) mora2(1) mora3(1)
    assert_eq!(inner[0].tone, 0); // i
    assert_eq!(inner[1].tone, 1); // m
    assert_eq!(inner[2].tone, 1); // o
}

#[test]
fn test_accent_type5_6mora() {
    let engine = Engine::default();
    let tokens = vec![tok("test", "名詞", "カキクケコサ")]; // 6 moras
    let mut nodes = engine.analyze(&tokens);
    nodes[0].accent_type = 5;
    let phrases = vec![AccentPhrase {
        nodes: vec![0],
        accent_type: 5,
        mora_count: 6,
        is_interrogative: false,
    }];
    let pts = engine.extract_phone_tones(&nodes, &phrases);
    let inner: Vec<_> = pts.iter().filter(|p| p.phone != "sil").collect();
    // mora0(0) mora1(1) mora2(1) mora3(1) mora4(1) mora5(0)
    assert_eq!(inner[0].tone, 0);  // k (mora0)
    assert_eq!(inner[1].tone, 0);  // a
    assert_eq!(inner[2].tone, 1);  // k (mora1)
    assert_eq!(inner[3].tone, 1);  // i
    assert_eq!(inner[10].tone, 0); // s (mora5)
    assert_eq!(inner[11].tone, 0); // a
}

// ============================================================
// AccentRuleTable tests
// ============================================================

#[test]
fn test_rule_table_meishi_joshi() {
    let table = AccentRuleTable::default_rules();
    let rule = table.find_rule("名詞", "助詞");
    assert!(rule.is_some());
    assert_eq!(rule.unwrap().rule_type, AccentRuleType::KeepLeft);
}

#[test]
fn test_rule_table_meishi_jodoushi() {
    let table = AccentRuleTable::default_rules();
    let rule = table.find_rule("名詞", "助動詞");
    assert!(rule.is_some());
    assert_eq!(rule.unwrap().rule_type, AccentRuleType::KeepLeft);
}

#[test]
fn test_rule_table_doushi_joshi() {
    let table = AccentRuleTable::default_rules();
    let rule = table.find_rule("動詞", "助詞");
    assert!(rule.is_some());
    assert_eq!(rule.unwrap().rule_type, AccentRuleType::KeepLeft);
}

#[test]
fn test_rule_table_doushi_jodoushi() {
    let table = AccentRuleTable::default_rules();
    let rule = table.find_rule("動詞", "助動詞");
    assert!(rule.is_some());
    assert_eq!(rule.unwrap().rule_type, AccentRuleType::KeepLeft);
}

#[test]
fn test_rule_table_keiyoushi_joshi() {
    let table = AccentRuleTable::default_rules();
    let rule = table.find_rule("形容詞", "助詞");
    assert!(rule.is_some());
    assert_eq!(rule.unwrap().rule_type, AccentRuleType::KeepLeft);
}

#[test]
fn test_rule_table_keiyoushi_jodoushi() {
    let table = AccentRuleTable::default_rules();
    let rule = table.find_rule("形容詞", "助動詞");
    assert!(rule.is_some());
    assert_eq!(rule.unwrap().rule_type, AccentRuleType::KeepLeft);
}

#[test]
fn test_rule_table_wildcard() {
    let table = AccentRuleTable::default_rules();
    let rule = table.find_rule("フィラー", "記号");
    assert!(rule.is_some());
    // フィラー,* ルールにマッチ (Flat) - より具体的な規則が優先される
    assert_eq!(rule.unwrap().rule_type, AccentRuleType::Flat);
}

#[test]
fn test_rule_table_meishi_meishi() {
    let table = AccentRuleTable::default_rules();
    let rule = table.find_rule("名詞", "名詞");
    assert!(rule.is_some());
    assert_eq!(rule.unwrap().rule_type, AccentRuleType::LeftMoraCount);
}

// ============================================================
// Interrogative detection
// ============================================================

#[test]
fn test_interrogative_ka() {
    let engine = Engine::default();
    let tokens = vec![tok("猫", "名詞", "ネコ"), tok("か", "助詞", "カ")];
    let mut nodes = engine.analyze(&tokens);
    let phrases = engine.estimate_accent(&mut nodes);
    assert!(phrases.last().unwrap().is_interrogative);
}

#[test]
fn test_interrogative_question_mark() {
    let engine = Engine::default();
    let tokens = vec![tok("猫", "名詞", "ネコ"), tok("？", "記号", "？")];
    let mut nodes = engine.analyze(&tokens);
    let phrases = engine.estimate_accent(&mut nodes);
    assert!(phrases.last().unwrap().is_interrogative);
}

#[test]
fn test_non_interrogative() {
    let engine = Engine::default();
    let tokens = vec![tok("猫", "名詞", "ネコ"), tok("です", "助動詞", "デス")];
    let mut nodes = engine.analyze(&tokens);
    let phrases = engine.estimate_accent(&mut nodes);
    assert!(!phrases.last().unwrap().is_interrogative);
}

// ============================================================
// Mora count in accent phrase
// ============================================================

#[test]
fn test_phrase_mora_count_single_word() {
    let engine = Engine::default();
    let tokens = vec![tok("猫", "名詞", "ネコ")];
    let mut nodes = engine.analyze(&tokens);
    let phrases = engine.estimate_accent(&mut nodes);
    assert_eq!(phrases[0].mora_count, 2);
}

#[test]
fn test_phrase_mora_count_word_plus_particle() {
    let engine = Engine::default();
    let tokens = vec![tok("猫", "名詞", "ネコ"), tok("が", "助詞", "ガ")];
    let mut nodes = engine.analyze(&tokens);
    let phrases = engine.estimate_accent(&mut nodes);
    // ネコ(2) + ガ(1) = 3
    assert_eq!(phrases[0].mora_count, 3);
}

#[test]
fn test_phrase_mora_count_word_plus_two_particles() {
    let engine = Engine::default();
    let tokens = vec![
        tok("学校", "名詞", "ガッコウ"),
        tok("に", "助詞", "ニ"),
        tok("は", "助詞", "ワ"),
    ];
    let mut nodes = engine.analyze(&tokens);
    let phrases = engine.estimate_accent(&mut nodes);
    // ガッコウ(4) + ニ(1) + ワ(1) = 6
    assert_eq!(phrases[0].mora_count, 6);
}

// ============================================================
// Complex sentence phrase splitting
// ============================================================

#[test]
fn test_complex_sentence_three_phrases() {
    let engine = Engine::default();
    let tokens = vec![
        tok("猫", "名詞", "ネコ"),
        tok("が", "助詞", "ガ"),
        tok("走り", "動詞", "ハシリ"),
        tok("犬", "名詞", "イヌ"),
        tok("が", "助詞", "ガ"),
        tok("吠える", "動詞", "ホエル"),
    ];
    let mut nodes = engine.analyze(&tokens);
    let phrases = engine.estimate_accent(&mut nodes);
    // 猫が | 走り | 犬が | 吠える -> at least 3-4 phrases
    assert!(phrases.len() >= 3);
}

#[test]
fn test_complex_sentence_four_content_words() {
    let engine = Engine::default();
    let tokens = vec![
        tok("春", "名詞", "ハル"),
        tok("夏", "名詞", "ナツ"),
        tok("秋", "名詞", "アキ"),
        tok("冬", "名詞", "フユ"),
    ];
    let mut nodes = engine.analyze(&tokens);
    let phrases = engine.estimate_accent(&mut nodes);
    assert_eq!(phrases.len(), 4);
}

// ============================================================
// AccentRuleType variants
// ============================================================

#[test]
fn test_accent_rule_type_eq() {
    assert_eq!(AccentRuleType::KeepLeft, AccentRuleType::KeepLeft);
    assert_eq!(AccentRuleType::KeepRight, AccentRuleType::KeepRight);
    assert_eq!(AccentRuleType::Flat, AccentRuleType::Flat);
    assert_eq!(AccentRuleType::LeftMoraCount, AccentRuleType::LeftMoraCount);
    assert_eq!(AccentRuleType::Fixed(3), AccentRuleType::Fixed(3));
    assert_ne!(AccentRuleType::Fixed(3), AccentRuleType::Fixed(4));
    assert_eq!(
        AccentRuleType::LeftMoraCountPlus(1),
        AccentRuleType::LeftMoraCountPlus(1)
    );
    assert_ne!(
        AccentRuleType::LeftMoraCountPlus(1),
        AccentRuleType::LeftMoraCountPlus(2)
    );
}

// ============================================================
// NjdNode construction
// ============================================================

#[test]
fn test_njd_node_from_token_basic() {
    let token = InputToken::new("東京", "名詞", "トウキョウ", "トーキョー");
    let node = NjdNode::from_token(&token);
    assert_eq!(node.surface, "東京");
    assert_eq!(node.pos, Pos::Meishi);
    assert_eq!(node.reading, "トウキョウ");
    assert_eq!(node.pronunciation, "トオキョオ"); // long vowels expanded
    assert_eq!(node.mora_count, 4);
}

#[test]
fn test_njd_node_chain_flag_initial() {
    let token = InputToken::new("猫", "名詞", "ネコ", "ネコ");
    let node = NjdNode::from_token(&token);
    assert_eq!(node.chain_flag, -1); // undecided initially
}

#[test]
fn test_njd_node_accent_type_initial() {
    let token = InputToken::new("猫", "名詞", "ネコ", "ネコ");
    let node = NjdNode::from_token(&token);
    assert_eq!(node.accent_type, 0); // default 0
}

// ============================================================
// expand_long_vowels
// ============================================================

#[test]
fn test_expand_long_vowels_koohii() {
    assert_eq!(kotonoha::njd::expand_long_vowels("コーヒー"), "コオヒイ");
}

#[test]
fn test_expand_long_vowels_tookyoo() {
    assert_eq!(kotonoha::njd::expand_long_vowels("トーキョー"), "トオキョオ");
}

#[test]
fn test_expand_long_vowels_no_change() {
    assert_eq!(kotonoha::njd::expand_long_vowels("カタカナ"), "カタカナ");
}

#[test]
fn test_expand_long_vowels_a_dan() {
    assert_eq!(kotonoha::njd::expand_long_vowels("カー"), "カア");
}

#[test]
fn test_expand_long_vowels_i_dan() {
    assert_eq!(kotonoha::njd::expand_long_vowels("キー"), "キイ");
}

#[test]
fn test_expand_long_vowels_u_dan() {
    assert_eq!(kotonoha::njd::expand_long_vowels("クー"), "クウ");
}

#[test]
fn test_expand_long_vowels_e_dan() {
    assert_eq!(kotonoha::njd::expand_long_vowels("ケー"), "ケエ");
}

#[test]
fn test_expand_long_vowels_o_dan() {
    assert_eq!(kotonoha::njd::expand_long_vowels("コー"), "コオ");
}

#[test]
fn test_expand_long_vowels_empty() {
    assert_eq!(kotonoha::njd::expand_long_vowels(""), "");
}

// ============================================================
// Edge: empty input
// ============================================================

#[test]
fn test_accent_empty_input() {
    let engine = Engine::default();
    let tokens: Vec<InputToken> = vec![];
    let mut nodes = engine.analyze(&tokens);
    let phrases = engine.estimate_accent(&mut nodes);
    assert!(phrases.is_empty());
}
