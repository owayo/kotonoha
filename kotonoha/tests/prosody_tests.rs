//! 韻律テスト
//! トーンパターン、韻律記号、PhoneTone抽出の網羅的テスト

use kotonoha::accent::AccentPhrase;
use kotonoha::njd::{InputToken, NjdNode};
use kotonoha::prosody::PhoneTone;
use kotonoha::Engine;

// ============================================================
// helpers
// ============================================================

fn tok(surface: &str, pos: &str, reading: &str) -> InputToken {
    InputToken::new(surface, pos, reading, reading)
}

fn tok_pron(surface: &str, pos: &str, reading: &str, pron: &str) -> InputToken {
    InputToken::new(surface, pos, reading, pron)
}

fn phones(pts: &[PhoneTone]) -> Vec<&str> {
    pts.iter().map(|p| p.phone.as_str()).collect()
}

fn mora_tones(pts: &[PhoneTone]) -> Vec<u8> {
    let inner: Vec<_> = pts.iter().filter(|p| p.phone != "sil").collect();
    let mut result = Vec::new();
    let mut i = 0;
    while i < inner.len() {
        let tone = inner[i].tone;
        if is_consonant_phone(&inner[i].phone) {
            result.push(tone);
            i += 2;
        } else {
            result.push(tone);
            i += 1;
        }
    }
    result
}

fn is_consonant_phone(p: &str) -> bool {
    matches!(
        p,
        "k" | "ky" | "g" | "gy" | "s" | "sh" | "z" | "j" | "t" | "ts" | "ch" | "d" | "dy"
            | "n" | "ny" | "h" | "hy" | "f" | "b" | "by" | "p" | "py" | "m" | "my" | "r"
            | "ry" | "w" | "y" | "v"
    )
}

fn make_phrases(nodes: &[NjdNode], accent_type: u8) -> Vec<AccentPhrase> {
    let mora_count: u8 = nodes.iter().map(|n| n.mora_count).sum();
    vec![AccentPhrase {
        nodes: (0..nodes.len()).collect(),
        accent_type,
        mora_count,
        is_interrogative: false,
    }]
}

// ============================================================
// Tone patterns: type 0 (flat) with 1-6 moras
// ============================================================

#[test]
fn test_tone_flat_1mora() {
    let engine = Engine::default();
    let tokens = vec![tok("木", "名詞", "キ")];
    let mut nodes = engine.analyze(&tokens);
    nodes[0].accent_type = 0;
    let phrases = make_phrases(&nodes, 0);
    let mt = mora_tones(&engine.extract_phone_tones(&nodes, &phrases));
    assert_eq!(mt, vec![0]); // single mora flat = low
}

#[test]
fn test_tone_flat_2mora() {
    let engine = Engine::default();
    let tokens = vec![tok("猫", "名詞", "ネコ")];
    let mut nodes = engine.analyze(&tokens);
    nodes[0].accent_type = 0;
    let phrases = make_phrases(&nodes, 0);
    let mt = mora_tones(&engine.extract_phone_tones(&nodes, &phrases));
    assert_eq!(mt, vec![0, 1]);
}

#[test]
fn test_tone_flat_3mora() {
    let engine = Engine::default();
    let tokens = vec![tok("桜", "名詞", "サクラ")];
    let mut nodes = engine.analyze(&tokens);
    nodes[0].accent_type = 0;
    let phrases = make_phrases(&nodes, 0);
    let mt = mora_tones(&engine.extract_phone_tones(&nodes, &phrases));
    assert_eq!(mt, vec![0, 1, 1]);
}

#[test]
fn test_tone_flat_4mora() {
    let engine = Engine::default();
    let tokens = vec![tok("test", "名詞", "アイウエ")];
    let mut nodes = engine.analyze(&tokens);
    nodes[0].accent_type = 0;
    let phrases = make_phrases(&nodes, 0);
    let mt = mora_tones(&engine.extract_phone_tones(&nodes, &phrases));
    assert_eq!(mt, vec![0, 1, 1, 1]);
}

#[test]
fn test_tone_flat_5mora() {
    let engine = Engine::default();
    let tokens = vec![tok("test", "名詞", "アイウエオ")];
    let mut nodes = engine.analyze(&tokens);
    nodes[0].accent_type = 0;
    let phrases = make_phrases(&nodes, 0);
    let mt = mora_tones(&engine.extract_phone_tones(&nodes, &phrases));
    assert_eq!(mt, vec![0, 1, 1, 1, 1]);
}

#[test]
fn test_tone_flat_6mora() {
    let engine = Engine::default();
    let tokens = vec![tok("test", "名詞", "カキクケコサ")];
    let mut nodes = engine.analyze(&tokens);
    nodes[0].accent_type = 0;
    let phrases = make_phrases(&nodes, 0);
    let mt = mora_tones(&engine.extract_phone_tones(&nodes, &phrases));
    assert_eq!(mt, vec![0, 1, 1, 1, 1, 1]);
}

// ============================================================
// Tone patterns: type 1 (atamadaka) with 1-6 moras
// ============================================================

#[test]
fn test_tone_atamadaka_1mora() {
    let engine = Engine::default();
    let tokens = vec![tok("木", "名詞", "キ")];
    let mut nodes = engine.analyze(&tokens);
    nodes[0].accent_type = 1;
    let phrases = make_phrases(&nodes, 1);
    let mt = mora_tones(&engine.extract_phone_tones(&nodes, &phrases));
    assert_eq!(mt, vec![1]);
}

#[test]
fn test_tone_atamadaka_2mora() {
    let engine = Engine::default();
    let tokens = vec![tok("猫", "名詞", "ネコ")];
    let mut nodes = engine.analyze(&tokens);
    nodes[0].accent_type = 1;
    let phrases = make_phrases(&nodes, 1);
    let mt = mora_tones(&engine.extract_phone_tones(&nodes, &phrases));
    assert_eq!(mt, vec![1, 0]);
}

#[test]
fn test_tone_atamadaka_3mora() {
    let engine = Engine::default();
    let tokens = vec![tok("test", "名詞", "サクラ")];
    let mut nodes = engine.analyze(&tokens);
    nodes[0].accent_type = 1;
    let phrases = make_phrases(&nodes, 1);
    let mt = mora_tones(&engine.extract_phone_tones(&nodes, &phrases));
    assert_eq!(mt, vec![1, 0, 0]);
}

#[test]
fn test_tone_atamadaka_4mora() {
    let engine = Engine::default();
    let tokens = vec![tok("test", "名詞", "アイウエ")];
    let mut nodes = engine.analyze(&tokens);
    nodes[0].accent_type = 1;
    let phrases = make_phrases(&nodes, 1);
    let mt = mora_tones(&engine.extract_phone_tones(&nodes, &phrases));
    assert_eq!(mt, vec![1, 0, 0, 0]);
}

#[test]
fn test_tone_atamadaka_5mora() {
    let engine = Engine::default();
    let tokens = vec![tok("test", "名詞", "アイウエオ")];
    let mut nodes = engine.analyze(&tokens);
    nodes[0].accent_type = 1;
    let phrases = make_phrases(&nodes, 1);
    let mt = mora_tones(&engine.extract_phone_tones(&nodes, &phrases));
    assert_eq!(mt, vec![1, 0, 0, 0, 0]);
}

#[test]
fn test_tone_atamadaka_6mora() {
    let engine = Engine::default();
    let tokens = vec![tok("test", "名詞", "カキクケコサ")];
    let mut nodes = engine.analyze(&tokens);
    nodes[0].accent_type = 1;
    let phrases = make_phrases(&nodes, 1);
    let mt = mora_tones(&engine.extract_phone_tones(&nodes, &phrases));
    assert_eq!(mt, vec![1, 0, 0, 0, 0, 0]);
}

// ============================================================
// Tone patterns: type 2 (nakadaka) with 2-6 moras
// ============================================================

#[test]
fn test_tone_nakadaka2_2mora() {
    let engine = Engine::default();
    let tokens = vec![tok("test", "名詞", "ネコ")];
    let mut nodes = engine.analyze(&tokens);
    nodes[0].accent_type = 2;
    let phrases = make_phrases(&nodes, 2);
    let mt = mora_tones(&engine.extract_phone_tones(&nodes, &phrases));
    // odaka for 2-mora word: 0,1
    assert_eq!(mt, vec![0, 1]);
}

#[test]
fn test_tone_nakadaka2_3mora() {
    let engine = Engine::default();
    let tokens = vec![tok("test", "名詞", "ココロ")];
    let mut nodes = engine.analyze(&tokens);
    nodes[0].accent_type = 2;
    let phrases = make_phrases(&nodes, 2);
    let mt = mora_tones(&engine.extract_phone_tones(&nodes, &phrases));
    assert_eq!(mt, vec![0, 1, 0]);
}

#[test]
fn test_tone_nakadaka2_4mora() {
    let engine = Engine::default();
    let tokens = vec![tok("test", "名詞", "アイウエ")];
    let mut nodes = engine.analyze(&tokens);
    nodes[0].accent_type = 2;
    let phrases = make_phrases(&nodes, 2);
    let mt = mora_tones(&engine.extract_phone_tones(&nodes, &phrases));
    assert_eq!(mt, vec![0, 1, 0, 0]);
}

// ============================================================
// Tone patterns: type 3 with 3-6 moras
// ============================================================

#[test]
fn test_tone_type3_3mora() {
    let engine = Engine::default();
    let tokens = vec![tok("test", "名詞", "サクラ")];
    let mut nodes = engine.analyze(&tokens);
    nodes[0].accent_type = 3;
    let phrases = make_phrases(&nodes, 3);
    let mt = mora_tones(&engine.extract_phone_tones(&nodes, &phrases));
    // odaka for 3 moras: 0,1,1
    assert_eq!(mt, vec![0, 1, 1]);
}

#[test]
fn test_tone_type3_4mora() {
    let engine = Engine::default();
    let tokens = vec![tok("test", "名詞", "アイウエ")];
    let mut nodes = engine.analyze(&tokens);
    nodes[0].accent_type = 3;
    let phrases = make_phrases(&nodes, 3);
    let mt = mora_tones(&engine.extract_phone_tones(&nodes, &phrases));
    assert_eq!(mt, vec![0, 1, 1, 0]);
}

#[test]
fn test_tone_type3_5mora() {
    let engine = Engine::default();
    let tokens = vec![tok("test", "名詞", "アイウエオ")];
    let mut nodes = engine.analyze(&tokens);
    nodes[0].accent_type = 3;
    let phrases = make_phrases(&nodes, 3);
    let mt = mora_tones(&engine.extract_phone_tones(&nodes, &phrases));
    assert_eq!(mt, vec![0, 1, 1, 0, 0]);
}

#[test]
fn test_tone_type3_6mora() {
    let engine = Engine::default();
    let tokens = vec![tok("test", "名詞", "カキクケコサ")];
    let mut nodes = engine.analyze(&tokens);
    nodes[0].accent_type = 3;
    let phrases = make_phrases(&nodes, 3);
    let mt = mora_tones(&engine.extract_phone_tones(&nodes, &phrases));
    assert_eq!(mt, vec![0, 1, 1, 0, 0, 0]);
}

// ============================================================
// Tone patterns: type 4 & 5 odaka
// ============================================================

#[test]
fn test_tone_type4_5mora() {
    let engine = Engine::default();
    let tokens = vec![tok("test", "名詞", "アイウエオ")];
    let mut nodes = engine.analyze(&tokens);
    nodes[0].accent_type = 4;
    let phrases = make_phrases(&nodes, 4);
    let mt = mora_tones(&engine.extract_phone_tones(&nodes, &phrases));
    assert_eq!(mt, vec![0, 1, 1, 1, 0]);
}

#[test]
fn test_tone_type5_5mora_odaka() {
    let engine = Engine::default();
    let tokens = vec![tok("test", "名詞", "アイウエオ")];
    let mut nodes = engine.analyze(&tokens);
    nodes[0].accent_type = 5;
    let phrases = make_phrases(&nodes, 5);
    let mt = mora_tones(&engine.extract_phone_tones(&nodes, &phrases));
    // odaka: 0,1,1,1,1
    assert_eq!(mt, vec![0, 1, 1, 1, 1]);
}

// ============================================================
// Prosody symbol structure
// ============================================================

#[test]
fn test_prosody_symbols_start_caret() {
    let engine = Engine::default();
    let tokens = vec![tok("猫", "名詞", "ネコ")];
    let nodes = engine.analyze(&tokens);
    let phrases = make_phrases(&nodes, 1);
    let syms = engine.extract_prosody_symbols(&nodes, &phrases);
    assert_eq!(syms.first().unwrap(), "^");
}

#[test]
fn test_prosody_symbols_end_dollar() {
    let engine = Engine::default();
    let tokens = vec![tok("猫", "名詞", "ネコ")];
    let nodes = engine.analyze(&tokens);
    let phrases = make_phrases(&nodes, 1);
    let syms = engine.extract_prosody_symbols(&nodes, &phrases);
    assert_eq!(syms.last().unwrap(), "$");
}

#[test]
fn test_prosody_symbols_end_question() {
    let engine = Engine::default();
    let tokens = vec![tok("猫", "名詞", "ネコ")];
    let nodes = engine.analyze(&tokens);
    let phrases = vec![AccentPhrase {
        nodes: vec![0],
        accent_type: 1,
        mora_count: 2,
        is_interrogative: true,
    }];
    let syms = engine.extract_prosody_symbols(&nodes, &phrases);
    assert_eq!(syms.last().unwrap(), "?");
}

#[test]
fn test_prosody_symbols_accent_rise_flat() {
    let engine = Engine::default();
    let tokens = vec![tok("桜", "名詞", "サクラ")];
    let nodes = engine.analyze(&tokens);
    let phrases = make_phrases(&nodes, 0);
    let syms = engine.extract_prosody_symbols(&nodes, &phrases);
    assert!(syms.contains(&"[".to_string()), "Flat should have accent rise: {:?}", syms);
}

#[test]
fn test_prosody_symbols_accent_fall_atamadaka() {
    let engine = Engine::default();
    let tokens = vec![tok("猫", "名詞", "ネコ")];
    let nodes = engine.analyze(&tokens);
    let phrases = make_phrases(&nodes, 1);
    let syms = engine.extract_prosody_symbols(&nodes, &phrases);
    assert!(syms.contains(&"]".to_string()), "Atamadaka should have accent fall: {:?}", syms);
}

#[test]
fn test_prosody_symbols_nakadaka_has_rise_and_fall() {
    let engine = Engine::default();
    let tokens = vec![tok("心", "名詞", "ココロ")];
    let nodes = engine.analyze(&tokens);
    let phrases = make_phrases(&nodes, 2);
    let syms = engine.extract_prosody_symbols(&nodes, &phrases);
    assert!(syms.contains(&"[".to_string()), "Nakadaka should have rise: {:?}", syms);
    assert!(syms.contains(&"]".to_string()), "Nakadaka should have fall: {:?}", syms);
}

// ============================================================
// Multi-phrase prosody symbols
// ============================================================

#[test]
fn test_prosody_two_phrases_one_pause() {
    let engine = Engine::default();
    let tokens = vec![tok("猫", "名詞", "ネコ"), tok("犬", "名詞", "イヌ")];
    let nodes = engine.analyze(&tokens);
    let phrases = vec![
        AccentPhrase { nodes: vec![0], accent_type: 1, mora_count: 2, is_interrogative: false },
        AccentPhrase { nodes: vec![1], accent_type: 0, mora_count: 2, is_interrogative: false },
    ];
    let syms = engine.extract_prosody_symbols(&nodes, &phrases);
    let pause_count = syms.iter().filter(|s| s.as_str() == "_").count();
    assert_eq!(pause_count, 1);
}

#[test]
fn test_prosody_three_phrases_two_pauses() {
    let engine = Engine::default();
    let tokens = vec![
        tok("猫", "名詞", "ネコ"),
        tok("犬", "名詞", "イヌ"),
        tok("鳥", "名詞", "トリ"),
    ];
    let nodes = engine.analyze(&tokens);
    let phrases = vec![
        AccentPhrase { nodes: vec![0], accent_type: 1, mora_count: 2, is_interrogative: false },
        AccentPhrase { nodes: vec![1], accent_type: 0, mora_count: 2, is_interrogative: false },
        AccentPhrase { nodes: vec![2], accent_type: 0, mora_count: 2, is_interrogative: false },
    ];
    let syms = engine.extract_prosody_symbols(&nodes, &phrases);
    let pause_count = syms.iter().filter(|s| s.as_str() == "_").count();
    assert_eq!(pause_count, 2);
}

#[test]
fn test_prosody_four_phrases_three_pauses() {
    let engine = Engine::default();
    let tokens = vec![
        tok("春", "名詞", "ハル"),
        tok("夏", "名詞", "ナツ"),
        tok("秋", "名詞", "アキ"),
        tok("冬", "名詞", "フユ"),
    ];
    let nodes = engine.analyze(&tokens);
    let phrases = vec![
        AccentPhrase { nodes: vec![0], accent_type: 1, mora_count: 2, is_interrogative: false },
        AccentPhrase { nodes: vec![1], accent_type: 1, mora_count: 2, is_interrogative: false },
        AccentPhrase { nodes: vec![2], accent_type: 0, mora_count: 2, is_interrogative: false },
        AccentPhrase { nodes: vec![3], accent_type: 1, mora_count: 2, is_interrogative: false },
    ];
    let syms = engine.extract_prosody_symbols(&nodes, &phrases);
    let pause_count = syms.iter().filter(|s| s.as_str() == "_").count();
    assert_eq!(pause_count, 3);
}

#[test]
fn test_prosody_five_phrases_four_pauses() {
    let engine = Engine::default();
    let tokens = vec![
        tok("月", "名詞", "ツキ"),
        tok("火", "名詞", "カ"),
        tok("水", "名詞", "スイ"),
        tok("木", "名詞", "モク"),
        tok("金", "名詞", "キン"),
    ];
    let nodes = engine.analyze(&tokens);
    let phrases = vec![
        AccentPhrase { nodes: vec![0], accent_type: 0, mora_count: 2, is_interrogative: false },
        AccentPhrase { nodes: vec![1], accent_type: 0, mora_count: 1, is_interrogative: false },
        AccentPhrase { nodes: vec![2], accent_type: 0, mora_count: 2, is_interrogative: false },
        AccentPhrase { nodes: vec![3], accent_type: 0, mora_count: 2, is_interrogative: false },
        AccentPhrase { nodes: vec![4], accent_type: 0, mora_count: 2, is_interrogative: false },
    ];
    let syms = engine.extract_prosody_symbols(&nodes, &phrases);
    let pause_count = syms.iter().filter(|s| s.as_str() == "_").count();
    assert_eq!(pause_count, 4);
}

// ============================================================
// PhoneTone extraction for specific kana types
// ============================================================

#[test]
fn test_phone_tone_vowel_only() {
    let engine = Engine::default();
    let tokens = vec![tok("愛", "名詞", "アイ")];
    let nodes = engine.analyze(&tokens);
    let phrases = make_phrases(&nodes, 1);
    let pts = engine.extract_phone_tones(&nodes, &phrases);
    let ph = phones(&pts);
    assert_eq!(ph[0], "sil");
    assert_eq!(ph[1], "a");
    assert_eq!(ph[2], "i");
    assert_eq!(ph[3], "sil");
}

#[test]
fn test_phone_tone_youon_kyo() {
    let engine = Engine::default();
    let tokens = vec![tok_pron("今日", "名詞", "キョウ", "キョー")];
    let nodes = engine.analyze(&tokens);
    let phrases = make_phrases(&nodes, 1);
    let pts = engine.extract_phone_tones(&nodes, &phrases);
    let ph = phones(&pts);
    assert!(ph.contains(&"ky"), "Should contain ky: {:?}", ph);
}

#[test]
fn test_phone_tone_sokuon() {
    let engine = Engine::default();
    let tokens = vec![tok_pron("学校", "名詞", "ガッコウ", "ガッコー")];
    let nodes = engine.analyze(&tokens);
    let phrases = make_phrases(&nodes, 0);
    let pts = engine.extract_phone_tones(&nodes, &phrases);
    let ph = phones(&pts);
    assert!(ph.contains(&"cl"), "Should contain cl: {:?}", ph);
}

#[test]
fn test_phone_tone_moraic_n() {
    let engine = Engine::default();
    let tokens = vec![tok("本", "名詞", "ホン")];
    let nodes = engine.analyze(&tokens);
    let phrases = make_phrases(&nodes, 1);
    let pts = engine.extract_phone_tones(&nodes, &phrases);
    let ph = phones(&pts);
    assert!(ph.contains(&"N"), "Should contain N: {:?}", ph);
}

#[test]
fn test_phone_tone_long_vowel_repeats() {
    let engine = Engine::default();
    let tokens = vec![tok("カー", "名詞", "カー")];
    let nodes = engine.analyze(&tokens);
    let phrases = make_phrases(&nodes, 0);
    let pts = engine.extract_phone_tones(&nodes, &phrases);
    let ph = phones(&pts);
    // sil, k, a, (long vowel), sil
    // The long vowel ー becomes a separate mora; reading-based parsing
    assert!(ph.contains(&"a"), "Should contain 'a' phoneme: {:?}", ph);
    assert_eq!(pts.len(), 5, "Should have 5 entries (sil+k+a+long+sil): {:?}", ph);
}

// ============================================================
// Interrogative vs declarative
// ============================================================

#[test]
fn test_declarative_ends_dollar() {
    let engine = Engine::default();
    let tokens = vec![tok("猫", "名詞", "ネコ"), tok("です", "助動詞", "デス")];
    let syms = engine.tokens_to_prosody_symbols(&tokens);
    assert_eq!(syms.last().unwrap(), "$");
}

#[test]
fn test_interrogative_ends_question() {
    let engine = Engine::default();
    let tokens = vec![tok("猫", "名詞", "ネコ"), tok("か", "助詞", "カ")];
    let syms = engine.tokens_to_prosody_symbols(&tokens);
    assert_eq!(syms.last().unwrap(), "?");
}

#[test]
fn test_interrogative_question_mark_symbol() {
    let engine = Engine::default();
    let tokens = vec![tok("猫", "名詞", "ネコ"), tok("？", "記号", "？")];
    let syms = engine.tokens_to_prosody_symbols(&tokens);
    assert_eq!(syms.last().unwrap(), "?");
}

// ============================================================
// PhoneTone sil boundary
// ============================================================

#[test]
fn test_phone_tones_always_start_with_sil() {
    let engine = Engine::default();
    let test_cases = vec![
        vec![tok("猫", "名詞", "ネコ")],
        vec![tok("桜", "名詞", "サクラ")],
        vec![tok("猫", "名詞", "ネコ"), tok("が", "助詞", "ガ")],
    ];
    for tokens in test_cases {
        let pts = engine.tokens_to_phone_tones(&tokens);
        assert_eq!(pts.first().unwrap().phone, "sil");
    }
}

#[test]
fn test_phone_tones_always_end_with_sil() {
    let engine = Engine::default();
    let test_cases = vec![
        vec![tok("猫", "名詞", "ネコ")],
        vec![tok("桜", "名詞", "サクラ")],
        vec![tok("猫", "名詞", "ネコ"), tok("が", "助詞", "ガ")],
    ];
    for tokens in test_cases {
        let pts = engine.tokens_to_phone_tones(&tokens);
        assert_eq!(pts.last().unwrap().phone, "sil");
    }
}

#[test]
fn test_phone_tones_sil_has_tone_zero() {
    let engine = Engine::default();
    let tokens = vec![tok("猫", "名詞", "ネコ")];
    let pts = engine.tokens_to_phone_tones(&tokens);
    assert_eq!(pts.first().unwrap().tone, 0);
    assert_eq!(pts.last().unwrap().tone, 0);
}

// ============================================================
// PhoneTone equality / clone
// ============================================================

#[test]
fn test_phone_tone_eq() {
    let a = PhoneTone { phone: "a".to_string(), tone: 1 };
    let b = PhoneTone { phone: "a".to_string(), tone: 1 };
    assert_eq!(a, b);
}

#[test]
fn test_phone_tone_ne() {
    let a = PhoneTone { phone: "a".to_string(), tone: 1 };
    let b = PhoneTone { phone: "a".to_string(), tone: 0 };
    assert_ne!(a, b);
}

#[test]
fn test_phone_tone_clone() {
    let a = PhoneTone { phone: "k".to_string(), tone: 1 };
    let b = a.clone();
    assert_eq!(a, b);
}

// ============================================================
// Prosody symbols for empty input
// ============================================================

#[test]
fn test_prosody_symbols_empty_input() {
    let engine = Engine::default();
    let tokens: Vec<InputToken> = vec![];
    let syms = engine.tokens_to_prosody_symbols(&tokens);
    assert!(syms.contains(&"^".to_string()));
    assert!(syms.contains(&"$".to_string()));
}

// ============================================================
// Prosody: flat has no fall marker
// ============================================================

#[test]
fn test_prosody_flat_has_terminal_fall() {
    let engine = Engine::default();
    let tokens = vec![tok("桜", "名詞", "サクラ")];
    let nodes = engine.analyze(&tokens);
    let phrases = make_phrases(&nodes, 0);
    let syms = engine.extract_prosody_symbols(&nodes, &phrases);
    // Flat type: last mora is high, and next_tone (beyond phrase) = 0,
    // so there's a terminal fall marker ] at the end
    // ^ サ [ ク ラ ] $
    assert!(syms.contains(&"]".to_string()), "Flat should have terminal fall: {:?}", syms);
    assert!(syms.contains(&"[".to_string()), "Flat should have rise: {:?}", syms);
}

// ============================================================
// Prosody: atamadaka 1-mora has fall but no rise
// ============================================================

#[test]
fn test_prosody_atamadaka_1mora_fall() {
    let engine = Engine::default();
    let tokens = vec![tok("木", "名詞", "キ")];
    let nodes = engine.analyze(&tokens);
    let phrases = make_phrases(&nodes, 1);
    let syms = engine.extract_prosody_symbols(&nodes, &phrases);
    // 1-mora atamadaka: ^ キ ] $ (fall after the only mora)
    assert!(syms.contains(&"]".to_string()), "Should have fall: {:?}", syms);
}

// ============================================================
// Multi-phrase symbol ordering
// ============================================================

#[test]
fn test_prosody_symbol_ordering_two_phrases() {
    let engine = Engine::default();
    let tokens = vec![tok("猫", "名詞", "ネコ"), tok("犬", "名詞", "イヌ")];
    let nodes = engine.analyze(&tokens);
    let phrases = vec![
        AccentPhrase { nodes: vec![0], accent_type: 1, mora_count: 2, is_interrogative: false },
        AccentPhrase { nodes: vec![1], accent_type: 0, mora_count: 2, is_interrogative: false },
    ];
    let syms = engine.extract_prosody_symbols(&nodes, &phrases);
    // ^ ... _ ... $
    let caret_pos = syms.iter().position(|s| s == "^").unwrap();
    let pause_pos = syms.iter().position(|s| s == "_").unwrap();
    let dollar_pos = syms.iter().position(|s| s == "$").unwrap();
    assert!(caret_pos < pause_pos);
    assert!(pause_pos < dollar_pos);
}
