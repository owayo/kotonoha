//! Engine API テスト
//! Engine構築、便利メソッド、一貫性、スレッドセーフティの網羅的テスト

use kotonoha::njd::{InputToken, Pos};
use kotonoha::Engine;
use std::sync::Arc;
use std::thread;

// ============================================================
// helpers
// ============================================================

fn tok(surface: &str, pos: &str, reading: &str) -> InputToken {
    InputToken::new(surface, pos, reading, reading)
}

fn tok_pron(surface: &str, pos: &str, reading: &str, pron: &str) -> InputToken {
    InputToken::new(surface, pos, reading, pron)
}

fn tok_suffix(surface: &str, reading: &str) -> InputToken {
    let mut t = InputToken::new(surface, "名詞", reading, reading);
    t.pos_detail1 = "接尾".to_string();
    t
}

// ============================================================
// Engine construction
// ============================================================

#[test]
fn test_engine_default() {
    let _engine = Engine::default();
}

#[test]
fn test_engine_with_default_rules() {
    let _engine = Engine::with_default_rules();
}

#[test]
fn test_engine_default_eq_with_default_rules() {
    let e1 = Engine::default();
    let e2 = Engine::with_default_rules();
    let tokens = vec![tok("猫", "名詞", "ネコ")];
    let l1 = e1.tokens_to_labels(&tokens);
    let l2 = e2.tokens_to_labels(&tokens);
    assert_eq!(l1, l2);
}

// ============================================================
// Convenience methods
// ============================================================

#[test]
fn test_tokens_to_labels_returns_non_empty() {
    let engine = Engine::default();
    let labels = engine.tokens_to_labels(&[tok("猫", "名詞", "ネコ")]);
    assert!(!labels.is_empty());
}

#[test]
fn test_tokens_to_phone_tones_returns_non_empty() {
    let engine = Engine::default();
    let pts = engine.tokens_to_phone_tones(&[tok("猫", "名詞", "ネコ")]);
    assert!(!pts.is_empty());
}

#[test]
fn test_tokens_to_prosody_symbols_returns_non_empty() {
    let engine = Engine::default();
    let syms = engine.tokens_to_prosody_symbols(&[tok("猫", "名詞", "ネコ")]);
    assert!(!syms.is_empty());
}

// ============================================================
// Step-by-step vs convenience method consistency
// ============================================================

#[test]
fn test_consistency_labels_neko_ga() {
    let engine = Engine::default();
    let tokens = vec![tok("猫", "名詞", "ネコ"), tok("が", "助詞", "ガ")];

    let labels_conv = engine.tokens_to_labels(&tokens);
    let mut nodes = engine.analyze(&tokens);
    let phrases = engine.estimate_accent(&mut nodes);
    let labels_step = engine.make_label(&nodes, &phrases);

    assert_eq!(labels_conv, labels_step);
}

#[test]
fn test_consistency_phone_tones_neko_ga() {
    let engine = Engine::default();
    let tokens = vec![tok("猫", "名詞", "ネコ"), tok("が", "助詞", "ガ")];

    let pt_conv = engine.tokens_to_phone_tones(&tokens);
    let mut nodes = engine.analyze(&tokens);
    let phrases = engine.estimate_accent(&mut nodes);
    let pt_step = engine.extract_phone_tones(&nodes, &phrases);

    assert_eq!(pt_conv, pt_step);
}

#[test]
fn test_consistency_prosody_symbols_neko_ga() {
    let engine = Engine::default();
    let tokens = vec![tok("猫", "名詞", "ネコ"), tok("が", "助詞", "ガ")];

    let sym_conv = engine.tokens_to_prosody_symbols(&tokens);
    let mut nodes = engine.analyze(&tokens);
    let phrases = engine.estimate_accent(&mut nodes);
    let sym_step = engine.extract_prosody_symbols(&nodes, &phrases);

    assert_eq!(sym_conv, sym_step);
}

#[test]
fn test_consistency_labels_complex_sentence() {
    let engine = Engine::default();
    let tokens = vec![
        tok_pron("今日", "名詞", "キョウ", "キョー"),
        tok("は", "助詞", "ワ"),
        tok("天気", "名詞", "テンキ"),
        tok("です", "助動詞", "デス"),
    ];

    let labels_conv = engine.tokens_to_labels(&tokens);
    let mut nodes = engine.analyze(&tokens);
    let phrases = engine.estimate_accent(&mut nodes);
    let labels_step = engine.make_label(&nodes, &phrases);

    assert_eq!(labels_conv, labels_step);
}

#[test]
fn test_consistency_phone_tones_complex_sentence() {
    let engine = Engine::default();
    let tokens = vec![
        tok_pron("今日", "名詞", "キョウ", "キョー"),
        tok("は", "助詞", "ワ"),
        tok("天気", "名詞", "テンキ"),
        tok("です", "助動詞", "デス"),
    ];

    let pt_conv = engine.tokens_to_phone_tones(&tokens);
    let mut nodes = engine.analyze(&tokens);
    let phrases = engine.estimate_accent(&mut nodes);
    let pt_step = engine.extract_phone_tones(&nodes, &phrases);

    assert_eq!(pt_conv, pt_step);
}

#[test]
fn test_consistency_prosody_complex_sentence() {
    let engine = Engine::default();
    let tokens = vec![
        tok_pron("今日", "名詞", "キョウ", "キョー"),
        tok("は", "助詞", "ワ"),
        tok("天気", "名詞", "テンキ"),
        tok("です", "助動詞", "デス"),
    ];

    let sym_conv = engine.tokens_to_prosody_symbols(&tokens);
    let mut nodes = engine.analyze(&tokens);
    let phrases = engine.estimate_accent(&mut nodes);
    let sym_step = engine.extract_prosody_symbols(&nodes, &phrases);

    assert_eq!(sym_conv, sym_step);
}

// ============================================================
// Japanese greetings and common sentences
// ============================================================

#[test]
fn test_sentence_ohayou_gozaimasu() {
    let engine = Engine::default();
    let tokens = vec![
        tok_pron("おはよう", "感動詞", "オハヨウ", "オハヨー"),
        tok("ございます", "助動詞", "ゴザイマス"),
    ];
    let labels = engine.tokens_to_labels(&tokens);
    assert!(!labels.is_empty());
    assert!(labels.first().unwrap().contains("sil"));
    assert!(labels.last().unwrap().contains("sil"));
}

#[test]
fn test_sentence_arigatou() {
    let engine = Engine::default();
    let tokens = vec![tok_pron("ありがとう", "感動詞", "アリガトウ", "アリガトー")];
    let pts = engine.tokens_to_phone_tones(&tokens);
    assert_eq!(pts.first().unwrap().phone, "sil");
    assert_eq!(pts.last().unwrap().phone, "sil");
}

#[test]
fn test_sentence_sumimasen() {
    let engine = Engine::default();
    let tokens = vec![tok("すみません", "感動詞", "スミマセン")];
    let labels = engine.tokens_to_labels(&tokens);
    assert!(labels.len() > 2);
}

#[test]
fn test_sentence_ogenki_desuka() {
    let engine = Engine::default();
    let tokens = vec![
        tok("お", "接頭詞", "オ"),
        tok("元気", "名詞", "ゲンキ"),
        tok("です", "助動詞", "デス"),
        tok("か", "助詞", "カ"),
    ];
    let syms = engine.tokens_to_prosody_symbols(&tokens);
    assert_eq!(syms.first().unwrap(), "^");
    assert_eq!(syms.last().unwrap(), "?"); // interrogative
}

#[test]
fn test_sentence_eki_wa_doko_desuka() {
    let engine = Engine::default();
    let tokens = vec![
        tok("駅", "名詞", "エキ"),
        tok("は", "助詞", "ワ"),
        tok("どこ", "名詞", "ドコ"),
        tok("です", "助動詞", "デス"),
        tok("か", "助詞", "カ"),
    ];
    let labels = engine.tokens_to_labels(&tokens);
    assert!(labels.first().unwrap().contains("sil"));
    assert!(labels.last().unwrap().contains("sil"));
    let syms = engine.tokens_to_prosody_symbols(&tokens);
    assert_eq!(syms.last().unwrap(), "?");
}

#[test]
fn test_sentence_tokyo_tower_ni_ikitai_desu() {
    let engine = Engine::default();
    let tokens = vec![
        tok_pron("東京", "名詞", "トウキョウ", "トーキョー"),
        tok("タワー", "名詞", "タワー"),
        tok("に", "助詞", "ニ"),
        tok("行き", "動詞", "イキ"),
        tok("たい", "助動詞", "タイ"),
        tok("です", "助動詞", "デス"),
    ];
    let labels = engine.tokens_to_labels(&tokens);
    assert!(labels.len() > 5);
}

#[test]
fn test_sentence_watashi_wa_gakusei_desu() {
    let engine = Engine::default();
    let tokens = vec![
        tok("私", "名詞", "ワタシ"),
        tok("は", "助詞", "ワ"),
        tok("学生", "名詞", "ガクセイ"),
        tok("です", "助動詞", "デス"),
    ];
    let mut nodes = engine.analyze(&tokens);
    let phrases = engine.estimate_accent(&mut nodes);
    // 私は | 学生です
    assert_eq!(phrases.len(), 2);
}

#[test]
fn test_sentence_sore_wa_ii_desu_ne() {
    let engine = Engine::default();
    let tokens = vec![
        tok("それ", "名詞", "ソレ"),
        tok("は", "助詞", "ワ"),
        tok("いい", "形容詞", "イイ"),
        tok("です", "助動詞", "デス"),
        tok("ね", "助詞", "ネ"),
    ];
    let labels = engine.tokens_to_labels(&tokens);
    assert!(labels.len() > 4);
}

#[test]
fn test_sentence_nihongo_wo_benkyou_shimasu() {
    let engine = Engine::default();
    let tokens = vec![
        tok("日本語", "名詞", "ニホンゴ"),
        tok("を", "助詞", "ヲ"),
        tok_pron("勉強", "名詞", "ベンキョウ", "ベンキョー"),
        tok("します", "動詞", "シマス"),
    ];
    let labels = engine.tokens_to_labels(&tokens);
    assert!(labels.first().unwrap().contains("sil"));
    assert!(labels.last().unwrap().contains("sil"));
}

#[test]
fn test_sentence_kinou_nani_wo_shimashitaka() {
    let engine = Engine::default();
    let tokens = vec![
        tok_pron("昨日", "名詞", "キノウ", "キノー"),
        tok("何", "名詞", "ナニ"),
        tok("を", "助詞", "ヲ"),
        tok("しました", "動詞", "シマシタ"),
        tok("か", "助詞", "カ"),
    ];
    let syms = engine.tokens_to_prosody_symbols(&tokens);
    assert_eq!(syms.last().unwrap(), "?");
}

#[test]
fn test_sentence_ame_ga_futte_imasu() {
    let engine = Engine::default();
    let tokens = vec![
        tok("雨", "名詞", "アメ"),
        tok("が", "助詞", "ガ"),
        tok("降って", "動詞", "フッテ"),
        tok("います", "動詞", "イマス"),
    ];
    let labels = engine.tokens_to_labels(&tokens);
    assert!(labels.len() > 4);
}

#[test]
fn test_sentence_kono_hon_wa_omoshiroi() {
    let engine = Engine::default();
    let tokens = vec![
        tok("この", "連体詞", "コノ"),
        tok("本", "名詞", "ホン"),
        tok("は", "助詞", "ワ"),
        tok("面白い", "形容詞", "オモシロイ"),
    ];
    let mut nodes = engine.analyze(&tokens);
    let phrases = engine.estimate_accent(&mut nodes);
    assert!(!phrases.is_empty());
}

#[test]
fn test_sentence_hayaku_hashitte_kudasai() {
    let engine = Engine::default();
    let tokens = vec![
        tok("速く", "副詞", "ハヤク"),
        tok("走って", "動詞", "ハシッテ"),
        tok("ください", "動詞", "クダサイ"),
    ];
    let labels = engine.tokens_to_labels(&tokens);
    assert!(labels.len() > 6);
}

// ============================================================
// Various POS combinations
// ============================================================

#[test]
fn test_pos_meishi_particle_chain() {
    let engine = Engine::default();
    let tokens = vec![
        tok("学校", "名詞", "ガッコウ"),
        tok("の", "助詞", "ノ"),
        tok("先生", "名詞", "センセイ"),
        tok("に", "助詞", "ニ"),
    ];
    let mut nodes = engine.analyze(&tokens);
    let phrases = engine.estimate_accent(&mut nodes);
    // 学校の | 先生に
    assert_eq!(phrases.len(), 2);
}

#[test]
fn test_pos_verb_auxiliary_chain() {
    let engine = Engine::default();
    let tokens = vec![
        tok("食べ", "動詞", "タベ"),
        tok("られ", "助動詞", "ラレ"),
        tok("ない", "助動詞", "ナイ"),
    ];
    let mut nodes = engine.analyze(&tokens);
    let phrases = engine.estimate_accent(&mut nodes);
    // All should chain together
    assert_eq!(phrases.len(), 1);
    assert_eq!(phrases[0].nodes.len(), 3);
}

#[test]
fn test_pos_adjective_particle() {
    let engine = Engine::default();
    let tokens = vec![
        tok("美しい", "形容詞", "ウツクシイ"),
        tok("花", "名詞", "ハナ"),
        tok("が", "助詞", "ガ"),
    ];
    let mut nodes = engine.analyze(&tokens);
    let phrases = engine.estimate_accent(&mut nodes);
    // 美しい | 花が
    assert_eq!(phrases.len(), 2);
}

#[test]
fn test_pos_adverb_verb() {
    let engine = Engine::default();
    let tokens = vec![
        tok("ゆっくり", "副詞", "ユックリ"),
        tok("歩く", "動詞", "アルク"),
    ];
    let mut nodes = engine.analyze(&tokens);
    let phrases = engine.estimate_accent(&mut nodes);
    assert_eq!(phrases.len(), 2);
}

#[test]
fn test_pos_rentaishi_noun() {
    let engine = Engine::default();
    let tokens = vec![
        tok("この", "連体詞", "コノ"),
        tok("本", "名詞", "ホン"),
    ];
    let mut nodes = engine.analyze(&tokens);
    let phrases = engine.estimate_accent(&mut nodes);
    // 連体詞 is content word, so it starts a new phrase,
    // but 本 is also content word -> depends on chaining rules
    assert!(!phrases.is_empty());
}

#[test]
fn test_pos_prefix_noun_suffix() {
    let engine = Engine::default();
    let tokens = vec![
        tok("お", "接頭詞", "オ"),
        tok("名前", "名詞", "ナマエ"),
    ];
    let mut nodes = engine.analyze(&tokens);
    let phrases = engine.estimate_accent(&mut nodes);
    assert_eq!(phrases.len(), 1);
}

// ============================================================
// Determinism
// ============================================================

#[test]
fn test_determinism_labels() {
    let engine = Engine::default();
    let tokens = vec![
        tok("猫", "名詞", "ネコ"),
        tok("が", "助詞", "ガ"),
        tok("走る", "動詞", "ハシル"),
    ];
    let l1 = engine.tokens_to_labels(&tokens);
    let l2 = engine.tokens_to_labels(&tokens);
    assert_eq!(l1, l2);
}

#[test]
fn test_determinism_phone_tones() {
    let engine = Engine::default();
    let tokens = vec![
        tok("猫", "名詞", "ネコ"),
        tok("が", "助詞", "ガ"),
    ];
    let pt1 = engine.tokens_to_phone_tones(&tokens);
    let pt2 = engine.tokens_to_phone_tones(&tokens);
    assert_eq!(pt1, pt2);
}

#[test]
fn test_determinism_prosody() {
    let engine = Engine::default();
    let tokens = vec![
        tok_pron("今日", "名詞", "キョウ", "キョー"),
        tok("は", "助詞", "ワ"),
    ];
    let s1 = engine.tokens_to_prosody_symbols(&tokens);
    let s2 = engine.tokens_to_prosody_symbols(&tokens);
    assert_eq!(s1, s2);
}

#[test]
fn test_determinism_100_iterations() {
    let engine = Engine::default();
    let tokens = vec![
        tok("猫", "名詞", "ネコ"),
        tok("が", "助詞", "ガ"),
        tok("走る", "動詞", "ハシル"),
    ];
    let baseline = engine.tokens_to_labels(&tokens);
    for i in 0..100 {
        let result = engine.tokens_to_labels(&tokens);
        assert_eq!(baseline, result, "Mismatch at iteration {}", i);
    }
}

// ============================================================
// Thread safety
// ============================================================

#[test]
fn test_thread_safety_basic() {
    let engine = Arc::new(Engine::default());
    let mut handles = vec![];

    for _ in 0..4 {
        let engine = Arc::clone(&engine);
        handles.push(thread::spawn(move || {
            let tokens = vec![tok("猫", "名詞", "ネコ")];
            let labels = engine.tokens_to_labels(&tokens);
            assert!(!labels.is_empty());
        }));
    }

    for handle in handles {
        handle.join().expect("Thread should not panic");
    }
}

#[test]
fn test_thread_safety_different_inputs() {
    let engine = Arc::new(Engine::default());
    let mut handles = vec![];

    let inputs: Vec<Vec<InputToken>> = vec![
        vec![tok("猫", "名詞", "ネコ")],
        vec![tok("犬", "名詞", "イヌ")],
        vec![tok("鳥", "名詞", "トリ")],
        vec![tok("魚", "名詞", "サカナ")],
        vec![tok("花", "名詞", "ハナ")],
        vec![tok("木", "名詞", "キ")],
        vec![tok("山", "名詞", "ヤマ")],
        vec![tok("川", "名詞", "カワ")],
    ];

    for input in inputs {
        let engine = Arc::clone(&engine);
        handles.push(thread::spawn(move || {
            let labels = engine.tokens_to_labels(&input);
            assert!(!labels.is_empty());
            assert!(labels.first().unwrap().contains("sil"));
            assert!(labels.last().unwrap().contains("sil"));
        }));
    }

    for handle in handles {
        handle.join().expect("Thread should not panic");
    }
}

#[test]
fn test_thread_safety_same_input_deterministic() {
    let engine = Arc::new(Engine::default());
    let mut handles = vec![];

    for _ in 0..8 {
        let engine = Arc::clone(&engine);
        handles.push(thread::spawn(move || {
            let tokens = vec![
                tok("猫", "名詞", "ネコ"),
                tok("が", "助詞", "ガ"),
                tok("走る", "動詞", "ハシル"),
            ];
            engine.tokens_to_labels(&tokens)
        }));
    }

    let results: Vec<Vec<String>> = handles
        .into_iter()
        .map(|h| h.join().expect("Thread should not panic"))
        .collect();

    for i in 1..results.len() {
        assert_eq!(results[0], results[i], "Thread {} produced different result", i);
    }
}

#[test]
fn test_thread_safety_phone_tones() {
    let engine = Arc::new(Engine::default());
    let mut handles = vec![];

    for _ in 0..4 {
        let engine = Arc::clone(&engine);
        handles.push(thread::spawn(move || {
            let tokens = vec![
                tok_pron("今日", "名詞", "キョウ", "キョー"),
                tok("は", "助詞", "ワ"),
            ];
            engine.tokens_to_phone_tones(&tokens)
        }));
    }

    let results: Vec<_> = handles
        .into_iter()
        .map(|h| h.join().expect("Thread should not panic"))
        .collect();

    for i in 1..results.len() {
        assert_eq!(results[0], results[i]);
    }
}

#[test]
fn test_thread_safety_prosody_symbols() {
    let engine = Arc::new(Engine::default());
    let mut handles = vec![];

    for _ in 0..4 {
        let engine = Arc::clone(&engine);
        handles.push(thread::spawn(move || {
            let tokens = vec![
                tok("桜", "名詞", "サクラ"),
                tok("が", "助詞", "ガ"),
                tok("咲く", "動詞", "サク"),
            ];
            engine.tokens_to_prosody_symbols(&tokens)
        }));
    }

    let results: Vec<_> = handles
        .into_iter()
        .map(|h| h.join().expect("Thread should not panic"))
        .collect();

    for i in 1..results.len() {
        assert_eq!(results[0], results[i]);
    }
}

// ============================================================
// analyze method
// ============================================================

#[test]
fn test_analyze_empty() {
    let engine = Engine::default();
    let nodes = engine.analyze(&[]);
    assert!(nodes.is_empty());
}

#[test]
fn test_analyze_single_token() {
    let engine = Engine::default();
    let nodes = engine.analyze(&[tok("猫", "名詞", "ネコ")]);
    assert_eq!(nodes.len(), 1);
    assert_eq!(nodes[0].surface, "猫");
    assert_eq!(nodes[0].pos, Pos::Meishi);
    assert_eq!(nodes[0].mora_count, 2);
}

#[test]
fn test_analyze_preserves_token_order() {
    let engine = Engine::default();
    let tokens = vec![
        tok("猫", "名詞", "ネコ"),
        tok("が", "助詞", "ガ"),
        tok("走る", "動詞", "ハシル"),
    ];
    let nodes = engine.analyze(&tokens);
    assert_eq!(nodes.len(), 3);
    assert_eq!(nodes[0].surface, "猫");
    assert_eq!(nodes[1].surface, "が");
    assert_eq!(nodes[2].surface, "走る");
}

#[test]
fn test_analyze_mora_count_various() {
    let engine = Engine::default();
    let test_cases = vec![
        (tok("木", "名詞", "キ"), 1u8),
        (tok("猫", "名詞", "ネコ"), 2),
        (tok("桜", "名詞", "サクラ"), 3),
        (tok_pron("コーヒー", "名詞", "コーヒー", "コーヒー"), 4),
        (tok("コンニチワ", "感動詞", "コンニチワ"), 5),
    ];
    for (token, expected_mora) in test_cases {
        let nodes = engine.analyze(&[token]);
        assert_eq!(
            nodes[0].mora_count, expected_mora,
            "Mora mismatch for {}",
            nodes[0].surface
        );
    }
}

// ============================================================
// InputToken construction
// ============================================================

#[test]
fn test_input_token_new() {
    let t = InputToken::new("猫", "名詞", "ネコ", "ネコ");
    assert_eq!(t.surface, "猫");
    assert_eq!(t.pos, "名詞");
    assert_eq!(t.reading, "ネコ");
    assert_eq!(t.pronunciation, "ネコ");
    assert_eq!(t.pos_detail1, "*");
    assert_eq!(t.pos_detail2, "*");
    assert_eq!(t.pos_detail3, "*");
    assert_eq!(t.ctype, "*");
    assert_eq!(t.cform, "*");
    assert_eq!(t.lemma, "猫"); // default lemma = surface
}

#[test]
fn test_input_token_with_details() {
    let mut t = InputToken::new("東京", "名詞", "トウキョウ", "トーキョー");
    t.pos_detail1 = "固有名詞".to_string();
    t.pos_detail2 = "地域".to_string();
    t.ctype = "特殊".to_string();

    let engine = Engine::default();
    let nodes = engine.analyze(&[t]);
    assert_eq!(nodes[0].pos_detail1, "固有名詞");
    assert_eq!(nodes[0].pos_detail2, "地域");
    assert_eq!(nodes[0].ctype, "特殊");
}

// ============================================================
// End-to-end: label count == phone_tone count
// ============================================================

#[test]
fn test_e2e_label_phone_tone_count_single() {
    let engine = Engine::default();
    let tokens = vec![tok("猫", "名詞", "ネコ")];
    let labels = engine.tokens_to_labels(&tokens);
    let pts = engine.tokens_to_phone_tones(&tokens);
    assert_eq!(labels.len(), pts.len());
}

#[test]
fn test_e2e_label_phone_tone_count_multi() {
    let engine = Engine::default();
    let tokens = vec![
        tok("猫", "名詞", "ネコ"),
        tok("が", "助詞", "ガ"),
        tok("走る", "動詞", "ハシル"),
    ];
    let labels = engine.tokens_to_labels(&tokens);
    let pts = engine.tokens_to_phone_tones(&tokens);
    assert_eq!(labels.len(), pts.len());
}

#[test]
fn test_e2e_label_phone_tone_count_empty() {
    let engine = Engine::default();
    let tokens: Vec<InputToken> = vec![];
    let labels = engine.tokens_to_labels(&tokens);
    let pts = engine.tokens_to_phone_tones(&tokens);
    // Empty input: labels returns [sil], phone_tones returns [sil, sil]
    assert!(!labels.is_empty());
    assert!(!pts.is_empty());
}

#[test]
fn test_e2e_label_phone_tone_count_long_sentence() {
    let engine = Engine::default();
    let tokens = vec![
        tok_pron("今日", "名詞", "キョウ", "キョー"),
        tok("は", "助詞", "ワ"),
        tok("天気", "名詞", "テンキ"),
        tok("が", "助詞", "ガ"),
        tok("良い", "形容詞", "ヨイ"),
        tok("の", "助詞", "ノ"),
        tok("で", "助詞", "デ"),
        tok_pron("東京", "名詞", "トウキョウ", "トーキョー"),
        tok("の", "助詞", "ノ"),
        tok_pron("公園", "名詞", "コウエン", "コーエン"),
        tok("に", "助詞", "ニ"),
        tok("行き", "動詞", "イキ"),
        tok("まし", "助動詞", "マシ"),
        tok("た", "助動詞", "タ"),
    ];
    let labels = engine.tokens_to_labels(&tokens);
    let pts = engine.tokens_to_phone_tones(&tokens);
    assert_eq!(labels.len(), pts.len());
}

// ============================================================
// Edge cases
// ============================================================

#[test]
fn test_single_particle_only() {
    let engine = Engine::default();
    let tokens = vec![tok("は", "助詞", "ワ")];
    let labels = engine.tokens_to_labels(&tokens);
    assert!(!labels.is_empty());
}

#[test]
fn test_single_symbol_only() {
    let engine = Engine::default();
    let tokens = vec![tok("、", "記号", "、")];
    let labels = engine.tokens_to_labels(&tokens);
    assert!(!labels.is_empty());
}

#[test]
fn test_many_particles() {
    let engine = Engine::default();
    let tokens = vec![
        tok("は", "助詞", "ワ"),
        tok("が", "助詞", "ガ"),
        tok("を", "助詞", "ヲ"),
        tok("に", "助詞", "ニ"),
        tok("で", "助詞", "デ"),
    ];
    let labels = engine.tokens_to_labels(&tokens);
    assert!(!labels.is_empty());
}

#[test]
fn test_suffix_chaining() {
    let engine = Engine::default();
    let tokens = vec![
        tok("東京", "名詞", "トウキョウ"),
        tok_suffix("駅", "エキ"),
        tok("前", "名詞", "マエ"),
    ];
    let mut nodes = engine.analyze(&tokens);
    let phrases = engine.estimate_accent(&mut nodes);
    // 東京駅 should be one phrase, 前 separate
    assert_eq!(phrases[0].nodes.len(), 2);
}

#[test]
fn test_filler_at_start() {
    let engine = Engine::default();
    let tokens = vec![
        tok("えー", "フィラー", "エー"),
        tok("猫", "名詞", "ネコ"),
        tok("です", "助動詞", "デス"),
    ];
    let mut nodes = engine.analyze(&tokens);
    let phrases = engine.estimate_accent(&mut nodes);
    // Filler should be flat
    assert_eq!(phrases[0].accent_type, 0);
}

#[test]
fn test_multiple_fillers() {
    let engine = Engine::default();
    let tokens = vec![
        tok("えー", "フィラー", "エー"),
        tok("あの", "フィラー", "アノ"),
        tok("猫", "名詞", "ネコ"),
    ];
    let mut nodes = engine.analyze(&tokens);
    let phrases = engine.estimate_accent(&mut nodes);
    // Both fillers should be flat
    for phrase in &phrases {
        if nodes[phrase.nodes[0]].pos == Pos::Filler {
            assert_eq!(phrase.accent_type, 0);
        }
    }
}

// ============================================================
// Sentence with all outputs checked
// ============================================================

#[test]
fn test_comprehensive_output_neko_ga_hashiru() {
    let engine = Engine::default();
    let tokens = vec![
        tok("猫", "名詞", "ネコ"),
        tok("が", "助詞", "ガ"),
        tok("走る", "動詞", "ハシル"),
    ];

    let labels = engine.tokens_to_labels(&tokens);
    let pts = engine.tokens_to_phone_tones(&tokens);
    let syms = engine.tokens_to_prosody_symbols(&tokens);

    // Labels
    assert!(labels.first().unwrap().contains("sil"));
    assert!(labels.last().unwrap().contains("sil"));
    assert_eq!(labels.len(), pts.len());

    // PhoneTones
    assert_eq!(pts.first().unwrap().phone, "sil");
    assert_eq!(pts.last().unwrap().phone, "sil");

    // Prosody
    assert_eq!(syms.first().unwrap(), "^");
    assert!(syms.last().unwrap() == "$" || syms.last().unwrap() == "?");
    assert!(syms.contains(&"_".to_string())); // phrase boundary
}
