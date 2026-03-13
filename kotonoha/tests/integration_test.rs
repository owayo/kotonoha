//! kotonoha 統合テスト
//! Engine API全体を通して、日本語韻律処理の正確性を検証する

use kotonoha::accent::AccentPhrase;
use kotonoha::njd::{InputToken, Pos};
use kotonoha::prosody::PhoneTone;
use kotonoha::Engine;

// ============================================================
// ヘルパー関数
// ============================================================

/// 簡易InputToken生成（reading == pronunciation）
fn tok(surface: &str, pos: &str, reading: &str) -> InputToken {
    InputToken::new(surface, pos, reading, reading)
}

/// InputToken生成（pronunciation指定あり）
fn tok_pron(surface: &str, pos: &str, reading: &str, pronunciation: &str) -> InputToken {
    InputToken::new(surface, pos, reading, pronunciation)
}

/// 接尾辞用のInputToken生成
fn tok_suffix(surface: &str, reading: &str) -> InputToken {
    let mut t = InputToken::new(surface, "名詞", reading, reading);
    t.pos_detail1 = "接尾".to_string();
    t
}

/// PhoneTone列から音素文字列だけ抽出
fn phones(pts: &[PhoneTone]) -> Vec<&str> {
    pts.iter().map(|p| p.phone.as_str()).collect()
}

/// 各モーラの先頭音素のトーンを取得（sil除く）
/// 子音がある場合はその子音のトーンを返す（母音と同じはず）
fn mora_tones(pts: &[PhoneTone]) -> Vec<u8> {
    let inner: Vec<_> = pts
        .iter()
        .filter(|p| p.phone != "sil")
        .collect();
    // モーラごとのトーンを取得: 母音のトーンを集める
    let mut result = Vec::new();
    let mut i = 0;
    while i < inner.len() {
        let tone = inner[i].tone;
        // 子音なら次が母音 → skip; 母音ならそのまま記録
        if is_consonant_phone(&inner[i].phone) {
            // 子音のトーン == 母音のトーン
            result.push(tone);
            i += 2; // 子音 + 母音
        } else {
            // 母音のみ (ア行, N, cl等)
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

// ============================================================
// 1. 基本的な文処理テスト
// ============================================================

#[test]
fn test_basic_sentence_kyou_wa_yoi_tenki_desu() {
    let engine = Engine::default();
    let tokens = vec![
        tok_pron("今日", "名詞", "キョウ", "キョー"),
        tok("は", "助詞", "ワ"),
        tok("良い", "形容詞", "ヨイ"),
        tok("天気", "名詞", "テンキ"),
        tok("です", "助動詞", "デス"),
    ];

    let labels = engine.tokens_to_labels(&tokens);
    let phone_tones = engine.tokens_to_phone_tones(&tokens);
    let symbols = engine.tokens_to_prosody_symbols(&tokens);

    // 各出力が空でないことを確認
    assert!(!labels.is_empty(), "Labels should not be empty");
    assert!(!phone_tones.is_empty(), "PhoneTones should not be empty");
    assert!(!symbols.is_empty(), "Prosody symbols should not be empty");

    // sil で始まり sil で終わる
    assert_eq!(phone_tones.first().unwrap().phone, "sil");
    assert_eq!(phone_tones.last().unwrap().phone, "sil");

    // 韻律記号は ^ で始まる
    assert_eq!(symbols.first().unwrap(), "^");
}

#[test]
fn test_basic_sentence_tokyo_tower_ni_ikimashita() {
    let engine = Engine::default();
    let tokens = vec![
        tok_pron("東京", "名詞", "トウキョウ", "トーキョー"),
        tok("タワー", "名詞", "タワー"),
        tok("に", "助詞", "ニ"),
        tok("行きました", "動詞", "イキマシタ"),
    ];

    let mut nodes = engine.analyze(&tokens);
    assert_eq!(nodes.len(), 4);

    // 東京: ト・ウ・キョ・ウ = 4モーラ
    assert_eq!(nodes[0].mora_count, 4);
    // タワー: タ・ワ・ー = 3モーラ
    assert_eq!(nodes[1].mora_count, 3);

    let phrases = engine.estimate_accent(&mut nodes);
    assert!(!phrases.is_empty(), "Should produce at least one accent phrase");

    let labels = engine.make_label(&nodes, &phrases);
    assert!(labels.first().unwrap().contains("sil"));
    assert!(labels.last().unwrap().contains("sil"));
}

#[test]
fn test_basic_sentence_neko_ga_hashiru() {
    let engine = Engine::default();
    let tokens = vec![
        tok("猫", "名詞", "ネコ"),
        tok("が", "助詞", "ガ"),
        tok("走る", "動詞", "ハシル"),
    ];

    let mut nodes = engine.analyze(&tokens);
    assert_eq!(nodes.len(), 3);
    assert_eq!(nodes[0].pos, Pos::Meishi);
    assert_eq!(nodes[1].pos, Pos::Joshi);
    assert_eq!(nodes[2].pos, Pos::Doushi);

    let phrases = engine.estimate_accent(&mut nodes);
    // 「猫が」(名詞+助詞) と 「走る」(動詞) で2つのアクセント句
    assert_eq!(phrases.len(), 2);
    assert_eq!(phrases[0].nodes.len(), 2); // 猫 + が
    assert_eq!(phrases[1].nodes.len(), 1); // 走る

    // モーラ数の確認
    // 猫が: ネ・コ・ガ = 3モーラ
    assert_eq!(phrases[0].mora_count, 3);
    // 走る: ハ・シ・ル = 3モーラ
    assert_eq!(phrases[1].mora_count, 3);
}

// ============================================================
// 2. アクセント句境界テスト
// ============================================================

#[test]
fn test_accent_boundary_content_plus_particle_same_phrase() {
    let engine = Engine::default();
    let tokens = vec![
        tok("東京", "名詞", "トウキョウ"),
        tok("に", "助詞", "ニ"),
    ];

    let mut nodes = engine.analyze(&tokens);
    let phrases = engine.estimate_accent(&mut nodes);

    // 内容語 + 助詞 → 同一アクセント句
    assert_eq!(phrases.len(), 1);
    assert_eq!(phrases[0].nodes.len(), 2);
}

#[test]
fn test_accent_boundary_content_plus_content_separate_phrases() {
    let engine = Engine::default();
    let tokens = vec![
        tok("猫", "名詞", "ネコ"),
        tok("走る", "動詞", "ハシル"),
    ];

    let mut nodes = engine.analyze(&tokens);
    let phrases = engine.estimate_accent(&mut nodes);

    // 内容語 + 内容語 → 別のアクセント句
    assert_eq!(phrases.len(), 2);
}

#[test]
fn test_accent_boundary_prefix_plus_noun_same_phrase() {
    let engine = Engine::default();
    let tokens = vec![
        tok("お", "接頭詞", "オ"),
        tok("茶", "名詞", "チャ"),
    ];

    let mut nodes = engine.analyze(&tokens);
    let phrases = engine.estimate_accent(&mut nodes);

    // 接頭詞 + 名詞 → 同一アクセント句
    assert_eq!(phrases.len(), 1);
    assert_eq!(phrases[0].nodes.len(), 2);
}

#[test]
fn test_accent_boundary_noun_plus_suffix_same_phrase() {
    let engine = Engine::default();
    let tokens = vec![
        tok("東京", "名詞", "トウキョウ"),
        tok_suffix("駅", "エキ"),
    ];

    let mut nodes = engine.analyze(&tokens);
    let phrases = engine.estimate_accent(&mut nodes);

    // 名詞 + 接尾辞 → 同一アクセント句
    assert_eq!(phrases.len(), 1);
    assert_eq!(phrases[0].nodes.len(), 2);
}

#[test]
fn test_accent_boundary_content_plus_jodoushi_same_phrase() {
    let engine = Engine::default();
    let tokens = vec![
        tok("食べ", "動詞", "タベ"),
        tok("ます", "助動詞", "マス"),
    ];

    let mut nodes = engine.analyze(&tokens);
    let phrases = engine.estimate_accent(&mut nodes);

    // 動詞 + 助動詞 → 同一アクセント句
    assert_eq!(phrases.len(), 1);
    assert_eq!(phrases[0].nodes.len(), 2);
}

#[test]
fn test_accent_boundary_multiple_particles_chain() {
    let engine = Engine::default();
    let tokens = vec![
        tok("学校", "名詞", "ガッコウ"),
        tok("に", "助詞", "ニ"),
        tok("は", "助詞", "ワ"),
    ];

    let mut nodes = engine.analyze(&tokens);
    let phrases = engine.estimate_accent(&mut nodes);

    // 名詞 + 助詞 + 助詞 → 全て同一アクセント句
    assert_eq!(phrases.len(), 1);
    assert_eq!(phrases[0].nodes.len(), 3);
}

#[test]
fn test_accent_boundary_three_content_words() {
    let engine = Engine::default();
    let tokens = vec![
        tok("猫", "名詞", "ネコ"),
        tok("犬", "名詞", "イヌ"),
        tok("鳥", "名詞", "トリ"),
    ];

    let mut nodes = engine.analyze(&tokens);
    let phrases = engine.estimate_accent(&mut nodes);

    // 3つの内容語 → 3つのアクセント句
    assert_eq!(phrases.len(), 3);
}

// ============================================================
// 3. アクセント型テスト
// ============================================================

#[test]
fn test_accent_type_heiban_flat_type0() {
    // 平板型 (0型): 低高高高...
    let engine = Engine::default();
    let tokens = vec![tok("桜", "名詞", "サクラ")];
    let mut nodes = engine.analyze(&tokens);
    nodes[0].accent_type = 0; // 平板型

    let phrases = vec![AccentPhrase {
        nodes: vec![0],
        accent_type: 0,
        mora_count: 3,
        is_interrogative: false,
    }];

    let phone_tones = engine.extract_phone_tones(&nodes, &phrases);
    let mt = mora_tones(&phone_tones);

    // サ=低(0), ク=高(1), ラ=高(1)
    assert_eq!(mt.len(), 3);
    assert_eq!(mt[0], 0, "First mora of heiban should be low");
    assert_eq!(mt[1], 1, "Second mora of heiban should be high");
    assert_eq!(mt[2], 1, "Third mora of heiban should be high");
}

#[test]
fn test_accent_type_atamadaka_head_high_type1() {
    // 頭高型 (1型): 高低低低...
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

    let phone_tones = engine.extract_phone_tones(&nodes, &phrases);
    let mt = mora_tones(&phone_tones);

    // ネ=高(1), コ=低(0)
    assert_eq!(mt.len(), 2);
    assert_eq!(mt[0], 1, "First mora of atamadaka should be high");
    assert_eq!(mt[1], 0, "Second mora of atamadaka should be low");
}

#[test]
fn test_accent_type_nakadaka_mid_high_type2() {
    // 中高型 (2型 for 3-mora word): 低高低
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

    let phone_tones = engine.extract_phone_tones(&nodes, &phrases);
    let mt = mora_tones(&phone_tones);

    // コ=低(0), コ=高(1), ロ=低(0)
    assert_eq!(mt.len(), 3);
    assert_eq!(mt[0], 0, "First mora of nakadaka should be low");
    assert_eq!(mt[1], 1, "Second mora of nakadaka should be high");
    assert_eq!(mt[2], 0, "Third mora of nakadaka should be low");
}

#[test]
fn test_accent_type_nakadaka_type3_four_moras() {
    // 中高型 (3型 for 4-mora word): 低高高低
    let engine = Engine::default();
    let tokens = vec![tok("タマゴ焼き", "名詞", "タマゴヤ")]; // 4 moras
    let mut nodes = engine.analyze(&tokens);
    nodes[0].accent_type = 3;

    let phrases = vec![AccentPhrase {
        nodes: vec![0],
        accent_type: 3,
        mora_count: 4,
        is_interrogative: false,
    }];

    let phone_tones = engine.extract_phone_tones(&nodes, &phrases);
    let mt = mora_tones(&phone_tones);

    assert_eq!(mt.len(), 4);
    assert_eq!(mt[0], 0, "Mora 1 should be low");
    assert_eq!(mt[1], 1, "Mora 2 should be high");
    assert_eq!(mt[2], 1, "Mora 3 should be high");
    assert_eq!(mt[3], 0, "Mora 4 should be low");
}

#[test]
fn test_accent_type_odaka_tail_high() {
    // 尾高型: accent_type == mora_count の場合
    // 例: 4モーラで4型 → 低高高高 (句内では下降しない)
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

    let phone_tones = engine.extract_phone_tones(&nodes, &phrases);
    let mt = mora_tones(&phone_tones);

    // 尾高: 低高高高 (最後のモーラの次に下がるが、句内では高のまま)
    assert_eq!(mt.len(), 4);
    assert_eq!(mt[0], 0);
    assert_eq!(mt[1], 1);
    assert_eq!(mt[2], 1);
    assert_eq!(mt[3], 1);
}

// ============================================================
// 4. 韻律記号テスト
// ============================================================

#[test]
fn test_prosody_symbols_start_and_end() {
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

    let symbols = engine.extract_prosody_symbols(&nodes, &phrases);

    assert_eq!(symbols.first().unwrap(), "^", "Should start with ^");
    assert_eq!(symbols.last().unwrap(), "$", "Should end with $");
}

#[test]
fn test_prosody_symbols_accent_fall() {
    let engine = Engine::default();
    let tokens = vec![tok("猫", "名詞", "ネコ")];
    let nodes = engine.analyze(&tokens);

    let phrases = vec![AccentPhrase {
        nodes: vec![0],
        accent_type: 1, // 頭高: ネ↓コ
        mora_count: 2,
        is_interrogative: false,
    }];

    let symbols = engine.extract_prosody_symbols(&nodes, &phrases);

    // ^ ネ ] コ $
    assert!(
        symbols.contains(&"]".to_string()),
        "Should contain accent fall marker ']'. Got: {:?}",
        symbols
    );

    // ] は ネ の後にある
    let ne_pos = symbols.iter().position(|s| s == "ネ").unwrap();
    let fall_pos = symbols.iter().position(|s| s == "]").unwrap();
    assert_eq!(fall_pos, ne_pos + 1, "Fall should come right after ネ");
}

#[test]
fn test_prosody_symbols_accent_rise() {
    let engine = Engine::default();
    let tokens = vec![tok("桜", "名詞", "サクラ")];
    let nodes = engine.analyze(&tokens);

    let phrases = vec![AccentPhrase {
        nodes: vec![0],
        accent_type: 0, // 平板: サ(低) → ク(高)ラ(高)
        mora_count: 3,
        is_interrogative: false,
    }];

    let symbols = engine.extract_prosody_symbols(&nodes, &phrases);

    // ^ サ [ クラ $  (平板なので下降マークなし)
    assert!(
        symbols.contains(&"[".to_string()),
        "Should contain accent rise marker '['. Got: {:?}",
        symbols
    );
}

#[test]
fn test_prosody_symbols_pause_between_phrases() {
    let engine = Engine::default();
    let tokens = vec![
        tok("猫", "名詞", "ネコ"),
        tok("走る", "動詞", "ハシル"),
    ];
    let nodes = engine.analyze(&tokens);

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
            mora_count: 3,
            is_interrogative: false,
        },
    ];

    let symbols = engine.extract_prosody_symbols(&nodes, &phrases);

    // アクセント句間に _ がある
    assert!(
        symbols.contains(&"_".to_string()),
        "Should contain pause marker '_' between phrases. Got: {:?}",
        symbols
    );
}

#[test]
fn test_prosody_symbols_interrogative() {
    let engine = Engine::default();
    let tokens = vec![tok("猫", "名詞", "ネコ")];
    let nodes = engine.analyze(&tokens);

    let phrases = vec![AccentPhrase {
        nodes: vec![0],
        accent_type: 1,
        mora_count: 2,
        is_interrogative: true, // 疑問
    }];

    let symbols = engine.extract_prosody_symbols(&nodes, &phrases);

    assert_eq!(
        symbols.last().unwrap(),
        "?",
        "Interrogative sentence should end with '?'. Got: {:?}",
        symbols
    );
}

#[test]
fn test_prosody_interrogative_detection_via_engine() {
    let engine = Engine::default();
    let tokens = vec![
        tok("猫", "名詞", "ネコ"),
        tok("か", "助詞", "カ"),
    ];

    let mut nodes = engine.analyze(&tokens);
    let phrases = engine.estimate_accent(&mut nodes);

    // 「か」で終わる文は疑問文として検出される
    assert!(
        phrases.last().unwrap().is_interrogative,
        "Sentence ending with か should be interrogative"
    );
}

// ============================================================
// 5. HTS Label フォーマットテスト
// ============================================================

#[test]
fn test_hts_label_starts_and_ends_with_sil() {
    let engine = Engine::default();
    let tokens = vec![
        tok("猫", "名詞", "ネコ"),
        tok("が", "助詞", "ガ"),
    ];

    let labels = engine.tokens_to_labels(&tokens);

    assert!(
        labels.first().unwrap().contains("sil"),
        "First label should contain sil"
    );
    assert!(
        labels.last().unwrap().contains("sil"),
        "Last label should contain sil"
    );
}

#[test]
fn test_hts_label_contains_required_fields() {
    let engine = Engine::default();
    let tokens = vec![tok("猫", "名詞", "ネコ")];
    let labels = engine.tokens_to_labels(&tokens);

    // 中間ラベル(非sil)を確認
    let mid_labels: Vec<_> = labels.iter().filter(|l| !l.starts_with("xx^xx-sil")).collect();
    assert!(!mid_labels.is_empty(), "Should have non-sil labels");

    for label in &mid_labels {
        assert!(label.contains("/A:"), "Label should contain /A: field: {}", label);
        assert!(label.contains("/B:"), "Label should contain /B: field: {}", label);
        assert!(label.contains("/C:"), "Label should contain /C: field: {}", label);
        assert!(label.contains("/D:"), "Label should contain /D: field: {}", label);
        assert!(label.contains("/E:"), "Label should contain /E: field: {}", label);
        assert!(label.contains("/K:"), "Label should contain /K: field: {}", label);
    }
}

#[test]
fn test_hts_label_phoneme_context_window() {
    let engine = Engine::default();
    let tokens = vec![tok("猫", "名詞", "ネコ")];
    let labels = engine.tokens_to_labels(&tokens);

    // 中間ラベルのフォーマット: p1^p2-p3+p4=p5/A:...
    for label in &labels {
        let context_part = label.split("/A:").next().unwrap();
        assert!(
            context_part.contains('^'),
            "Context should contain ^ separator: {}",
            context_part
        );
        assert!(
            context_part.contains('-'),
            "Context should contain - separator: {}",
            context_part
        );
        assert!(
            context_part.contains('+'),
            "Context should contain + separator: {}",
            context_part
        );
        assert!(
            context_part.contains('='),
            "Context should contain = separator: {}",
            context_part
        );
    }
}

#[test]
fn test_hts_label_sil_label_format() {
    let engine = Engine::default();
    let tokens = vec![tok("猫", "名詞", "ネコ")];
    let labels = engine.tokens_to_labels(&tokens);

    let sil_label = &labels[0];
    // sil label should have xx for all context positions
    assert!(
        sil_label.starts_with("xx^xx-sil+xx=xx"),
        "Sil label should start with xx^xx-sil+xx=xx, got: {}",
        sil_label
    );
}

#[test]
fn test_hts_label_count_matches_phonemes() {
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

    let labels = engine.make_label(&nodes, &phrases);
    let phone_tones = engine.extract_phone_tones(&nodes, &phrases);

    // ラベル数 == PhoneTone数 (sil + phonemes + sil)
    assert_eq!(
        labels.len(),
        phone_tones.len(),
        "Label count should match phone_tone count"
    );
}

// ============================================================
// 6. エッジケーステスト
// ============================================================

#[test]
fn test_empty_input() {
    let engine = Engine::default();
    let tokens: Vec<InputToken> = vec![];

    let nodes = engine.analyze(&tokens);
    assert!(nodes.is_empty(), "Empty input should produce empty nodes");

    let labels = engine.tokens_to_labels(&tokens);
    // 空入力でも最低1つのsilラベルが生成される
    assert!(!labels.is_empty(), "Even empty input should produce labels");
    assert!(labels[0].contains("sil"));

    let phone_tones = engine.tokens_to_phone_tones(&tokens);
    // 空入力: sil + sil
    assert!(phone_tones.len() >= 2);
    assert_eq!(phone_tones.first().unwrap().phone, "sil");
    assert_eq!(phone_tones.last().unwrap().phone, "sil");

    let symbols = engine.tokens_to_prosody_symbols(&tokens);
    // 最低限 ^ と $ がある
    assert!(symbols.contains(&"^".to_string()));
    assert!(symbols.contains(&"$".to_string()));
}

#[test]
fn test_single_mora_word() {
    let engine = Engine::default();
    let tokens = vec![tok("木", "名詞", "キ")];

    let mut nodes = engine.analyze(&tokens);
    assert_eq!(nodes.len(), 1);
    assert_eq!(nodes[0].mora_count, 1);

    nodes[0].accent_type = 1; // 頭高型 (1モーラなら高のみ)

    let phrases = vec![AccentPhrase {
        nodes: vec![0],
        accent_type: 1,
        mora_count: 1,
        is_interrogative: false,
    }];

    let phone_tones = engine.extract_phone_tones(&nodes, &phrases);
    let mt = mora_tones(&phone_tones);

    assert_eq!(mt.len(), 1);
    assert_eq!(mt[0], 1, "Single mora with type 1 should be high");

    let labels = engine.make_label(&nodes, &phrases);
    assert!(labels.len() >= 3); // sil + ki (k,i) + sil
}

#[test]
fn test_long_vowel_word_koohii() {
    // コーヒー → 長音展開のテスト
    let engine = Engine::default();
    let tokens = vec![tok_pron("コーヒー", "名詞", "コーヒー", "コーヒー")];

    let mut nodes = engine.analyze(&tokens);
    assert_eq!(nodes.len(), 1);
    // コーヒー: コ・ー・ヒ・ー = 4モーラ
    assert_eq!(nodes[0].mora_count, 4);
    // pronunciation should have long vowels expanded
    assert_eq!(nodes[0].pronunciation, "コオヒイ");

    nodes[0].accent_type = 3; // コーヒー is type 3

    let phrases = vec![AccentPhrase {
        nodes: vec![0],
        accent_type: 3,
        mora_count: 4,
        is_interrogative: false,
    }];

    let phone_tones = engine.extract_phone_tones(&nodes, &phrases);
    let ph = phones(&phone_tones);
    // sil, k, o, o, h, i, i, sil  (reading-based mora parsing)
    // Note: phonemes come from reading, not pronunciation
    assert_eq!(ph.first().unwrap(), &"sil");
    assert_eq!(ph.last().unwrap(), &"sil");
    // Should contain vowels 'o' and 'i'
    assert!(ph.contains(&"o"), "Should contain 'o' phoneme");
    assert!(ph.contains(&"i"), "Should contain 'i' phoneme");
}

#[test]
fn test_geminate_word_gakkou() {
    // がっこう (学校) → 促音のテスト
    let engine = Engine::default();
    let tokens = vec![tok_pron("学校", "名詞", "ガッコウ", "ガッコー")];

    let mut nodes = engine.analyze(&tokens);
    assert_eq!(nodes.len(), 1);
    // ガッコウ: ガ・ッ・コ・ウ = 4モーラ
    assert_eq!(nodes[0].mora_count, 4);

    nodes[0].accent_type = 0; // 平板型

    let phrases = vec![AccentPhrase {
        nodes: vec![0],
        accent_type: 0,
        mora_count: 4,
        is_interrogative: false,
    }];

    let phone_tones = engine.extract_phone_tones(&nodes, &phrases);
    let ph = phones(&phone_tones);

    // Should contain "cl" for 促音
    assert!(
        ph.contains(&"cl"),
        "Should contain 'cl' for geminate. Got: {:?}",
        ph
    );

    // トーン: 平板(0型) → 低高高高
    let mt = mora_tones(&phone_tones);
    assert_eq!(mt.len(), 4);
    assert_eq!(mt[0], 0); // ガ=低
    assert_eq!(mt[1], 1); // ッ=高
    assert_eq!(mt[2], 1); // コ=高
    assert_eq!(mt[3], 1); // ウ=高
}

#[test]
fn test_word_with_n_moraic_nasal() {
    // ン (撥音) のテスト
    let engine = Engine::default();
    let tokens = vec![tok("本", "名詞", "ホン")];

    let mut nodes = engine.analyze(&tokens);
    assert_eq!(nodes.len(), 1);
    // ホン: ホ・ン = 2モーラ
    assert_eq!(nodes[0].mora_count, 2);

    nodes[0].accent_type = 1; // 頭高型

    let phrases = vec![AccentPhrase {
        nodes: vec![0],
        accent_type: 1,
        mora_count: 2,
        is_interrogative: false,
    }];

    let phone_tones = engine.extract_phone_tones(&nodes, &phrases);
    let ph = phones(&phone_tones);

    // sil, h, o, N, sil
    assert!(
        ph.contains(&"N"),
        "Should contain 'N' for moraic nasal. Got: {:?}",
        ph
    );

    let mt = mora_tones(&phone_tones);
    assert_eq!(mt.len(), 2);
    assert_eq!(mt[0], 1); // ホ=高 (頭高)
    assert_eq!(mt[1], 0); // ン=低
}

#[test]
fn test_word_with_youon_kyou() {
    // 拗音のテスト: キョウ
    let engine = Engine::default();
    let tokens = vec![tok_pron("今日", "名詞", "キョウ", "キョー")];

    let nodes = engine.analyze(&tokens);
    assert_eq!(nodes.len(), 1);
    // キョウ: キョ・ウ = 2モーラ
    assert_eq!(nodes[0].mora_count, 2);

    let phone_tones = engine.tokens_to_phone_tones(&tokens);
    let ph = phones(&phone_tones);

    // Should contain "ky" for 拗音子音
    assert!(
        ph.contains(&"ky"),
        "Should contain 'ky' for youon consonant. Got: {:?}",
        ph
    );
}

#[test]
fn test_only_particles() {
    // 助詞のみ（特殊ケース）
    let engine = Engine::default();
    let tokens = vec![
        tok("は", "助詞", "ワ"),
        tok("が", "助詞", "ガ"),
    ];

    let mut nodes = engine.analyze(&tokens);
    let phrases = engine.estimate_accent(&mut nodes);

    // 最初の助詞は chain_flag=0 で新しいアクセント句
    // 2番目の助詞は機能語なので接続
    assert!(!phrases.is_empty());

    let labels = engine.make_label(&nodes, &phrases);
    assert!(labels.first().unwrap().contains("sil"));
    assert!(labels.last().unwrap().contains("sil"));
}

#[test]
fn test_kigou_symbol_not_chained() {
    // 記号は接続しない
    let engine = Engine::default();
    let tokens = vec![
        tok("猫", "名詞", "ネコ"),
        tok("、", "記号", "、"),
        tok("犬", "名詞", "イヌ"),
    ];

    let mut nodes = engine.analyze(&tokens);
    let phrases = engine.estimate_accent(&mut nodes);

    // 記号は新しいアクセント句を開始するため、3つのフレーズ
    assert!(
        phrases.len() >= 2,
        "Symbol should break accent phrase. Got {} phrases",
        phrases.len()
    );
}

#[test]
fn test_filler_is_flat() {
    // フィラーは平板型
    let engine = Engine::default();
    let tokens = vec![tok("えー", "フィラー", "エー")];

    let mut nodes = engine.analyze(&tokens);
    nodes[0].accent_type = 1; // 仮にtype 1を設定

    let phrases = engine.estimate_accent(&mut nodes);

    // フィラーは特殊規則で平板型に
    assert_eq!(
        phrases[0].accent_type, 0,
        "Filler should be forced to flat (type 0)"
    );
}

// ============================================================
// 7. スレッドセーフティテスト
// ============================================================

#[test]
fn test_thread_safety_concurrent_processing() {
    use std::sync::Arc;
    use std::thread;

    let engine = Arc::new(Engine::default());
    let mut handles = vec![];

    // 異なる入力を複数スレッドで同時処理
    let inputs: Vec<Vec<InputToken>> = vec![
        vec![tok("猫", "名詞", "ネコ"), tok("が", "助詞", "ガ")],
        vec![tok_pron("東京", "名詞", "トウキョウ", "トーキョー")],
        vec![tok("走る", "動詞", "ハシル")],
        vec![
            tok("良い", "形容詞", "ヨイ"),
            tok("天気", "名詞", "テンキ"),
            tok("です", "助動詞", "デス"),
        ],
        vec![tok("コーヒー", "名詞", "コーヒー")],
        vec![tok("学校", "名詞", "ガッコウ"), tok("に", "助詞", "ニ")],
        vec![tok("本", "名詞", "ホン"), tok("を", "助詞", "ヲ")],
        vec![tok("食べ", "動詞", "タベ"), tok("ます", "助動詞", "マス")],
    ];

    for input in inputs {
        let engine = Arc::clone(&engine);
        handles.push(thread::spawn(move || {
            let labels = engine.tokens_to_labels(&input);
            assert!(!labels.is_empty(), "Labels should not be empty");
            assert!(labels.first().unwrap().contains("sil"));
            assert!(labels.last().unwrap().contains("sil"));

            let phone_tones = engine.tokens_to_phone_tones(&input);
            assert!(!phone_tones.is_empty());
            assert_eq!(phone_tones.first().unwrap().phone, "sil");
            assert_eq!(phone_tones.last().unwrap().phone, "sil");

            let symbols = engine.tokens_to_prosody_symbols(&input);
            assert!(!symbols.is_empty());
            assert_eq!(symbols.first().unwrap(), "^");
        }));
    }

    for handle in handles {
        handle.join().expect("Thread should not panic");
    }
}

#[test]
fn test_thread_safety_same_input_deterministic() {
    use std::sync::Arc;
    use std::thread;

    let engine = Arc::new(Engine::default());
    let mut handles = vec![];

    // 同一入力を複数スレッドで処理し、結果が一致することを確認
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

    // 全スレッドの結果が同一であることを確認
    for i in 1..results.len() {
        assert_eq!(
            results[0], results[i],
            "Thread {} produced different result from thread 0",
            i
        );
    }
}

// ============================================================
// 追加: エンドツーエンドの統合テスト
// ============================================================

#[test]
fn test_end_to_end_pipeline_step_by_step() {
    let engine = Engine::default();
    let tokens = vec![
        tok("猫", "名詞", "ネコ"),
        tok("が", "助詞", "ガ"),
        tok("走る", "動詞", "ハシル"),
    ];

    // Step 1: analyze
    let mut nodes = engine.analyze(&tokens);
    assert_eq!(nodes.len(), 3);
    assert_eq!(nodes[0].surface, "猫");
    assert_eq!(nodes[0].pos, Pos::Meishi);
    assert_eq!(nodes[1].pos, Pos::Joshi);
    assert_eq!(nodes[2].pos, Pos::Doushi);

    // Step 2: estimate_accent
    let phrases = engine.estimate_accent(&mut nodes);
    assert_eq!(phrases.len(), 2); // 猫が | 走る

    // Step 3: make_label
    let labels = engine.make_label(&nodes, &phrases);
    assert!(labels.len() >= 3); // at least sil + something + sil
    assert!(labels.first().unwrap().contains("sil"));
    assert!(labels.last().unwrap().contains("sil"));

    // Step 4: extract_phone_tones
    let phone_tones = engine.extract_phone_tones(&nodes, &phrases);
    assert_eq!(phone_tones.first().unwrap().phone, "sil");
    assert_eq!(phone_tones.last().unwrap().phone, "sil");

    // Step 5: extract_prosody_symbols
    let symbols = engine.extract_prosody_symbols(&nodes, &phrases);
    assert_eq!(symbols.first().unwrap(), "^");
    // Should have _ between the two accent phrases
    assert!(symbols.contains(&"_".to_string()));
    // Should end with $
    assert_eq!(symbols.last().unwrap(), "$");
}

#[test]
fn test_end_to_end_convenience_methods_consistency() {
    // 便利メソッドとステップバイステップの結果が一致することを確認
    let engine = Engine::default();
    let tokens = vec![
        tok_pron("今日", "名詞", "キョウ", "キョー"),
        tok("は", "助詞", "ワ"),
        tok("天気", "名詞", "テンキ"),
        tok("です", "助動詞", "デス"),
    ];

    // 便利メソッド
    let labels_conv = engine.tokens_to_labels(&tokens);
    let pt_conv = engine.tokens_to_phone_tones(&tokens);
    let sym_conv = engine.tokens_to_prosody_symbols(&tokens);

    // ステップバイステップ
    let mut nodes = engine.analyze(&tokens);
    let phrases = engine.estimate_accent(&mut nodes);
    let labels_step = engine.make_label(&nodes, &phrases);
    let pt_step = engine.extract_phone_tones(&nodes, &phrases);
    let sym_step = engine.extract_prosody_symbols(&nodes, &phrases);

    assert_eq!(labels_conv, labels_step, "Labels should match");
    assert_eq!(pt_conv, pt_step, "PhoneTones should match");
    assert_eq!(sym_conv, sym_step, "Prosody symbols should match");
}

#[test]
fn test_multi_phrase_prosody_symbol_structure() {
    // 複数アクセント句の韻律記号の構造を検証
    let engine = Engine::default();
    let tokens = vec![
        tok("猫", "名詞", "ネコ"),
        tok("犬", "名詞", "イヌ"),
        tok("鳥", "名詞", "トリ"),
    ];

    let mut nodes = engine.analyze(&tokens);
    let phrases = engine.estimate_accent(&mut nodes);
    assert_eq!(phrases.len(), 3);

    let symbols = engine.extract_prosody_symbols(&nodes, &phrases);

    // 構造: ^ ... _ ... _ ... $
    assert_eq!(symbols.first().unwrap(), "^");
    assert_eq!(symbols.last().unwrap(), "$");

    // _ が2つ (3フレーズ間に2つの境界)
    let pause_count = symbols.iter().filter(|s| s.as_str() == "_").count();
    assert_eq!(
        pause_count, 2,
        "Should have 2 pauses between 3 phrases. Got: {:?}",
        symbols
    );
}

#[test]
fn test_node_pos_detail_preserved() {
    // InputTokenの詳細情報がNjdNodeに正しく伝搬されることを確認
    let mut token = InputToken::new("東京", "名詞", "トウキョウ", "トーキョー");
    token.pos_detail1 = "固有名詞".to_string();
    token.pos_detail2 = "地域".to_string();
    token.ctype = "特殊".to_string();

    let engine = Engine::default();
    let nodes = engine.analyze(&[token]);

    assert_eq!(nodes[0].pos_detail1, "固有名詞");
    assert_eq!(nodes[0].pos_detail2, "地域");
    assert_eq!(nodes[0].ctype, "特殊");
}

#[test]
fn test_long_sentence() {
    // 長い文のテスト（パフォーマンスと安定性）
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
        tok("公園", "名詞", "コウエン"),
        tok("に", "助詞", "ニ"),
        tok("行き", "動詞", "イキ"),
        tok("まし", "助動詞", "マシ"),
        tok("た", "助動詞", "タ"),
    ];

    let labels = engine.tokens_to_labels(&tokens);
    let phone_tones = engine.tokens_to_phone_tones(&tokens);
    let symbols = engine.tokens_to_prosody_symbols(&tokens);

    assert!(!labels.is_empty());
    assert!(labels.first().unwrap().contains("sil"));
    assert!(labels.last().unwrap().contains("sil"));

    assert_eq!(phone_tones.first().unwrap().phone, "sil");
    assert_eq!(phone_tones.last().unwrap().phone, "sil");

    assert_eq!(symbols.first().unwrap(), "^");
    assert_eq!(symbols.last().unwrap(), "$");

    // ラベル数 == PhoneTone数
    assert_eq!(labels.len(), phone_tones.len());
}
