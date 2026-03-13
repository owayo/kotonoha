//! HTS Label生成テスト
//! ラベルフォーマット、フィールド値、無声母音の網羅的テスト

use kotonoha::accent::AccentPhrase;
use kotonoha::njd::{InputToken, NjdNode};
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

fn make_phrases(nodes: &[NjdNode], accent_type: u8) -> Vec<AccentPhrase> {
    let mora_count: u8 = nodes.iter().map(|n| n.mora_count).sum();
    vec![AccentPhrase {
        nodes: (0..nodes.len()).collect(),
        accent_type,
        mora_count,
        is_interrogative: false,
    }]
}

fn non_sil_labels(labels: &[String]) -> Vec<&String> {
    labels.iter().filter(|l| !l.starts_with("xx^xx-sil")).collect()
}

// ============================================================
// Basic sil label tests
// ============================================================

#[test]
fn test_label_first_is_sil() {
    let engine = Engine::default();
    let labels = engine.tokens_to_labels(&[tok("猫", "名詞", "ネコ")]);
    assert!(labels.first().unwrap().contains("sil"));
}

#[test]
fn test_label_last_is_sil() {
    let engine = Engine::default();
    let labels = engine.tokens_to_labels(&[tok("猫", "名詞", "ネコ")]);
    assert!(labels.last().unwrap().contains("sil"));
}

#[test]
fn test_sil_label_full_format() {
    let engine = Engine::default();
    let labels = engine.tokens_to_labels(&[tok("猫", "名詞", "ネコ")]);
    let sil = &labels[0];
    assert!(sil.starts_with("xx^xx-sil+xx=xx"));
}

#[test]
fn test_sil_label_has_all_fields() {
    let engine = Engine::default();
    let labels = engine.tokens_to_labels(&[tok("猫", "名詞", "ネコ")]);
    let sil = &labels[0];
    for field in ["/A:", "/B:", "/C:", "/D:", "/E:", "/K:"] {
        assert!(sil.contains(field), "Sil label missing {}: {}", field, sil);
    }
}

// ============================================================
// Label contains required fields
// ============================================================

#[test]
fn test_label_fields_a_through_k() {
    let engine = Engine::default();
    let labels = engine.tokens_to_labels(&[tok("猫", "名詞", "ネコ")]);
    let mid = non_sil_labels(&labels);
    assert!(!mid.is_empty());
    for label in &mid {
        for field in ["/A:", "/B:", "/C:", "/D:", "/E:", "/K:"] {
            assert!(label.contains(field), "Label missing {}: {}", field, label);
        }
    }
}

#[test]
fn test_label_fields_for_multi_word() {
    let engine = Engine::default();
    let labels = engine.tokens_to_labels(&[
        tok("猫", "名詞", "ネコ"),
        tok("が", "助詞", "ガ"),
        tok("走る", "動詞", "ハシル"),
    ]);
    let mid = non_sil_labels(&labels);
    for label in &mid {
        assert!(label.contains("/A:"), "Missing /A: in {}", label);
        assert!(label.contains("/K:"), "Missing /K: in {}", label);
    }
}

// ============================================================
// Phoneme context window (p1-p5 = p_prev2^p_prev-p_curr+p_next=p_next2)
// ============================================================

#[test]
fn test_label_context_separators() {
    let engine = Engine::default();
    let labels = engine.tokens_to_labels(&[tok("猫", "名詞", "ネコ")]);
    for label in &labels {
        let ctx = label.split("/A:").next().unwrap();
        assert!(ctx.contains('^'), "Missing ^ in context: {}", ctx);
        assert!(ctx.contains('-'), "Missing - in context: {}", ctx);
        assert!(ctx.contains('+'), "Missing + in context: {}", ctx);
        assert!(ctx.contains('='), "Missing = in context: {}", ctx);
    }
}

#[test]
fn test_label_context_window_for_neko() {
    let engine = Engine::default();
    let tokens = vec![tok("猫", "名詞", "ネコ")];
    let mut nodes = engine.analyze(&tokens);
    nodes[0].accent_type = 1;
    let phrases = make_phrases(&nodes, 1);
    let labels = engine.make_label(&nodes, &phrases);

    // Labels: sil, n, e, k, o, sil
    // For 'n' (idx 0 in phonemes): p1=xx, p2=xx (from sil), p3=n, p4=e, p5=k
    // The second label (index 1) corresponds to 'n'
    let n_label = &labels[1];
    let ctx = n_label.split("/A:").next().unwrap();
    assert!(ctx.contains("-n+"), "n label should have -n+: {}", ctx);
}

#[test]
fn test_label_context_for_sakura() {
    let engine = Engine::default();
    let tokens = vec![tok("桜", "名詞", "サクラ")];
    let mut nodes = engine.analyze(&tokens);
    nodes[0].accent_type = 0;
    let phrases = make_phrases(&nodes, 0);
    let labels = engine.make_label(&nodes, &phrases);

    // Should have: sil, s, a, k, u, r, a, sil = 8 labels
    assert_eq!(labels.len(), 8);
}

// ============================================================
// A field values (accent)
// ============================================================

#[test]
fn test_label_a_field_present() {
    let engine = Engine::default();
    let tokens = vec![tok("猫", "名詞", "ネコ")];
    let mut nodes = engine.analyze(&tokens);
    nodes[0].accent_type = 1;
    let phrases = make_phrases(&nodes, 1);
    let labels = engine.make_label(&nodes, &phrases);
    let mid = non_sil_labels(&labels);
    for label in &mid {
        let a_part = label.split("/A:").nth(1).unwrap().split("/B:").next().unwrap();
        // A field has format: a1+a2+a3
        let parts: Vec<&str> = a_part.split('+').collect();
        assert_eq!(parts.len(), 3, "A field should have 3 parts: {}", a_part);
    }
}

// ============================================================
// B field (prev phrase info)
// ============================================================

#[test]
fn test_label_b_field_xx_for_first_phrase() {
    let engine = Engine::default();
    let tokens = vec![tok("猫", "名詞", "ネコ")];
    let mut nodes = engine.analyze(&tokens);
    nodes[0].accent_type = 1;
    let phrases = make_phrases(&nodes, 1);
    let labels = engine.make_label(&nodes, &phrases);
    let mid = non_sil_labels(&labels);
    // First phrase has no prev phrase -> B field should have xx
    for label in &mid {
        let b_part = label.split("/B:").nth(1).unwrap().split("/C:").next().unwrap();
        assert!(b_part.starts_with("xx"), "B field should start with xx for first phrase: {}", b_part);
    }
}

#[test]
fn test_label_b_field_has_values_for_second_phrase() {
    let engine = Engine::default();
    let tokens = vec![
        tok("猫", "名詞", "ネコ"),
        tok("犬", "名詞", "イヌ"),
    ];
    let mut nodes = engine.analyze(&tokens);
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
    let labels = engine.make_label(&nodes, &phrases);
    // Find a label in the second phrase (イヌ)
    // It should have B field with actual values (from first phrase)
    let mid = non_sil_labels(&labels);
    let second_phrase_label = mid.last().unwrap();
    let b_part = second_phrase_label.split("/B:").nth(1).unwrap().split("/C:").next().unwrap();
    assert!(!b_part.starts_with("xx"), "B field of second phrase should have values: {}", b_part);
}

// ============================================================
// C field (current phrase info)
// ============================================================

#[test]
fn test_label_c_field_mora_count() {
    let engine = Engine::default();
    let tokens = vec![tok("猫", "名詞", "ネコ")];
    let mut nodes = engine.analyze(&tokens);
    nodes[0].accent_type = 1;
    let phrases = make_phrases(&nodes, 1);
    let labels = engine.make_label(&nodes, &phrases);
    let mid = non_sil_labels(&labels);
    for label in &mid {
        let c_part = label.split("/C:").nth(1).unwrap().split("/D:").next().unwrap();
        // C: mora_count_accent_type+interrogative
        assert!(c_part.starts_with("2_"), "C field should start with mora_count 2: {}", c_part);
    }
}

#[test]
fn test_label_c_field_accent_type() {
    let engine = Engine::default();
    let tokens = vec![tok("猫", "名詞", "ネコ")];
    let mut nodes = engine.analyze(&tokens);
    nodes[0].accent_type = 1;
    let phrases = make_phrases(&nodes, 1);
    let labels = engine.make_label(&nodes, &phrases);
    let mid = non_sil_labels(&labels);
    let label = mid[0];
    let c_part = label.split("/C:").nth(1).unwrap().split("/D:").next().unwrap();
    // Should contain accent_type=1
    assert!(c_part.contains("1+"), "C field should contain accent_type 1: {}", c_part);
}

// ============================================================
// D field (next phrase info)
// ============================================================

#[test]
fn test_label_d_field_xx_for_single_phrase() {
    let engine = Engine::default();
    let tokens = vec![tok("猫", "名詞", "ネコ")];
    let mut nodes = engine.analyze(&tokens);
    nodes[0].accent_type = 1;
    let phrases = make_phrases(&nodes, 1);
    let labels = engine.make_label(&nodes, &phrases);
    let mid = non_sil_labels(&labels);
    for label in &mid {
        let d_part = label.split("/D:").nth(1).unwrap().split("/E:").next().unwrap();
        assert!(d_part.starts_with("xx"), "D field should be xx for single phrase: {}", d_part);
    }
}

#[test]
fn test_label_d_field_has_values_for_first_of_two_phrases() {
    let engine = Engine::default();
    let tokens = vec![
        tok("猫", "名詞", "ネコ"),
        tok("犬", "名詞", "イヌ"),
    ];
    let mut nodes = engine.analyze(&tokens);
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
    let labels = engine.make_label(&nodes, &phrases);
    // First phrase labels should have D field with second phrase info
    let mid = non_sil_labels(&labels);
    let first_phrase_label = mid[0];
    let d_part = first_phrase_label.split("/D:").nth(1).unwrap().split("/E:").next().unwrap();
    assert!(!d_part.starts_with("xx"), "D field of first phrase should have values: {}", d_part);
}

// ============================================================
// K field (total phrases)
// ============================================================

#[test]
fn test_label_k_field_total_phrases() {
    let engine = Engine::default();
    let tokens = vec![tok("猫", "名詞", "ネコ")];
    let mut nodes = engine.analyze(&tokens);
    nodes[0].accent_type = 1;
    let phrases = make_phrases(&nodes, 1);
    let labels = engine.make_label(&nodes, &phrases);
    let mid = non_sil_labels(&labels);
    for label in &mid {
        let k_part = label.split("/K:").nth(1).unwrap();
        assert!(k_part.starts_with("1+"), "K field should start with 1 for single phrase: {}", k_part);
    }
}

#[test]
fn test_label_k_field_two_phrases() {
    let engine = Engine::default();
    let tokens = vec![
        tok("猫", "名詞", "ネコ"),
        tok("犬", "名詞", "イヌ"),
    ];
    let nodes = engine.analyze(&tokens);
    let phrases = vec![
        AccentPhrase { nodes: vec![0], accent_type: 1, mora_count: 2, is_interrogative: false },
        AccentPhrase { nodes: vec![1], accent_type: 0, mora_count: 2, is_interrogative: false },
    ];
    let labels = engine.make_label(&nodes, &phrases);
    let mid = non_sil_labels(&labels);
    for label in &mid {
        let k_part = label.split("/K:").nth(1).unwrap();
        // K: k1+k2-k3 = breath_group_count+accent_phrase_count-mora_count
        assert_eq!(k_part, "1+2-4", "K field should be 1+2-4 (1 breath group, 2 phrases, 4 moras): {}", k_part);
    }
}

// ============================================================
// Voiceless vowel detection
// ============================================================

#[test]
fn test_voiceless_kitsune() {
    // きつね: ki-tsu-ne -> ki should be voiceless (k[I]tsune)
    let engine = Engine::default();
    let tokens = vec![tok("狐", "名詞", "キツネ")];
    let mut nodes = engine.analyze(&tokens);
    nodes[0].accent_type = 0;
    let phrases = make_phrases(&nodes, 0);
    let labels = engine.make_label(&nodes, &phrases);
    // Check if there's an uppercase I in the labels
    let all_labels = labels.join(" ");
    // ki -> k, i/I; tsu -> ts, u; ne -> n, e
    // With voiceless: k, I, ts, u, n, e (ki before ts = voiceless)
    assert!(
        all_labels.contains("-I+") || all_labels.contains("-i+"),
        "kitsune should have i or I phoneme: {}",
        all_labels
    );
}

#[test]
fn test_voiceless_sushi() {
    // すし: su-shi -> su should be voiceless (s[U]shi)
    let engine = Engine::default();
    let tokens = vec![tok("寿司", "名詞", "スシ")];
    let mut nodes = engine.analyze(&tokens);
    nodes[0].accent_type = 2;
    let phrases = make_phrases(&nodes, 2);
    let labels = engine.make_label(&nodes, &phrases);
    let mid = non_sil_labels(&labels);
    // Look for U or u in labels
    let all = mid.iter().map(|l| l.as_str()).collect::<Vec<_>>().join(" ");
    assert!(
        all.contains("-U+") || all.contains("-u+"),
        "sushi should process su vowel: {}",
        all
    );
}

#[test]
fn test_voiceless_hitsuji() {
    // ひつじ: hi-tsu-ji
    let engine = Engine::default();
    let tokens = vec![tok("羊", "名詞", "ヒツジ")];
    let mut nodes = engine.analyze(&tokens);
    nodes[0].accent_type = 0;
    let phrases = make_phrases(&nodes, 0);
    let labels = engine.make_label(&nodes, &phrases);
    assert!(labels.len() > 2, "Should have multiple labels");
}

#[test]
fn test_voiceless_chikuwa() {
    // ちくわ: chi-ku-wa
    let engine = Engine::default();
    let tokens = vec![tok("竹輪", "名詞", "チクワ")];
    let mut nodes = engine.analyze(&tokens);
    nodes[0].accent_type = 0;
    let phrases = make_phrases(&nodes, 0);
    let labels = engine.make_label(&nodes, &phrases);
    assert!(labels.len() > 2, "Should have multiple labels");
}

#[test]
fn test_voiceless_kusuri() {
    // くすり: ku-su-ri -> ku before su (voiceless candidates)
    let engine = Engine::default();
    let tokens = vec![tok("薬", "名詞", "クスリ")];
    let mut nodes = engine.analyze(&tokens);
    nodes[0].accent_type = 0;
    let phrases = make_phrases(&nodes, 0);
    let labels = engine.make_label(&nodes, &phrases);
    let mid = non_sil_labels(&labels);
    assert!(!mid.is_empty());
}

// ============================================================
// Label count consistency with phone_tone count
// ============================================================

#[test]
fn test_label_count_matches_phone_tones_single_word() {
    let engine = Engine::default();
    let tokens = vec![tok("猫", "名詞", "ネコ")];
    let mut nodes = engine.analyze(&tokens);
    nodes[0].accent_type = 1;
    let phrases = make_phrases(&nodes, 1);
    let labels = engine.make_label(&nodes, &phrases);
    let pts = engine.extract_phone_tones(&nodes, &phrases);
    assert_eq!(labels.len(), pts.len(), "Label count != phone_tone count");
}

#[test]
fn test_label_count_matches_phone_tones_multi_word() {
    let engine = Engine::default();
    let tokens = vec![
        tok("猫", "名詞", "ネコ"),
        tok("が", "助詞", "ガ"),
        tok("走る", "動詞", "ハシル"),
    ];
    let mut nodes = engine.analyze(&tokens);
    let phrases = engine.estimate_accent(&mut nodes);
    let labels = engine.make_label(&nodes, &phrases);
    let pts = engine.extract_phone_tones(&nodes, &phrases);
    assert_eq!(labels.len(), pts.len());
}

#[test]
fn test_label_count_matches_phone_tones_long_vowel() {
    let engine = Engine::default();
    let tokens = vec![tok_pron("コーヒー", "名詞", "コーヒー", "コーヒー")];
    let mut nodes = engine.analyze(&tokens);
    nodes[0].accent_type = 3;
    let phrases = make_phrases(&nodes, 3);
    let labels = engine.make_label(&nodes, &phrases);
    let pts = engine.extract_phone_tones(&nodes, &phrases);
    assert_eq!(labels.len(), pts.len());
}

#[test]
fn test_label_count_matches_phone_tones_geminate() {
    let engine = Engine::default();
    let tokens = vec![tok_pron("学校", "名詞", "ガッコウ", "ガッコー")];
    let mut nodes = engine.analyze(&tokens);
    nodes[0].accent_type = 0;
    let phrases = make_phrases(&nodes, 0);
    let labels = engine.make_label(&nodes, &phrases);
    let pts = engine.extract_phone_tones(&nodes, &phrases);
    assert_eq!(labels.len(), pts.len());
}

#[test]
fn test_label_count_matches_phone_tones_n() {
    let engine = Engine::default();
    let tokens = vec![tok("本", "名詞", "ホン")];
    let mut nodes = engine.analyze(&tokens);
    nodes[0].accent_type = 1;
    let phrases = make_phrases(&nodes, 1);
    let labels = engine.make_label(&nodes, &phrases);
    let pts = engine.extract_phone_tones(&nodes, &phrases);
    assert_eq!(labels.len(), pts.len());
}

// ============================================================
// Empty input
// ============================================================

#[test]
fn test_label_empty_input() {
    let engine = Engine::default();
    let labels = engine.tokens_to_labels(&[]);
    assert!(!labels.is_empty());
    assert!(labels[0].contains("sil"));
}

// ============================================================
// Various input lengths
// ============================================================

#[test]
fn test_label_single_vowel_word() {
    let engine = Engine::default();
    let labels = engine.tokens_to_labels(&[tok("愛", "名詞", "アイ")]);
    // sil + a + i + sil = 4
    assert_eq!(labels.len(), 4);
}

#[test]
fn test_label_single_mora_word() {
    let engine = Engine::default();
    let labels = engine.tokens_to_labels(&[tok("木", "名詞", "キ")]);
    // sil + k + i + sil = 4
    assert_eq!(labels.len(), 4);
}

#[test]
fn test_label_three_mora_word() {
    let engine = Engine::default();
    let labels = engine.tokens_to_labels(&[tok("桜", "名詞", "サクラ")]);
    // sil + s,a + k,u + r,a + sil = 8
    assert_eq!(labels.len(), 8);
}

// ============================================================
// Multi-phrase labels
// ============================================================

#[test]
fn test_label_two_phrases() {
    let engine = Engine::default();
    let tokens = vec![
        tok("猫", "名詞", "ネコ"),
        tok("犬", "名詞", "イヌ"),
    ];
    let labels = engine.tokens_to_labels(&tokens);
    assert!(labels.len() >= 6); // sil + ne + ko + i + nu + sil
}

#[test]
fn test_label_three_phrases() {
    let engine = Engine::default();
    let tokens = vec![
        tok("猫", "名詞", "ネコ"),
        tok("犬", "名詞", "イヌ"),
        tok("鳥", "名詞", "トリ"),
    ];
    let labels = engine.tokens_to_labels(&tokens);
    assert!(labels.len() >= 8); // sil + multiple phonemes + sil
}

// ============================================================
// Label format integrity: no double slashes or empty fields
// ============================================================

#[test]
fn test_label_no_double_slashes() {
    let engine = Engine::default();
    let labels = engine.tokens_to_labels(&[tok("猫", "名詞", "ネコ"), tok("が", "助詞", "ガ")]);
    for label in &labels {
        assert!(!label.contains("//"), "Label should not have //: {}", label);
    }
}

#[test]
fn test_label_no_empty_segments() {
    let engine = Engine::default();
    let labels = engine.tokens_to_labels(&[tok("猫", "名詞", "ネコ")]);
    for label in &labels {
        let ctx = label.split("/A:").next().unwrap();
        // Check that separators are not adjacent: ^- or -+ or += etc.
        assert!(!ctx.contains("^-"), "Empty p1: {}", ctx);
        assert!(!ctx.contains("-+"), "Empty p3: {}", ctx);
        assert!(!ctx.contains("+="), "Empty p4: {}", ctx);
    }
}

// ============================================================
// Interrogative in label
// ============================================================

#[test]
fn test_label_interrogative_c_field() {
    let engine = Engine::default();
    let tokens = vec![tok("猫", "名詞", "ネコ")];
    let mut nodes = engine.analyze(&tokens);
    nodes[0].accent_type = 1;
    let phrases = vec![AccentPhrase {
        nodes: vec![0],
        accent_type: 1,
        mora_count: 2,
        is_interrogative: true,
    }];
    let labels = engine.make_label(&nodes, &phrases);
    let mid = non_sil_labels(&labels);
    for label in &mid {
        let c_part = label.split("/C:").nth(1).unwrap().split("/D:").next().unwrap();
        // C field last element should be 1 for interrogative
        assert!(c_part.ends_with("+1"), "C field should end with +1 for interrogative: {}", c_part);
    }
}

#[test]
fn test_label_non_interrogative_c_field() {
    let engine = Engine::default();
    let tokens = vec![tok("猫", "名詞", "ネコ")];
    let mut nodes = engine.analyze(&tokens);
    nodes[0].accent_type = 1;
    let phrases = make_phrases(&nodes, 1);
    let labels = engine.make_label(&nodes, &phrases);
    let mid = non_sil_labels(&labels);
    for label in &mid {
        let c_part = label.split("/C:").nth(1).unwrap().split("/D:").next().unwrap();
        assert!(c_part.ends_with("+0"), "C field should end with +0 for declarative: {}", c_part);
    }
}

// ============================================================
// Voiceless vowel: devoicing should occur
// ============================================================

#[test]
fn test_voiceless_kitsune_ki_devoices() {
    // きつね: ki-tsu-ne → ki should devoice (k before ts, both voiceless)
    let engine = Engine::default();
    let tokens = vec![tok("狐", "名詞", "キツネ")];
    let mut nodes = engine.analyze(&tokens);
    nodes[0].accent_type = 0; // 平板型
    let phrases = make_phrases(&nodes, 0);
    let labels = engine.make_label(&nodes, &phrases);
    let all = labels.join(" ");
    assert!(
        all.contains("-I+"),
        "kitsune: 'ki' should devoice to I: {}",
        all
    );
}

#[test]
fn test_voiceless_sushi_su_devoices() {
    // すし: su-shi → su should devoice (s before sh, both voiceless)
    // accent_type=2 (odaka: low-high) so su is NOT at accent nucleus
    let engine = Engine::default();
    let tokens = vec![tok("寿司", "名詞", "スシ")];
    let mut nodes = engine.analyze(&tokens);
    nodes[0].accent_type = 2;
    let phrases = make_phrases(&nodes, 2);
    let labels = engine.make_label(&nodes, &phrases);
    let all = labels.join(" ");
    assert!(
        all.contains("-U+"),
        "sushi: 'su' should devoice to U: {}",
        all
    );
}

#[test]
fn test_voiceless_hitsuji_hi_devoices() {
    // ひつじ: hi-tsu-ji → hi should devoice (h before ts, both voiceless)
    let engine = Engine::default();
    let tokens = vec![tok("羊", "名詞", "ヒツジ")];
    let mut nodes = engine.analyze(&tokens);
    nodes[0].accent_type = 0;
    let phrases = make_phrases(&nodes, 0);
    let labels = engine.make_label(&nodes, &phrases);
    let all = labels.join(" ");
    assert!(
        all.contains("-I+"),
        "hitsuji: 'hi' should devoice to I: {}",
        all
    );
}

#[test]
fn test_voiceless_chikuwa_chi_devoices() {
    // ちくわ: chi-ku-wa → chi should devoice (ch before k, both voiceless)
    // ku should NOT devoice (alternation rule: chi already devoiced)
    let engine = Engine::default();
    let tokens = vec![tok("竹輪", "名詞", "チクワ")];
    let mut nodes = engine.analyze(&tokens);
    nodes[0].accent_type = 0;
    let phrases = make_phrases(&nodes, 0);
    let labels = engine.make_label(&nodes, &phrases);
    let all = labels.join(" ");
    // chi's vowel should be I (devoiced)
    assert!(
        all.contains("-I+"),
        "chikuwa: 'chi' should devoice to I: {}",
        all
    );
}

#[test]
fn test_voiceless_chikuwa_ku_no_devoice_alternation() {
    // ちくわ: chi-ku-wa → ku should NOT devoice due to alternation rule
    // (chi already devoiced, consecutive devoicing avoided)
    let engine = Engine::default();
    let tokens = vec![tok("竹輪", "名詞", "チクワ")];
    let mut nodes = engine.analyze(&tokens);
    nodes[0].accent_type = 0;
    let phrases = make_phrases(&nodes, 0);
    let labels = engine.make_label(&nodes, &phrases);
    let mid = non_sil_labels(&labels);
    let all = mid.iter().map(|l| l.as_str()).collect::<Vec<_>>().join(" ");
    // ku's vowel should remain lowercase u (not devoiced)
    // Find the 'u' that comes after 'k' — should be lowercase
    assert!(
        all.contains("-u+"),
        "chikuwa: 'ku' should NOT devoice (alternation): {}",
        all
    );
}

#[test]
fn test_voiceless_kusuri_ku_devoices() {
    // くすり: ku-su-ri → ku should devoice (k before s, both voiceless)
    // su should NOT devoice (alternation rule)
    let engine = Engine::default();
    let tokens = vec![tok("薬", "名詞", "クスリ")];
    let mut nodes = engine.analyze(&tokens);
    nodes[0].accent_type = 0;
    let phrases = make_phrases(&nodes, 0);
    let labels = engine.make_label(&nodes, &phrases);
    let all = labels.join(" ");
    assert!(
        all.contains("-U+"),
        "kusuri: 'ku' should devoice to U: {}",
        all
    );
}

#[test]
fn test_voiceless_desu_su_devoices() {
    // です: de-su → su should devoice at utterance end
    let engine = Engine::default();
    let tokens = vec![
        tok("猫", "名詞", "ネコ"),
        tok("です", "助動詞", "デス"),
    ];
    let mut nodes = engine.analyze(&tokens);
    nodes[0].accent_type = 1;
    nodes[1].accent_type = 1;
    let phrases = vec![AccentPhrase {
        nodes: vec![0, 1],
        accent_type: 1,
        mora_count: 4, // ne-ko-de-su
        is_interrogative: false,
    }];
    let labels = engine.make_label(&nodes, &phrases);
    let all = labels.join(" ");
    // 'su' at utterance end after voiceless consonant 's' should devoice
    assert!(
        all.contains("-U+") || all.contains("-U="),
        "desu: 'su' should devoice at utterance end: {}",
        all
    );
}

#[test]
fn test_voiceless_masu_su_devoices() {
    // ます: ma-su → su should devoice at utterance end
    let engine = Engine::default();
    let tokens = vec![
        tok("走り", "動詞", "ハシリ"),
        tok("ます", "助動詞", "マス"),
    ];
    let mut nodes = engine.analyze(&tokens);
    nodes[0].accent_type = 0;
    nodes[1].accent_type = 1;
    let phrases = vec![AccentPhrase {
        nodes: vec![0, 1],
        accent_type: 0,
        mora_count: 5, // ha-shi-ri-ma-su
        is_interrogative: false,
    }];
    let labels = engine.make_label(&nodes, &phrases);
    let all = labels.join(" ");
    assert!(
        all.contains("-U+") || all.contains("-U="),
        "masu: 'su' should devoice at utterance end: {}",
        all
    );
}

// ============================================================
// Voiceless vowel: devoicing should NOT occur
// ============================================================

#[test]
fn test_voiceless_hashi_no_devoice() {
    // はし: ha-shi → ha has 'h' consonant which is voiceless, but vowel is 'a' not i/u
    // shi has voiceless 'sh' but no following voiceless consonant → should not devoice
    let engine = Engine::default();
    let tokens = vec![tok("橋", "名詞", "ハシ")];
    let mut nodes = engine.analyze(&tokens);
    nodes[0].accent_type = 2;
    let phrases = make_phrases(&nodes, 2);
    let labels = engine.make_label(&nodes, &phrases);
    let all = labels.join(" ");
    // No uppercase vowels expected
    assert!(
        !all.contains("-I+") && !all.contains("-U+") && !all.contains("-I=") && !all.contains("-U="),
        "hashi: no vowel should devoice: {}",
        all
    );
}

#[test]
fn test_voiceless_ki_single_mora_no_devoice() {
    // 木 (き): single-mora word, should NOT devoice
    let engine = Engine::default();
    let tokens = vec![tok("木", "名詞", "キ")];
    let mut nodes = engine.analyze(&tokens);
    nodes[0].accent_type = 0;
    let phrases = make_phrases(&nodes, 0);
    let labels = engine.make_label(&nodes, &phrases);
    let all = labels.join(" ");
    assert!(
        !all.contains("-I+") && !all.contains("-I="),
        "ki (single mora): should NOT devoice: {}",
        all
    );
}

#[test]
fn test_voiceless_shi_single_mora_no_devoice() {
    // 詩 (し): single-mora word, should NOT devoice
    let engine = Engine::default();
    let tokens = vec![tok("詩", "名詞", "シ")];
    let mut nodes = engine.analyze(&tokens);
    nodes[0].accent_type = 1;
    let phrases = make_phrases(&nodes, 1);
    let labels = engine.make_label(&nodes, &phrases);
    let all = labels.join(" ");
    assert!(
        !all.contains("-I+") && !all.contains("-I="),
        "shi (single mora): should NOT devoice: {}",
        all
    );
}

#[test]
fn test_voiceless_su_single_mora_no_devoice() {
    // 酢 (す): single-mora word, should NOT devoice
    let engine = Engine::default();
    let tokens = vec![tok("酢", "名詞", "ス")];
    let mut nodes = engine.analyze(&tokens);
    nodes[0].accent_type = 1;
    let phrases = make_phrases(&nodes, 1);
    let labels = engine.make_label(&nodes, &phrases);
    let all = labels.join(" ");
    assert!(
        !all.contains("-U+") && !all.contains("-U="),
        "su (single mora): should NOT devoice: {}",
        all
    );
}

// ============================================================
// Voiceless vowel: accent nucleus rule
// ============================================================

#[test]
fn test_voiceless_accent_nucleus_no_devoice() {
    // accent_type=1 → mora 0 (1st mora) is the accent nucleus
    // き (ki) at position 0 in accent_type=1 should NOT devoice
    let engine = Engine::default();
    let tokens = vec![tok("北", "名詞", "キタ")];
    let mut nodes = engine.analyze(&tokens);
    nodes[0].accent_type = 1; // accent nucleus on first mora (ki)
    let phrases = make_phrases(&nodes, 1);
    let labels = engine.make_label(&nodes, &phrases);
    let all = labels.join(" ");
    // ki at nucleus should not devoice even if followed by voiceless ta
    assert!(
        !all.contains("-I+"),
        "accent nucleus ki should NOT devoice: {}",
        all
    );
}

// ============================================================
// Voiceless vowel: long vowel rule
// ============================================================

#[test]
fn test_voiceless_long_vowel_no_devoice() {
    // すう (su-u): long vowel → su should NOT devoice
    // Build a reading that produces s, u, u
    let engine = Engine::default();
    let tokens = vec![tok_pron("数", "名詞", "スウ", "スウ")];
    let mut nodes = engine.analyze(&tokens);
    nodes[0].accent_type = 1;
    let phrases = make_phrases(&nodes, 1);
    let labels = engine.make_label(&nodes, &phrases);
    let all = labels.join(" ");
    assert!(
        !all.contains("-U+"),
        "su before same vowel (long vowel) should NOT devoice: {}",
        all
    );
}

// ============================================================
// Various word types
// ============================================================

#[test]
fn test_label_with_youon() {
    let engine = Engine::default();
    let labels = engine.tokens_to_labels(&[tok_pron("今日", "名詞", "キョウ", "キョー")]);
    let mid = non_sil_labels(&labels);
    let all = mid.iter().map(|l| l.as_str()).collect::<Vec<_>>().join(" ");
    assert!(all.contains("ky"), "Labels should contain ky: {}", all);
}

#[test]
fn test_label_with_geminate() {
    let engine = Engine::default();
    let labels = engine.tokens_to_labels(&[tok_pron("学校", "名詞", "ガッコウ", "ガッコー")]);
    let mid = non_sil_labels(&labels);
    let all = mid.iter().map(|l| l.as_str()).collect::<Vec<_>>().join(" ");
    assert!(all.contains("cl"), "Labels should contain cl: {}", all);
}

#[test]
fn test_label_with_moraic_nasal() {
    let engine = Engine::default();
    let labels = engine.tokens_to_labels(&[tok("本", "名詞", "ホン")]);
    let mid = non_sil_labels(&labels);
    let all = mid.iter().map(|l| l.as_str()).collect::<Vec<_>>().join(" ");
    assert!(all.contains("-N+"), "Labels should contain N: {}", all);
}
