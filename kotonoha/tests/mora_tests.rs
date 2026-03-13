//! モーラ解析テスト
//! count_mora, parse_mora の網羅的テスト

use kotonoha::mora::{count_mora, parse_mora};

// ============================================================
// count_mora: 1モーラ語
// ============================================================

#[test]
fn test_mora_count_single_vowel_a() {
    assert_eq!(count_mora("ア"), 1);
}

#[test]
fn test_mora_count_single_vowel_i() {
    assert_eq!(count_mora("イ"), 1);
}

#[test]
fn test_mora_count_single_vowel_u() {
    assert_eq!(count_mora("ウ"), 1);
}

#[test]
fn test_mora_count_single_vowel_e() {
    assert_eq!(count_mora("エ"), 1);
}

#[test]
fn test_mora_count_single_vowel_o() {
    assert_eq!(count_mora("オ"), 1);
}

#[test]
fn test_mora_count_single_n() {
    assert_eq!(count_mora("ン"), 1);
}

#[test]
fn test_mora_count_single_sokuon() {
    assert_eq!(count_mora("ッ"), 1);
}

#[test]
fn test_mora_count_single_consonant_ka() {
    assert_eq!(count_mora("カ"), 1);
}

#[test]
fn test_mora_count_single_youon_kya() {
    assert_eq!(count_mora("キャ"), 1);
}

#[test]
fn test_mora_count_single_long_vowel() {
    assert_eq!(count_mora("ー"), 1);
}

// ============================================================
// count_mora: 2モーラ語
// ============================================================

#[test]
fn test_mora_count_2_neko() {
    assert_eq!(count_mora("ネコ"), 2);
}

#[test]
fn test_mora_count_2_hon() {
    assert_eq!(count_mora("ホン"), 2);
}

#[test]
fn test_mora_count_2_ai() {
    assert_eq!(count_mora("アイ"), 2);
}

// ============================================================
// count_mora: 3モーラ語
// ============================================================

#[test]
fn test_mora_count_3_sakura() {
    assert_eq!(count_mora("サクラ"), 3);
}

#[test]
fn test_mora_count_3_hashiru() {
    assert_eq!(count_mora("ハシル"), 3);
}

#[test]
fn test_mora_count_3_tenki() {
    // テ・ン・キ = 3
    assert_eq!(count_mora("テンキ"), 3);
}

// ============================================================
// count_mora: 4モーラ語
// ============================================================

#[test]
fn test_mora_count_4_gakkou() {
    // ガ・ッ・コ・ウ = 4
    assert_eq!(count_mora("ガッコウ"), 4);
}

#[test]
fn test_mora_count_4_tokyo() {
    // ト・ー・キョ・ー = 4
    assert_eq!(count_mora("トーキョー"), 4);
}

#[test]
fn test_mora_count_4_koohii() {
    // コ・ー・ヒ・ー = 4
    assert_eq!(count_mora("コーヒー"), 4);
}

#[test]
fn test_mora_count_4_toukyou() {
    // ト・ウ・キョ・ウ = 4
    assert_eq!(count_mora("トウキョウ"), 4);
}

// ============================================================
// count_mora: 5モーラ語
// ============================================================

#[test]
fn test_mora_count_5_konnichiwa() {
    assert_eq!(count_mora("コンニチワ"), 5);
}

#[test]
fn test_mora_count_5_kakikukeko() {
    assert_eq!(count_mora("カキクケコ"), 5);
}

// ============================================================
// count_mora: 6+ モーラ語
// ============================================================

#[test]
fn test_mora_count_6_arigatou() {
    // ア・リ・ガ・ト・ウ・ゴ = 6
    assert_eq!(count_mora("アリガトウゴ"), 6);
}

#[test]
fn test_mora_count_6_konpyuutaa() {
    // コ(1) ン(2) ピュ(3) ー(4) タ(5) ー(6) = 6
    assert_eq!(count_mora("コンピューター"), 6);
}

#[test]
fn test_mora_count_6_shimyureeshon() {
    // シ(1) ミュ(2) レ(3) ー(4) ショ(5) ン(6) = 6
    assert_eq!(count_mora("シミュレーション"), 6);
}

// ============================================================
// count_mora: edge cases
// ============================================================

#[test]
fn test_mora_count_empty() {
    assert_eq!(count_mora(""), 0);
}

#[test]
fn test_mora_count_only_long_vowel() {
    assert_eq!(count_mora("ー"), 1);
}

#[test]
fn test_mora_count_only_sokuon() {
    assert_eq!(count_mora("ッ"), 1);
}

#[test]
fn test_mora_count_only_n() {
    assert_eq!(count_mora("ン"), 1);
}

#[test]
fn test_mora_count_consecutive_long_vowels() {
    // アーー = ア(1) ー(2) ー(3)
    assert_eq!(count_mora("アーー"), 3);
}

#[test]
fn test_mora_count_consecutive_sokuon() {
    // ッッ = 2
    assert_eq!(count_mora("ッッ"), 2);
}

#[test]
fn test_mora_count_mixed_special() {
    // ンッー = 3
    assert_eq!(count_mora("ンッー"), 3);
}

// ============================================================
// parse_mora: basic vowels
// ============================================================

#[test]
fn test_parse_mora_a() {
    let moras = parse_mora("ア");
    assert_eq!(moras.len(), 1);
    assert_eq!(moras[0].consonant, None);
    assert_eq!(moras[0].vowel, "a");
    assert_eq!(moras[0].text, "ア");
}

#[test]
fn test_parse_mora_i() {
    let moras = parse_mora("イ");
    assert_eq!(moras.len(), 1);
    assert_eq!(moras[0].consonant, None);
    assert_eq!(moras[0].vowel, "i");
}

// ============================================================
// parse_mora: CV split for consonant types
// ============================================================

#[test]
fn test_parse_mora_cv_k() {
    let moras = parse_mora("カ");
    assert_eq!(moras[0].consonant, Some("k".to_string()));
    assert_eq!(moras[0].vowel, "a");
}

#[test]
fn test_parse_mora_cv_s() {
    let moras = parse_mora("サ");
    assert_eq!(moras[0].consonant, Some("s".to_string()));
    assert_eq!(moras[0].vowel, "a");
}

#[test]
fn test_parse_mora_cv_sh() {
    let moras = parse_mora("シ");
    assert_eq!(moras[0].consonant, Some("sh".to_string()));
    assert_eq!(moras[0].vowel, "i");
}

#[test]
fn test_parse_mora_cv_t() {
    let moras = parse_mora("タ");
    assert_eq!(moras[0].consonant, Some("t".to_string()));
    assert_eq!(moras[0].vowel, "a");
}

#[test]
fn test_parse_mora_cv_ch() {
    let moras = parse_mora("チ");
    assert_eq!(moras[0].consonant, Some("ch".to_string()));
    assert_eq!(moras[0].vowel, "i");
}

#[test]
fn test_parse_mora_cv_ts() {
    let moras = parse_mora("ツ");
    assert_eq!(moras[0].consonant, Some("ts".to_string()));
    assert_eq!(moras[0].vowel, "u");
}

#[test]
fn test_parse_mora_cv_n_consonant() {
    let moras = parse_mora("ナ");
    assert_eq!(moras[0].consonant, Some("n".to_string()));
    assert_eq!(moras[0].vowel, "a");
}

#[test]
fn test_parse_mora_cv_h() {
    let moras = parse_mora("ハ");
    assert_eq!(moras[0].consonant, Some("h".to_string()));
    assert_eq!(moras[0].vowel, "a");
}

#[test]
fn test_parse_mora_cv_f() {
    let moras = parse_mora("フ");
    assert_eq!(moras[0].consonant, Some("f".to_string()));
    assert_eq!(moras[0].vowel, "u");
}

#[test]
fn test_parse_mora_cv_m() {
    let moras = parse_mora("マ");
    assert_eq!(moras[0].consonant, Some("m".to_string()));
    assert_eq!(moras[0].vowel, "a");
}

#[test]
fn test_parse_mora_cv_y() {
    let moras = parse_mora("ヤ");
    assert_eq!(moras[0].consonant, Some("y".to_string()));
    assert_eq!(moras[0].vowel, "a");
}

#[test]
fn test_parse_mora_cv_r() {
    let moras = parse_mora("ラ");
    assert_eq!(moras[0].consonant, Some("r".to_string()));
    assert_eq!(moras[0].vowel, "a");
}

#[test]
fn test_parse_mora_cv_w() {
    let moras = parse_mora("ワ");
    assert_eq!(moras[0].consonant, Some("w".to_string()));
    assert_eq!(moras[0].vowel, "a");
}

#[test]
fn test_parse_mora_cv_g() {
    let moras = parse_mora("ガ");
    assert_eq!(moras[0].consonant, Some("g".to_string()));
    assert_eq!(moras[0].vowel, "a");
}

#[test]
fn test_parse_mora_cv_z() {
    let moras = parse_mora("ザ");
    assert_eq!(moras[0].consonant, Some("z".to_string()));
    assert_eq!(moras[0].vowel, "a");
}

#[test]
fn test_parse_mora_cv_d() {
    let moras = parse_mora("ダ");
    assert_eq!(moras[0].consonant, Some("d".to_string()));
    assert_eq!(moras[0].vowel, "a");
}

#[test]
fn test_parse_mora_cv_b() {
    let moras = parse_mora("バ");
    assert_eq!(moras[0].consonant, Some("b".to_string()));
    assert_eq!(moras[0].vowel, "a");
}

#[test]
fn test_parse_mora_cv_p() {
    let moras = parse_mora("パ");
    assert_eq!(moras[0].consonant, Some("p".to_string()));
    assert_eq!(moras[0].vowel, "a");
}

// ============================================================
// parse_mora: youon CV split
// ============================================================

#[test]
fn test_parse_mora_cv_ky() {
    let moras = parse_mora("キャ");
    assert_eq!(moras.len(), 1);
    assert_eq!(moras[0].consonant, Some("ky".to_string()));
    assert_eq!(moras[0].vowel, "a");
}

#[test]
fn test_parse_mora_cv_sh_youon() {
    let moras = parse_mora("シャ");
    assert_eq!(moras.len(), 1);
    assert_eq!(moras[0].consonant, Some("sh".to_string()));
    assert_eq!(moras[0].vowel, "a");
}

#[test]
fn test_parse_mora_cv_ny() {
    let moras = parse_mora("ニュ");
    assert_eq!(moras.len(), 1);
    assert_eq!(moras[0].consonant, Some("ny".to_string()));
    assert_eq!(moras[0].vowel, "u");
}

#[test]
fn test_parse_mora_cv_hy() {
    let moras = parse_mora("ヒョ");
    assert_eq!(moras.len(), 1);
    assert_eq!(moras[0].consonant, Some("hy".to_string()));
    assert_eq!(moras[0].vowel, "o");
}

#[test]
fn test_parse_mora_cv_my() {
    let moras = parse_mora("ミャ");
    assert_eq!(moras.len(), 1);
    assert_eq!(moras[0].consonant, Some("my".to_string()));
    assert_eq!(moras[0].vowel, "a");
}

#[test]
fn test_parse_mora_cv_ry() {
    let moras = parse_mora("リョ");
    assert_eq!(moras.len(), 1);
    assert_eq!(moras[0].consonant, Some("ry".to_string()));
    assert_eq!(moras[0].vowel, "o");
}

#[test]
fn test_parse_mora_cv_gy() {
    let moras = parse_mora("ギュ");
    assert_eq!(moras.len(), 1);
    assert_eq!(moras[0].consonant, Some("gy".to_string()));
    assert_eq!(moras[0].vowel, "u");
}

#[test]
fn test_parse_mora_cv_by() {
    let moras = parse_mora("ビュ");
    assert_eq!(moras.len(), 1);
    assert_eq!(moras[0].consonant, Some("by".to_string()));
    assert_eq!(moras[0].vowel, "u");
}

#[test]
fn test_parse_mora_cv_py() {
    let moras = parse_mora("ピョ");
    assert_eq!(moras.len(), 1);
    assert_eq!(moras[0].consonant, Some("py".to_string()));
    assert_eq!(moras[0].vowel, "o");
}

// ============================================================
// parse_mora: special moras
// ============================================================

#[test]
fn test_parse_mora_n_special() {
    let moras = parse_mora("ン");
    assert_eq!(moras.len(), 1);
    assert_eq!(moras[0].consonant, None);
    assert_eq!(moras[0].vowel, "N");
}

#[test]
fn test_parse_mora_sokuon() {
    let moras = parse_mora("ッ");
    assert_eq!(moras.len(), 1);
    assert_eq!(moras[0].consonant, None);
    assert_eq!(moras[0].vowel, "cl");
}

// ============================================================
// parse_mora: multi-mora words
// ============================================================

#[test]
fn test_parse_mora_neko() {
    let moras = parse_mora("ネコ");
    assert_eq!(moras.len(), 2);
    assert_eq!(moras[0].text, "ネ");
    assert_eq!(moras[1].text, "コ");
}

#[test]
fn test_parse_mora_sakura() {
    let moras = parse_mora("サクラ");
    assert_eq!(moras.len(), 3);
    assert_eq!(moras[0].text, "サ");
    assert_eq!(moras[1].text, "ク");
    assert_eq!(moras[2].text, "ラ");
}

#[test]
fn test_parse_mora_gakkou() {
    let moras = parse_mora("ガッコウ");
    assert_eq!(moras.len(), 4);
    assert_eq!(moras[0].text, "ガ");
    assert_eq!(moras[1].text, "ッ");
    assert_eq!(moras[1].vowel, "cl");
    assert_eq!(moras[2].text, "コ");
    assert_eq!(moras[3].text, "ウ");
}

#[test]
fn test_parse_mora_empty() {
    let moras = parse_mora("");
    assert_eq!(moras.len(), 0);
}

#[test]
fn test_parse_mora_hon() {
    let moras = parse_mora("ホン");
    assert_eq!(moras.len(), 2);
    assert_eq!(moras[0].consonant, Some("h".to_string()));
    assert_eq!(moras[0].vowel, "o");
    assert_eq!(moras[1].consonant, None);
    assert_eq!(moras[1].vowel, "N");
}

#[test]
fn test_parse_mora_long_vowel() {
    let moras = parse_mora("カー");
    assert_eq!(moras.len(), 2);
    assert_eq!(moras[0].text, "カ");
    assert_eq!(moras[1].text, "ー");
}

#[test]
fn test_parse_mora_count_consistency() {
    // count_mora and parse_mora should agree on length
    let words = vec![
        "ア", "ネコ", "サクラ", "ガッコウ", "コンニチワ",
        "トーキョー", "コーヒー", "シミュレーション",
    ];
    for word in words {
        let count = count_mora(word);
        let parsed = parse_mora(word);
        assert_eq!(
            count as usize,
            parsed.len(),
            "Mismatch for '{}': count_mora={}, parse_mora.len={}",
            word,
            count,
            parsed.len()
        );
    }
}
