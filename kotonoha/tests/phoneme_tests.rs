//! 音素変換テスト
//! katakana_to_phonemes および音素分類関数の網羅的テスト

use kotonoha::phoneme::{
    is_consonant, is_special, is_voiceless_vowel, is_vowel, katakana_to_phonemes, ALL_PHONEMES,
    CONSONANTS, SPECIAL_SYMBOLS, VOWELS,
};

// ============================================================
// ア行 (vowels only)
// ============================================================

#[test]
fn test_phoneme_a() {
    assert_eq!(katakana_to_phonemes("ア"), vec!["a"]);
}

#[test]
fn test_phoneme_i() {
    assert_eq!(katakana_to_phonemes("イ"), vec!["i"]);
}

#[test]
fn test_phoneme_u() {
    assert_eq!(katakana_to_phonemes("ウ"), vec!["u"]);
}

#[test]
fn test_phoneme_e() {
    assert_eq!(katakana_to_phonemes("エ"), vec!["e"]);
}

#[test]
fn test_phoneme_o() {
    assert_eq!(katakana_to_phonemes("オ"), vec!["o"]);
}

// ============================================================
// カ行
// ============================================================

#[test]
fn test_phoneme_ka() {
    assert_eq!(katakana_to_phonemes("カ"), vec!["k", "a"]);
}

#[test]
fn test_phoneme_ki() {
    assert_eq!(katakana_to_phonemes("キ"), vec!["k", "i"]);
}

#[test]
fn test_phoneme_ku() {
    assert_eq!(katakana_to_phonemes("ク"), vec!["k", "u"]);
}

#[test]
fn test_phoneme_ke() {
    assert_eq!(katakana_to_phonemes("ケ"), vec!["k", "e"]);
}

#[test]
fn test_phoneme_ko() {
    assert_eq!(katakana_to_phonemes("コ"), vec!["k", "o"]);
}

// ============================================================
// サ行
// ============================================================

#[test]
fn test_phoneme_sa() {
    assert_eq!(katakana_to_phonemes("サ"), vec!["s", "a"]);
}

#[test]
fn test_phoneme_shi() {
    assert_eq!(katakana_to_phonemes("シ"), vec!["sh", "i"]);
}

#[test]
fn test_phoneme_su() {
    assert_eq!(katakana_to_phonemes("ス"), vec!["s", "u"]);
}

#[test]
fn test_phoneme_se() {
    assert_eq!(katakana_to_phonemes("セ"), vec!["s", "e"]);
}

#[test]
fn test_phoneme_so() {
    assert_eq!(katakana_to_phonemes("ソ"), vec!["s", "o"]);
}

// ============================================================
// タ行
// ============================================================

#[test]
fn test_phoneme_ta() {
    assert_eq!(katakana_to_phonemes("タ"), vec!["t", "a"]);
}

#[test]
fn test_phoneme_chi() {
    assert_eq!(katakana_to_phonemes("チ"), vec!["ch", "i"]);
}

#[test]
fn test_phoneme_tsu() {
    assert_eq!(katakana_to_phonemes("ツ"), vec!["ts", "u"]);
}

#[test]
fn test_phoneme_te() {
    assert_eq!(katakana_to_phonemes("テ"), vec!["t", "e"]);
}

#[test]
fn test_phoneme_to() {
    assert_eq!(katakana_to_phonemes("ト"), vec!["t", "o"]);
}

// ============================================================
// ナ行
// ============================================================

#[test]
fn test_phoneme_na() {
    assert_eq!(katakana_to_phonemes("ナ"), vec!["n", "a"]);
}

#[test]
fn test_phoneme_ni() {
    assert_eq!(katakana_to_phonemes("ニ"), vec!["n", "i"]);
}

#[test]
fn test_phoneme_nu() {
    assert_eq!(katakana_to_phonemes("ヌ"), vec!["n", "u"]);
}

#[test]
fn test_phoneme_ne() {
    assert_eq!(katakana_to_phonemes("ネ"), vec!["n", "e"]);
}

#[test]
fn test_phoneme_no() {
    assert_eq!(katakana_to_phonemes("ノ"), vec!["n", "o"]);
}

// ============================================================
// ハ行
// ============================================================

#[test]
fn test_phoneme_ha() {
    assert_eq!(katakana_to_phonemes("ハ"), vec!["h", "a"]);
}

#[test]
fn test_phoneme_hi() {
    assert_eq!(katakana_to_phonemes("ヒ"), vec!["h", "i"]);
}

#[test]
fn test_phoneme_fu() {
    assert_eq!(katakana_to_phonemes("フ"), vec!["f", "u"]);
}

#[test]
fn test_phoneme_he() {
    assert_eq!(katakana_to_phonemes("ヘ"), vec!["h", "e"]);
}

#[test]
fn test_phoneme_ho() {
    assert_eq!(katakana_to_phonemes("ホ"), vec!["h", "o"]);
}

// ============================================================
// マ行
// ============================================================

#[test]
fn test_phoneme_ma() {
    assert_eq!(katakana_to_phonemes("マ"), vec!["m", "a"]);
}

#[test]
fn test_phoneme_mi() {
    assert_eq!(katakana_to_phonemes("ミ"), vec!["m", "i"]);
}

#[test]
fn test_phoneme_mu() {
    assert_eq!(katakana_to_phonemes("ム"), vec!["m", "u"]);
}

#[test]
fn test_phoneme_me() {
    assert_eq!(katakana_to_phonemes("メ"), vec!["m", "e"]);
}

#[test]
fn test_phoneme_mo() {
    assert_eq!(katakana_to_phonemes("モ"), vec!["m", "o"]);
}

// ============================================================
// ヤ行
// ============================================================

#[test]
fn test_phoneme_ya() {
    assert_eq!(katakana_to_phonemes("ヤ"), vec!["y", "a"]);
}

#[test]
fn test_phoneme_yu() {
    assert_eq!(katakana_to_phonemes("ユ"), vec!["y", "u"]);
}

#[test]
fn test_phoneme_yo() {
    assert_eq!(katakana_to_phonemes("ヨ"), vec!["y", "o"]);
}

// ============================================================
// ラ行
// ============================================================

#[test]
fn test_phoneme_ra() {
    assert_eq!(katakana_to_phonemes("ラ"), vec!["r", "a"]);
}

#[test]
fn test_phoneme_ri() {
    assert_eq!(katakana_to_phonemes("リ"), vec!["r", "i"]);
}

#[test]
fn test_phoneme_ru() {
    assert_eq!(katakana_to_phonemes("ル"), vec!["r", "u"]);
}

#[test]
fn test_phoneme_re() {
    assert_eq!(katakana_to_phonemes("レ"), vec!["r", "e"]);
}

#[test]
fn test_phoneme_ro() {
    assert_eq!(katakana_to_phonemes("ロ"), vec!["r", "o"]);
}

// ============================================================
// ワ行
// ============================================================

#[test]
fn test_phoneme_wa() {
    assert_eq!(katakana_to_phonemes("ワ"), vec!["w", "a"]);
}

#[test]
fn test_phoneme_wo() {
    assert_eq!(katakana_to_phonemes("ヲ"), vec!["o"]);
}

#[test]
fn test_phoneme_n() {
    assert_eq!(katakana_to_phonemes("ン"), vec!["N"]);
}

// ============================================================
// 濁音 ガ行
// ============================================================

#[test]
fn test_phoneme_ga() {
    assert_eq!(katakana_to_phonemes("ガ"), vec!["g", "a"]);
}

#[test]
fn test_phoneme_gi() {
    assert_eq!(katakana_to_phonemes("ギ"), vec!["g", "i"]);
}

#[test]
fn test_phoneme_gu() {
    assert_eq!(katakana_to_phonemes("グ"), vec!["g", "u"]);
}

#[test]
fn test_phoneme_ge() {
    assert_eq!(katakana_to_phonemes("ゲ"), vec!["g", "e"]);
}

#[test]
fn test_phoneme_go() {
    assert_eq!(katakana_to_phonemes("ゴ"), vec!["g", "o"]);
}

// ============================================================
// 濁音 ザ行
// ============================================================

#[test]
fn test_phoneme_za() {
    assert_eq!(katakana_to_phonemes("ザ"), vec!["z", "a"]);
}

#[test]
fn test_phoneme_ji() {
    assert_eq!(katakana_to_phonemes("ジ"), vec!["j", "i"]);
}

#[test]
fn test_phoneme_zu() {
    assert_eq!(katakana_to_phonemes("ズ"), vec!["z", "u"]);
}

#[test]
fn test_phoneme_ze() {
    assert_eq!(katakana_to_phonemes("ゼ"), vec!["z", "e"]);
}

#[test]
fn test_phoneme_zo() {
    assert_eq!(katakana_to_phonemes("ゾ"), vec!["z", "o"]);
}

// ============================================================
// 濁音 ダ行
// ============================================================

#[test]
fn test_phoneme_da() {
    assert_eq!(katakana_to_phonemes("ダ"), vec!["d", "a"]);
}

#[test]
fn test_phoneme_di_as_ji() {
    // ヂ = ji (ジと同音)
    assert_eq!(katakana_to_phonemes("ヂ"), vec!["j", "i"]);
}

#[test]
fn test_phoneme_du_as_zu() {
    // ヅ = zu (ズと同音)
    assert_eq!(katakana_to_phonemes("ヅ"), vec!["z", "u"]);
}

#[test]
fn test_phoneme_de() {
    assert_eq!(katakana_to_phonemes("デ"), vec!["d", "e"]);
}

#[test]
fn test_phoneme_do() {
    assert_eq!(katakana_to_phonemes("ド"), vec!["d", "o"]);
}

// ============================================================
// 半濁音 パ行
// ============================================================

#[test]
fn test_phoneme_pa() {
    assert_eq!(katakana_to_phonemes("パ"), vec!["p", "a"]);
}

#[test]
fn test_phoneme_pi() {
    assert_eq!(katakana_to_phonemes("ピ"), vec!["p", "i"]);
}

#[test]
fn test_phoneme_pu() {
    assert_eq!(katakana_to_phonemes("プ"), vec!["p", "u"]);
}

#[test]
fn test_phoneme_pe() {
    assert_eq!(katakana_to_phonemes("ペ"), vec!["p", "e"]);
}

#[test]
fn test_phoneme_po() {
    assert_eq!(katakana_to_phonemes("ポ"), vec!["p", "o"]);
}

// ============================================================
// 濁音 バ行
// ============================================================

#[test]
fn test_phoneme_ba() {
    assert_eq!(katakana_to_phonemes("バ"), vec!["b", "a"]);
}

#[test]
fn test_phoneme_bi() {
    assert_eq!(katakana_to_phonemes("ビ"), vec!["b", "i"]);
}

#[test]
fn test_phoneme_bu() {
    assert_eq!(katakana_to_phonemes("ブ"), vec!["b", "u"]);
}

#[test]
fn test_phoneme_be() {
    assert_eq!(katakana_to_phonemes("ベ"), vec!["b", "e"]);
}

#[test]
fn test_phoneme_bo() {
    assert_eq!(katakana_to_phonemes("ボ"), vec!["b", "o"]);
}

// ============================================================
// 拗音 (youon) キャ行
// ============================================================

#[test]
fn test_phoneme_kya() {
    assert_eq!(katakana_to_phonemes("キャ"), vec!["ky", "a"]);
}

#[test]
fn test_phoneme_kyu() {
    assert_eq!(katakana_to_phonemes("キュ"), vec!["ky", "u"]);
}

#[test]
fn test_phoneme_kyo() {
    assert_eq!(katakana_to_phonemes("キョ"), vec!["ky", "o"]);
}

// ============================================================
// 拗音 シャ行
// ============================================================

#[test]
fn test_phoneme_sha() {
    assert_eq!(katakana_to_phonemes("シャ"), vec!["sh", "a"]);
}

#[test]
fn test_phoneme_shu() {
    assert_eq!(katakana_to_phonemes("シュ"), vec!["sh", "u"]);
}

#[test]
fn test_phoneme_sho() {
    assert_eq!(katakana_to_phonemes("ショ"), vec!["sh", "o"]);
}

// ============================================================
// 拗音 チャ行
// ============================================================

#[test]
fn test_phoneme_cha() {
    assert_eq!(katakana_to_phonemes("チャ"), vec!["ch", "a"]);
}

#[test]
fn test_phoneme_chu() {
    assert_eq!(katakana_to_phonemes("チュ"), vec!["ch", "u"]);
}

#[test]
fn test_phoneme_cho() {
    assert_eq!(katakana_to_phonemes("チョ"), vec!["ch", "o"]);
}

// ============================================================
// 拗音 ニャ行
// ============================================================

#[test]
fn test_phoneme_nya() {
    assert_eq!(katakana_to_phonemes("ニャ"), vec!["ny", "a"]);
}

#[test]
fn test_phoneme_nyu() {
    assert_eq!(katakana_to_phonemes("ニュ"), vec!["ny", "u"]);
}

#[test]
fn test_phoneme_nyo() {
    assert_eq!(katakana_to_phonemes("ニョ"), vec!["ny", "o"]);
}

// ============================================================
// 拗音 ヒャ行
// ============================================================

#[test]
fn test_phoneme_hya() {
    assert_eq!(katakana_to_phonemes("ヒャ"), vec!["hy", "a"]);
}

#[test]
fn test_phoneme_hyu() {
    assert_eq!(katakana_to_phonemes("ヒュ"), vec!["hy", "u"]);
}

#[test]
fn test_phoneme_hyo() {
    assert_eq!(katakana_to_phonemes("ヒョ"), vec!["hy", "o"]);
}

// ============================================================
// 拗音 ミャ行
// ============================================================

#[test]
fn test_phoneme_mya() {
    assert_eq!(katakana_to_phonemes("ミャ"), vec!["my", "a"]);
}

#[test]
fn test_phoneme_myu() {
    assert_eq!(katakana_to_phonemes("ミュ"), vec!["my", "u"]);
}

#[test]
fn test_phoneme_myo() {
    assert_eq!(katakana_to_phonemes("ミョ"), vec!["my", "o"]);
}

// ============================================================
// 拗音 リャ行
// ============================================================

#[test]
fn test_phoneme_rya() {
    assert_eq!(katakana_to_phonemes("リャ"), vec!["ry", "a"]);
}

#[test]
fn test_phoneme_ryu() {
    assert_eq!(katakana_to_phonemes("リュ"), vec!["ry", "u"]);
}

#[test]
fn test_phoneme_ryo() {
    assert_eq!(katakana_to_phonemes("リョ"), vec!["ry", "o"]);
}

// ============================================================
// 拗音 ギャ行
// ============================================================

#[test]
fn test_phoneme_gya() {
    assert_eq!(katakana_to_phonemes("ギャ"), vec!["gy", "a"]);
}

#[test]
fn test_phoneme_gyu() {
    assert_eq!(katakana_to_phonemes("ギュ"), vec!["gy", "u"]);
}

#[test]
fn test_phoneme_gyo() {
    assert_eq!(katakana_to_phonemes("ギョ"), vec!["gy", "o"]);
}

// ============================================================
// 拗音 ジャ行
// ============================================================

#[test]
fn test_phoneme_ja() {
    assert_eq!(katakana_to_phonemes("ジャ"), vec!["j", "a"]);
}

#[test]
fn test_phoneme_ju() {
    assert_eq!(katakana_to_phonemes("ジュ"), vec!["j", "u"]);
}

#[test]
fn test_phoneme_jo() {
    assert_eq!(katakana_to_phonemes("ジョ"), vec!["j", "o"]);
}

// ============================================================
// 拗音 ビャ行
// ============================================================

#[test]
fn test_phoneme_bya() {
    assert_eq!(katakana_to_phonemes("ビャ"), vec!["by", "a"]);
}

#[test]
fn test_phoneme_byu() {
    assert_eq!(katakana_to_phonemes("ビュ"), vec!["by", "u"]);
}

#[test]
fn test_phoneme_byo() {
    assert_eq!(katakana_to_phonemes("ビョ"), vec!["by", "o"]);
}

// ============================================================
// 拗音 ピャ行
// ============================================================

#[test]
fn test_phoneme_pya() {
    assert_eq!(katakana_to_phonemes("ピャ"), vec!["py", "a"]);
}

#[test]
fn test_phoneme_pyu() {
    assert_eq!(katakana_to_phonemes("ピュ"), vec!["py", "u"]);
}

#[test]
fn test_phoneme_pyo() {
    assert_eq!(katakana_to_phonemes("ピョ"), vec!["py", "o"]);
}

// ============================================================
// 特殊外来音
// ============================================================

#[test]
fn test_phoneme_ti_special() {
    // ティ
    assert_eq!(katakana_to_phonemes("ティ"), vec!["t", "i"]);
}

#[test]
fn test_phoneme_di_special() {
    // ディ
    assert_eq!(katakana_to_phonemes("ディ"), vec!["dy", "i"]);
}

#[test]
fn test_phoneme_fa() {
    assert_eq!(katakana_to_phonemes("ファ"), vec!["f", "a"]);
}

#[test]
fn test_phoneme_fi() {
    assert_eq!(katakana_to_phonemes("フィ"), vec!["f", "i"]);
}

#[test]
fn test_phoneme_fe() {
    assert_eq!(katakana_to_phonemes("フェ"), vec!["f", "e"]);
}

#[test]
fn test_phoneme_fo() {
    assert_eq!(katakana_to_phonemes("フォ"), vec!["f", "o"]);
}

#[test]
fn test_phoneme_va() {
    assert_eq!(katakana_to_phonemes("ヴァ"), vec!["v", "a"]);
}

#[test]
fn test_phoneme_vi() {
    assert_eq!(katakana_to_phonemes("ヴィ"), vec!["v", "i"]);
}

#[test]
fn test_phoneme_vu() {
    assert_eq!(katakana_to_phonemes("ヴ"), vec!["v", "u"]);
}

#[test]
fn test_phoneme_ve() {
    assert_eq!(katakana_to_phonemes("ヴェ"), vec!["v", "e"]);
}

#[test]
fn test_phoneme_vo() {
    assert_eq!(katakana_to_phonemes("ヴォ"), vec!["v", "o"]);
}

#[test]
fn test_phoneme_tsa() {
    assert_eq!(katakana_to_phonemes("ツァ"), vec!["ts", "a"]);
}

#[test]
fn test_phoneme_tsi() {
    assert_eq!(katakana_to_phonemes("ツィ"), vec!["ts", "i"]);
}

#[test]
fn test_phoneme_tse() {
    assert_eq!(katakana_to_phonemes("ツェ"), vec!["ts", "e"]);
}

#[test]
fn test_phoneme_tso() {
    assert_eq!(katakana_to_phonemes("ツォ"), vec!["ts", "o"]);
}

#[test]
fn test_phoneme_she() {
    assert_eq!(katakana_to_phonemes("シェ"), vec!["sh", "e"]);
}

#[test]
fn test_phoneme_che() {
    assert_eq!(katakana_to_phonemes("チェ"), vec!["ch", "e"]);
}

#[test]
fn test_phoneme_kye() {
    assert_eq!(katakana_to_phonemes("キェ"), vec!["ky", "e"]);
}

#[test]
fn test_phoneme_wi_special() {
    assert_eq!(katakana_to_phonemes("ウィ"), vec!["w", "i"]);
}

#[test]
fn test_phoneme_we_special() {
    assert_eq!(katakana_to_phonemes("ウェ"), vec!["w", "e"]);
}

#[test]
fn test_phoneme_dyu() {
    assert_eq!(katakana_to_phonemes("デュ"), vec!["dy", "u"]);
}

// ============================================================
// 特殊記号
// ============================================================

#[test]
fn test_phoneme_sokuon() {
    assert_eq!(katakana_to_phonemes("ッ"), vec!["cl"]);
}

// ============================================================
// 長音展開
// ============================================================

#[test]
fn test_phoneme_long_vowel_a() {
    // カー → k, a, a
    assert_eq!(katakana_to_phonemes("カー"), vec!["k", "a", "a"]);
}

#[test]
fn test_phoneme_long_vowel_i() {
    // キー → k, i, i
    assert_eq!(katakana_to_phonemes("キー"), vec!["k", "i", "i"]);
}

#[test]
fn test_phoneme_long_vowel_u() {
    // クー → k, u, u
    assert_eq!(katakana_to_phonemes("クー"), vec!["k", "u", "u"]);
}

#[test]
fn test_phoneme_long_vowel_e() {
    // ケー → k, e, e
    assert_eq!(katakana_to_phonemes("ケー"), vec!["k", "e", "e"]);
}

#[test]
fn test_phoneme_long_vowel_o() {
    // コー → k, o, o
    assert_eq!(katakana_to_phonemes("コー"), vec!["k", "o", "o"]);
}

// ============================================================
// 促音 (ッ) in context
// ============================================================

#[test]
fn test_phoneme_geminate_in_word_gakkou() {
    assert_eq!(
        katakana_to_phonemes("ガッコウ"),
        vec!["g", "a", "cl", "k", "o", "u"]
    );
}

#[test]
fn test_phoneme_geminate_at_start_kitto() {
    assert_eq!(
        katakana_to_phonemes("キット"),
        vec!["k", "i", "cl", "t", "o"]
    );
}

#[test]
fn test_phoneme_geminate_in_bikkuri() {
    assert_eq!(
        katakana_to_phonemes("ビックリ"),
        vec!["b", "i", "cl", "k", "u", "r", "i"]
    );
}

// ============================================================
// ン before various consonants
// ============================================================

#[test]
fn test_phoneme_n_before_ka() {
    assert_eq!(
        katakana_to_phonemes("アンカ"),
        vec!["a", "N", "k", "a"]
    );
}

#[test]
fn test_phoneme_n_before_sa() {
    assert_eq!(
        katakana_to_phonemes("アンサ"),
        vec!["a", "N", "s", "a"]
    );
}

#[test]
fn test_phoneme_n_before_ta() {
    assert_eq!(
        katakana_to_phonemes("アンタ"),
        vec!["a", "N", "t", "a"]
    );
}

#[test]
fn test_phoneme_n_before_na() {
    assert_eq!(
        katakana_to_phonemes("アンナ"),
        vec!["a", "N", "n", "a"]
    );
}

#[test]
fn test_phoneme_n_before_pa() {
    assert_eq!(
        katakana_to_phonemes("アンパ"),
        vec!["a", "N", "p", "a"]
    );
}

#[test]
fn test_phoneme_n_at_end() {
    assert_eq!(
        katakana_to_phonemes("カン"),
        vec!["k", "a", "N"]
    );
}

// ============================================================
// Multi-character words
// ============================================================

#[test]
fn test_phoneme_konpyuutaa() {
    assert_eq!(
        katakana_to_phonemes("コンピューター"),
        vec!["k", "o", "N", "py", "u", "u", "t", "a", "a"]
    );
}

#[test]
fn test_phoneme_shimyureeshon() {
    assert_eq!(
        katakana_to_phonemes("シミュレーション"),
        vec!["sh", "i", "my", "u", "r", "e", "e", "sh", "o", "N"]
    );
}

#[test]
fn test_phoneme_konnichiwa() {
    assert_eq!(
        katakana_to_phonemes("コンニチワ"),
        vec!["k", "o", "N", "n", "i", "ch", "i", "w", "a"]
    );
}

#[test]
fn test_phoneme_koohii() {
    assert_eq!(
        katakana_to_phonemes("コーヒー"),
        vec!["k", "o", "o", "h", "i", "i"]
    );
}

#[test]
fn test_phoneme_tokyo() {
    assert_eq!(
        katakana_to_phonemes("トウキョウ"),
        vec!["t", "o", "u", "ky", "o", "u"]
    );
}

#[test]
fn test_phoneme_choucho() {
    assert_eq!(
        katakana_to_phonemes("チョウチョ"),
        vec!["ch", "o", "u", "ch", "o"]
    );
}

// ============================================================
// is_vowel / is_consonant / is_special classification
// ============================================================

#[test]
fn test_is_vowel_all_lowercase() {
    for v in &["a", "i", "u", "e", "o"] {
        assert!(is_vowel(v), "{} should be a vowel", v);
    }
}

#[test]
fn test_is_vowel_all_uppercase() {
    for v in &["A", "I", "U", "E", "O"] {
        assert!(is_vowel(v), "{} should be a vowel", v);
    }
}

#[test]
fn test_is_vowel_rejects_consonants() {
    for c in CONSONANTS {
        assert!(!is_vowel(c), "{} should not be a vowel", c);
    }
}

#[test]
fn test_is_voiceless_vowel_all() {
    for v in &["A", "I", "U", "E", "O"] {
        assert!(is_voiceless_vowel(v), "{} should be voiceless", v);
    }
    for v in &["a", "i", "u", "e", "o"] {
        assert!(!is_voiceless_vowel(v), "{} should not be voiceless", v);
    }
}

#[test]
fn test_is_consonant_all() {
    for c in CONSONANTS {
        assert!(is_consonant(c), "{} should be consonant", c);
    }
}

#[test]
fn test_is_consonant_rejects_vowels() {
    for v in VOWELS {
        assert!(!is_consonant(v), "{} should not be consonant", v);
    }
}

#[test]
fn test_is_special_all() {
    for s in SPECIAL_SYMBOLS {
        assert!(is_special(s), "{} should be special", s);
    }
}

#[test]
fn test_is_special_rejects_vowels() {
    for v in VOWELS {
        assert!(!is_special(v), "{} should not be special", v);
    }
}

#[test]
fn test_all_phonemes_list_not_empty() {
    assert!(!ALL_PHONEMES.is_empty());
    // Every phoneme should be classifiable
    for p in ALL_PHONEMES {
        let classified = is_vowel(p) || is_consonant(p) || is_special(p);
        assert!(classified, "Phoneme {} is not classified", p);
    }
}

// ============================================================
// Empty / edge cases
// ============================================================

#[test]
fn test_phoneme_empty_string() {
    assert_eq!(katakana_to_phonemes(""), Vec::<String>::new());
}

#[test]
fn test_phoneme_unknown_character_skipped() {
    // ASCII characters are skipped
    assert_eq!(katakana_to_phonemes("ABC"), Vec::<String>::new());
}

#[test]
fn test_phoneme_mixed_katakana_and_unknown() {
    // Unknown chars are skipped, katakana is processed
    assert_eq!(katakana_to_phonemes("アXイ"), vec!["a", "i"]);
}
