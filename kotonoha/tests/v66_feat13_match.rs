//! Python ↔ Rust feat13 数値完全一致テスト
//!
//! `training/dump_v66_fixture.py` で生成した fixture を読み込み、Rust 側の
//! `nn::v66::features::extract_feat13` が同じ値を返すことを確認する。
//!
//! fixture が無い場合 (CI 等) はスキップする。

use std::path::Path;

use kotonoha::nn::v66::features::{FEATURE13_DIM, MorphemeView, extract_feat13};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct FixtureMorpheme {
    surface: String,
    pos: String,
    pos_detail1: String,
    pos_detail2: String,
    conjugation_type: String,
    conjugation_form: String,
    #[allow(dead_code)]
    lemma: String,
    reading: String,
    #[allow(dead_code)]
    pronunciation: String,
    dict_accent_type: String,
}

#[derive(Debug, Deserialize)]
struct FixtureUtterance {
    utterance_id: String,
    morphemes: Vec<FixtureMorpheme>,
    feat13: Vec<Vec<f32>>,
}

#[derive(Debug, Deserialize)]
struct Fixture {
    utterances: Vec<FixtureUtterance>,
}

fn parse_dict_str(s: &str) -> Option<u8> {
    if s == "*" || s.is_empty() {
        return None;
    }
    let trimmed = s.strip_prefix('"').unwrap_or(s);
    trimmed.parse::<u8>().ok()
}

#[test]
fn feat13_matches_python_fixture() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("v66_feat13.json");
    if !path.exists() {
        eprintln!("fixture missing, skipping: {}", path.display());
        return;
    }

    let json = std::fs::read_to_string(&path).expect("read fixture");
    let fixture: Fixture = serde_json::from_str(&json).expect("parse fixture");

    let mut total_morphemes = 0usize;
    for utt in &fixture.utterances {
        let n = utt.morphemes.len();
        assert_eq!(
            utt.feat13.len(),
            n,
            "utt {}: feat13 row count mismatch",
            utt.utterance_id
        );
        for (i, (m, expected)) in utt.morphemes.iter().zip(utt.feat13.iter()).enumerate() {
            assert_eq!(expected.len(), FEATURE13_DIM);
            let position = i as f32 / (n - 1).max(1) as f32;
            let view = MorphemeView {
                surface: &m.surface,
                pos: &m.pos,
                pos_detail1: &m.pos_detail1,
                pos_detail2: &m.pos_detail2,
                conjugation_type: &m.conjugation_type,
                conjugation_form: &m.conjugation_form,
                reading: &m.reading,
                dict_accent_type: parse_dict_str(&m.dict_accent_type),
            };
            let got = extract_feat13(&view, position);
            for (j, (a, b)) in got.iter().zip(expected.iter()).enumerate() {
                let diff = (a - b).abs();
                assert!(
                    diff < 1e-7,
                    "utt {} morph {} dim {}: rust={a} python={b} diff={diff}",
                    utt.utterance_id,
                    i,
                    j,
                );
            }
            total_morphemes += 1;
        }
    }
    println!(
        "feat13 match: {} utts, {} morphemes",
        fixture.utterances.len(),
        total_morphemes
    );
}
