//! V66Pipeline (Rust) ↔ Python `dump_v66_pipeline_fixture.py` argmax 一致テスト
//!
//! 環境要求 (満たさない場合スキップ):
//! - `ORT_DYLIB_PATH`: libonnxruntime.so の絶対パス
//! - `V66_BUNDLE_DIR` (省略時 `/mnt/c/GitHub/kotonoha-models`)
//! - fixture: `tests/fixtures/v66_pipeline.json`

#![cfg(feature = "cuda")]

use std::path::{Path, PathBuf};

use kotonoha::nn::FeatureMorpheme;
use kotonoha::nn::v66::{V66Bundle, V66Pipeline};
use kotonoha::njd::{InputToken, build_njd_nodes};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct FixMorpheme {
    surface: String,
    pos: String,
    pos_detail1: String,
    pos_detail2: String,
    pos_detail3: String,
    conjugation_type: String,
    conjugation_form: String,
    lemma: String,
    reading: String,
    pronunciation: String,
    dict_accent_type: String,
}

fn parse_dict_str(s: &str) -> Option<u8> {
    if s == "*" || s.is_empty() {
        return None;
    }
    let trimmed = s.strip_prefix('"').unwrap_or(s);
    trimmed.parse::<u8>().ok()
}

#[derive(Debug, Deserialize)]
struct FixUtt {
    utterance_id: String,
    morphemes: Vec<FixMorpheme>,
    predicted_accent_types: Vec<u8>,
}

#[derive(Debug, Deserialize)]
struct Fixture {
    utterances: Vec<FixUtt>,
}

fn skip_with(reason: &str) {
    eprintln!("v66_pipeline_match: SKIP — {reason}");
}

#[test]
fn pipeline_argmax_matches_python() {
    if std::env::var("ORT_DYLIB_PATH").is_err() {
        skip_with("ORT_DYLIB_PATH not set");
        return;
    }
    let fixture_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("v66_pipeline.json");
    if !fixture_path.exists() {
        skip_with("v66_pipeline.json fixture missing");
        return;
    }
    let bundle_dir = std::env::var("V66_BUNDLE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/mnt/c/GitHub/kotonoha-models"));
    if !bundle_dir.is_dir() {
        skip_with(&format!("bundle dir missing: {}", bundle_dir.display()));
        return;
    }

    // accent_dict 2 ファイルをマージしてロード
    let dict1 = PathBuf::from("/mnt/c/GitHub/kotonoha/data/accent_dict.csv");
    let dict2 = PathBuf::from("/mnt/c/GitHub/kotonoha-training-data/train/accent_dict.csv");
    let mut dict_paths: Vec<&Path> = Vec::new();
    if dict1.exists() {
        dict_paths.push(&dict1);
    }
    if dict2.exists() {
        dict_paths.push(&dict2);
    }
    let bundle = match V66Bundle::from_paths(&bundle_dir, &dict_paths) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("v66_pipeline_match: SKIP — bundle load failed: {e}");
            return;
        }
    };
    let pipeline = V66Pipeline::new(bundle);

    let json = std::fs::read_to_string(&fixture_path).expect("read fixture");
    let fixture: Fixture = serde_json::from_str(&json).expect("parse fixture");

    let mut total = 0usize;
    let mut matched = 0usize;
    let mut mismatched_utts = 0usize;
    for utt in &fixture.utterances {
        let tokens: Vec<InputToken> = utt
            .morphemes
            .iter()
            .map(|m| InputToken {
                surface: m.surface.clone(),
                pos: m.pos.clone(),
                pos_detail1: m.pos_detail1.clone(),
                pos_detail2: m.pos_detail2.clone(),
                pos_detail3: m.pos_detail3.clone(),
                ctype: m.conjugation_type.clone(),
                cform: m.conjugation_form.clone(),
                lemma: m.lemma.clone(),
                reading: m.reading.clone(),
                pronunciation: m.pronunciation.clone(),
            })
            .collect();
        let nodes = build_njd_nodes(&tokens);
        let ctx: Vec<FeatureMorpheme<'_>> = tokens
            .iter()
            .zip(nodes.iter())
            .map(|(token, node)| FeatureMorpheme { token, node })
            .collect();
        let dict_accents: Vec<Option<u8>> = utt
            .morphemes
            .iter()
            .map(|m| parse_dict_str(&m.dict_accent_type))
            .collect();
        let pred = pipeline
            .predict_with_dict_accents(&ctx, &dict_accents)
            .expect("predict failed");
        assert_eq!(
            pred.len(),
            utt.predicted_accent_types.len(),
            "utt {} length mismatch",
            utt.utterance_id
        );
        let mut utt_match = true;
        for (i, (&got, &expected)) in pred.iter().zip(utt.predicted_accent_types.iter()).enumerate()
        {
            total += 1;
            if got == expected {
                matched += 1;
            } else {
                if utt_match {
                    eprintln!("utt {} first mismatch:", utt.utterance_id);
                    utt_match = false;
                    mismatched_utts += 1;
                }
                eprintln!(
                    "  morph {} surface={} expected={expected} got={got}",
                    i, tokens[i].surface
                );
            }
        }
    }

    let acc = matched as f64 / total as f64;
    eprintln!(
        "pipeline match: {matched}/{total} morphemes ({:.2}%), {mismatched_utts}/{} utts",
        acc * 100.0,
        fixture.utterances.len()
    );
    // 数値ずれの許容: 99.9% 一致 (softmax 数値差で稀に argmax が割れる程度を許容)
    assert!(
        acc >= 0.999,
        "pipeline argmax match {acc:.4} below 99.9% threshold"
    );
}
