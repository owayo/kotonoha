//! 配置済み KNM を、分割済み注釈データで評価するオフラインの検証器。
//! 句の曖昧な核位置は候補集合として採点し、推論と結合規則の結果を分けて返す。

use kotonoha::Engine;
use kotonoha::njd::InputToken;
use kotonoha::nn::FeatureMorpheme;
use kotonoha::nn::native::NativeModel;
use kotonoha::nn::v66::{V66Bundle, V66Pipeline};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::path::PathBuf;

fn token(m: &Value) -> InputToken {
    let s = |key: &str| m[key].as_str().unwrap_or("*").to_owned();
    InputToken {
        surface: s("surface"),
        pos: s("pos"),
        pos_detail1: s("pos_detail1"),
        pos_detail2: s("pos_detail2"),
        pos_detail3: s("pos_detail3"),
        ctype: s("conjugation_type"),
        cform: s("conjugation_form"),
        lemma: s("lemma"),
        reading: s("reading"),
        pronunciation: s("pronunciation"),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args_os().collect();
    if !(3..=4).contains(&args.len()) {
        return Err("usage: knm_accuracy MODEL_DIR PREPARED_DATASET.json [ACCENT_KNM]".into());
    }
    let mut bundle = V66Bundle::from_dir(&PathBuf::from(&args[1]))?;
    if args.len() == 4 {
        let model = NativeModel::load(&PathBuf::from(&args[3]))?;
        if model.input_dim() != 103 {
            return Err("expected a 103-dimensional final model".into());
        }
        bundle.models.v66_split1 = model;
    }
    let pipeline = V66Pipeline::new(bundle);
    let dataset: Value = serde_json::from_reader(std::fs::File::open(&args[2])?)?;
    let engine = Engine::with_default_rules();
    let mut totals: BTreeMap<String, [usize; 3]> = BTreeMap::new();
    let mut cases = Vec::new();
    for row in dataset["records"].as_array().ok_or("records missing")? {
        if row["split"] != "test" {
            continue;
        }
        let ms = row["record"]["morphemes"]
            .as_array()
            .ok_or("morphemes missing")?;
        let tokens: Vec<_> = ms.iter().map(token).collect();
        let mut nodes = engine.analyze(&tokens);
        let ctx: Vec<_> = tokens
            .iter()
            .zip(&nodes)
            .map(|(token, node)| FeatureMorpheme { token, node })
            .collect();
        let predicted = pipeline.predict(&ctx)?;
        if predicted.len() != nodes.len() {
            return Err("prediction length mismatch".into());
        }
        for (node, &accent) in nodes.iter_mut().zip(&predicted) {
            node.accent_type = accent;
        }
        let phrases = engine.estimate_accent(&mut nodes);
        let mut combined = vec![0; nodes.len()];
        for phrase in phrases {
            for i in phrase.nodes {
                combined[i] = phrase.accent_type;
            }
        }
        let source = row["source"].as_str().ok_or("source missing")?;
        let mut count = [0; 3];
        let mut details = Vec::new();
        for (i, m) in ms.iter().enumerate() {
            if matches!(m["pos"].as_str(), Some("記号" | "補助記号")) {
                continue;
            }
            let candidates = if let Some(values) = m["accent_candidates"].as_array() {
                values
                    .iter()
                    .map(|v| v.as_u64().ok_or("invalid candidate"))
                    .collect::<Result<Vec<_>, _>>()?
            } else {
                vec![m["accent_type"].as_u64().ok_or("invalid label")?]
            };
            let raw_ok = candidates.contains(&u64::from(predicted[i]));
            let combined_ok = candidates.contains(&(combined[i] as u64));
            count[0] += 1;
            count[1] += usize::from(raw_ok);
            count[2] += usize::from(combined_ok);
            details.push(json!({"index":i,"raw":predicted[i],"combined":combined[i],"allowed":candidates,"raw_correct":raw_ok,"combined_correct":combined_ok}));
        }
        let total = totals.entry(source.to_owned()).or_default();
        for (a, b) in total.iter_mut().zip(count) {
            *a += b;
        }
        cases.push(json!({"id":row["id"],"source":source,"counts":count,"tokens":details}));
    }
    let metrics: BTreeMap<_, _> = totals
        .iter()
        .map(|(source, c)| {
            (
                source,
                json!({"tokens":c[0],"raw_accuracy":c[1] as f64/c[0] as f64,"combined_accuracy":c[2] as f64/c[0] as f64}),
            )
        })
        .collect();
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({"metric":"allowed-target accuracy on supplied morphology; not an HSD text benchmark","by_source":metrics,"cases":cases})
        )?
    );
    Ok(())
}
