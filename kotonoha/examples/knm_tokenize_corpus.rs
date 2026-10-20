//! 学習入力の本文だけを、同じ HSD・NJD 経路でまとめて解析するオフライン補助ツール。
//! 正解ラベルは参照せず、InputToken と採用した発音を返す。

use kotonoha::Engine;
use kotonoha::njd::InputToken;
use serde_json::{Value, json};
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args_os().collect();
    if args.len() != 3 {
        return Err("usage: knm_tokenize_corpus DICTIONARY.hsd PREPARED_DATASET.json".into());
    }
    let mut analyzer = hasami::Analyzer::load(PathBuf::from(&args[1]))?;
    let dataset: Value = serde_json::from_reader(std::fs::File::open(&args[2])?)?;
    let records = dataset["records"].as_array().ok_or("records missing")?;
    let engine = Engine::with_default_rules();
    let mut output = Vec::with_capacity(records.len());
    for row in records {
        let id = row["id"].as_str().ok_or("record id missing")?;
        let source = row["source"].as_str().ok_or("record source missing")?;
        let text = row["record"]["text"]
            .as_str()
            .ok_or("record text missing")?;
        let tokens: Vec<_> = analyzer
            .try_tokenize(text)?
            .into_iter()
            .map(InputToken::from)
            .collect();
        let pronunciations: Vec<_> = engine
            .analyze(&tokens)
            .into_iter()
            .map(|node| node.pronunciation)
            .collect();
        output.push(json!({
            "id": id, "source": source, "text": text,
            "tokens": tokens, "pronunciations": pronunciations,
        }));
    }
    serde_json::to_writer(
        std::io::stdout().lock(),
        &json!({"format_version":1,"records":output}),
    )?;
    Ok(())
}
