//! 実発話 fixture による専用パイプラインの起動/推論ベンチマーク。
use kotonoha::njd::{InputToken, build_njd_nodes};
use kotonoha::nn::FeatureMorpheme;
use kotonoha::nn::v66::{V66Bundle, V66Pipeline};
use serde_json::{Value, json};
use std::path::PathBuf;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args_os().collect();
    if args.len() != 3 {
        return Err("usage: knm_pipeline_probe MODEL_DIR PIPELINE_FIXTURE.json".into());
    }
    let root = PathBuf::from(&args[1]);
    let t = Instant::now();
    let bundle = V66Bundle::from_dir(&root)?;
    let load_ms = t.elapsed().as_secs_f64() * 1000.;
    let pipeline = V66Pipeline::new(bundle);
    let fixture: Value = serde_json::from_reader(std::fs::File::open(&args[2])?)?;
    let mut results = Vec::new();
    for utt in fixture["utterances"]
        .as_array()
        .ok_or("utterances missing")?
    {
        let mut tokens = Vec::new();
        let mut accents = Vec::new();
        for m in utt["morphemes"].as_array().ok_or("morphemes missing")? {
            let s = |key: &str| m[key].as_str().unwrap_or("*").to_string();
            tokens.push(InputToken {
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
            });
            accents.push(
                s("dict_accent_type")
                    .trim_start_matches('"')
                    .parse::<u8>()
                    .ok(),
            );
        }
        let nodes = build_njd_nodes(&tokens);
        let ctx: Vec<_> = tokens
            .iter()
            .zip(&nodes)
            .map(|(token, node)| FeatureMorpheme { token, node })
            .collect();
        let output = pipeline.predict_with_dict_accents(&ctx, &accents)?;
        let expected: Vec<u8> = serde_json::from_value(utt["predicted_accent_types"].clone())?;
        if output != expected {
            return Err(format!("pipeline mismatch: {}", utt["utterance_id"]).into());
        }
        let mut times = Vec::new();
        for _ in 0..5 {
            let t = Instant::now();
            pipeline.predict_with_dict_accents(&ctx, &accents)?;
            times.push(t.elapsed().as_secs_f64() * 1000.);
        }
        times.sort_by(f64::total_cmp);
        results.push(json!({"utterance_id":utt["utterance_id"],"seq":tokens.len(),"inference_ms":times[2],"predictions":output}));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({"load_ms":load_ms,"workers":rayon::current_num_threads(),"cases":results})
        )?
    );
    Ok(())
}
