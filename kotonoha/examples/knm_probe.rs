//! KNM と参照出力の数値比較・起動/推論計測。参照 JSON の生成手順は models 側を参照。
use kotonoha::nn::native::NativeModel;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Deserialize)]
struct Case {
    file: String,
    seq: usize,
    input: Vec<f32>,
    expected: Vec<f32>,
}
#[derive(Serialize)]
struct Measurement {
    file: String,
    seq: usize,
    load_ms: f64,
    inference_ms: f64,
    max_abs_error: f32,
    argmax_equal: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args_os().collect();
    if args.len() != 3 {
        return Err("usage: knm_probe MODEL_DIR REFERENCE.json".into());
    }
    let root = PathBuf::from(&args[1]);
    let cases: Vec<Case> = serde_json::from_reader(std::fs::File::open(&args[2])?)?;
    let mut results = Vec::new();
    for case in cases {
        let t = Instant::now();
        let model = NativeModel::load(&root.join(&case.file))?;
        let load_ms = t.elapsed().as_secs_f64() * 1000.;
        model.predict(&case.input, case.seq)?;
        let mut times = Vec::new();
        let mut actual = Vec::new();
        for _ in 0..5 {
            let t = Instant::now();
            actual = model.predict(&case.input, case.seq)?;
            times.push(t.elapsed().as_secs_f64() * 1000.);
        }
        times.sort_by(f64::total_cmp);
        if actual.len() != case.expected.len() {
            return Err("output length mismatch".into());
        }
        let max_abs_error = actual
            .iter()
            .zip(&case.expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let argmax = |row: &[f32]| {
            row.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(i, _)| i)
        };
        let argmax_equal = actual
            .as_chunks::<21>()
            .0
            .iter()
            .zip(case.expected.as_chunks::<21>().0.iter())
            .all(|(a, b)| argmax(a) == argmax(b));
        results.push(Measurement {
            file: case.file,
            seq: case.seq,
            load_ms,
            inference_ms: times[2],
            max_abs_error,
            argmax_equal,
        });
    }
    println!("{}", serde_json::to_string_pretty(&results)?);
    if results
        .iter()
        .any(|r| !r.argmax_equal || r.max_abs_error > 0.0001)
    {
        return Err("KNM reference mismatch".into());
    }
    Ok(())
}
