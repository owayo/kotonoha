//! 同じ専用エンジンで KNM 保存プロファイルを比較する。重みとカーネルは変更しない。
use kotonoha::njd::{InputToken, build_njd_nodes};
use kotonoha::nn::FeatureMorpheme;
use kotonoha::nn::native::NativeModel;
use kotonoha::nn::v66::{V66Bundle, V66Pipeline};
use serde_json::{Value, json};
use std::io::Write;
use std::path::Path;
use std::time::Instant;

#[cfg(windows)]
fn memory() -> Value {
    // PROCESS_MEMORY_COUNTERS_EX: private commit と working set は別々に記録する。
    #[repr(C)]
    #[derive(Default)]
    struct Counters {
        cb: u32,
        page_faults: u32,
        peak_working_set: usize,
        working_set: usize,
        quota_peak_paged: usize,
        quota_paged: usize,
        quota_peak_nonpaged: usize,
        quota_nonpaged: usize,
        pagefile: usize,
        peak_pagefile: usize,
        private_usage: usize,
    }
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentProcess() -> *mut std::ffi::c_void;
        fn K32GetProcessMemoryInfo(
            process: *mut std::ffi::c_void,
            counters: *mut Counters,
            size: u32,
        ) -> i32;
    }
    let mut counters = Counters {
        cb: std::mem::size_of::<Counters>() as u32,
        ..Counters::default()
    };
    // SAFETY: current-process pseudo handle, initialized C-layout struct of the advertised size.
    let success =
        unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb) };
    if success == 0 {
        return Value::Null;
    }
    json!({"working_set_bytes": counters.working_set,
        "peak_working_set_bytes": counters.peak_working_set,
        "private_commit_bytes": counters.private_usage,
        "peak_pagefile_bytes": counters.peak_pagefile})
}

#[cfg(not(windows))]
fn memory() -> Value {
    Value::Null
}

fn single(root: &Path, request: &Value) -> Result<Value, Box<dyn std::error::Error>> {
    let mut models = Vec::new();
    let mut loads = Vec::new();
    let start = Instant::now();
    for name in request["files"].as_array().ok_or("files missing")? {
        let name = name.as_str().ok_or("file name missing")?;
        let tick = Instant::now();
        let model = NativeModel::load(&root.join(name))?;
        loads.push(json!({"file": name, "ms": tick.elapsed().as_secs_f64()*1000.}));
        models.push((name, model));
    }
    let load_ms = start.elapsed().as_secs_f64() * 1000.;
    let after_load = memory();
    let mut cases = Vec::new();
    for (name, model) in &models {
        for seq in [0, 1, 2, 7, 8, 20, 33, 64, 128] {
            let input: Vec<f32> = (0..seq * model.input_dim())
                .map(|i| ((i * 17 + 13) % 97) as f32 / 97.)
                .collect();
            let tick = Instant::now();
            let logits = model.predict(&input, seq)?;
            let first_ms = tick.elapsed().as_secs_f64() * 1000.;
            let mut times = Vec::new();
            for _ in 0..3 {
                let tick = Instant::now();
                let repeated = model.predict(&input, seq)?;
                times.push(tick.elapsed().as_secs_f64() * 1000.);
                if repeated
                    .iter()
                    .map(|v| v.to_bits())
                    .ne(logits.iter().map(|v| v.to_bits()))
                {
                    return Err("non-deterministic logits".into());
                }
            }
            cases.push(json!({"file": name, "seq": seq, "first_ms": first_ms,
                "steady_ms": times, "logits_bits": logits.iter().map(|v| v.to_bits()).collect::<Vec<_>>() }));
        }
    }
    Ok(
        json!({"load_ms":load_ms, "model_loads": loads, "memory_after_load":after_load,
        "memory_after_inference":memory(), "cases":cases}),
    )
}

fn pipeline(root: &Path, fixture: &Value) -> Result<Value, Box<dyn std::error::Error>> {
    let start = Instant::now();
    let bundle = V66Bundle::from_dir(root)?;
    let load_ms = start.elapsed().as_secs_f64() * 1000.;
    let pipeline = V66Pipeline::new(bundle);
    let after_load = memory();
    let mut cases = Vec::new();
    let mut after_first = Value::Null;
    for (index, utt) in fixture["utterances"]
        .as_array()
        .ok_or("utterances missing")?
        .iter()
        .enumerate()
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
        let tick = Instant::now();
        let output = pipeline.predict_with_dict_accents(&ctx, &accents)?;
        let first_ms = tick.elapsed().as_secs_f64() * 1000.;
        if index == 0 {
            after_first = memory();
            println!(
                "{}",
                json!({"event":"first_output", "ready_ms":start.elapsed().as_secs_f64()*1000.})
            );
            std::io::stdout().flush()?;
        }
        if let Some(expected) = utt.get("predicted_accent_types") {
            let expected: Vec<u8> = serde_json::from_value(expected.clone())?;
            if output != expected {
                return Err("fixture prediction mismatch".into());
            }
        }
        let mut times = Vec::new();
        for _ in 0..3 {
            let tick = Instant::now();
            let repeated = pipeline.predict_with_dict_accents(&ctx, &accents)?;
            times.push(tick.elapsed().as_secs_f64() * 1000.);
            if repeated != output {
                return Err("non-deterministic pipeline".into());
            }
        }
        cases.push(
            json!({"id":utt["utterance_id"], "seq":tokens.len(), "first_ms":first_ms,
            "steady_ms":times, "predictions":output}),
        );
    }
    Ok(
        json!({"load_ms":load_ms,"memory_after_load":after_load,"memory_after_first":after_first,
        "memory_after_steady":memory(),"cases":cases}),
    )
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args_os().collect();
    if args.len() != 4 {
        return Err("usage: knm_format_probe single|pipeline BUNDLE REQUEST.json".into());
    }
    let request: Value = serde_json::from_reader(std::fs::File::open(&args[3])?)?;
    let result = if args[1] == "single" {
        single(Path::new(&args[2]), &request)?
    } else if args[1] == "pipeline" {
        pipeline(Path::new(&args[2]), &request)?
    } else {
        return Err("unknown probe mode".into());
    };
    #[cfg(target_arch = "x86_64")]
    let avx2_fma = std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma");
    #[cfg(not(target_arch = "x86_64"))]
    let avx2_fma = false;
    println!(
        "{}",
        serde_json::to_string(
            &json!({"event":"result", "workers":rayon::current_num_threads(), "avx2_fma":avx2_fma, "measurements":result})
        )?
    );
    Ok(())
}
