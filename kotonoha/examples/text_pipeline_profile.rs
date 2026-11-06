//! 実際の HSD でテキストから韻律出力までを段階別に計測する。KNM の推論も任意で通す。
//!
//! ```bash
//! cargo run --release --locked -p kotonoha --features neural --example text_pipeline_profile -- \
//!   DICT.hsd CORPUS.txt [MODEL_DIR]
//! ```
//!
//! CORPUS は 1 行 1 文の UTF-8。各段階の時間は全行の合計で、最初の 1 周をウォームアップとして捨て、
//! `KOTONOHA_PROFILE_ROUNDS` 周 (既定 5) の生データと中央値を JSON で出す。
//! `digest` は正規化結果・トークン・ラベル・PhoneTone・韻律記号を区切り付きで FNV-1a に通した値で、
//! 変更前後の出力が一致するかの照合に使う。時間は壁時計であり、ホストの負荷を受ける。

use std::path::Path;
use std::time::{Duration, Instant};

use kotonoha::Engine;
use kotonoha::njd::InputToken;
use kotonoha::normalizer::DictionaryNormalizer;

const STAGES: [&str; 8] = [
    "normalize",
    "tokenize",
    "njd",
    "predict",
    "accent",
    "label",
    "phone_tones",
    "prosody",
];
/// 正規化で照合する記号付き語の記号。計測用の既定値で、利用側の設定とは無関係。
const NORMALIZER_SYMBOLS: &str = "+#&";

struct Digest(u64);

impl Digest {
    fn new() -> Self {
        Self(0xcbf29ce484222325)
    }

    fn push(&mut self, text: &str) {
        for byte in text.bytes().chain([0]) {
            self.0 = (self.0 ^ u64::from(byte)).wrapping_mul(0x100000001b3);
        }
    }
}

#[cfg(feature = "neural")]
type Predictor = kotonoha::nn::v66::V66Pipeline;
#[cfg(not(feature = "neural"))]
type Predictor = ();

fn load_predictor(dir: Option<&std::ffi::OsString>) -> Result<Option<Predictor>, String> {
    let Some(dir) = dir else {
        return Ok(None);
    };
    #[cfg(feature = "neural")]
    {
        let bundle = kotonoha::nn::v66::V66Bundle::from_dir(Path::new(dir))
            .map_err(|e| format!("model bundle: {e}"))?;
        Ok(Some(kotonoha::nn::v66::V66Pipeline::new(bundle)))
    }
    #[cfg(not(feature = "neural"))]
    {
        let _ = dir;
        Err("MODEL_DIR requires --features neural".into())
    }
}

/// Engine の文脈付き予測と同じく、全トークンの予測が揃ったときだけアクセント型を書き換える。
fn apply_predictor(
    predictor: Option<&Predictor>,
    tokens: &[InputToken],
    nodes: &mut [kotonoha::njd::NjdNode],
) -> Result<bool, Box<dyn std::error::Error>> {
    let Some(predictor) = predictor else {
        return Ok(false);
    };
    #[cfg(feature = "neural")]
    {
        let context: Vec<_> = tokens
            .iter()
            .zip(nodes.iter())
            .map(|(token, node)| kotonoha::nn::FeatureMorpheme { token, node })
            .collect();
        let accents = predictor.predict(&context)?;
        if accents.len() != nodes.len() {
            return Err("prediction length mismatch".into());
        }
        for (node, accent) in nodes.iter_mut().zip(accents) {
            node.accent_type = accent;
        }
        Ok(true)
    }
    #[cfg(not(feature = "neural"))]
    {
        let _ = (predictor, tokens, nodes);
        Ok(false)
    }
}

fn milliseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if !(2..=3).contains(&args.len()) {
        return Err("usage: text_pipeline_profile DICT.hsd CORPUS.txt [MODEL_DIR]".into());
    }
    let rounds: usize = std::env::var("KOTONOHA_PROFILE_ROUNDS")
        .ok()
        .map(|value| value.parse())
        .transpose()?
        .unwrap_or(5);
    if rounds == 0 {
        return Err("KOTONOHA_PROFILE_ROUNDS must be positive".into());
    }
    let dictionary = Path::new(&args[0]);
    let corpus = std::fs::read_to_string(&args[1])?;
    let lines: Vec<&str> = corpus
        .lines()
        .filter(|line| !line.trim().is_empty())
        .collect();

    let started = Instant::now();
    let mut analyzer = hasami::Analyzer::load(dictionary)?;
    let mut normalizer = DictionaryNormalizer::load(dictionary, NORMALIZER_SYMBOLS)?;
    let predictor = load_predictor(args.get(2))?;
    let load_ms = milliseconds(started.elapsed());
    let engine = Engine::with_default_rules();

    let mut samples: Vec<[f64; STAGES.len()]> = Vec::with_capacity(rounds);
    let mut expected_digest = None;
    let mut token_count = 0;
    for round in 0..=rounds {
        let mut elapsed = [Duration::ZERO; STAGES.len()];
        let mut digest = Digest::new();
        token_count = 0;
        for line in &lines {
            let started = Instant::now();
            let normalized = normalizer.normalize(line)?;
            elapsed[0] += started.elapsed();

            let started = Instant::now();
            let tokens: Vec<InputToken> = analyzer
                .try_tokenize(line)?
                .into_iter()
                .map(InputToken::from)
                .collect();
            elapsed[1] += started.elapsed();

            let started = Instant::now();
            let mut nodes = engine.analyze(&tokens);
            elapsed[2] += started.elapsed();

            let started = Instant::now();
            if apply_predictor(predictor.as_ref(), &tokens, &mut nodes)? {
                elapsed[3] += started.elapsed();
            }

            let started = Instant::now();
            let phrases = engine.estimate_accent(&mut nodes);
            elapsed[4] += started.elapsed();

            let started = Instant::now();
            let labels = engine.make_label(&nodes, &phrases);
            elapsed[5] += started.elapsed();

            let started = Instant::now();
            let tones = engine.extract_phone_tones_with_punct(&nodes, &phrases);
            elapsed[6] += started.elapsed();

            let started = Instant::now();
            let symbols = engine.extract_prosody_symbols(&nodes, &phrases);
            elapsed[7] += started.elapsed();

            token_count += tokens.len();
            digest.push(&normalized);
            for token in &tokens {
                digest.push(&format!("{token:?}"));
            }
            for label in &labels {
                digest.push(label);
            }
            for tone in &tones {
                digest.push(&tone.phone);
                digest.push(&tone.tone.to_string());
            }
            digest.push(&symbols.concat());
        }
        match expected_digest {
            None => expected_digest = Some(digest.0),
            Some(expected) if expected != digest.0 => {
                return Err(format!("round {round} changed the output digest").into());
            }
            Some(_) => {}
        }
        // 1 周目はページの読み込みや遅延初期化を含むので捨てる。
        if round > 0 {
            samples.push(elapsed.map(milliseconds));
        }
    }

    let mut stages = serde_json::Map::new();
    let mut total_samples: Vec<f64> = samples.iter().map(|round| round.iter().sum()).collect();
    for (index, name) in STAGES.iter().enumerate() {
        let mut values: Vec<f64> = samples.iter().map(|round| round[index]).collect();
        let raw = values.clone();
        stages.insert(
            (*name).into(),
            serde_json::json!({ "median_ms": median(&mut values), "samples_ms": raw }),
        );
    }
    let raw_total = total_samples.clone();
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "lines": lines.len(),
            "tokens": token_count,
            "rounds": rounds,
            "predictor": predictor.is_some(),
            "worker_threads": rayon::current_num_threads(),
            "load_ms": load_ms,
            "total_median_ms": median(&mut total_samples),
            "total_samples_ms": raw_total,
            "stages": stages,
            "digest": format!("{:016x}", expected_digest.unwrap_or_default()),
        }))?
    );
    Ok(())
}
