//! kotonoha ベンチマーク
//!
//! モーラ計算、音素変換、アクセント推定、ラベル生成、韻律抽出の性能を測定する。

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use kotonoha::accent::AccentPhrase;
use kotonoha::njd::InputToken;
use std::hint::black_box;

/// ベンチマーク用の5トークン文を生成する
fn sample_tokens() -> Vec<InputToken> {
    vec![
        InputToken::new("今日", "名詞", "キョウ", "キョー"),
        InputToken::new("は", "助詞", "ワ", "ワ"),
        InputToken::new("良い", "形容詞", "ヨイ", "ヨイ"),
        InputToken::new("天気", "名詞", "テンキ", "テンキ"),
        InputToken::new("です", "助動詞", "デス", "デス"),
    ]
}

fn bench_mora_count(c: &mut Criterion) {
    let words = ["キョウ", "コンニチワ", "トーキョー", "ガッコー", "シャシン"];

    c.bench_function("mora/count_mora", |b| {
        b.iter(|| {
            for word in &words {
                black_box(kotonoha::mora::count_mora(word));
            }
        });
    });
}

fn bench_katakana_to_phonemes(c: &mut Criterion) {
    let words = [
        "カキクケコ",
        "シャシン",
        "コンニチワ",
        "トーキョー",
        "ガッコー",
    ];

    c.bench_function("phoneme/katakana_to_phonemes", |b| {
        b.iter(|| {
            for word in &words {
                black_box(kotonoha::phoneme::katakana_to_phonemes(word));
            }
        });
    });
}

fn bench_analyze_tokens(c: &mut Criterion) {
    let engine = kotonoha::Engine::with_default_rules();
    let tokens = sample_tokens();

    c.bench_function("engine/analyze", |b| {
        b.iter(|| {
            black_box(engine.analyze(black_box(&tokens)));
        });
    });
}

fn bench_estimate_accent(c: &mut Criterion) {
    let engine = kotonoha::Engine::with_default_rules();
    let tokens = sample_tokens();

    c.bench_function("engine/estimate_accent", |b| {
        b.iter(|| {
            let mut nodes = engine.analyze(&tokens);
            black_box(engine.estimate_accent(black_box(&mut nodes)));
        });
    });
}

fn bench_generate_labels(c: &mut Criterion) {
    let engine = kotonoha::Engine::with_default_rules();
    let tokens = sample_tokens();

    c.bench_function("engine/generate_labels", |b| {
        b.iter(|| {
            let mut nodes = engine.analyze(&tokens);
            let phrases = engine.estimate_accent(&mut nodes);
            black_box(engine.make_label(black_box(&nodes), black_box(&phrases)));
        });
    });
}

fn bench_label_scaling(c: &mut Criterion) {
    let engine = kotonoha::Engine::with_default_rules();
    let mut group = c.benchmark_group("label_scaling");
    for repetitions in [1, 20, 200] {
        let tokens: Vec<_> = (0..repetitions).flat_map(|_| sample_tokens()).collect();
        let mut nodes = engine.analyze(&tokens);
        let phrases = engine.estimate_accent(&mut nodes);
        group.bench_with_input(BenchmarkId::new("phrases", repetitions), &(), |b, ()| {
            b.iter(|| black_box(engine.make_label(black_box(&nodes), black_box(&phrases))));
        });

        // 同じ音素数でアクセント句の長さだけを変え、句単位の処理費用を比較する。
        let single_phrase = [AccentPhrase {
            nodes: (0..nodes.len()).collect(),
            accent_type: 0,
            mora_count: nodes.iter().map(|node| node.mora_count).sum(),
            is_interrogative: false,
        }];
        group.bench_with_input(BenchmarkId::new("single", repetitions), &(), |b, ()| {
            b.iter(|| black_box(engine.make_label(black_box(&nodes), black_box(&single_phrase))));
        });
    }
    group.finish();
}

fn bench_tokens_to_labels(c: &mut Criterion) {
    let engine = kotonoha::Engine::with_default_rules();
    let tokens = sample_tokens();

    c.bench_function("engine/tokens_to_labels", |b| {
        b.iter(|| {
            black_box(engine.tokens_to_labels(black_box(&tokens)));
        });
    });
}

fn bench_prosody_extraction(c: &mut Criterion) {
    let engine = kotonoha::Engine::with_default_rules();
    let tokens = sample_tokens();

    c.bench_function("engine/prosody_extraction", |b| {
        b.iter(|| {
            black_box(engine.tokens_to_prosody_symbols(black_box(&tokens)));
        });
    });
}

criterion_group!(mora_benches, bench_mora_count,);

criterion_group!(phoneme_benches, bench_katakana_to_phonemes,);

criterion_group!(
    engine_benches,
    bench_analyze_tokens,
    bench_estimate_accent,
    bench_generate_labels,
    bench_label_scaling,
    bench_tokens_to_labels,
    bench_prosody_extraction,
);

criterion_main!(mora_benches, phoneme_benches, engine_benches);
