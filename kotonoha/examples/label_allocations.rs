//! ラベル生成1回のアロケーション数・要求バイト数と出力の一致を調べる。
//! `cargo run --release -p kotonoha --example label_allocations` で実行する。

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use kotonoha::Engine;
use kotonoha::accent::AccentPhrase;
use kotonoha::njd::InputToken;

struct CountingAllocator;

static COUNTING: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);

// SAFETY: 同じレイアウト・ポインタをそのまま System へ渡す。カウンタはメモリに触れない。
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        }
        // SAFETY: GlobalAlloc の呼び出し元が保証する有効なレイアウトを転送する。
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: System が確保したポインタを、元のレイアウトで解放する。
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            BYTES.fetch_add(new_size, Ordering::Relaxed);
        }
        // SAFETY: ポインタ・元のレイアウト・新サイズは呼び出し元が保証する。
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn main() {
    let engine = Engine::with_default_rules();
    let sentence = [
        InputToken::new("今日", "名詞", "キョウ", "キョー"),
        InputToken::new("は", "助詞", "ワ", "ワ"),
        InputToken::new("良い", "形容詞", "ヨイ", "ヨイ"),
        InputToken::new("天気", "名詞", "テンキ", "テンキ"),
        InputToken::new("です", "助動詞", "デス", "デス"),
    ];
    let mut results = Vec::new();
    for repetitions in [1, 20, 200] {
        let tokens: Vec<_> = (0..repetitions)
            .flat_map(|_| sentence.iter().cloned())
            .collect();
        let mut nodes = engine.analyze(&tokens);
        let phrases = engine.estimate_accent(&mut nodes);
        let single = vec![AccentPhrase {
            nodes: (0..nodes.len()).collect(),
            accent_type: 0,
            mora_count: nodes.iter().map(|node| node.mora_count).sum(),
            is_interrogative: false,
        }];
        for (kind, phrases) in [("phrases", phrases), ("single", single)] {
            // 初期化を済ませてから、ラベル生成中だけを数える。
            drop(engine.make_label(&nodes, &phrases));
            ALLOCATIONS.store(0, Ordering::Relaxed);
            BYTES.store(0, Ordering::Relaxed);
            COUNTING.store(true, Ordering::Relaxed);
            let labels = engine.make_label(&nodes, &phrases);
            COUNTING.store(false, Ordering::Relaxed);
            // 区切りを含む FNV-1a。変更前後で全ラベルのバイト列を照合する。
            let digest = labels.iter().fold(0xcbf29ce484222325u64, |hash, label| {
                label.bytes().chain([0]).fold(hash, |hash, byte| {
                    (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3)
                })
            });
            results.push(serde_json::json!({
                "case": format!("{kind}/{repetitions}"),
                "labels": labels.len(),
                "digest": format!("{digest:016x}"),
                "allocations": ALLOCATIONS.load(Ordering::Relaxed),
                "requested_bytes": BYTES.load(Ordering::Relaxed),
            }));
        }
    }
    println!("{}", serde_json::to_string_pretty(&results).unwrap());
}
