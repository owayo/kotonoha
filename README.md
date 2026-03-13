# kotonoha

Rust 製日本語韻律エンジン。OpenJTalk の機能を置き換え、HTS Full-Context Label および PhoneTone を生成する。

## 機能

- 形態素解析（[hasami](https://github.com/owayo/hasami) 連携）
- アクセント句境界検出・アクセント型推定（ルールベース + CRF + ONNX）
- HTS Full-Context Label 生成
- PhoneTone / 韻律記号列の抽出
- CUDA 対応ニューラルアクセント予測（optional）

## クレート構成

| クレート | 説明 |
|---------|------|
| `kotonoha` | コアライブラリ |
| `kotonoha-cli` | CLI ツール (`kotonoha` コマンド) |
| `kotonoha-python` | Python バインディング (PyO3) |

## 関連リポジトリ

| リポジトリ | 説明 |
|-----------|------|
| [kotonoha-models](https://github.com/owayo/kotonoha-models) | ONNX アクセント予測モデル (Git LFS) |
| [kotonoha-training-data](https://github.com/owayo/kotonoha-training-data) | 訓練データ・辞書・LLM データ生成ツール |
| [hasami](https://github.com/owayo/hasami) | 形態素解析エンジン |

## 使い方

```rust
use kotonoha::Engine;
use kotonoha::njd::InputToken;
use std::path::Path;

let mut engine = Engine::with_default_rules();

// アクセント辞書を読み込み（kotonoha-training-data/data/dicts/）
engine.load_accent_dict(Path::new("accent_dict.csv"))?;

// 形態素解析辞書を読み込み（hasami）
engine.load_dictionary(Path::new("ipadic.hsd"))?;

// テキストから HTS ラベルを生成
let labels = engine.text_to_labels("今日は良い天気です")?;
```

## ビルド

```bash
cargo build --release
```

CUDA 対応（ONNX アクセント予測）を有効にする場合:

```bash
cargo build --release --features cuda
```

## パフォーマンス

| パイプライン | 処理時間 (5トークン文) |
|------------|---------------------|
| フルパイプライン (analyze → accent → label) | ~25μs |
| 韻律抽出 | ~5μs |
| スループット | ~40,000 文/秒 |

## ライセンス

MIT
