<p align="center">
  <img src="docs/images/app.png" width="128" alt="kotonoha">
</p>

<h1 align="center">kotonoha</h1>

<p align="center">
  HTS Full-Context Label と PhoneTone を生成する Rust 製日本語韻律エンジン
</p>

<p align="center">
  <a href="https://github.com/owayo/kotonoha/actions/workflows/ci.yml"><img src="https://github.com/owayo/kotonoha/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/owayo/kotonoha" alt="License"></a>
</p>

---

kotonoha は OpenJTalk の韻律処理を Rust で実装したライブラリです。日本語のテキストや形態素トークン列から、音声合成に使うラベル・音素とトーン・韻律記号を生成します。CLI と Python バインディングも同梱しています。

## 機能

- **形態素解析**: [hasami](https://github.com/owayo/hasami) の辞書を使ったテキスト解析
- **アクセント推定**: アクセント句境界の検出と、ルール・CRF・ONNX によるアクセント型の推定
- **ラベル生成**: HTS Full-Context Label の出力
- **韻律抽出**: PhoneTone、句読点を保持した PhoneTone、韻律記号列の出力
- **ニューラル推論**: オプションの `cuda` feature による ONNX モデルの利用

## 動作環境

- Rust 1.98 以上と、OS に対応する C/C++ ビルドツール
- 開発用ツールの管理には [mise](https://mise.jdx.dev/)。Rust・Python・uv の版は `mise.toml` に固定しています
- Python バインディングは Python 3.9 以上。通常の開発・CI は Python 3.12 を使います
- テキストの形態素解析には hasami の `.hsd` 辞書が必要です。トークン列を直接渡す処理は辞書なしで使えます

ONNX 推論にはモデルと ONNX Runtime の共有ライブラリも必要です。`ORT_DYLIB_PATH` にライブラリのパスを指定します。CUDA を使う場合は対応する GPU・ドライバ・実行環境を用意してください。

## インストール

### Rust ライブラリ

利用するプロジェクトで追加します。

```bash
cargo add kotonoha --git https://github.com/owayo/kotonoha
```

### CLI

```bash
cargo install --git https://github.com/owayo/kotonoha --locked kotonoha-cli
```

### ソースから

mise を用意してから実行します。

```bash
git clone https://github.com/owayo/kotonoha.git
cd kotonoha
make setup
make install
```

`make install` は CLI を `/usr/local/bin` に配置します。配置先は `make install INSTALL_PATH="$HOME/.local/bin"` で変更できます。`make setup` は Python バインディングも `kotonoha-python/.venv` にビルドします。

### 辞書の準備

hasami で作成した `.hsd` 辞書を `--dict` に指定できます。IPAdic から作る場合は、同梱のスクリプトを使います。

```bash
mise exec -- bash scripts/setup_dict.sh
```

生成先は `data/ipadic.hsd` です。アクセント辞書の CSV は `--accent-dict` で別途指定します。

## 使い方

### Rust API

トークン列を直接渡す例です。

```rust
use kotonoha::Engine;
use kotonoha::njd::InputToken;

let engine = Engine::with_default_rules();
let tokens = vec![InputToken::new("猫", "名詞", "ネコ", "ネコ")];
let labels = engine.tokens_to_labels(&tokens);
let phone_tones = engine.tokens_to_phone_tones(&tokens);
```

テキストを解析する場合は、先に辞書を読み込みます。

```rust
use kotonoha::Engine;
use std::path::Path;

let mut engine = Engine::with_default_rules();
engine.load_accent_dict(Path::new("accent_dict.csv"))?;
engine.load_dictionary(Path::new("ipadic.hsd"))?;
let labels = engine.text_to_labels("今日は良い天気です")?;
```

アクセント辞書は [kotonoha-training-data](https://github.com/owayo/kotonoha-training-data) の `data/dicts/` でも管理しています。

### CLI

```bash
kotonoha analyze '今日は良い天気です' --dict data/ipadic.hsd
kotonoha analyze '今日は良い天気です' --dict data/ipadic.hsd --format phone-tones
kotonoha tokenize '東京都に住む' --dict data/ipadic.hsd --format wakachi
kotonoha --help
```

`label` は解析済みトークンの JSON、`build-dict` は MeCab 辞書ソースからの構築、`train-crf` は CRF モデルの学習を扱います。各サブコマンドの引数は `kotonoha <サブコマンド> --help` で確認できます。

### Python API

`make setup` 後は `mise exec -- uv run --locked --project kotonoha-python python` から利用できます。トークンには属性を持つオブジェクトを渡します。

```python
from types import SimpleNamespace
import kotonoha

engine = kotonoha.KotonohaEngine()
tokens = [SimpleNamespace(surface="猫", pos="名詞", reading="ネコ")]
labels = engine.make_label(tokens)
phone_tones = engine.phone_tones(tokens)
```

`dict_path` をコンストラクタへ渡すと、`text_to_labels()` などでテキストを直接解析できます。`model_bundle` による ONNX 推論や学習・評価の手順は [training/README.md](training/README.md) を参照してください。

## クレート構成

| クレート | 説明 |
|---|---|
| `kotonoha` | コアライブラリ |
| `kotonoha-cli` | CLI ツール (`kotonoha` コマンド) |
| `kotonoha-python` | Python バインディング (PyO3) |

## 関連リポジトリ

| リポジトリ | 説明 |
|---|---|
| [kotonoha-models](https://github.com/owayo/kotonoha-models) | ONNX アクセント予測モデル (Git LFS) |
| [kotonoha-training-data](https://github.com/owayo/kotonoha-training-data) | 訓練データ・辞書・LLM データ生成ツール |
| [hasami](https://github.com/owayo/hasami) | 形態素解析エンジン |

## パフォーマンス

5 トークン文の計測値です。モデルや辞書、実行環境によって処理時間は変わります。

| パイプライン | 処理時間 (5 トークン文) |
|---|---|
| フルパイプライン (analyze → accent → label) | 約 25 μs |
| 韻律抽出 | 約 5 μs |
| スループット | 約 40,000 文/秒 |

`make run ARGS="bench"` で内部ベンチマークを実行できます。

## 開発

```bash
make setup
make ci
```

| コマンド | 内容 |
|---|---|
| `make build` / `make release` | ワークスペースのデバッグ版 / リリース版をビルド |
| `make build-cuda` | ONNX 推論を有効にしてリリース版をビルド |
| `make run ARGS="--help"` | CLI の起動 |
| `make test` / `make test-python` | Rust と Python のテスト / Python API のテスト |
| `make lint` | 既定 feature と全 feature の clippy 検査 |
| `make fmt` / `make fmt-check` | 整形 / 書き換えずに整形を確認 |
| `make setup-training` | 学習・評価用 Python 依存を取得 |

`make` で全ターゲットを表示します。CI は Linux・macOS で `make setup` と `make ci`、Windows で同じ Rust/Python テストとビルドを実行します。モデルを必要とする推論の一致テストは `ORT_DYLIB_PATH` と `V66_BUNDLE_DIR` がある環境で実行します。

学習環境は通常の `make setup` に含めません。macOS は CPU 版 ONNX Runtime と PyPI の PyTorch、Linux・Windows は GPU 版 ONNX Runtime と CUDA 12.6 向け PyTorch を取得します。学習スクリプトごとの GPU 要件は [training/README.md](training/README.md) を確認してください。

## ライセンス

[MIT](LICENSE)
