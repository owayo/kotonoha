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
- Python バインディングは Python 3.9 以上。通常の開発・CI は Python 3.14 を使います
- テキストの形態素解析には hasami の `.hsd` 辞書が必要です。トークン列を直接渡す処理は辞書なしで使えます

ONNX 推論にはモデルと ONNX Runtime の共有ライブラリも必要です。`ORT_DYLIB_PATH` にライブラリのパスを指定します。CUDA を使う場合は対応する GPU・ドライバ・実行環境を用意してください。
GPU 版では ONNX Runtime が必要とする CUDA・cuDNN の共有ライブラリも必要です。
Windows で `cudnn64_9.dll` などが見つからない場合の確認方法は
[CUDA Execution Provider の公式ドキュメント](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements)を参照してください。

v66 系の各 ONNX セッションはノード内の実行を1スレッドに固定します。順番に呼ぶモデルごとに CPU のスレッドプールを作らず、短い発話でのスレッドの競合を抑えます。

## インストール

### Rust ライブラリ

利用するプロジェクトで追加します。

```bash
cargo add kotonoha --git https://github.com/owayo/kotonoha
```

### 記号付きの既知語を前処理から保護する

記号を算術の読みへ変えたり削除したりする前に、`normalizer::DictionaryNormalizer` を使えます。
記号を含む既知語だけを hasami 辞書の発音へ写し、未知語・数式・識別子内部・数字から始まる語は変更しません。
利用側が変換・削除する記号を `symbols` に指定します。指定外の記号語は変更せず、
括弧を含まない原形へ正規化された記号注記は、指定にかかわらず辞書の読みで保護します。
単なる顔文字の分類名などを読み上げへ展開しないため、括弧のままの原形は対象にしません。
外側の丸括弧・角括弧は残します。語の読みはコードに列挙せず、指定した `.hsd` に従います。
数字・字幅・句読点の正規化や、注記を省略するなどの演出は行いません。
`Engine::text_to_*` へは元のテキストを渡せるため、この処理は利用側の前処理が必要な場合だけ使います。

```python
from kotonoha import DictionaryNormalizer

normalizer = DictionaryNormalizer("ipadic-neologd-sudachi.hsd", symbols="+#")
text = normalizer.normalize("C++とC#")
```

ONNX モデルや ONNX Runtime は不要です。Rust では `DictionaryNormalizer::load(path, symbols)`、
メモリ上の辞書では `DictionaryNormalizer::from_dictionary(dictionary, symbols)` を使います。

### CLI

[Releases](https://github.com/owayo/kotonoha/releases) から OS・CPU に対応するアーカイブを取得し、
展開した `kotonoha`（Windows は `kotonoha.exe`）を PATH の通った場所に配置します。
`SHA256SUMS` で各添付の SHA-256 を検証できます。配布 CLI は ONNX 推論に対応しています。

| OS | CPU | 添付ファイル |
|---|---|---|
| Linux | x64 | `kotonoha-x86_64-unknown-linux-gnu.tar.gz` |
| Linux | ARM64 | `kotonoha-aarch64-unknown-linux-gnu.tar.gz` |
| macOS | Apple Silicon | `kotonoha-aarch64-apple-darwin.tar.gz` |
| macOS | Intel | `kotonoha-x86_64-apple-darwin.tar.gz` |
| Windows | x64 | `kotonoha-x86_64-pc-windows-msvc.zip` |

Cargo からインストールする場合は次を実行します。ONNX 推論も使う場合は `--features cuda` を付けます。

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

### モデルの取得・共有

hasami の辞書取得と同様に、モデルをプロジェクト外の共有ディレクトリへ取得できます。
v66_split1 に必要な 13 ONNX（約 245 MB）とアクセント辞書だけを取得します。
Git や Git LFS、ONNX Runtime はダウンロードには不要です。

```bash
kotonoha model download
kotonoha model path
kotonoha model download --dir /shared/kotonoha/models
```

既定の保存先は `~/.local/share/kotonoha/models`、Windows は
`%LOCALAPPDATA%\kotonoha\models` です。`XDG_DATA_HOME` を設定している場合は
`$XDG_DATA_HOME/kotonoha/models` を使います。`KOTONOHA_MODEL_BUNDLE` を設定すると、
取得先と推論時の参照先をまとめて変更できます。`--dir` はその取得の保存先だけを変更します。

```powershell
$env:KOTONOHA_MODEL_BUNDLE = 'C:\shared\kotonoha\models'
kotonoha model download
```

配布元 `owayo/kotonoha-models` は非公開のため、アクセス権のある GitHub 認証が必要です。
`GH_TOKEN`、`GITHUB_TOKEN`、`gh auth login` で保存した github.com の認証の順に使います。
取得元のコミット・サイズ・SHA-256 は固定し、検証が済んだファイルだけを配置します。
検証済みのファイルは再取得せず、途中で失敗した場合も再実行で残りを取得できます。
中身が異なる既存ファイルはエラーになり、`--force` を付けたときだけ置き換えます。
`--base-url URL` でミラーも使えます（配布元のファイル名を使用、GitHub 認証は送信しません）。

ONNX 推論を使う CLI は `cuda` feature を有効にしてインストールします。
`ORT_DYLIB_PATH` を指定すると、`analyze` と `label` は配置済みの共有モデルを使います。
`--model-bundle DIR` で明示したパスは環境変数・既定の共有先より優先します。
推論時にモデルを自動ダウンロードすることはありません。

```bash
cargo install --git https://github.com/owayo/kotonoha --locked --features cuda kotonoha-cli
kotonoha analyze '今日は良い天気です' --dict data/ipadic.hsd
kotonoha analyze '今日は良い天気です' --dict data/ipadic.hsd --model-bundle /shared/kotonoha/models
```

Python も同じ共有先を使います。`model_bundle`、`KOTONOHA_MODEL_BUNDLE` が未指定で、
legacy の `model_path` / `KOTONOHA_MODEL_PATH` もなければ、配置済みの共有モデルを検出します。
独自の保存先を使う場合は、戻り値を `model_bundle` に渡せます。

```python
import kotonoha

print(kotonoha.model_dir())
bundle = kotonoha.download_models()  # dir=..., force=True, base_url=... も指定可
engine = kotonoha.KotonohaEngine(model_bundle=bundle)
```

Rust から取得する場合は `download` feature を有効にし、
`kotonoha::model::download(None, Default::default())` を呼びます。
`default_model_dir()` / `installed_model_dir()` により同じ共有先を参照できます。

### 辞書の準備

hasami で作成した `.hsd` 辞書を `--dict` に指定できます。IPAdic から作る場合は、同梱のスクリプトを使います。

```bash
mise exec -- bash scripts/setup_dict.sh
```

生成先は `data/ipadic.hsd` です。アクセント辞書の CSV は `--accent-dict` で別途指定します。

hasami の配布辞書に加えたユーザ語彙・読み修正を使う場合は、修正後の `.hsd` を指定します。
hasami の依存更新だけでは、手元の `.hsd` に新しい語彙は追加されません。

### ONNX モデルの準備

[Releases](https://github.com/owayo/kotonoha/releases) の添付から `kotonoha-models-v66.tar.gz` と
`kotonoha-models-v66.tar.gz.sha256` を取得し、検証して展開します。

```bash
sha256sum --check kotonoha-models-v66.tar.gz.sha256
tar -xzf kotonoha-models-v66.tar.gz
```

展開先の `kotonoha-models/` を Python API の `model_bundle`、または環境変数
`KOTONOHA_MODEL_BUNDLE` に指定します。v66 系の推論に必要な 13 個の ONNX、
`data/accent_dict.csv` に JSUT 辞書を後勝ちでマージした `accent_dict.csv`、
元の `accent_dict_jsut.csv` を同梱しています。`models.json` には取得元コミットと各ファイルの
SHA-256 を記録します。ONNX Runtime の共有ライブラリは別途用意し、`ORT_DYLIB_PATH` に指定してください。

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

`mora::count_mora`、`NjdNode::mora_count`、`AccentPhrase::mora_count` と
`AccentPhrase::accent_type` は `usize` です。長い発音やトークン列でも、
モーラ数とラベル内の位置を保持します。

hasami からの活用型・活用形は `InputToken` と `NjdNode` に保持します。
意志・推量の長音化では辞書の活用形を優先し、基本形の「映ろう（うつろう）」や
「まよう（迷う）」の末尾の「う」を長音に変えません。活用情報がない入力では、
表層形と原形による判定を使います。

アクセント型の予測失敗を検出するには `engine.try_predict_accent_types(&tokens)` を使います。
推論エラーや予測結果の長さの不一致を `AccentPredictionError` として返します。
`predict_accent_types` と `tokens_to_*` は、失敗を標準エラーへ記録し、既存のアクセント型で処理を続けます。
予測器が未設定の場合、どちらの予測 API も辞書由来のアクセント型を返します。

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
`Engine` は `(lemma, reading)` の完全一致でアクセント型を取得します。
同じ見出し語でも読みが一致しなければ、別の読みのアクセントを流用せず、予測器による推定または既定値を使います。
アクセント辞書 CSV の `accent_type` は `u8` の整数、アクセント規則 CSV の規則名と数値は
有効な値を指定してください。不正な値は読み込み時にエラーになります。
結合規則 CSV の列不足や、品詞階層から計算する優先度が255を超える場合もエラーとして返します。

### CLI

```bash
kotonoha analyze '今日は良い天気です' --dict data/ipadic.hsd
kotonoha analyze '今日は良い天気です' --dict data/ipadic.hsd --format phone-tones
kotonoha tokenize '東京都に住む' --dict data/ipadic.hsd --format wakachi
kotonoha --help
```

`label` は解析済みトークンの JSON、`build-dict` は MeCab 辞書ソースからの構築、`train-crf` は CRF モデルの学習を扱います。各サブコマンドの引数は `kotonoha <サブコマンド> --help` で確認できます。
`train-crf` の入力は発話を空行で区切り、各行を4列または9列にしてください。
列数やアクセント型が不正な場合はモデルを作らずにエラーで終了します。
エポック数は1以上、学習率は有限の正数、L2 正則化係数は有限の非負数を指定してください。
学習率と L2 正則化係数の積は1未満が必要です。
CRF 重みファイルの非有限値・途中切れ・末尾の余分なデータも読み込み時にエラーになります。

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

`engine.predict_accent_types(tokens)` は、推論失敗や予測結果の長さの不一致で `RuntimeError` を返します。
評価処理で辞書値へのフォールバックをモデルの予測として数えないための仕様です。
戻り値は整数のリスト（`list[int]`）で、空入力では `[]` を返します。

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

`make` で全ターゲットを表示します。CI は Linux・macOS で `make setup` と `make ci`、Windows で同じ Rust/Python テストを実行し、配布する5ターゲットの CLI を ONNX 推論を有効にしてビルドします。モデルを必要とする推論の一致テストは `ORT_DYLIB_PATH` と `V66_BUNDLE_DIR` がある環境で実行します。

学習環境は通常の `make setup` に含めません。macOS は CPU 版 ONNX Runtime と PyPI の PyTorch、Linux・Windows は GPU 版 ONNX Runtime と CUDA 12.6 向け PyTorch を取得します。学習スクリプトごとの GPU 要件は [training/README.md](training/README.md) を確認してください。

### リリース

Actions の [Release](.github/workflows/release.yml) → **Run workflow** を既定ブランチから実行すると、
hasami と同じく次の処理をまとめて行います。

1. JST の `yy.m.counter` で採番します。同じ年月では連番を増やし、年月が変わると100から始めます。
2. Rust ワークスペースと Python パッケージの版・lockfile を更新し、コミットと `v<version>` タグを同時に push します。
3. 同じタグから5ターゲットの CLI をビルドし、モデル配布物を生成します。
4. バイナリ5個、`kotonoha-models-v66.tar.gz`、その `.sha256`、全添付用の `SHA256SUMS` を Release に添付して公開します。
5. 公開した添付の名前・サイズ・状態を確認し、再ダウンロードして全ファイルの SHA-256 を照合します。

モデル配布物は、そのタグの `.github/model-bundle.json` に固定した学習済み ONNX と辞書から作ります。
学習はリリース処理に含みません。入力の欠損・Git LFS ポインタ・SHA-256 不一致を検出すると失敗します。
CLI とモデルは並列に生成し、すべて成功してから Release を作成します。各ジョブの結果は Actions の Summary に残ります。

初回のみ、非公開の `owayo/kotonoha-models` に読み取り専用の deploy key を登録し、
その秘密鍵を kotonoha の Actions secret `KOTONOHA_MODELS_READ_KEY` に設定してください。
モデルを更新するときは `.github/model-bundle.json` のコミット・ファイル一覧・サイズ・SHA-256 も更新します。

Actions には、既定ブランチへのコミット・タグの push と Release 作成ができる `GITHUB_TOKEN` の権限が必要です。
ブランチ保護で bot の push を制限している場合は、リポジトリのルールで許可してください。

`dry_run=true` を選ぶと、任意のブランチで採番・メタデータ更新・タグの重複確認だけを試せます。
コミット・push・タグ・バイナリ・モデル配布物・Release は作りません。モデル取得用の秘密鍵も不要です。
通常の実行（`dry_run=false`、既定）は既定ブランチに限定しています。
ビルド・モデル生成・添付の検証が失敗した場合は、同じ実行の失敗したジョブを再実行できます。
添付を再実行するときは、同じタグの Release にある同名ファイルを置き換えます。

ローカルでも、Git LFS の実体を取得したモデルリポジトリから同じ配布物を作れます。

```bash
python scripts/package_models.py --models-dir ../kotonoha-models --source-revision "$(git rev-parse HEAD)"
python -m unittest discover -s scripts/tests -v
```

生成先は `target/model-release/` です。

## ライセンス

[MIT](LICENSE)
