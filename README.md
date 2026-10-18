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
- **アクセント推定**: アクセント句境界の検出と、ルール・CRF・専用ニューラル推論によるアクセント型の推定
- **ラベル生成**: HTS Full-Context Label の出力
- **韻律抽出**: PhoneTone、句読点を保持した PhoneTone、韻律記号列の出力
- **ニューラル推論**: `neural` feature による KNM モデルの専用 CPU 推論

## 動作環境

- ソースからのビルドには Rust 1.98 以上と、OS に対応する C/C++ ビルドツールが必要です
- 開発用ツールの管理には [mise](https://mise.jdx.dev/)。Rust・Python・uv の版は `mise.toml` に固定しています
- Python バインディングは Python 3.9 以上。通常の開発・CI は Python 3.14 を使います
- テキストの形態素解析には hasami の `.hsd` 辞書が必要です。トークン列を直接渡す処理は辞書なしで使えます

ニューラル推論は独自形式 **KNM** と専用 Rust CPU エンジンを使用します。ONNX Runtime、
`ORT_DYLIB_PATH`、CUDA/cuDNN の導入は不要です。`neural` feature で有効にします。
旧 `cuda` feature は `neural` の互換エイリアスであり、現在のエンジンは CPU 実装です。
AVX2/FMA を実行時に検出し、9 students と 2 meta models を Rayon で並列実行します。
スレッド数は `RAYON_NUM_THREADS` で調整できます。

重みは FP32 のまま可逆圧縮し、本番 13 モデルで約 245 MB → 204.5 MB に削減しています（辞書を除く、10進 MB）。

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
`SHA256SUMS` で各添付の SHA-256 を検証できます。KNM 対応版の配布 CLI は専用 CPU 推論を有効にしてビルドしています。

| OS | CPU | 添付ファイル |
|---|---|---|
| Linux | x64 | `kotonoha-x86_64-unknown-linux-gnu.tar.gz` |
| Linux | ARM64 | `kotonoha-aarch64-unknown-linux-gnu.tar.gz` |
| macOS | Apple Silicon | `kotonoha-aarch64-apple-darwin.tar.gz` |
| macOS | Intel | `kotonoha-x86_64-apple-darwin.tar.gz` |
| Windows | x64 | `kotonoha-x86_64-pc-windows-msvc.zip` |

Cargo から KNM 推論に対応する CLI をインストールする場合は次を実行します。

```bash
cargo install --git https://github.com/owayo/kotonoha --locked --features neural kotonoha-cli
```

ルール処理だけを使う場合は `--features neural` を省けます。モデルの取得はどちらのビルドでも使えます。

### ソースから

mise を用意してから実行します。

```bash
git clone https://github.com/owayo/kotonoha.git
cd kotonoha
make setup
make install
```

`make install` はルール処理用 CLI を `/usr/local/bin` に配置します。配置先は `make install INSTALL_PATH="$HOME/.local/bin"` で変更できます。
KNM 推論用の CLI は `make build-neural` で生成した `target/release/kotonoha` を使うか、上記の Cargo インストールを使います。
`make setup` は KNM 推論を有効にした Python バインディングも `kotonoha-python/.venv` にビルドします。
Makefile は POSIX シェル用です。Windows では Cargo のコマンドを直接実行できます。

### モデルの取得・共有

hasami の辞書取得と同様に、モデルをプロジェクト外の共有ディレクトリへ取得できます。
Kotonoha の最新安定版 Release から `kotonoha-models.tar.gz` を直接取得し、
13 KNM（約 204.5 MB）とアクセント辞書を配置します。ONNX の取得・変換は行いません。
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

呼び出すたびに最新 Release を確認します。同じアーカイブを配置済みで、全ファイルの整合性も
確認できた場合はアーカイブの再転送を省きます。正常な管理済み旧版は自動更新し、
手動で配置・変更したファイルの上書きは `--force` で指定します。
アーカイブの SHA-256、各ファイルのサイズ・SHA-256、KNM と辞書の読み込みを検証してから置き換えます。
通信・検証に失敗した場合は既存モデルを保護し、配置途中の失敗では元に戻します。

公開 Release の取得に認証は不要です。API の認証には `GH_TOKEN`、`GITHUB_TOKEN`、
`gh auth login` の順に利用可能な認証を使えます。添付の取得やミラーへ認証は送りません。
`--base-url URL` のミラーには `kotonoha-models.tar.gz` と同名の `.sha256` を配置します。
最新 Release に KNM がなければエラーを返します。既存の ONNX は `model convert` で通信なしに変換できます。

KNM 推論を使う CLI は `neural` feature を有効にしてインストールします。
`analyze` と `label` は配置済みの共有モデルを使います。
`--model-bundle DIR` で明示したパスは環境変数・既定の共有先より優先します。
推論時にモデルを自動ダウンロードすることはありません。

```bash
cargo install --git https://github.com/owayo/kotonoha --locked --features neural kotonoha-cli
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

テキスト解析には hasami の `.hsd` 辞書を `--dict`（Python は `dict_path`）へ指定します。
モデル Release に添付・同梱された `hasami-dictionaries.json` に、検証した hasami のタグ、
推奨辞書、各辞書のサイズ・SHA-256 を記録しています。
そのタグの [hasami Release](https://github.com/owayo/hasami/releases) から対応する `.hsd` を取得してください。
HSD 本体は KNM アーカイブには含まれず、`model download` でも取得しません。

ソースのチェックアウトから、記録済みの推奨 HSD を取得・検証する場合は次を使えます（Python 3.12 以上と GitHub CLI が必要）。

```bash
python scripts/hasami_release.py download --snapshot .github/hasami-dictionaries.json --output-dir target/hsd-release
```

IPAdic だけの辞書を自分で作る場合は、同梱のスクリプトを使います。

```bash
mise exec -- bash scripts/setup_dict.sh
```

生成先は `data/ipadic.hsd` です。アクセント辞書の CSV は `--accent-dict` で別途指定します。

hasami の配布辞書に加えたユーザ語彙・読み修正を使う場合は、修正後の `.hsd` を指定します。
hasami の依存更新だけでは、手元の `.hsd` に新しい語彙は追加されません。

### KNM モデルの準備

[Releases](https://github.com/owayo/kotonoha/releases) の添付から `kotonoha-models.tar.gz` と
`kotonoha-models.tar.gz.sha256` を取得し、検証して展開します。

```bash
sha256sum --check kotonoha-models.tar.gz.sha256
tar -xzf kotonoha-models.tar.gz
```

展開先の `kotonoha-models/` を Python API の `model_bundle`、または環境変数
`KOTONOHA_MODEL_BUNDLE` に指定します。推論に必要な 13 個の KNM、
`data/accent_dict.csv` に JSUT 辞書を後勝ちでマージした `accent_dict.csv`、
元の `accent_dict_jsut.csv` を同梱しています。`models.json` には取得元コミットと各ファイルの
サイズ・SHA-256、`hasami-dictionaries.json` には検証した HSD の情報を記録します。

KNM は `accent.knm`、`teacher.knm`、`student-01.knm`〜`student-09.knm`、
`meta-01.knm`・`meta-02.knm` の役割名で配置します。配布名に学習時の版番号は付けず、
モデルの版は Release タグ、取得元コミットと SHA-256 で識別します。
Python の `model_variant` は省略するか `"default"` を指定します。

固定版の既存 ONNX を通信なしで変換する場合は、次を実行します。元ファイルは保持されます。

```bash
kotonoha model convert --dir /path/to/model-source
```

変換元の固定値は `.github/model-bundle.json` と `kotonoha/src/model/manifest.rs` にあります。
`model convert` は重みと `knm-manifest.json` を生成する処理で、Release 用のアクセント辞書のマージは行いません。
配布と同じ辞書を含む構成は、後述の [ローカル梱包](#ローカルでのモデル梱包)で作れます。

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

アクセント辞書は配布モデルに同梱しています。追加の辞書も CSV で指定できます。
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

`dict_path` をコンストラクタへ渡すと、`text_to_labels()` などでテキストを直接解析できます。
KNM の取得・`model_bundle` の指定は [モデルの取得・共有](#モデルの取得共有)、
学習環境と過去の評価記録は [training/README.md](training/README.md) を参照してください。

`engine.predict_accent_types(tokens)` は、推論失敗や予測結果の長さの不一致で `RuntimeError` を返します。
評価処理で辞書値へのフォールバックをモデルの予測として数えないための仕様です。
戻り値は整数のリスト（`list[int]`）で、空入力では `[]` を返します。

## クレート構成

| クレート | 説明 |
|---|---|
| `kotonoha` | コアライブラリ |
| `kotonoha-cli` | CLI ツール (`kotonoha` コマンド) |
| `kotonoha-python` | Python バインディング (PyO3) |

## パフォーマンス

`cargo run --release --locked -p kotonoha-cli -- bench` で内部ベンチマークを実行できます。
固定の5トークンを渡し、ルールによるアクセント処理・ラベル生成・韻律抽出などを測定します。
HSD の形態素解析、KNM の読込・ニューラル推論は含みません。

KNM 移行時の測定では、Windows / Ryzen 7 5700X、モデル間8並列、実発話20件の中央値で、
ONNX Runtime の参照パイプライン32.32 ms/発話に対し、専用 Rust パイプラインは29.31 ms/発話でした。
特徴量処理を含む比較であり、単一モデル・逐次実行・長い入力では ONNX Runtime の方が速い場合があります。
すべての環境・入力での速度優位を示すものではありません。

## 開発

```bash
make setup
make ci
```

| コマンド | 内容 |
|---|---|
| `make build` / `make release` | ワークスペースのデバッグ版 / リリース版をビルド |
| `make build-neural` | 専用 CPU 推論を有効にしてリリース版をビルド |
| `make build-cuda` | `make build-neural` の互換エイリアス |
| `make run ARGS="--help"` | CLI の起動 |
| `make test` / `make test-python` | Rust と Python のテスト / Python API のテスト |
| `make lint` | 既定 feature と全 feature の clippy 検査 |
| `make fmt` / `make fmt-check` | 整形 / 書き換えずに整形を確認 |
| `make setup-training` | 学習・評価用 Python 依存を取得 |

`make` で全ターゲットを表示します。CI は Linux・macOS で `make setup` と `make ci`、Windows で同じ Rust/Python テストを実行し、配布する5ターゲットの CLI を専用ニューラル推論を有効にしてビルドします。モデルを必要とする一致テストは KNM を配置した `V66_BUNDLE_DIR` がある環境で実行します。

学習環境は通常の `make setup` に含めません。macOS は CPU 版 ONNX Runtime と PyPI の PyTorch、Linux・Windows は GPU 版 ONNX Runtime と CUDA 12.6 向け PyTorch を取得します。学習スクリプトごとの GPU 要件は [training/README.md](training/README.md) を確認してください。

### リリース

Actions の [Release](.github/workflows/release.yml) → **Run workflow** を既定ブランチから実行すると、
hasami と同じく次の処理をまとめて行います。

1. JST の `yy.m.counter` で採番します。同じ年月では連番を増やし、年月が変わると100から始めます。
2. Rust ワークスペースと Python パッケージの版・lockfile を更新し、コミットと `v<version>` タグを同時に push します。
3. 同じタグから5ターゲットの CLI をビルドします。
4. 固定版の学習済み ONNX から 13 個の KNM を毎回ビルドし、辞書とマニフェストを `kotonoha-models.tar.gz` に梱包します。
5. バイナリ5個、`kotonoha-models.tar.gz`、その `.sha256`、検証に使った HSD の記録 `hasami-dictionaries.json`、全添付用の `SHA256SUMS` を Release に添付して公開します。
6. 公開した添付の名前・サイズ・状態を確認し、再ダウンロードして全ファイルの SHA-256 を照合します。

モデル配布物は、そのタグの `.github/model-bundle.json` に固定した学習済み ONNX を Rust で KNM へ変換し、辞書と一緒に梱包します。
学習はリリース処理に含みません。入力の欠損・Git LFS ポインタ・SHA-256 不一致を検出すると失敗します。
CLI とモデルは並列に生成し、すべて成功してから Release を作成します。各ジョブの結果は Actions の Summary に残ります。

#### HSD 更新に合わせた自動 Release

hasami の [Release ワークフロー](https://github.com/owayo/hasami/blob/main/.github/workflows/release.yml)は、
HSD の公開と添付の検証に成功した直後、そのタグを指定して Kotonoha の [Release](.github/workflows/release.yml) を起動します。
毎時チェックは行いません。通知では `hasami_tag` と `check_hsd_changes=true` を指定します。
Kotonoha は Release の実行枠を取得してから、指定タグの3辞書のサイズ・SHA-256 を
最後に公開できたモデルの `hasami-dictionaries.json` と比較します。

新しい HSD を固定タグから取得して検証し、Kotonoha の専用 KNM 推論と組み合わせて解析を実行します。
成功した場合だけモデルを梱包・公開します。HSD の記録はアーカイブ内にも含めます。
HSD 本体は該当する hasami Release から取得します。KNM は固定した学習済み重みから再ビルドし、再学習は行いません。

同じ HSD の重複通知や、公開済みモデルが使う HSD より古い通知では Release を増やしません。
実行中の Release があればキューで待機します（最大100件）。待機後の既定ブランチから採番するため、前の Release の版更新も取り込みます。
初回は公開済みモデルに HSD の記録がない場合もビルドします。失敗した生成は処理済みとせず、Actions の失敗ジョブを再実行できます。
自動の定期再試行はありません。通常の手動 Release は `check_hsd_changes=false`（既定）のままなら毎回ビルドし、
`hasami_tag` を省くと最新安定版を使います。`dry_run=true` は従来どおり採番・メタデータ更新の確認だけです。

リポジトリをまたぐ通知用に、初回だけ次を設定します。

1. 専用 GitHub App に **Actions: Read and write** だけを許可し、**`owayo/kotonoha` のみ**にインストールします。
2. **hasami 側**の Actions variable `KOTONOHA_APP_CLIENT_ID` に Client ID、Actions secret `KOTONOHA_APP_PRIVATE_KEY` に秘密鍵を登録します。
3. Kotonoha の受信側を先に既定ブランチへ反映し、App の設定後に hasami の通知側を反映します。

同じ所有者でも、hasami の `GITHUB_TOKEN` は Kotonoha の workflow を起動できないため、この認証が必要です。
[GitHub の workflow dispatch 権限](https://docs.github.com/en/rest/actions/workflows#create-a-workflow-dispatch-event)を参照してください。
通知ジョブで Kotonoha の Actions だけに絞った短命の installation token を生成し、ジョブ終了時に失効させます。
App の設定不備や通知失敗は hasami の `notify-kotonoha` ジョブを失敗にします。
HSD の公開は完了済みなので、設定を直してその失敗ジョブだけ再実行すれば再ビルド通知を送れます。

Release のモデル入力とその取得設定はリリース環境の管理者が用意します。
モデルを更新するときは `.github/model-bundle.json` のコミット・ファイル一覧・サイズ・SHA-256 も更新します。

Actions には、既定ブランチへのコミット・タグの push と Release 作成ができる `GITHUB_TOKEN` の権限が必要です。
ブランチ保護で bot の push を制限している場合は、リポジトリのルールで許可してください。

`dry_run=true` を選ぶと、任意のブランチで採番・メタデータ更新・タグの重複確認だけを試せます。
コミット・push・タグ・バイナリ・モデル配布物・Release は作りません。モデル取得用の秘密鍵も不要です。
通常の実行（`dry_run=false`、既定）は既定ブランチに限定しています。
ビルド・モデル生成・添付の検証が失敗した場合は、同じ実行の失敗したジョブを再実行できます。
添付を再実行するときは、同じタグの Release にある同名ファイルを置き換えます。

#### ローカルでのモデル梱包

固定版のモデル入力を `/path/to/model-source` に配置し、Python 3.12 以上を用意して Kotonoha リポジトリで実行します。
固定版の ONNX から KNM を生成してから、辞書と HSD の記録を梱包します。

```bash
cargo run --release --locked -p kotonoha-cli --features neural -- model convert --dir /path/to/model-source --force
python scripts/package_models.py --native --models-dir /path/to/model-source --source-revision "$(git rev-parse HEAD)" --hasami-release .github/hasami-dictionaries.json
python -m unittest discover -s scripts/tests -v
```

Windows の PowerShell では `$(git rev-parse HEAD)` をダブルクォートの中でそのまま使えます。
生成先は `target/model-release/` で、アーカイブ・`.sha256`・`hasami-dictionaries.json` を出力します。
梱包スクリプト自体は HSD を取得・検証しません。Release ワークフローでは別途、記録した推奨 HSD と
生成した KNM を使う結合テストと CLI 解析を通してから梱包します。

## ライセンス

[MIT](LICENSE)
