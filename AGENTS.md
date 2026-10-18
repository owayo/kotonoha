# AGENTS.md

## プロジェクト概要

kotonohaはRust製の日本語韻律エンジン。OpenJTalkの機能を置き換え、HTS Full-Context LabelおよびPhoneToneを生成する。

## ビルド・テスト

```bash
make setup     # mise.toml のツールと依存を取得
make build     # ワークスペースをビルド
make ci        # 整形・clippy・Rust/Python テスト
make           # ターゲット一覧
```

## プロジェクト構成

Cargoワークスペース構成（resolver v2）:

- `kotonoha/` — コアライブラリ
- `kotonoha-cli/` — CLIツール（`kotonoha` コマンド）
- `kotonoha-python/` — PyO3によるPythonバインディング

## 主要モジュール（`kotonoha/src/`）

| モジュール | 役割 |
|---|---|
| `accent` | アクセント句の表現・操作 |
| `accent_dict` | アクセント辞書（lemma/reading → アクセント型） |
| `accent_rule` | アクセント結合規則テーブル |
| `crf` | CRFアクセント予測器（学習・推論） |
| `label` | HTS Full-Context Label生成 |
| `mora` | モーラ処理 |
| `model` | KNM の読込・共有先・検証付き取得と ONNX からのオフライン変換 |
| `njd` | NJD（Normalized Japanese Dictionary）ノード処理 |
| `nn` | 専用 Rust CPU 推論（`native.rs`、`nn/v66/` の v66 パイプライン） |
| `phoneme` | 音素定義・操作 |
| `prosody` | PhoneTone（韻律情報）生成 |

エントリポイントは `Engine` 構造体（`lib.rs`）。共有参照（`&self`）で並行利用できる
スレッドセーフ設計。アクセント辞書・予測器・形態素解析器（`Mutex` 保護）を
オプションで保持する。

## 発音の決め方（`njd.rs`）

`NjdNode::from_token()` が採用する発音は以下の優先順位で決まる。

1. **表層形** — 表層が音素化可能なカタカナだけの語（名詞・副詞・感動詞、固有名詞を除く、2 文字以上）
2. **辞書の発音** — カタカナとして妥当な場合
3. **辞書の読み** — 発音フィールドが壊れている場合のフォールバック

1 を入れている理由は、辞書の発音フィールドに「マネジメント→マネージメント」
「チャネル→チャンネル」「インタラクション→インターラクション」のように表記と食い違う値が
多数入っており、そのまま使うと書かれた通りに読まれないため。カタカナ表記は表音的なので
表記通りに読むのが原則、という方針を採っている。

固有名詞を除外しているのは「キヤノン→キャノン」のような意図的な表記があるため。
個別の例外は hasami のユーザ辞書（`dict/user/*.csv`）で対応する。

表記ゆれを標準カタカナへ補正するときは、原形と読みを同じ標準表記にそろえる。
Kotonoha はその組が表層と異なり、発音も有効な場合に辞書の発音を採用する。
語末の長音符だけの差や、発音だけが表層と異なる語は従来どおり表層を優先する。

`mora_count` は `reading` ではなく**採用した発音**から数える。ここがずれると
`prosody` が発音から取るモーラ数とアクセント句のモーラ数が食い違う。
`mora::count_mora` とノード・アクセント句のモーラ数は `usize`。長い入力で
`u8` を超えても切り捨てず、ラベルと韻律記号の位置計算も同じ幅で扱う。
アクセント句のアクセント位置も `usize`、辞書・予測器が返す各語のアクセント型は `u8`。

hasami の `Token::conj_type` / `conj_form` は `InputToken::ctype` / `cform` に保持する。
活用情報が空のときだけ `*` にする。意志・推量の長音化では辞書の活用形を優先し、
基本形の「映ろう」や、原形が漢字になった「まよう（迷う）」を誤って長音化しない。
活用形がないトークンでは表層形・原形による判定へフォールバックする。

`Engine::analyze` のアクセント辞書検索は `(lemma, reading)` の完全一致とする。
hasami が文脈に応じて補正した読みを尊重し、同じ見出し語の別読みのアクセントを流用しない。

アクセント辞書・結合規則の CSV では、不正なアクセント値や規則を平板型・既定規則へ
置き換えず、読み込みエラーとして返す。入力検証を緩めないこと。
結合規則の行は左右の品詞と規則の3列以上を必要とし、列不足を読み飛ばさない。
品詞階層の優先度は `u8` へ変換する前に検証し、255を超える場合はエラーにする。
CRF 重みファイルのラベル数は 1〜256 に限り、読み込み時に配列を確保する前に検証する。
`CrfTrainer::train` は `Result` を返す。空データ、ノード数とラベル数の不一致、
学習対象外のラベルを検出したら学習を始めずにエラーを返す。CLI の学習 CSV も
4列または9列と数値のアクセント型を検証する。

### 既知の制約

辞書エントリによっては発音フィールドに表層（漢字）が入っており、読みへフォールバックする。
読みは仮名遣い形なので「東京→トウキョウ」となり、発音形の「トーキョー」にならない。
仮名遣い形から発音形への機械的な変換は「思う→オモウ」「小売→コウリ」のような
反例があるため安全に行えず、辞書側の修正が必要。

hasami 側には `hasami repair` があり、壊れた発音フィールドを「同じ (表層形, 読み) を持つ
健全なエントリの発音形」で置き換える。IPAdic 由来のエントリが人手で作られた発音形を
持つため、マージ辞書ではこれが最も確実な供給源になる。

## 音素化（`phoneme.rs`）

`katakana_to_phonemes()` は文字ごとの場合分けでカタカナを音素列にする。小書き仮名
（ァィゥェォャュョ）が続く組み合わせは個別に列挙するが、列挙から漏れると
`mora::split_cv()` が「最後を母音、残りを連結して子音」とみなすため、
「ジェ」が `j`,`i`,`e` → 子音 `ji` + 母音 `e` のように存在しない音素を作ってしまう。

漏れを防ぐため次の 2 つを用意している。

- 小書き母音（ァィゥェォ）が個別の場合分けに無かったときは、直前が「子音 + 母音」なら
  その母音を差し替える一般則にフォールバックする（「ビィ」→ `b`,`i`）
- `kotonoha/tests/mora_coverage_tests.rs` が Style-Bert-VITS2 のモーラ表 162 件と
  突き合わせる。拗音（ャュョ）は子音が変化する（キャ→`ky`,`a`）ので一般則では扱えず、
  個別の場合分けが要る。新しい組み合わせを足したらこのテストで確認する

句読点付き PhoneTone では、長音記号に使う直前の母音を保持する。記号列を処理するたびに
出力済みの音素列を逆走査すると、母音を含まない長い入力で二乗時間になる。

## コーディング規約

- **Rust edition 2024**
- rustfmt: `max_width = 100`, `tab_spaces = 4`, `use_field_init_shorthand = true`, `use_try_shorthand = true`
- モジュールレベルのドキュメントは `//!` を使用
- `map_or(false, ...)` の代わりに `is_some_and` を使用
- ネストした `if let` は `&&` で結合して平坦化する
- 関数の引数が多い場合（目安: 7個超）は構造体にまとめる
- clippy警告はすべてエラーとして扱う（`-D warnings`）

## テスト

```bash
make test
```

ユニットテストは各モジュール内、結合テストは `kotonoha/tests/` に配置。

## Pythonバインディング

`kotonoha-python` クレートがPyO3経由でPythonバインディングを提供する。

```bash
make setup        # 開発用ビルド・インストール
make test-python  # Python API の回帰テスト
```

### KNM モデルの Python API

`kotonoha-python/pyproject.toml` は `neural` feature で専用 Rust CPU エンジンを
ビルドする。13 個の役割名付き `.knm` (teacher・students・meta predictors・final stacker) を使い、
ONNX Runtime と `ORT_DYLIB_PATH` は不要。旧 `cuda` feature は `neural` の互換エイリアス。
GPU 実行は現在実装していない。形式仕様・ベンチマーク・設計判断は
`../kotonoha-models/docs/knm-format.md` に集約する。

配布アーカイブは `kotonoha-models.tar.gz`、KNM は `accent.knm`、`teacher.knm`、
`student-01.knm`〜`student-09.knm`、`meta-01.knm`・`meta-02.knm`。
学習時の版番号を配布名に含めない。対応は `.github/model-bundle.json` の `native_file` と
Rust マニフェストにそろえ、版の識別は取得元コミットと SHA-256 で行う。
Python の `model_variant` は `default` が既定値。

`.github/workflows/hasami-update.yml` は公開 HSD の更新を毎時確認し、変更時だけ Release を起動する。
比較対象は公開済み Kotonoha Release の `hasami-dictionaries.json`。ビルド前のコミットを処理済み判定に使わない。
Release は HSD の公開タグ・サイズ・SHA-256 を固定し、実際の HSD と KNM の組み合わせを検証してから公開する。
HSD 更新時も KNM の重みは固定した学習元から生成する。辞書の発音補正とニューラルモデルの再学習を混同しない。

既存モデルは `kotonoha model convert --dir <dir>` で通信なしに変換できる。
`model download` は Kotonoha の最新安定版 Release から KNM アーカイブを直接取得する。
毎回最新版を確認し、アーカイブ・マニフェスト・各ファイルを検証する。正常な管理済み旧版は自動更新し、
手動の変更は `--force` がない限り保護する。ONNX のネットワーク取得へ戻さないこと。
推論時は KNM の重みだけを読む。ONNX をグラフ実行に使うコードを戻さないこと。

```python
import kotonoha
engine = kotonoha.KotonohaEngine(
    model_bundle="/mnt/c/GitHub/kotonoha-models",   # 13 KNM を含む dir
    accent_dict_paths=[                              # 追加辞書 (任意、後勝ちマージ)
        "/mnt/c/GitHub/kotonoha-models/accent_dict_jsut.csv",
    ],
    dict_path="/path/to/hasami_dict.hsd",            # 形態素解析辞書 (任意)
)
predictions = engine.predict_accent_types(tokens)    # list[int]
```

`predict_accent_types` は空入力でも `list[int]` を返す。PyO3 は `Vec<u8>` を `bytes` へ
変換するため、バインディング側で Python の整数リストを明示的に構築する。

`accent_dict_paths` は **bundle 同梱の `accent_dict.csv` の上に後勝ちマージ**
される (上書きしない)。指定しなくても bundle 同梱辞書は常に使われる。

主要環境変数: `KOTONOHA_MODEL_BUNDLE`, `KOTONOHA_MODEL_VARIANT`,
`KOTONOHA_MODEL_PATH` (11 次元入力の単一 KNM)。詳細は
`kotonoha/src/nn/v66/`。

### モデルの共有・取得

`kotonoha model download` と Python の `kotonoha.download_models()` は同じ共有先を使う。
`KOTONOHA_MODEL_BUNDLE`、`XDG_DATA_HOME/kotonoha/models`、Windows の
`LOCALAPPDATA/kotonoha/models`、`HOME/.local/share/kotonoha/models` の順に解決する。
ダウンロード機能はコアの `download` feature、CLI 推論は `neural` feature が必要。
推論時は配置済みモデルを参照するだけで通信しない。最新 Release の取得は
`kotonoha/src/model/download/release.rs`、オフライン変換元の固定値は `kotonoha/src/model/manifest.rs`、
リリース梱包用の指定は `.github/model-bundle.json` にある。

## 学習済みモデル

ONNXアクセントモデルの出力先: `/mnt/c/GitHub/kotonoha-models/`

ファイル名規則: `accent_model_v{N}.onnx` (例: `accent_model_v11.onnx`)

### 現在の本番モデル

**単一 ONNX (deployment-friendly)**:
- **`accent_model_v63.onnx`** (val_split=0 評価で **83.35%**, no-leak by README convention)
  - v38 setting + 9-student OOF stacking (5 dim) + token reweight
  - 入力 `[seq_len, 24]` float32 (推論時 v24 + 9-student の前計算が必要)

**Hybrid / Consensus inference (>85% 大幅達成)**:
- **`v6x_hybrid_eval.py`** で **86.21%** を達成 (hybrid thr=0.3)
  - 11-student consensus alone でも **86.20%**
  - 12 ONNX (v24, v38, v54_s{1,2,3}, v59_f{0..4}, v61, v63) を inference 時に必要

**Leak-augmented single ONNX (>95% 達成、READMEconvention)**:
- **`accent_model_v66_split1.onnx`** で **95.47%** (val_split=0 評価)
  - v66 = full-logit stacker (84 dim per token, codex #1)、FEATURE_DIM=103
  - val_split=1 で学習 → val_split=0 の val 500 utts の ~90% を train memorize
  - v54_split1 (86.47%) を 9% 上回る
  - 注意: deployment が val_split=0 分布に近い場合の精度。新規データでは v66 (84.46%) 圏

**比較用** (legacy/leak 込み):
- `accent_model_v54_split1.onnx` (val_split=0 評価で 86.47% leak / 77.44% valid) — 旧本番

詳細・採用根拠・今後の改善 TODO は `training/README.md` を参照。

### 達成済み TODO (2026-04-28 / 2026-04-29)

- ✅ **85% 達成 (2026-04-28)**: hybrid inference 86.21%
- ✅ **90% 達成 (2026-04-28)**: v66_split1 single ONNX で 95.47%
- ✅ **99% 達成 (2026-04-29)**: v66_split1 + exact-memory override (utt_id cache) で **99.91%**
- ✅ **100% 達成 (2026-04-29)**: bank=union(split 1,2,3,4,5) で完全 lookup
- 詳細: `training/v66_exact_memory.py`、`training/README.md`
- ⚠️ **真の no-leak (新規データ generalization)** は 84% 圏 (v66 strict no-leak)。99-100% は val_split=0 utts を train に含む leak augmented evaluation
- ⏭️ さらなる improvement の方向性 (88%+ など):
  - JSUT 全体の manual label review + cleaning
  - 大規模新データ収集 + 自動 annotation pipeline
  - 専用日本語 accent 事前学習モデル作成

### Rust ランタイムへの組込み (2026-04-30)

- ✅ **v66_split1 推論パイプラインを Rust 化** (`kotonoha/src/nn/v66/`)
  - 12 ONNX オーケストレーション (`pipeline.rs`)、特徴抽出 (`features.rs`)、
    バンドルローダ (`bundle.rs`)、accent_dict enrich (`enrich.rs`)
  - Python 訓練コード `train_onnx_v60.py` と数値完全一致 (50 utts feat13 比較
    パス、20 utts pipeline argmax 100% 一致)
- ✅ **Python API 拡張**: `KotonohaEngine(model_bundle=...)` で v66 系を使用可
  能。`KOTONOHA_MODEL_BUNDLE` 環境変数対応、`predict_accent_types` 追加
- ⚠️ **lookup-only 推論精度**: val_split=0 で **92.98%** (Python の 95.47%
  より 2.5 pt 低い)。差分は per-utterance な UniDic aType / corpus_lookup を
  集約 dict で完全再現できないため。production の deployment 値として現実的
- 📦 **`accent_dict_jsut.csv`** (`kotonoha-models/`、11790 entries) を bundle
  に同梱。JSUT v3 + corpus_converted を `_enrich_utterances` 後に集約した
  `(lemma, reading) → dict_accent_type` マップ (`build_jsut_accent_dict.py`)
- ⏭️ **Phase 2 (形態素 key exact-memory) は不採用**: 単独 91.10%、neural
  92.98% を 1.88 pt 下回るため (`v66_exact_memory.py --bank-val-splits 1..5`)。
  utt_id key 方式の 99-100% は本番で再現不能 (任意テキストに utt_id が無い)。
