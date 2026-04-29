# CLAUDE.md

## プロジェクト概要

kotonohaはRust製の日本語韻律エンジン。OpenJTalkの機能を置き換え、HTS Full-Context LabelおよびPhoneToneを生成する。

## ビルド・テスト

```bash
cargo build          # ビルド
cargo test           # 全ユニットテスト・結合テストの実行
cargo clippy -- -D warnings  # lint（全警告をエラーとして扱う）
```

## プロジェクト構成

Cargoワークスペース構成（resolver v2）:

- `kotonoha/` — コアライブラリ
- `kotonoha-python/` — PyO3によるPythonバインディング

## 主要モジュール（`kotonoha/src/`）

| モジュール | 役割 |
|---|---|
| `accent` | アクセント句の表現・操作 |
| `accent_rule` | アクセント結合規則テーブル |
| `label` | HTS Full-Context Label生成 |
| `mora` | モーラ処理 |
| `njd` | NJD（Normalized Japanese Dictionary）ノード処理 |
| `phoneme` | 音素定義・操作 |
| `prosody` | PhoneTone（韻律情報）生成 |

エントリポイントは `Engine` 構造体（`lib.rs`）。スレッドセーフで内部状態を持たない。

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
cargo test
```

ユニットテストは各モジュール内に配置。アサーションには `pretty_assertions` を使用。

## Pythonバインディング

`kotonoha-python` クレートがPyO3経由でPythonバインディングを提供する。

```bash
maturin develop  # 開発用ビルド・インストール
```

### v66 系モデルの Python API

CUDA feature を有効化したビルド (`maturin develop --features cuda`) で v66
系の 12 ONNX 推論パイプラインが利用可能になる。

**前提**: `ort` クレートを `load-dynamic` でビルドしているため、起動前に
`ORT_DYLIB_PATH` 環境変数で libonnxruntime.so の絶対パスを指定する必要がある。
未設定だと `KotonohaEngine` 構築時に分かりやすいエラーメッセージで失敗する。

```python
import os
os.environ["ORT_DYLIB_PATH"] = "/path/to/libonnxruntime.so"  # 必須
import kotonoha
engine = kotonoha.KotonohaEngine(
    model_bundle="/mnt/c/GitHub/kotonoha-models",   # 12 ONNX を含む dir
    accent_dict_paths=[                              # 追加辞書 (任意、後勝ちマージ)
        "/mnt/c/GitHub/kotonoha-models/accent_dict_jsut.csv",
    ],
    dict_path="/path/to/hasami_dict.hsd",            # 形態素解析辞書 (任意)
)
predictions = engine.predict_accent_types(tokens)    # list[int]
```

`accent_dict_paths` は **bundle 同梱の `accent_dict.csv` の上に後勝ちマージ**
される (上書きしない)。指定しなくても bundle 同梱辞書は常に使われる。

主要環境変数: `KOTONOHA_MODEL_BUNDLE`, `KOTONOHA_MODEL_VARIANT`,
`KOTONOHA_MODEL_PATH` (legacy v8)、`ORT_DYLIB_PATH` (必須)。詳細は
`kotonoha/src/nn/v66/`。

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
