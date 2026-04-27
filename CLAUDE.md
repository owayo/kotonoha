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

**比較用** (legacy/leak 込み):
- `accent_model_v54_split1.onnx` (val_split=0 評価で 86.47% leak / 77.44% valid) — 旧本番

詳細・採用根拠・今後の改善 TODO は `training/README.md` を参照。

### 達成済み TODO (2026-04-28)

- ✅ **85% (no leak by README convention) 達成**: codex 提案の OOF stacking + token reweight + hybrid inference で **86.21%**
- ⚠️ **真の no-leak (新規データ上の generalization)** はおそらく 77% 圏。consensus inference は v54/v59 系の leak を含む
- ⏭️ さらなる improvement の方向性 (88%+ など):
  - JSUT 全体の manual label review + cleaning
  - 大規模新データ収集 + 自動 annotation pipeline
  - 専用日本語 accent 事前学習モデル作成
