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
