# 2026-10-30 定期点検

## 同期と依存

- 開始時の `main` は未コミット変更なしで、`origin/main` の `9adb5b8`（v26.10.108 公開後）と一致。
  取得のみで rebase・stash は不要だった。開始前から残る 2026-10-08 の
  GitHub Desktop stash（Cargo.lock の clap 4.6.7 断片）は今回と無関係のため保持した。
- `mise exec -- depup --install --include-pinned` を実行。採用したのは
  uv 0.13.4 → 0.13.5 と、Cargo.lock の推移依存の更新（icu 2.3 → 2.4 など、
  depup の `--age` 方針による 8 件の巻き戻しを含む）のみ。
- depup が提案した `zstd` の下限 0.14 → 0.14.3 は、Issue #9（`dfc9115`）で
  「下流が選べるよう 0.14 系の下限を保つ」と決めた経緯に反するため戻した。
  ロックの解決は従来どおり 0.14.3 のまま。
- depup が提案した Python 3.14.7 → 3.15.1 は、前日の点検と同じ理由で見送った。
  `training/uv.lock` の `onnxruntime-gpu` 1.31.0 は cp314 までの wheel しか持たず、
  cp315 では学習環境を導入できないことを再確認した。uv 0.13.5 + Python 3.14.7 で
  Python バインディング 12 件・配布スクリプト 21 件のテストが成功した
  （バインディングは Windows 用の一時 venv でビルド。共有 `.venv` は WSL 用のため触れない）。

## 構造・リファクタリング

- `nn.rs` の v2 特徴量用 `pos_detail1_to_id` / `pos_detail2_to_id` /
  `conj_type_to_id` / `conj_form_to_id` / `reading_hash` / `char_hash` は
  `nn/v66/vocab.rs`・`nn/v66/features.rs` と完全重複だったため削除し、v66 側へ委譲した
  （約 140 行削減）。訓練側 Python と一致させるべき変換表の正本が 1 箇所になった。
  `Pos` enum を取る `pos_to_id` と `crf.rs` の `pos_to_id` は特徴空間が異なる別物なので残した。
- `accent.rs`: `set_chain_flags` の借用回避クローン（ノードごとに String 2 本）を判定先行で除去、
  `build_accent_phrases` の句ごとの `Vec` クローンを `mem::take` に置換、
  `build_full_pos_str` を単一確保の `push_str` 構築にし、`combine_accent_types` では
  右側の品詞文字列を次反復の左側へ持ち回して構築回数を半減した。
- `label.rs`: `build_label_context` の `node_mora_counts` と `mora_to_single_mora_word` は
  「句が単独ノードかつ 1 モーラ」という句単位の単一 bool と等価だったため、
  ベクタ 2 本を削除して 1 つの bool にした。
- 見送り: `phoneme.rs` の大きな match はモーラ表カバレッジテストに支えられた表引きで分割の利益がない。
  `crf.rs::train`・`model/knm.rs::load` は検証順序と形式仕様の境界が明確で、前日の判断を維持。
  `mora.rs::count_mora` と `v66::features::count_mora` は音韻規則と訓練側互換という別仕様のため統合しない。
  `njd.rs::expand_long_vowels` の `Vec<char>` 除去は効果が小さく読みやすさと相殺するため見送り。

## 計測

Windows x64 / Ryzen 7 5700X、Rust 1.99.1 release。固定文「今日は良い天気です」5 トークン、
HSD・KNM は使用しない。生データは [benchmarks/maintenance-20261030.json](benchmarks/maintenance-20261030.json)。

`label_allocations`（`make_label` 1 回分、ウォームアップ後）は 6 条件すべてで
割当回数・累積要求バイト数が減り、全ラベルの FNV-1a ダイジェストが変更前後で一致した。

| 条件 | 割当回数 変更前 → 変更後 | 要求バイト数 変更前 → 変更後 |
|---|---:|---:|
| phrases / 1 | 223 → 217 | 12,416 → 12,296 |
| single / 1 | 184 → 180 | 12,410 → 12,290 |
| phrases / 20 | 4,337 → 4,217 | 232,694 → 230,294 |
| single / 20 | 3,031 → 3,019 | 213,350 → 210,830 |
| phrases / 200 | 43,826 → 42,626 | 2,333,629 → 2,309,629 |
| single / 200 | 29,863 → 29,845 | 1,991,442 → 1,971,002 |

ダイジェスト一致は、変更後の `estimate_accent` が組んだ句でラベルを生成して得たものなので、
アクセント句構築の同値性も同時に確認している。`estimate_accent` 内のクローン除去は
計数窓の外にあり、この表には現れない（割当が構造的に減ることはコードから直接いえる）。

Criterion の時間計測（30 標本・ウォームアップ 0.5 秒・測定 1 秒）は、触れていない
`engine/analyze` が +20%、`engine/prosody_extraction` が −38% 振れる程度のホスト起因の
ばらつきがあり、前日と同様、壁時計からの向上率・退行有無の断定はしない。
採用根拠は全条件での割当回数・要求バイト数の削減と出力ダイジェストの一致である。

```bash
cargo bench --locked -p kotonoha --bench benchmark -- 'engine/|label_scaling' \
  --sample-size 30 --warm-up-time 0.5 --measurement-time 1 --nresamples 1000 --noplot
cargo run --release --locked -p kotonoha --example label_allocations
```

## レビューと回帰検証

`astro-sight review --git --base HEAD~1` で公開 API の追加・削除・変更なし、
未解決の呼出元なし、dead symbol なしを確認した。影響シンボルは
`set_chain_flags` / `build_accent_phrases` / `build_full_pos_str` /
`combine_accent_types` / `build_label_context` のモジュール内private関数のみ。
今回の点検で新たな不具合と断定できる事象はなかった。

1 モーラ語の無声化除外（規則 5）は、既存テストが「酢」（頭高型）でアクセント核の規則にも
守られる条件だったため、平板型で例外だけが効く「木」と、2 ノード句では例外が効かず
無声化する「木 + 戸」の回帰テスト 2 件を `label_tests.rs` へ追加した。

次の検証が成功した。

- `cargo fmt --all -- --check`
- `cargo clippy --locked --workspace --all-targets -- -D warnings`（既定・全 feature の両構成)
- `cargo test --locked --workspace --exclude kotonoha-python --no-fail-fast`（既定・全 feature の両構成）
- Python 公開 API テスト 12 件、配布処理の Python テスト 21 件（Python 3.14.7 / uv 0.13.5）

## 未実施項目と記録

- 対象リポジトリの open Issue は開始時 0 件で、点検中の起票もない。
- コード変更（依存・nn 重複排除・accent / label の改善）は点検作業の途中で
  利用者自身が GitHub Desktop からコミットした（`2d3e5c7`）。履歴の書き換えは行わない。
- GPU 学習・モデル再学習、Linux・macOS でのローカル実行は行っていない。
  明示的な ignored テスト（`KOTONOHA_TEST_DICT` 2 件、`KOTONOHA_RELEASE_HSD` /
  `KOTONOHA_RELEASE_MODELS` 1 件)は今回の通常テストでは実行していない。
- ドキュメント用リンク更新スクリプトは macOS パスで、Windows の同等品がないため未実行。
  `CLAUDE.md` は「AGENTS.md」とだけ書かれた通常ファイルのままで、直接編集していない。
  README は日本語のため `README.ja.md` の追加は不要。
