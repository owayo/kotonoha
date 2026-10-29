# 2026-10-29 定期点検

## 同期と依存

- 開始時の `main` は未コミット変更なし。`origin/main` を fetch・rebase し、
  `ffe81e2`（v26.10.108）まで fast-forward した。stash は作成していない。
- Rust 1.99.1、Python 3.14.7、uv 0.13.4 を mise で確認。
  Windows では POSIX 用の `make setup` が失敗するため、CI と同じ直接コマンドで準備・検証した。
  既存の共有 `.venv` は Windows から更新できず、一時ディレクトリに別環境を作成した。
- `mise exec -- depup --install --include-pinned` を実行。
  `memmap2` 0.9.11 → 0.9.12、`ureq` 3.4.2 → 3.5.0、
  学習用 `pyopenjtalk-plus` 0.4.1.post9 → post10 を採用した。
  zstd は利用 API の互換性を保つため、従来の `0.14` の下限指定を維持した。
- Python 3.15.1 は、[公式移行情報](https://docs.python.org/3.15/whatsnew/3.15.html#porting-to-python-3-15)
  と [PyO3 の対応状況](https://pyo3.rs/v0.30.1/changelog) を確認した上で採用を見送った。
  学習環境の `uv sync --locked --dry-run --project training` は
  `onnxruntime-gpu==1.31.0` に cp315 wheel がないため失敗した。Python 3.14.7 を維持する。
- depup が一時的な `exclude-newer` を学習用 `uv.lock` に残し、通常の
  `uv lock --check` が失敗することを再現した。[depup #68](https://github.com/owayo/depup/issues/68)
  を起票し、通常の `uv lock --project training` で再生成して `--check` の成功を確認した。
  ロックの実質的な依存更新は上記の pyopenjtalk-plus のみ。

## 構造・正確性の確認

astro-sight の `symbols` でコア全体、CLI、Python バインディング、配布スクリプトを調べ、
`calls`・`refs`・`sequence` で Engine、ラベル生成、CRF 学習、KNM 読込、推論の経路を確認した。
KNM の確保前の形状・範囲検証とダイジェスト検証、CRF の入力検証、ダウンロードの保護・
ロールバックも確認した。この点検で Kotonoha に新たな不具合と断定できる事象はなかった。

ラベルの組み立てでは、音素窓と A フィールドは音素ごとに変わる一方、B〜K は句内で一定だった。
これらを別関数に分離し、共通フィールドの生成を句ごとに1回にした。保持するのは直前の1句分のみ。
結果配列も音素数から必要数を確保する。公開 API、ラベル書式、無声化、句位置の計算は維持する。

CRF 学習と KNM 読込は長い関数だが、検証順序と学習・形式仕様の境界が明確であり、今回の
ラベル生成のボトルネックにも関与しないため分割を見送った。特徴抽出の品詞 ID は
元のトークンと変換後の品詞で扱いが異なるため、品詞変換全体を一括で共通化しなかった。

`context`・`impact`・`review --git` で公開 API の変更、未対応の呼出元、未使用シンボルは検出されなかった。
同時変更履歴から挙がった prosody・CLI・Python バインディングは、出力と API を維持する今回の変更では
追従不要と判断し、既存テストで確認した。

## アロケーション

Windows x64 / Ryzen 7 5700X、Rust 1.99.1 の release ビルド。固定文は
「今日は良い天気です」の5トークンで、HSD・KNM は使用しない。
`phrases` は通常のアクセント句、`single` は同じトークン列を1句にまとめた条件。
数値はウォームアップ後の `make_label` 1回分で、再確保も1回として数える。
バイト数はアロケータへの累積要求量であり、RSS やピークメモリではない。

| 条件 / 文の反復数 | 割当回数 変更前 → 変更後 | 削減率 | 要求バイト数 変更前 → 変更後 |
|---|---:|---:|---:|
| phrases / 1 | 402 → 223 | 44.5% | 16,307 → 12,416 |
| single / 1 | 391 → 184 | 52.9% | 16,847 → 12,410 |
| phrases / 20 | 8,292 → 4,337 | 47.7% | 316,718 → 232,694 |
| single / 20 | 7,441 → 3,031 | 59.3% | 302,948 → 213,350 |
| phrases / 200 | 86,224 → 43,826 | 49.2% | 3,140,322 → 2,333,629 |
| single / 200 | 74,056 → 29,863 | 59.7% | 2,815,888 → 1,991,442 |

6条件すべてでラベル数と、各ラベルを区切り付きで結合した FNV-1a ハッシュが一致した。
割当回数と要求バイト数の退行はなかった。

## 時間計測の範囲と制約

Criterion 0.8.2 で通常の測定後、同じ変更前・変更後の実行ファイルを3組測定した。
CPU affinity は同じ論理 CPU（mask 2）、各条件30標本、ウォームアップ0.5秒、測定目標1秒、
再標本化1,000回。実行順は前→後、後→前、前→後。タイマは壁時計である。
ホストでは別の処理も動作しており、実行ごとのばらつきが大きかった。

各実行の中央値をさらに3回で集計した値、範囲、全標本と割当結果は
[生データ](benchmarks/label-20261029.json) に保存した。
以下は参考値（μs、括弧は3回の中央値の最小〜最大）である。

| 条件 | 変更前 | 変更後 |
|---|---:|---:|
| engine/generate_labels | 170.19 (60.25〜284.44) | 88.03 (52.88〜168.91) |
| engine/tokens_to_labels | 124.81 (87.89〜309.51) | 103.31 (33.60〜198.63) |
| phrases / 1 | 220.92 (141.20〜236.16) | 78.47 (67.29〜135.75) |
| phrases / 20 | 2,281.69 (1,436.89〜5,352.04) | 753.00 (525.53〜2,059.16) |
| phrases / 200 | 94,287.85 (64,578.95〜118,131.45) | 28,256.05 (5,158.55〜52,062.80) |
| single / 1 | 61.79 (54.19〜144.74) | 68.44 (27.49〜100.78) |
| single / 20 | 5,061.70 (1,166.17〜6,718.68) | 1,487.57 (463.14〜2,038.42) |
| single / 200 | 81,319.50 (76,131.25〜85,373.40) | 24,682.75 (9,065.12〜37,678.80) |

この時間計測から入力全体の速度向上率や時間の退行の有無は断定しない。
採用根拠は、全条件で確認した44.5〜59.7%の割当回数削減と、累積要求量の削減である。
待機時間を含む壁時計の改善率を、計算処理そのものの改善率として扱わない。

```bash
cargo bench --locked -p kotonoha --bench benchmark -- \
  'label_scaling|engine/(generate_labels|tokens_to_labels)' \
  --sample-size 30 --warm-up-time 0.5 --measurement-time 1 --nresamples 1000 --noplot
cargo run --release --locked -p kotonoha --example label_allocations
```

比較時は変更前後で同じ依存・ツールチェーン・入力を使い、ホストの他の処理もそろえる。

## 回帰検証と未実施項目

空の句を挟んだ複数句・疑問文・句境界でラベル全文を比較し、Engine 再利用時に句の情報が
持ち越されないことを確認するテストを追加した。既存のラベル関連テストを含め57件が成功した。

次の検証が成功した。

- `cargo fmt --all -- --check`
- `cargo clippy --locked --workspace --all-targets -- -D warnings`
- `cargo clippy --locked --workspace --all-targets --all-features -- -D warnings`
- `cargo test --locked --workspace --exclude kotonoha-python --no-fail-fast`（693件成功）
- 同じ Rust テストの `--all-features` 構成（700件成功、配置済み KNM の推論一致テストを含む）
- KNM の推論一致テストの明示実行でも、20発話・323語の最終予測が参照値と全件一致
- Python 公開 API テスト12件、配布処理の Python テスト21件
- `cargo build --release --locked -p kotonoha-cli --features neural`
- 学習用 `uv lock --check --project training`
- 独立環境で pyopenjtalk-plus post10 の import と日本語 g2p

GPU 学習全体・モデルの再学習、Linux・macOS でのローカル実行は行っていない。
Rust テストの明示的な ignored は既定構成で2件、全 feature で3件残る。
監査済み配布辞書を指定する `KOTONOHA_TEST_DICT` の2件と、公開 HSD・KNM の組合せを指定する
`KOTONOHA_RELEASE_HSD` / `KOTONOHA_RELEASE_MODELS` の1件であり、今回の通常テストでは実行していない。
pyopenjtalk-plus の単独確認には ONNX Runtime を追加しておらず、Nani 推定は対象外。
ドキュメント用リンク更新スクリプトは、指定の macOS パスと確認可能な Windows の同等品が
なかったため未実行。`CLAUDE.md` は通常ファイルだったため、置換・直接編集していない。
README は日本語のため `README.ja.md` の追加は不要。
