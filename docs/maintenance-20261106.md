# 2026-11-06 定期点検

## 同期と hasami 26.11.101

- 開始時の `main` は未コミット変更なし。`git pull --rebase` で `d024124` から
  `b7b7fb0`（v26.11.100 公開）まで fast-forward した。stash は作成していない。
  開始前からある GitHub Desktop の stash は今回と無関係のため保持した。
- 依頼された hasami 26.11.101 は、v26.11.100 の Release 自動化（`b7b7fb0`）で
  `Cargo.toml`・`Cargo.lock`（`6b30d3d`）・`.github/hasami-dictionaries.json`（形式 v6）とも固定済みだった。
  v26.11.100..v26.11.101 の変更は HSD 形式 v6（左右の文脈 ID が同じエントリの 4B 格納）への移行と
  計測資料だけで、`Token`・公開 API・解析結果は変わらない。hasami 側の記録では、Kotonoha `d024124` の
  実 Engine で 1,360 文 × 3 辞書の NJD〜HTS ラベル・PhoneTone・韻律記号が v5 と v6 で一致している。
  Kotonoha の発音・NJD・アクセント処理の追従は不要と判断した。
- Release ボットのコミットでは CI が走らないため、手元で同等の検証を行った。テスト内で構築する辞書は
  v6 で書き出されて実際のロード経路を通る。加えて v6 の推奨辞書（手元の再構築品。公開カタログとサイズは同じで
  SHA-256 は異なる）と配置済み KNM で、ignored の実辞書・推論テスト 6 件が成功した。
- v5 以前の `.hsd` は `dictionary format v5 is no longer supported` で読み込みに失敗する。
  利用者向けに README の「辞書の準備」へ追記した。

## 依存

`mise exec -- depup --install --include-pinned` の提案から次を採用した（`96486e9`）。

- uv 0.13.5 → 0.13.6、Python 3.14.7 → 3.14.10。depup は Python 3.15.2 を提案したが、
  学習用の onnxruntime-gpu 1.31.1 と torch 2.12.1+cu126 は cp315 の wheel を持たないため見送った。
- Cargo.lock の推移依存（cc 1.10.0、rustix 1.1.6、smallvec 1.16.4、utf8_iter 1.1.0 と utf_types 1.0.0、
  webpki-roots 1.0.11 など。windows-sys は 0.61.2 にそろった）。
- 学習用の onnxruntime-gpu / onnxruntime 1.31.1、transformers 5.22.0、maturin 1.16.0。
  学習用 venv で CUDA プロバイダの onnxruntime と torch を import できることを確認した。
- zstd の下限 0.14 → 0.14.3 は、下流が 0.14 系を選べるよう下限を保つ既存方針に従って戻した。

## GitHub Issue

owayo/kotonoha の open Issue は開始時 0 件、open PR も 0 件。点検中の起票はない。

## リファクタリング

- `nn/v66/pipeline.rs` の `predict_with_dict_accents`（210 行、循環的複雑度 28）を、feat13・teacher・
  9 student・meta・stacker84・最終段の関数に分けた。feat14 / feat24 / feat103 の列の配置
  （teacher の argmax、`teacher_soft_stats`、末尾の meta_9・stacker84）を定数とコンパイル時の検査で固定した。
  stacker の平均・標準偏差・投票は、3 か所に分かれていたループを 11 モデルの反復 1 つにまとめた。
  加算順は student 1〜9、v61、v63 のままで、Python 参照との一致テスト（20 発話の argmax 完全一致、
  feat13 一致）と、968 文を KNM まで通した出力ダイジェスト（`36bce5791b714cd7`）が変更前と一致した。
  stacker と teacher の列を検証する単体テストを 3 件追加した。
- CLI の `analyze` は、`--dict` の有無を確かめる前に共有先の KNM 13 個を読み込んでいた。辞書と
  アクセント辞書を先に検査し、辞書の読み込みと形態素解析のエラー処理を `tokenize` と共通にした。
  辞書未指定のエラーで終了するまでの中央値は 465.8 ms から 28.6 ms になった（各 7 回）。
  変更前の CLI で失敗する回帰テストを追加した。
- 見送り: `phoneme.rs` の大きな match はモーラ表のカバレッジテストに支えられた表引きで、分割の利益がない。
  `crf.rs`・`model/knm.rs::load` は前回までの判断を維持した。`cmd_bench` の測定ブロックの重複は
  開発用の出力で効果が小さい。`DictionaryNormalizer::normalize` は記号や括弧を含む文を 2 回解析するが、
  解消には hasami の API 追加が要り、ルール経路でも全体の約 2.6% にとどまるため見送った。

## 性能

### 計測方法

`kotonoha/examples/text_pipeline_profile.rs` を追加した。実際の HSD（任意で KNM）を使い、正規化・
形態素解析・NJD・KNM 推論・アクセント句・ラベル・PhoneTone・韻律記号の時間と出力のダイジェストを返す。

- Windows 11 / Ryzen 7 5700X（8 コア 16 スレッド）、Rust 1.99.1 release、Rayon 16 スレッド
- 推奨 HSD（ipadic-neologd-sudachi、形式 v6）、共有先に配置済みの 13 KNM
  （v26.10.104 の Release から取得。models.json の 16 ファイルのサイズ・SHA-256 と一致）
- 入力は 968 文・15,787 トークン。リポジトリ内の日本語文書と、テスト fixture にある JSUT 由来の 50 発話から
  作り、本文は格納しない（SHA-256 `e6a121dd…`）
- 1 周目を捨て、変更前後のバイナリを交互に実行した。生データは
  [benchmarks/maintenance-20261106.json](benchmarks/maintenance-20261106.json)

変更前の内訳は、ルール経路（KNM なし）で全体 132 ms のうちラベル 51.3 ms、正規化 23.0 ms、
形態素解析 19.6 ms、アクセント句 11.6 ms。KNM を使う経路では推論が全体の 99.1% を占めた。

### 双方向 LSTM の 2 方向の並列化

KNM 1 モデルの推論は約 3.7 ms/文で、各層・各方向の再帰行列（1 MiB）が L2 に収まらず、
時刻ごとに読み直すことが律速だった。2 方向は独立なので `rayon::join` で並列に計算する。
各方向の演算順は変えておらず、13 モデル × 8 系列長の生ロジット（各 2,751 値）がビット単位で一致した。

全経路（優先度 High、各 6 標本）:

| 指標 | 変更前 中央値（範囲） | 変更後 中央値（範囲） | 変化 |
|---|---:|---:|---:|
| KNM 推論 | 27,510.8 ms（25,657〜28,207） | 23,677.5 ms（21,495〜25,130） | −13.9% |
| 全体 | 27,829.1 ms | 23,990.2 ms | −13.8% |

段別（コーパスと同じ系列長の合成入力、優先度 High、各 6 標本）:

| 段 | 変更前 中央値 | 変更後 中央値 | 変化 |
|---|---:|---:|---:|
| teacher | 4,450.7 ms | 2,923.0 ms | −34.3% |
| 9 student（並列） | 10,861.4 ms | 10,024.7 ms | −7.7% |
| meta 2 個（並列） | 5,386.1 ms | 3,494.6 ms | −35.1% |
| 最終段 | 4,550.3 ms | 2,750.9 ms | −39.5% |
| 合計 | 25,304.2 ms | 19,209.4 ms | −24.1% |

単独で動く段ほど効果が大きい。97 文に連結した長い入力（平均 163 トークン）では合計 −16.0%、
1,420 個の短い句（平均 10 トークン）では合計 −10.4% だった。短い句では 9 student の並列段が
中央値で +4.2%（範囲は重なる）とわずかに遅くなるが、単独段の短縮で相殺される。
ピーク Working Set は変更前 352.6 / 351.0 MiB、変更後 351.1 / 352.5 MiB で差はない。

最初に通常の優先度で 3 版を交互に測ったときは、推論段が変更前 22,045 / 23,660 ms、
変更後 24,220 / 21,251 ms で、ホストの負荷による揺れに差が埋もれた。この値は参考として生データに残し、
採用の根拠は優先度 High での範囲が重ならない比較とする。

### 結合規則の検索

`AccentRuleTable::find_rule` は、品詞の階層ごとに `String` を作り、索引を引くたびに左右の品詞を
複製していた。階層はすべて元の文字列の接頭辞なので借用で列挙し、索引を「左品詞 → 右品詞」の
入れ子にした。候補の順序と同じ優先度での選び方は変えていない。

ルール経路（KNM なし、優先度 High、各 80 標本）で、アクセント句段は中央値 13.50 ms から 4.75 ms
（−64.8%、変更前の最小 9.95 ms > 変更後の最大 7.44 ms）になった。出力ダイジェスト
`d1df88f02a69ca2d` は一致した。ルール経路の全体は −11.2% だが、変更していない段も −3〜−6% 揺れている。

```bash
cargo run --release --locked -p kotonoha --features neural --example text_pipeline_profile -- \
  DICT.hsd CORPUS.txt [MODEL_DIR]
```

## 不具合の修正

- **同じ CSV 内の重複キー**: `AccentDict::lookup_exact` と `scripts/package_models.py` は、
  同じ CSV に同じ `(lemma, reading)` が複数あれば先頭を採用する。一方、`V66Bundle` の
  `merge_accent_dict` は各行を `set` で上書きするため、同じ CSV でも末尾の値になっていた。
  `猫,ネコ,1` と `猫,ネコ,2` を含む CSV で、`Engine::load_accent_dict` は 1、`V66Bundle`
  （Python の `accent_dict_paths` を含む）は 2 を返す。`data/accent_dict.csv` には値の食い違う重複キーが
  45 件あり、これを追加辞書に渡すとニューラル推論の辞書特徴量が別の経路と食い違う。
  `AccentDict::from_csv` で後続の重複を読み飛ばし（値の検証は続ける）、すべての経路を先頭採用にそろえた。
  配布の `accent_dict.csv`・`accent_dict_jsut.csv` には重複がなく、既定の構成の出力は変わらない。
- **列不足のアクセント辞書**: ヘッダが 2 列の CSV では各行も 2 列になり、`from_csv` がすべて読み飛ばして
  空の辞書を返していた。辞書を指定したのに効かないことに気づけないため、3 列未満の行はエラーにした。

どちらも修正前に失敗し、修正後に成功する回帰テストを追加した（`accent_dict.rs` 2 件、
`nn/v66/bundle.rs` 1 件。後続の重複も検証することを確かめるテストも 1 件追加）。

## テスト

追加した 11 件のうち、上記以外は次のとおり。

- `nn/native.rs`: タイル配置を解いた逐次計算と、並列化した双方向 LSTM の出力を比べる。
  順方向と逆方向の出力を入れ替えると失敗することを確かめた。
- `accent_rule.rs`: 品詞バリアントの空の階層・`*` で始まる品詞の扱いと、同じ優先度の規則が
  複数当たるときの選び方。
- `nn/v66/pipeline.rs`: 上記の 3 件。CLI: 上記の 1 件。

## 検証

- `cargo fmt --all -- --check`、`cargo clippy --locked --workspace --all-targets -- -D warnings`
  （既定・全 feature の両構成）
- `cargo test --locked --workspace --exclude kotonoha-python --no-fail-fast`（既定・全 feature の両構成）
- Python API のテストと配布処理の Python テスト（Python 3.14.10 / uv 0.13.6。Python 拡張は
  Windows 用の一時 venv でビルドし直した）
- 推奨 HSD（v6）と配置済み KNM で、ignored の実辞書・推論テスト
- `astro-sight review --git`: 公開 API の追加・削除・変更、未解決の呼出元、未使用シンボルはなし

件数は点検を締めくくるコミットの本文に記録する。

## 未実施項目と記録

- Linux・macOS でのローカル実行、GPU 学習・モデルの再学習は行っていない。
  Super-TTS の環境へ Python 拡張を入れ直す結合確認は、TTS の環境を変えないため行っていない。
  hasami 26.11.101 は解析結果と API を変えず、Kotonoha の Python 経路は一時 venv で確認した。
- 公開 Release の HSD そのものではなく、同じ版の手元の再構築品で確認した。
- ドキュメント用リンク更新スクリプトは macOS のパスで、Windows に同等品がないため未実行。
  `CLAUDE.md` はリポジトリ上はシンボリックリンク（mode 120000）だが、`core.symlinks=false` のため
  「AGENTS.md」とだけ書かれた通常ファイルとしてチェックアウトされている。直接編集していない。
- 作業ツリーでは点検前から 102 個のファイルが CRLF だった（インデックスは LF。stat キャッシュのため
  `git status` には出ない）。今回編集した 3 ファイルだけを LF に戻し、ほかは変更していない。
- README は日本語のため `README.ja.md` の追加は不要。

## 追記: hasami 26.11.102

- 点検後に hasami v26.11.102 が公開され、Kotonoha の Release 自動化（`0420af5`、Kotonoha v26.11.102）で
  `Cargo.toml`・`Cargo.lock`（`471321e`）・`.github/hasami-dictionaries.json` が更新済みだった。
- v26.11.101..v26.11.102 の変更は、`Token` の判定を `pub(crate)` の述語へまとめたリファクタリング
  （hasami 側で 3 辞書 × 2 入力の全トークン・全フィールドの一致を確認済み）と、「なら」の回帰テスト、
  文書と開発用ツールの更新だけ。公開 API と HSD 形式（v6）は変わらず、Kotonoha の追従修正は不要と判断した。
  HSD は 3 辞書ともサイズが同じで、SHA-256 だけが変わった。
- 検証: `cargo fmt --check`、clippy（既定・全 feature）、Rust テスト 既定 729 件・全 feature 741 件、
  Python API 17 件、配布処理 22 件、v6 の推奨 HSD と KNM による ignored の実辞書・推論テスト 6 件が成功。
- 同じ手元の HSD で `text_pipeline_profile` の出力ダイジェストを v26.11.101 と比べ、ルール経路の
  3 コーパス（968 文・97 の長文・1,420 の短い句）と KNM 経路の 968 文がすべて一致した。
  公開 HSD そのものと KNM の組み合わせは、Release ワークフローが検証する。
