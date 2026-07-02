//! `dict_accent_type` 補完
//!
//! 訓練側 `train_onnx_v60.py::_enrich_utterances` (461-488 行) と等価ロジック。
//! 本番の hasami → InputToken パスでは `dict_accent_type` が未設定なので、
//! `(lemma, reading)` から `AccentDict` を引き、無ければ `(lemma.split('-')[0], reading)`
//! にフォールバックして補完する。

use crate::accent_dict::AccentDict;
use crate::njd::InputToken;

/// `tokens` の各 `InputToken` に対して `dict_accent_type` を解決する
///
/// `tokens.len()` と同じ長さの `Vec<Option<u8>>` を返す。各要素は `None` (辞書ヒット
/// 無し) または `Some(accent_type)` (0..21 の範囲)。
///
/// ルックアップは `lookup_exact` による `(lemma, reading)` 完全一致のみで、Python 側
/// `accent_dict[(lemma, reading)]` (タプル key) と同一セマンティクス。読み不一致時に
/// 同一 lemma の別読みエントリへフォールバックすることはない。
pub fn resolve_dict_accents(tokens: &[InputToken], accent_dict: &AccentDict) -> Vec<Option<u8>> {
    tokens
        .iter()
        .map(|t| resolve_one(&t.lemma, &t.reading, accent_dict))
        .collect()
}

/// 単一形態素のアクセント辞書ルックアップ
///
/// 訓練側 `train_onnx_v60.py::_enrich_utterances` (461-488) と同じシンプルな
/// 2 キー戦略を採用:
/// 1. `(lemma, reading)` を試す
/// 2. lemma に `-` が含まれる場合は base 部 (`lemma.split('-')[0]`, reading) を試す
///
/// `(surface, reading)` 等の追加候補は **採用しない**。これは、本番が
/// JSUT v3 build 時の `text_to_accent_json.py::lookup_accent` (4 候補 +
/// corpus_lookup + UniDic aType) を直接再現することは出来ないため、代わりに
/// JSUT v3 で確定した `(lemma, reading) → dict_accent_type` を集約した
/// `accent_dict_jsut.csv` を bundle に同梱して使う設計のため。
/// surface ベースの fallback は false-positive (別語の値拾い) を生むリスクがある。
pub fn resolve_one(lemma: &str, reading: &str, accent_dict: &AccentDict) -> Option<u8> {
    if let Some(acc) = accent_dict.lookup_exact(lemma, reading) {
        return Some(acc);
    }
    if let Some(dash_pos) = lemma.find('-') {
        let base = &lemma[..dash_pos];
        if !base.is_empty()
            && let Some(acc) = accent_dict.lookup_exact(base, reading)
        {
            return Some(acc);
        }
    }
    None
}

/// `dict_accent_type` 解決時の hit/miss 統計
///
/// 本番運用で「accent_dict カバレッジが足りているか」を観測するために使う。
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct EnrichStats {
    /// 形態素総数
    pub total: usize,
    /// (lemma, reading) で直接 hit した件数
    pub hit_lemma: usize,
    /// dash fallback (`(base_lemma, reading)`) で hit した件数
    pub hit_dash: usize,
    /// 一切 hit せず None になった件数
    pub miss: usize,
}

impl EnrichStats {
    /// hit 率 (=`(hit_lemma + hit_dash) / total`)、`total == 0` のときは `0.0`。
    pub fn hit_rate(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        (self.hit_lemma + self.hit_dash) as f64 / self.total as f64
    }
}

/// `resolve_dict_accents` と同じ結果を返しつつ、各 token がどの経路で
/// 解決されたかを `EnrichStats` で集計する観測用ヘルパ。
pub fn resolve_dict_accents_with_stats(
    tokens: &[InputToken],
    accent_dict: &AccentDict,
) -> (Vec<Option<u8>>, EnrichStats) {
    let mut stats = EnrichStats {
        total: tokens.len(),
        ..EnrichStats::default()
    };
    let out = tokens
        .iter()
        .map(|t| {
            if let Some(acc) = accent_dict.lookup_exact(&t.lemma, &t.reading) {
                stats.hit_lemma += 1;
                return Some(acc);
            }
            if let Some(dash_pos) = t.lemma.find('-') {
                let base = &t.lemma[..dash_pos];
                if !base.is_empty()
                    && let Some(acc) = accent_dict.lookup_exact(base, &t.reading)
                {
                    stats.hit_dash += 1;
                    return Some(acc);
                }
            }
            stats.miss += 1;
            None
        })
        .collect();
    (out, stats)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_token(lemma: &str, reading: &str) -> InputToken {
        InputToken {
            surface: lemma.to_string(),
            pos: "名詞".to_string(),
            pos_detail1: "*".to_string(),
            pos_detail2: "*".to_string(),
            pos_detail3: "*".to_string(),
            ctype: "*".to_string(),
            cform: "*".to_string(),
            lemma: lemma.to_string(),
            reading: reading.to_string(),
            pronunciation: reading.to_string(),
        }
    }

    #[test]
    fn resolve_direct_hit() {
        let mut dict = AccentDict::new();
        dict.insert("猫", "ネコ", 1);
        dict.insert("犬", "イヌ", 2);
        let tokens = vec![make_token("猫", "ネコ"), make_token("犬", "イヌ")];
        let out = resolve_dict_accents(&tokens, &dict);
        assert_eq!(out, vec![Some(1), Some(2)]);
    }

    #[test]
    fn resolve_no_hit() {
        let dict = AccentDict::new();
        let tokens = vec![make_token("鳥", "トリ")];
        let out = resolve_dict_accents(&tokens, &dict);
        assert_eq!(out, vec![None]);
    }

    #[test]
    fn resolve_dash_fallback() {
        // 訓練データにある `マレーシア-Malaysia` ような複合 lemma に対し、
        // 辞書には `マレーシア` 単体しか登録されていない場合のフォールバック。
        let mut dict = AccentDict::new();
        dict.insert("マレーシア", "マレーシア", 2);
        let tokens = vec![make_token("マレーシア-Malaysia", "マレーシア")];
        let out = resolve_dict_accents(&tokens, &dict);
        assert_eq!(out, vec![Some(2)]);
    }

    #[test]
    fn resolve_reading_mismatch_misses() {
        // Python `accent_dict[(lemma, reading)]` と同じく、lemma が存在しても
        // 読みが一致しなければ None (別読みエントリへのフォールバック禁止)。
        let mut dict = AccentDict::new();
        dict.insert("本", "ホン", 1);
        let tokens = vec![make_token("本", "モト")];
        let out = resolve_dict_accents(&tokens, &dict);
        assert_eq!(out, vec![None]);
    }

    #[test]
    fn resolve_dash_fallback_still_misses() {
        let dict = AccentDict::new();
        let tokens = vec![make_token("X-Y", "エックスワイ")];
        let out = resolve_dict_accents(&tokens, &dict);
        assert_eq!(out, vec![None]);
    }

    #[test]
    fn enrich_stats_counts_each_path() {
        let mut dict = AccentDict::new();
        dict.insert("猫", "ネコ", 1);
        dict.insert("マレーシア", "マレーシア", 2);
        let tokens = vec![
            make_token("猫", "ネコ"),                        // hit_lemma
            make_token("マレーシア-Malaysia", "マレーシア"), // hit_dash
            make_token("鳥", "トリ"),                        // miss
        ];
        let (out, stats) = resolve_dict_accents_with_stats(&tokens, &dict);
        assert_eq!(out, vec![Some(1), Some(2), None]);
        assert_eq!(stats.total, 3);
        assert_eq!(stats.hit_lemma, 1);
        assert_eq!(stats.hit_dash, 1);
        assert_eq!(stats.miss, 1);
        assert!((stats.hit_rate() - 2.0 / 3.0).abs() < 1e-9);
    }
}
