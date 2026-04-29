//! v66 系特徴抽出用の vocab テーブル
//!
//! `kotonoha/training/train_onnx_v60.py` の以下と完全一致させる:
//! - POS_VOCAB (15 entries)
//! - PD1_VOCAB (21 entries)
//! - PD2_VOCAB (11 entries)
//! - CONJ_TYPE_GROUPS (9 entries)
//! - CONJ_FORM_GROUPS (10 entries)

/// 品詞文字列 → ID
///
/// 訓練側 train_onnx_v60.py:67-83 と一致。`<unk>` は 0。
pub fn pos_to_id(pos: &str) -> u32 {
    match pos {
        "名詞" => 1,
        "助詞" => 2,
        "動詞" => 3,
        "助動詞" => 4,
        "接尾辞" => 5,
        "形容詞" => 6,
        "代名詞" => 7,
        "副詞" => 8,
        "形状詞" => 9,
        "連体詞" => 10,
        "接頭辞" => 11,
        "接続詞" => 12,
        "感動詞" => 13,
        "記号" => 14,
        _ => 0,
    }
}

/// 品詞詳細1 → ID
///
/// 訓練側 train_onnx_v60.py:86-108 と一致。`<unk>` は 0。
pub fn pos_detail1_to_id(detail: &str) -> u32 {
    match detail {
        "*" => 1,
        "普通名詞" => 2,
        "格助詞" => 3,
        "非自立可能" => 4,
        "固有名詞" => 5,
        "接続助詞" => 6,
        "係助詞" => 7,
        "副助詞" => 8,
        "一般" => 9,
        "終助詞" => 10,
        "準体助詞" => 11,
        "数詞" => 12,
        "助動詞語幹" => 13,
        "タリ" => 14,
        "フィラー" => 15,
        "動詞的" => 16,
        "名詞的" => 17,
        "形容詞的" => 18,
        "形状詞的" => 19,
        "文字" => 20,
        _ => 0,
    }
}

/// 品詞詳細2 → ID
///
/// 訓練側 train_onnx_v60.py:111-123 と一致。`<unk>` は 0。
pub fn pos_detail2_to_id(detail: &str) -> u32 {
    match detail {
        "*" => 1,
        "一般" => 2,
        "サ変可能" => 3,
        "サ変形状詞可能" => 4,
        "副詞可能" => 5,
        "形状詞可能" => 6,
        "人名" => 7,
        "地名" => 8,
        "助数詞" => 9,
        "助数詞可能" => 10,
        _ => 0,
    }
}

/// 活用型 → グループ ID
///
/// 訓練側 train_onnx_v60.py:217-229 (`_get_conj_type_group`) と一致。
/// 完全一致 `*` → 0、prefix 一致 → グループ ID、それ以外 → 0。
pub fn conj_type_group(ctype: &str) -> u32 {
    if ctype == "*" {
        return 0;
    }
    // 訓練側は dict 反復順 (Python 3.7+ 保証) で先頭 prefix が一致した時点で抜ける。
    // 同 dict は `*=0` 以外に「五段=1, 上一段=2, 下一段=3, カ行変格=4, サ行変格=5,
    // 形容詞=6, 助動詞=7, 文語=8」の順で並ぶ。
    if ctype.starts_with("五段") {
        1
    } else if ctype.starts_with("上一段") {
        2
    } else if ctype.starts_with("下一段") {
        3
    } else if ctype.starts_with("カ行変格") {
        4
    } else if ctype.starts_with("サ行変格") {
        5
    } else if ctype.starts_with("形容詞") {
        6
    } else if ctype.starts_with("助動詞") {
        7
    } else if ctype.starts_with("文語") {
        8
    } else {
        0
    }
}

/// 活用形 → グループ ID
///
/// 訓練側 train_onnx_v60.py:232-244 (`_get_conj_form_group`) と一致。
pub fn conj_form_group(cform: &str) -> u32 {
    if cform == "*" {
        return 0;
    }
    if cform.starts_with("未然形") {
        1
    } else if cform.starts_with("連用形") {
        2
    } else if cform.starts_with("終止形") {
        3
    } else if cform.starts_with("連体形") {
        4
    } else if cform.starts_with("仮定形") {
        5
    } else if cform.starts_with("命令形") {
        6
    } else if cform.starts_with("已然形") {
        7
    } else if cform.starts_with("意志推量形") {
        8
    } else if cform.starts_with("語幹") {
        9
    } else {
        0
    }
}

/// アクセント類数（モデル出力のクラス数 0..20）
pub const NUM_CLASSES: usize = 21;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pos_vocab_matches_python() {
        assert_eq!(pos_to_id("名詞"), 1);
        assert_eq!(pos_to_id("代名詞"), 7);
        assert_eq!(pos_to_id("形状詞"), 9);
        assert_eq!(pos_to_id("接尾辞"), 5);
        assert_eq!(pos_to_id("記号"), 14);
        assert_eq!(pos_to_id("BOS/EOS"), 0); // unknown -> 0
        assert_eq!(pos_to_id(""), 0);
    }

    #[test]
    fn pd1_vocab_matches_python() {
        assert_eq!(pos_detail1_to_id("*"), 1);
        assert_eq!(pos_detail1_to_id("普通名詞"), 2);
        assert_eq!(pos_detail1_to_id("文字"), 20);
        assert_eq!(pos_detail1_to_id("未知タグ"), 0);
    }

    #[test]
    fn pd2_vocab_matches_python() {
        assert_eq!(pos_detail2_to_id("*"), 1);
        assert_eq!(pos_detail2_to_id("助数詞可能"), 10);
        assert_eq!(pos_detail2_to_id("未知タグ"), 0);
    }

    #[test]
    fn conj_type_group_matches_python() {
        assert_eq!(conj_type_group("*"), 0);
        assert_eq!(conj_type_group("五段-カ行"), 1);
        assert_eq!(conj_type_group("助動詞-ナイ"), 7);
        assert_eq!(conj_type_group("形容詞-イ"), 6);
        assert_eq!(conj_type_group("未知"), 0);
    }

    #[test]
    fn conj_form_group_matches_python() {
        assert_eq!(conj_form_group("*"), 0);
        assert_eq!(conj_form_group("連用形-一般"), 2);
        assert_eq!(conj_form_group("意志推量形"), 8);
        assert_eq!(conj_form_group("語幹"), 9);
        assert_eq!(conj_form_group("未知"), 0);
    }
}
