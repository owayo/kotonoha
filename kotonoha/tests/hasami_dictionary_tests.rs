//! hasami の辞書情報を取り込んだ発音・韻律の回帰テスト。

use hasami::dict::{DictBuilder, DictEntry};
use kotonoha::Engine;
use kotonoha::njd::InputToken;

fn analyzer() -> hasami::Analyzer {
    let mut builder = DictBuilder::new();
    for (surface, lemma, reading, ctype, cform, pos) in [
        (
            "映ろう",
            "映ろう",
            "ウツロウ",
            "五段・ワ行促音便",
            "基本形",
            "動詞,自立,*,*",
        ),
        (
            "作ろ",
            "作る",
            "ツクロ",
            "五段・ラ行",
            "未然ウ接続",
            "動詞,自立,*,*",
        ),
        ("う", "う", "ウ", "不変化型", "基本形", "助動詞,*,*,*"),
        ("猫", "猫", "ネコ", "*", "*", "名詞,一般,*,*"),
    ] {
        builder.add_entry(DictEntry {
            surface: surface.into(),
            base_form: lemma.into(),
            reading: reading.into(),
            pronunciation: reading.into(),
            conj_type: ctype.into(),
            conj_form: cform.into(),
            pos: pos.into(),
            cost: -10000,
            ..Default::default()
        });
    }
    hasami::Analyzer::from_dict(builder.build().unwrap())
}

#[test]
fn dictionary_conjugation_reaches_njd_and_preserves_dictionary_form_u() {
    let tokens: Vec<InputToken> = analyzer()
        .try_tokenize("映ろう 作ろう 猫")
        .unwrap()
        .into_iter()
        .map(InputToken::from)
        .collect();
    assert_eq!(
        tokens
            .iter()
            .map(|t| t.surface.as_str())
            .collect::<Vec<_>>(),
        ["映ろう", "作ろ", "う", "猫"]
    );
    assert_eq!(tokens[0].ctype, "五段・ワ行促音便");
    assert_eq!(tokens[0].cform, "基本形");
    assert_eq!(tokens[1].cform, "未然ウ接続");
    assert_eq!(tokens[3].ctype, "*");
    assert_eq!(tokens[3].cform, "*");

    let engine = Engine::default();
    let nodes = engine.analyze(&tokens);
    assert_eq!(
        nodes
            .iter()
            .map(|n| n.pronunciation.as_str())
            .collect::<Vec<_>>(),
        ["ウツロウ", "ツクロ", "オ", "ネコ"]
    );
    assert_eq!(nodes[0].cform, "基本形");
    assert_eq!(nodes[1].ctype, "五段・ラ行");
    assert_eq!(
        engine
            .tokens_to_phone_tones(&tokens[..1])
            .iter()
            .map(|pt| pt.phone.as_str())
            .collect::<Vec<_>>(),
        ["sil", "u", "ts", "u", "r", "o", "u", "sil"]
    );
}

#[test]
fn unknown_halfwidth_kana_is_pronounced_using_hasami_reading() {
    let tokens: Vec<InputToken> = analyzer()
        .try_tokenize("ｶﾞﾌﾞﾋﾟ")
        .unwrap()
        .into_iter()
        .map(InputToken::from)
        .collect();
    assert_eq!(
        tokens
            .iter()
            .map(|t| t.reading.as_str())
            .collect::<String>(),
        "ガブピ"
    );
    assert_eq!(
        Engine::default()
            .tokens_to_phone_tones(&tokens)
            .iter()
            .map(|pt| pt.phone.as_str())
            .collect::<Vec<_>>(),
        ["sil", "g", "a", "b", "u", "p", "i", "sil"]
    );
}
