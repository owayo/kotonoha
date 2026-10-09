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

#[test]
fn normalized_katakana_lemma_reaches_pronunciation_and_prosody() {
    let mut builder = DictBuilder::new();
    for (surface, lemma, reading, pronunciation) in [
        ("ストゥリング", "ストリング", "ストリング", "ストリング"),
        ("クレディット", "クレジット", "クレジット", "クレジット"),
        ("キンドゥル", "キンドル", "キンドル", "キンドル"),
        ("イマックス", "イーマックス", "イーマックス", "イーマックス"),
        ("チャネル", "チャネル", "チャネル", "チャンネル"),
        ("ユーザ", "ユーザー", "ユーザー", "ユーザー"),
    ] {
        builder.add_entry(DictEntry {
            surface: surface.into(),
            base_form: lemma.into(),
            reading: reading.into(),
            pronunciation: pronunciation.into(),
            pos: "名詞,一般,*,*".into(),
            cost: -10000,
            ..Default::default()
        });
    }
    let mut analyzer = hasami::Analyzer::from_dict(builder.build().unwrap());
    let engine = Engine::default();
    for (surface, expected) in [
        ("ストゥリング", "ストリング"),
        ("クレディット", "クレジット"),
        ("キンドゥル", "キンドル"),
        ("イマックス", "イイマックス"),
        ("チャネル", "チャネル"),
        ("ユーザ", "ユウザ"),
    ] {
        let tokens: Vec<InputToken> = analyzer
            .try_tokenize(surface)
            .unwrap()
            .into_iter()
            .map(InputToken::from)
            .collect();
        let nodes = engine.analyze(&tokens);
        assert_eq!(nodes[0].pronunciation, expected, "{surface}");
        assert_eq!(nodes[0].mora_count, kotonoha::mora::count_mora(expected));
        let phones: Vec<String> = engine
            .tokens_to_phone_tones(&tokens)
            .into_iter()
            .map(|pt| pt.phone)
            .collect();
        let mut canonical = InputToken::new("語", "名詞", expected, expected);
        canonical.lemma = "語".to_string();
        assert_eq!(
            phones,
            engine
                .tokens_to_phone_tones(&[canonical])
                .into_iter()
                .map(|pt| pt.phone)
                .collect::<Vec<_>>(),
            "{surface}"
        );
        assert!(!engine.tokens_to_labels(&tokens).is_empty());
    }
}

#[test]
fn numeric_unit_reading_reaches_njd_without_changing_acronyms() {
    let mut builder = DictBuilder::new();
    for (surface, pos, reading) in [
        ("3", "名詞,数,*,*", "サン"),
        ("mL", "名詞,接尾,助数詞,*", "ミリリットル"),
        ("ML", "名詞,一般,*,*", "エムエル"),
    ] {
        builder.add_entry(DictEntry {
            surface: surface.into(),
            base_form: surface.into(),
            reading: reading.into(),
            pronunciation: reading.into(),
            pos: pos.into(),
            cost: -10000,
            ..Default::default()
        });
    }
    let mut analyzer = hasami::Analyzer::from_dict(builder.build().unwrap());
    let engine = Engine::default();
    for (text, expected) in [("3mL", "ミリリットル"), ("3ML", "エムエル")] {
        let tokens: Vec<InputToken> = analyzer
            .try_tokenize(text)
            .unwrap()
            .into_iter()
            .map(InputToken::from)
            .collect();
        let nodes = engine.analyze(&tokens);
        assert_eq!(nodes[1].pronunciation, expected);
        assert_eq!(nodes[1].mora_count, kotonoha::mora::count_mora(expected));
        assert!(!engine.tokens_to_phone_tones(&tokens).is_empty());
        assert!(!engine.tokens_to_labels(&tokens).is_empty());
    }
}
