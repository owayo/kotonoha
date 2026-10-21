//! hasami の辞書情報を取り込んだ発音・韻律の回帰テスト。

use hasami::dict::{DictBuilder, DictEntry};
use kotonoha::Engine;
use kotonoha::njd::InputToken;

#[test]
#[ignore = "KOTONOHA_TEST_DICT に監査済みの配布辞書を指定する"]
fn distributed_kana_boundaries_reach_pronunciation_and_labels() {
    let path = std::env::var("KOTONOHA_TEST_DICT").expect("KOTONOHA_TEST_DICT is required");
    let mut engine = Engine::default();
    engine.load_dictionary(std::path::Path::new(&path)).unwrap();
    for (text, expected) in [
        ("髙", "タカ"),
        ("進化形", "シンカケイ"),
        ("もうすでに", "モオスデニ"),
        ("くがつよっか", "クガツヨッカ"),
        ("くがつはつか", "クガツハツカ"),
        ("みっか", "ミッカ"),
        ("むいか", "ムイカ"),
        ("三がつついたちから", "サンガツツイタチカラ"),
        ("塩はつかわない", "シオワツカワナイ"),
        ("いっかいは", "イッカイワ"),
        ("よじ", "ヨジ"),
        ("二かい", "ニカイ"),
        ("十二こ", "ジュウニコ"),
    ] {
        let mut nodes = engine.text_to_analyze(text).unwrap();
        assert_eq!(
            nodes
                .iter()
                .map(|n| n.pronunciation.as_str())
                .collect::<String>(),
            expected,
            "{text}"
        );
        for node in &nodes {
            assert_eq!(
                node.mora_count,
                kotonoha::mora::parse_mora(&node.pronunciation).len(),
                "{text}"
            );
        }
        let phrases = engine.estimate_accent(&mut nodes);
        assert_eq!(
            phrases.iter().map(|p| p.mora_count).sum::<usize>(),
            kotonoha::mora::count_mora(expected)
        );
        let phones = engine.extract_phone_tones(&nodes, &phrases);
        let labels = engine.make_label(&nodes, &phrases);
        assert_eq!(phones.len(), labels.len(), "{text}");
        assert_eq!(
            phones
                .iter()
                .filter(|p| !matches!(p.phone.as_str(), "sil" | "pau"))
                .map(|p| p.phone.as_str())
                .collect::<Vec<_>>(),
            kotonoha::phoneme::katakana_to_phonemes(expected)
        );
        assert!(!engine.text_to_prosody_symbols(text).unwrap().is_empty());
    }
}

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

#[test]
fn quantity_and_counter_readings_reach_pronunciation_and_labels() {
    let mut builder = DictBuilder::new();
    for (surface, pos, reading) in [
        ("0", "名詞,数,*,*", "ゼロ"),
        ("1", "名詞,数,*,*", "イチ"),
        ("2", "名詞,数,*,*", "ニ"),
        ("3", "名詞,数,*,*", "サン"),
        ("5", "名詞,数,*,*", "ゴ"),
        ("三", "名詞,数,*,*", "サン"),
        ("一", "名詞,数,*,*", "イチ"),
        ("六", "名詞,数,*,*", "ロク"),
        ("本", "名詞,接尾,助数詞,*", "ホン"),
        ("分", "名詞,接尾,助数詞,*", "フン"),
        ("回", "名詞,接尾,助数詞,*", "カイ"),
        ("円", "名詞,接尾,助数詞,*", "エン"),
        ("倍", "名詞,接尾,助数詞,*", "バイ"),
        ("F", "名詞,一般,*,*", "ファラド"),
        ("ISO", "名詞,固有名詞,一般,*", "アイエスオー"),
        (",", "記号,読点,*,*", ""),
        (".", "記号,句点,*,*", ""),
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
    for (text, expected) in [
        ("三本", "サンボン"),
        ("一本", "イッポン"),
        ("一回", "イッカイ"),
        ("六分", "ロップン"),
        ("10分", "ジュップン"),
        ("1,000円", "センエン"),
        ("1000本", "センボン"),
        ("2.5倍", "ニテンゴバイ"),
        ("3 F", "サンエフ"),
        ("3\tF", "サンエフ"),
        ("3F", "サンファラド"),
        ("ISO 10分", "アイエスオーイチゼロフン"),
        ("10 分", "イチゼロフン"),
    ] {
        let tokens: Vec<InputToken> = analyzer
            .try_tokenize(text)
            .unwrap()
            .into_iter()
            .map(InputToken::from)
            .collect();
        let mut nodes = engine.analyze(&tokens);
        let canonical = engine.analyze(&[InputToken::new("語", "名詞", expected, expected)]);
        assert_eq!(
            nodes
                .iter()
                .map(|n| n.pronunciation.as_str())
                .collect::<String>(),
            canonical[0].pronunciation,
            "{text}"
        );
        for node in &nodes {
            assert_eq!(
                node.mora_count,
                kotonoha::mora::count_mora(&node.pronunciation)
            );
        }
        let phrases = engine.estimate_accent(&mut nodes);
        let phones = engine.extract_phone_tones(&nodes, &phrases);
        let labels = engine.make_label(&nodes, &phrases);
        assert!(!phrases.is_empty() && phones.len() > 2, "{text}");
        assert_eq!(phones.len(), labels.len(), "{text}");
    }
}

#[test]
fn spaced_month_readings_reach_pronunciation_and_labels() {
    let mut builder = DictBuilder::new();
    for (surface, pos, reading, pronunciation) in [
        ("1", "名詞,数,*,*", "イチ", "イチ"),
        ("2", "名詞,数,*,*", "ニ", "ニ"),
        ("4", "名詞,数,*,*", "ヨン", "ヨン"),
        ("７", "名詞,数,*,*", "ナナ", "ナナ"),
        ("九", "名詞,数,*,*", "キュウ", "キュー"),
        ("十", "名詞,数,*,*", "ジュウ", "ジュー"),
        ("月", "名詞,一般,*,*", "ツキ", "ツキ"),
        ("間", "名詞,接尾,一般,*", "カン", "カン"),
    ] {
        builder.add_entry(DictEntry {
            surface: surface.into(),
            base_form: surface.into(),
            reading: reading.into(),
            pronunciation: pronunciation.into(),
            pos: pos.into(),
            cost: -10000,
            ..Default::default()
        });
    }
    let mut analyzer = hasami::Analyzer::from_dict(builder.build().unwrap());
    let engine = Engine::default();
    for (text, expected) in [
        ("4 月", "シガツ"),
        ("７\t月", "シチガツ"),
        ("九 月", "クガツ"),
        ("十 月", "ジュウガツ"),
        ("12 月", "ジュウニガツ"),
        ("月", "ツキ"),
        ("九 月間", "キュウツキカン"),
    ] {
        let tokens: Vec<InputToken> = analyzer
            .try_tokenize(text)
            .unwrap()
            .into_iter()
            .map(InputToken::from)
            .collect();
        let mut nodes = engine.analyze(&tokens);
        assert_eq!(
            nodes
                .iter()
                .map(|node| node.pronunciation.as_str())
                .collect::<String>(),
            expected,
            "{text}"
        );
        let phrases = engine.estimate_accent(&mut nodes);
        assert_eq!(
            phrases
                .iter()
                .map(|phrase| phrase.mora_count)
                .sum::<usize>(),
            kotonoha::mora::count_mora(expected),
            "{text}"
        );
        let phones = engine.extract_phone_tones(&nodes, &phrases);
        assert_eq!(phones.len(), engine.make_label(&nodes, &phrases).len());
        assert_eq!(
            phones
                .iter()
                .filter(|phone| phone.phone != "sil")
                .map(|phone| phone.phone.as_str())
                .collect::<Vec<_>>(),
            kotonoha::phoneme::katakana_to_phonemes(expected),
            "{text}"
        );
    }
}
