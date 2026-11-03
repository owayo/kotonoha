//! 段位の実辞書候補が、姓の素性を残さず下流へ届くことを検証する。

use hasami::dict::{DictBuilder, DictEntry};
use kotonoha::Engine;
use kotonoha::njd::InputToken;
use kotonoha::normalizer::DictionaryNormalizer;

fn dictionary() -> DictBuilder {
    let mut builder = DictBuilder::new();
    for (reading, pos, id, cost) in [
        ("ハッタン", "名詞,固有名詞,人名,姓", 1290, 3582),
        ("ハチダン", "名詞,一般,*,*", 1285, 5606),
    ] {
        builder.add_entry(DictEntry {
            surface: "八段".into(),
            base_form: "八段".into(),
            reading: reading.into(),
            pronunciation: reading.into(),
            pos: pos.into(),
            left_id: id,
            right_id: id,
            cost,
            ..Default::default()
        });
    }
    for (surface, lemma, reading, pronunciation, pos) in [
        ("段位", "段位", "ダンイ", "ダンイ", "名詞,一般,*,*"),
        ("姓", "姓", "セイ", "セイ", "名詞,一般,*,*"),
        ("名字", "名字", "ミョウジ", "ミョージ", "名詞,一般,*,*"),
        ("棋士", "棋士", "キシ", "キシ", "名詞,一般,*,*"),
        ("田中", "田中", "タナカ", "タナカ", "名詞,固有名詞,人名,姓"),
        ("太郎", "太郎", "タロウ", "タロー", "名詞,固有名詞,人名,名"),
        (
            "ハッタン",
            "ハッタン",
            "ハッタン",
            "ハッタン",
            "名詞,一般,*,*",
        ),
        ("は", "は", "ハ", "ワ", "助詞,係助詞,*,*"),
        ("の", "の", "ノ", "ノ", "助詞,格助詞,一般,*"),
        ("を", "を", "ヲ", "ヲ", "助詞,格助詞,一般,*"),
        ("と", "と", "ト", "ト", "助詞,格助詞,一般,*"),
        ("読む", "読む", "ヨム", "ヨム", "動詞,自立,*,*"),
        ("という", "という", "トイウ", "トユウ", "助詞,格助詞,連語,*"),
        ("です", "です", "デス", "デス", "助動詞,*,*,*"),
        ("さん", "さん", "サン", "サン", "名詞,接尾,人名,*"),
        ("目指す", "目指す", "メザス", "メザス", "動詞,自立,*,*"),
        ("。", "。", "。", "。", "記号,句点,*,*"),
        ("「", "「", "「", "「", "記号,括弧開,*,*"),
        ("」", "」", "」", "」", "記号,括弧閉,*,*"),
        ("『", "『", "『", "『", "記号,括弧開,*,*"),
        ("』", "』", "』", "』", "記号,括弧閉,*,*"),
    ] {
        builder.add_entry(DictEntry {
            surface: surface.into(),
            base_form: lemma.into(),
            reading: reading.into(),
            pronunciation: pronunciation.into(),
            pos: pos.into(),
            cost: -10000,
            ..Default::default()
        });
    }
    builder
}

#[test]
fn actual_rank_candidate_reaches_njd_normalizer_phones_and_labels() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("rank.hsd");
    let builder = dictionary();
    builder
        .write_hsd(&path, &builder.write_options(), |_, _| {})
        .unwrap();
    let mut analyzer = hasami::Analyzer::load(&path).unwrap();
    let mut engine = Engine::default();
    engine.load_dictionary(&path).unwrap();
    let mut normalizer = DictionaryNormalizer::load(&path, "").unwrap();

    for (text, expected, normalized) in [
        (
            "段位は八段です。",
            &["ハチダン"][..],
            "段位はハチダンです。",
        ),
        (
            "段位は「八段」です。",
            &["ハチダン"],
            "段位は「ハチダン」です。",
        ),
        (
            "段位は『八段』です。",
            &["ハチダン"],
            "段位は『ハチダン』です。",
        ),
        ("八段を目指す。", &["ハチダン"], "ハチダンを目指す。"),
        ("棋士の田中八段。", &["ハチダン"], "棋士の田中ハチダン。"),
        ("田中太郎八段。", &["ハチダン"], "田中太郎ハチダン。"),
        ("姓は八段です。", &["ハッタン"], "姓は八段です。"),
        (
            "「八段」という名字です。",
            &["ハッタン"],
            "「八段」という名字です。",
        ),
        ("八段さん。", &["ハッタン"], "八段さん。"),
        ("八段太郎。", &["ハッタン"], "八段太郎。"),
        ("姓の八段を目指す。", &["ハッタン"], "姓の八段を目指す。"),
        (
            "ハッタンと読む八段を目指す。",
            &["ハッタン"],
            "ハッタンと読む八段を目指す。",
        ),
        (
            "ハッタンと読む 「八段」を目指す。",
            &["ハッタン"],
            "ハッタンと読む 「八段」を目指す。",
        ),
        ("八段。", &["ハッタン"], "八段。"),
        ("段位は。八段です。", &["ハッタン"], "段位は。八段です。"),
        ("段位は\n八段です。", &["ハッタン"], "段位は\n八段です。"),
        ("八段です。段位は。", &["ハッタン"], "八段です。段位は。"),
        (
            "段位は八段。八段さん。",
            &["ハチダン", "ハッタン"],
            "段位はハチダン。八段さん。",
        ),
    ] {
        let raw = analyzer.try_tokenize(text).unwrap();
        let rank_tokens: Vec<_> = raw
            .iter()
            .filter(|t| t.surface.as_ref() == "八段")
            .collect();
        assert_eq!(rank_tokens.len(), expected.len(), "{text}");
        for (token, reading) in rank_tokens.iter().zip(expected) {
            assert_eq!(token.reading.as_ref(), *reading, "{text}");
            assert_eq!(token.pronunciation.as_ref(), *reading, "{text}");
            assert_eq!(token.base_form.as_ref(), "八段", "{text}");
            assert!(token.is_known, "{text}");
            assert!(
                token.conj_type.is_empty() && token.conj_form.is_empty(),
                "{text}"
            );
            let (pos, cost) = if *reading == "ハチダン" {
                ("名詞,一般,*,*", 5606)
            } else {
                ("名詞,固有名詞,人名,姓", 3582)
            };
            assert_eq!(token.pos.as_ref(), pos, "{text}");
            assert_eq!(token.word_cost, cost, "{text}");
            assert_eq!(&text[token.start..token.end], "八段", "{text}");
            let input = InputToken::from((**token).clone());
            let parts: Vec<_> = pos.split(',').collect();
            assert_eq!(input.pos, parts[0]);
            assert_eq!(input.pos_detail1, parts[1]);
            assert_eq!(input.pos_detail2, parts[2]);
            assert_eq!(input.pos_detail3, parts[3]);
            assert_eq!(input.ctype, "*");
            assert_eq!(input.cform, "*");
            assert_eq!(input.lemma, "八段");
            assert_eq!(input.reading, *reading);
            assert_eq!(input.pronunciation, *reading);
        }

        let inputs: Vec<_> = raw.into_iter().map(InputToken::from).collect();
        let nodes = engine.text_to_analyze(text).unwrap();
        assert_eq!(nodes.len(), inputs.len(), "{text}");
        for (node, input) in nodes.iter().zip(&inputs) {
            assert_eq!(node.surface, input.surface, "{text}");
            assert_eq!(node.reading, input.reading, "{text}");
            assert_eq!(node.pos_detail1, input.pos_detail1, "{text}");
            assert_eq!(node.pos_detail2, input.pos_detail2, "{text}");
            assert_eq!(node.pos_detail3, input.pos_detail3, "{text}");
            assert_eq!(node.lemma, input.lemma, "{text}");
            if node.surface == "八段" {
                assert_eq!(node.pronunciation, node.reading, "{text}");
                assert_eq!(node.mora_count, 4, "{text}");
            }
        }
        let phones = engine.text_to_phone_tones(text).unwrap();
        let labels = engine.text_to_labels(text).unwrap();
        assert_eq!(phones.len(), labels.len(), "{text}");
        assert_eq!(
            phones
                .iter()
                .filter(|phone| !matches!(phone.phone.as_str(), "sil" | "pau"))
                .map(|phone| phone.phone.as_str())
                .collect::<Vec<_>>(),
            kotonoha::phoneme::katakana_to_phonemes(
                &nodes
                    .iter()
                    .map(|n| n.pronunciation.as_str())
                    .collect::<String>()
            ),
            "{text}"
        );
        assert!(
            !engine.text_to_prosody_symbols(text).unwrap().is_empty(),
            "{text}"
        );
        assert_eq!(normalizer.normalize(text).unwrap(), normalized, "{text}");
        let projected = normalizer.token_pronunciations(text).unwrap();
        let ranks: Vec<_> = projected
            .iter()
            .filter(|(start, end, _)| &text[*start..*end] == "八段")
            .map(|(_, _, pronunciation)| pronunciation.as_str())
            .collect();
        assert_eq!(ranks, expected, "{text}");
    }
    verify_explicit_surnames_from_split_tokens(&mut engine, dir.path());
}

fn input10(token: &InputToken) -> [&str; 10] {
    [
        &token.surface,
        &token.pos,
        &token.pos_detail1,
        &token.pos_detail2,
        &token.pos_detail3,
        &token.ctype,
        &token.cform,
        &token.lemma,
        &token.reading,
        &token.pronunciation,
    ]
}

fn split_dictionary() -> DictBuilder {
    let mut builder = dictionary();
    for (surface, reading, pos) in [
        ("八", "ハチ", "名詞,数,*,*"),
        ("段", "ダン", "名詞,接尾,助数詞,*"),
    ] {
        builder.add_entry(DictEntry {
            surface: surface.into(),
            base_form: surface.into(),
            reading: reading.into(),
            pronunciation: reading.into(),
            pos: pos.into(),
            cost: -1000,
            ..Default::default()
        });
    }
    for (surface, reading, pos) in [
        ("苗字", "ミョウジ", "名詞,一般,*,*"),
        ("が", "ガ", "助詞,格助詞,一般,*"),
        ("だ", "ダ", "助動詞,*,*,*"),
        ("、", "、", "記号,読点,*,*"),
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
    builder
}

fn verify_explicit_surnames_from_split_tokens(engine: &mut Engine, dir: &std::path::Path) {
    let path = dir.join("split-rank.hsd");
    let builder = split_dictionary();
    builder
        .write_hsd(&path, &builder.write_options(), |_, _| {})
        .unwrap();
    let mut analyzer = hasami::Analyzer::load(&path).unwrap();
    engine.load_dictionary(&path).unwrap();
    let mut normalizer = DictionaryNormalizer::load(&path, "").unwrap();

    // -1000 + -1000 は実姓3582・一般5606より安い。新分岐を通る前提を実解析でも確認する。
    let bare = analyzer.try_tokenize("八段").unwrap();
    assert_eq!(
        bare.iter().map(|t| t.surface.as_ref()).collect::<Vec<_>>(),
        ["八", "段"]
    );
    assert!(
        analyzer
            .contextual_reading_spans("八段")
            .unwrap()
            .is_empty()
    );

    // trueは明示姓の一語、falseは合法な数詞＋助数詞の分割。文脈ごとに期待を分ける。
    for (text, surnames, normalized) in [
        ("姓は八段です。", &[true][..], "姓はハッタンです。"),
        ("名字が『八段』だ。", &[true], "名字が『ハッタン』だ。"),
        ("苗字は八段だ。", &[true], "苗字はハッタンだ。"),
        ("段位は八段です。", &[false], "段位は八段です。"),
        (
            "姓は八段です、段位は八段です。",
            &[true, false],
            "姓はハッタンです、段位は八段です。",
        ),
        (
            "段位は八段です、姓は八段です。",
            &[false, true],
            "段位は八段です、姓はハッタンです。",
        ),
        (
            "姓は田中です。八段を目指す。",
            &[false],
            "姓は田中です。八段を目指す。",
        ),
        (
            "姓は田中です。\n八段を目指す。",
            &[false],
            "姓は田中です。\n八段を目指す。",
        ),
        (
            "八段を目指す。姓は田中です。",
            &[false],
            "八段を目指す。姓は田中です。",
        ),
    ] {
        let raw = analyzer.try_tokenize(text).unwrap();
        let inputs: Vec<_> = raw.iter().cloned().map(InputToken::from).collect();
        let occurrences: Vec<_> = text.match_indices("八段").map(|(start, _)| start).collect();
        assert_eq!(occurrences.len(), surnames.len(), "{text}");
        let mut expected_spans = Vec::new();
        for (start, surname) in occurrences.into_iter().zip(surnames) {
            let end = start + "八段".len();
            let selected: Vec<_> = raw
                .iter()
                .zip(&inputs)
                .filter(|(token, _)| token.start >= start && token.end <= end)
                .collect();
            let expected_fields = if *surname {
                expected_spans.push((start, end, "ハッタン".to_owned()));
                vec![[
                    "八段",
                    "名詞",
                    "固有名詞",
                    "人名",
                    "姓",
                    "*",
                    "*",
                    "八段",
                    "ハッタン",
                    "ハッタン",
                ]]
            } else {
                vec![
                    ["八", "名詞", "数", "*", "*", "*", "*", "八", "ハチ", "ハチ"],
                    [
                        "段",
                        "名詞",
                        "接尾",
                        "助数詞",
                        "*",
                        "*",
                        "*",
                        "段",
                        "ダン",
                        "ダン",
                    ],
                ]
            };
            assert_eq!(selected.len(), expected_fields.len(), "{text}");
            assert_eq!(selected.first().unwrap().0.start, start, "{text}");
            assert_eq!(selected.last().unwrap().0.end, end, "{text}");
            for ((token, input), fields) in selected.iter().zip(expected_fields) {
                assert!(token.is_known, "{text}");
                assert_eq!(
                    token.word_cost,
                    if *surname { 3582 } else { -1000 },
                    "{text}"
                );
                assert_eq!(
                    &text[token.start..token.end],
                    token.surface.as_ref(),
                    "{text}"
                );
                assert_eq!(input10(input), fields, "{text}");
            }
        }
        let actual_spans: Vec<_> = analyzer
            .contextual_reading_spans(text)
            .unwrap()
            .into_iter()
            .map(|span| (span.start, span.end, span.pronunciation))
            .collect();
        assert_eq!(actual_spans, expected_spans, "{text}");
        // 既に正しいハチ＋ダンのrankを、span無しで仮名へ投影する必要はない。
        assert_eq!(normalizer.normalize(text).unwrap(), normalized, "{text}");
        let projected = normalizer.token_pronunciations(text).unwrap();
        assert_eq!(
            projected,
            raw.iter()
                .map(|t| (t.start, t.end, t.pronunciation.to_string()))
                .collect::<Vec<_>>(),
            "{text}"
        );
        let nodes = engine.text_to_analyze(text).unwrap();
        assert_eq!(nodes.len(), inputs.len(), "{text}");
        for (node, input) in nodes.iter().zip(&inputs) {
            assert_eq!(node.surface, input.surface, "{text}");
            assert_eq!(node.reading, input.reading, "{text}");
            assert_eq!(node.lemma, input.lemma, "{text}");
            assert_eq!(node.pos_detail1, input.pos_detail1, "{text}");
            assert_eq!(node.pos_detail2, input.pos_detail2, "{text}");
            assert_eq!(node.pos_detail3, input.pos_detail3, "{text}");
            if matches!(node.surface.as_str(), "八段" | "八" | "段") {
                assert_eq!(node.pronunciation, input.pronunciation, "{text}");
                assert_eq!(
                    node.mora_count,
                    if node.surface == "八段" { 4 } else { 2 },
                    "{text}"
                );
            }
        }
        let phones = engine.text_to_phone_tones(text).unwrap();
        assert_eq!(
            phones.len(),
            engine.text_to_labels(text).unwrap().len(),
            "{text}"
        );
        assert_eq!(
            phones
                .iter()
                .filter(|p| !matches!(p.phone.as_str(), "sil" | "pau"))
                .map(|p| p.phone.as_str())
                .collect::<Vec<_>>(),
            kotonoha::phoneme::katakana_to_phonemes(
                &nodes
                    .iter()
                    .map(|n| n.pronunciation.as_str())
                    .collect::<String>()
            ),
            "{text}"
        );
        assert!(
            !engine.text_to_prosody_symbols(text).unwrap().is_empty(),
            "{text}"
        );
    }
}
