use hasami::dict::{DictBuilder, DictEntry};
use kotonoha::normalizer::DictionaryNormalizer;
use kotonoha::{Engine, njd::InputToken};

fn dictionary() -> hasami::Dictionary {
    let mut builder = DictBuilder::new();
    for (surface, pronunciation, pos) in [
        ("C++", "シープラスプラス", "名詞,固有名詞,一般,*"),
        ("C#", "シーシャープ", "名詞,固有名詞,一般,*"),
        ("AT&T", "エーティーアンドティー", "名詞,固有名詞,組織,*"),
        ("(笑)", "ワライ", "記号,一般,*,*"),
        ("（笑）", "ワライ", "記号,一般,*,*"),
        ("笑", "エミ", "名詞,固有名詞,人名,名"),
        ("(opaque)", "opaque", "記号,一般,*,*"),
        ("1.5倍", "イチテンゴバイ", "名詞,固有名詞,一般,*"),
        ("(A)", "カオモジ", "記号,一般,*,*"),
    ] {
        builder.add_entry(DictEntry {
            surface: surface.into(),
            base_form: if matches!(surface, "(笑)" | "（笑）") {
                "笑い"
            } else {
                surface
            }
            .into(),
            reading: pronunciation.into(),
            pronunciation: pronunciation.into(),
            pos: pos.into(),
            cost: -1000,
            ..Default::default()
        });
    }
    builder.build().unwrap()
}

#[test]
fn symbolic_words_use_dictionary_pronunciations_and_keep_input_gaps() {
    let mut normalizer = DictionaryNormalizer::from_dictionary(dictionary(), "+#&");
    assert_eq!(
        normalizer.normalize("  C++\tC#\nAT&T C++20  ").unwrap(),
        "  シープラスプラス\tシーシャープ\nエーティーアンドティー シープラスプラス20  "
    );
    assert_eq!(
        normalizer.normalize("猫😀C#、C++。猫").unwrap(),
        "猫😀シーシャープ、シープラスプラス。猫"
    );
}

#[test]
fn only_selected_symbols_and_symbol_annotations_are_rewritten() {
    let mut normalizer = DictionaryNormalizer::from_dictionary(dictionary(), "+");
    assert_eq!(
        normalizer.normalize("C++ C# AT&T (笑)").unwrap(),
        "シープラスプラス C# AT&T (ワライ)"
    );
}

#[test]
fn identifiers_and_arithmetic_are_not_rewritten() {
    let mut normalizer = DictionaryNormalizer::from_dictionary(dictionary(), "+#&");
    for text in [
        "C+1",
        "ABC++",
        "C++++",
        "AC#",
        "_C++",
        "C++Parser",
        "C#_name",
        "2C++",
        "AT&T&T",
        "1.5倍",
    ] {
        assert_eq!(normalizer.normalize(text).unwrap(), text);
    }
}

#[test]
fn annotations_keep_brackets_without_changing_person_names() {
    let mut normalizer = DictionaryNormalizer::from_dictionary(dictionary(), "+#&");
    for (text, expected) in [
        ("(笑)", "(ワライ)"),
        ("（笑）", "（ワライ）"),
        ("笑さん", "笑さん"),
        ("(笑顔)", "(笑顔)"),
        ("(opaque)", "(opaque)"),
        ("(A)", "(A)"),
        ("未登録#", "未登録#"),
        ("", ""),
    ] {
        assert_eq!(normalizer.normalize(text).unwrap(), expected);
    }
}

#[test]
fn normalization_is_idempotent_and_uses_the_supplied_dictionary() {
    let text = "C++ (笑) AT&T";
    let mut normalizer = DictionaryNormalizer::from_dictionary(dictionary(), "+#&");
    let once = normalizer.normalize(text).unwrap();
    assert_eq!(normalizer.normalize(&once).unwrap(), once);
    let mut builder = DictBuilder::new();
    builder.add_entry(DictEntry {
        surface: "猫".into(),
        pos: "名詞,一般,*,*".into(),
        ..Default::default()
    });
    let mut other = DictionaryNormalizer::from_dictionary(builder.build().unwrap(), "+#&");
    assert_eq!(other.normalize(text).unwrap(), text);
}

#[test]
fn actual_tokens_preserve_symbol_pronunciations_in_njd_and_prosody() {
    let mut analyzer = hasami::Analyzer::from_dict(dictionary());
    let engine = Engine::default();
    for (text, expected) in [("C#", "シイシャアプ"), ("(笑)", "ワライ")] {
        let tokens: Vec<InputToken> = analyzer
            .try_tokenize(text)
            .unwrap()
            .into_iter()
            .map(InputToken::from)
            .collect();
        let nodes = engine.analyze(&tokens);
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0].pronunciation, expected);
        assert_eq!(nodes[0].mora_count, kotonoha::mora::count_mora(expected));
        assert!(!engine.tokens_to_labels(&tokens).is_empty());
        assert!(engine.tokens_to_phone_tones(&tokens).len() > 2);
        assert!(!engine.tokens_to_prosody_symbols(&tokens).is_empty());
    }
}

fn contextual_dictionary(
    value_pos: &str,
    reading: &str,
    pronunciation: &str,
) -> hasami::Dictionary {
    let mut builder = DictBuilder::new();
    for (surface, pos, r, p, kind, form) in [
        ("関数", "名詞,一般,*,*", "カンスウ", "カンスー", "", ""),
        ("式", "名詞,一般,*,*", "シキ", "シキ", "", ""),
        ("自体", "名詞,一般,*,*", "ジタイ", "ジタイ", "", ""),
        ("が", "助詞,格助詞,一般,*", "ガ", "ガ", "", ""),
        (
            "返す",
            "動詞,自立,*,*",
            "カエス",
            "カエス",
            "五段・サ行",
            "基本形",
        ),
        (
            "とり",
            "動詞,自立,*,*",
            "トリ",
            "トリ",
            "五段・ラ行",
            "連用形",
        ),
        (
            "うる",
            "動詞,非自立,*,*",
            "ウル",
            "ウル",
            "一段・得ル",
            "基本形",
        ),
        ("最大", "名詞,一般,*,*", "サイダイ", "サイダイ", "", ""),
        ("最も", "副詞,一般,*,*", "モットモ", "モットモ", "", ""),
        ("頻度", "名詞,一般,*,*", "ヒンド", "ヒンド", "", ""),
        (
            "高い",
            "形容詞,自立,*,*",
            "タカイ",
            "タカイ",
            "形容詞・アウオ段",
            "基本形",
        ),
        (
            "です",
            "助動詞,*,*,*",
            "デス",
            "デス",
            "特殊・デス",
            "基本形",
        ),
        ("アタイ", "名詞,固有名詞,一般,*", "アタイ", "アタイ", "", ""),
        ("の", "助詞,連体化,*,*", "ノ", "ノ", "", ""),
        ("値", value_pos, reading, pronunciation, "", ""),
        ("は", "助詞,係助詞,*,*", "ハ", "ワ", "", ""),
        ("を", "助詞,格助詞,一般,*", "ヲ", "ヲ", "", ""),
        (
            "求める",
            "動詞,自立,*,*",
            "モトメル",
            "モトメル",
            "一段",
            "基本形",
        ),
        ("f", "記号,アルファベット,*,*", "エフ", "エフ", "", ""),
        ("8", "名詞,数,*,*", "ハチ", "ハチ", "", ""),
        ("魚", "名詞,一般,*,*", "サカナ", "サカナ", "", ""),
        ("。", "記号,句点,*,*", "。", "。", "", ""),
        ("、", "記号,読点,*,*", "、", "、", "", ""),
        (",", "記号,読点,*,*", ",", ",", "", ""),
        ("(", "記号,括弧開,*,*", "(", "(", "", ""),
        (")", "記号,括弧閉,*,*", ")", ")", "", ""),
        ("「", "記号,括弧開,*,*", "「", "「", "", ""),
        ("」", "記号,括弧閉,*,*", "」", "」", "", ""),
        (
            "C++",
            "名詞,固有名詞,一般,*",
            "シープラスプラス",
            "シープラスプラス",
            "",
            "",
        ),
        ("(笑)", "記号,一般,*,*", "ワライ", "ワライ", "", ""),
    ] {
        builder.add_entry(DictEntry {
            surface: surface.into(),
            pos: pos.into(),
            base_form: match surface {
                "(笑)" => "笑い",
                "とり" => "とる",
                _ => surface,
            }
            .into(),
            reading: r.into(),
            pronunciation: p.into(),
            conj_type: kind.into(),
            conj_form: form.into(),
            cost: -1000,
            ..Default::default()
        });
    }
    builder.build().unwrap()
}

#[test]
fn contextual_corrections_reach_normalization_without_symbol_characters() {
    let mut analyzer =
        hasami::Analyzer::from_dict(contextual_dictionary("名詞,一般,*,*", "ネ", "ネ"));
    let mut normalizer = DictionaryNormalizer::from_dictionary(
        contextual_dictionary("名詞,一般,*,*", "ネ", "ネ"),
        "",
    );
    let text = "関数の値。";
    let receipts = analyzer.contextual_reading_spans(text).unwrap();
    assert_eq!(receipts.len(), 1);
    assert_eq!(&text[receipts[0].start..receipts[0].end], "値");
    let selected = normalizer.token_pronunciations(text).unwrap();
    assert_eq!(normalizer.normalize(text).unwrap(), "関数のアタイ。");
    assert_eq!(normalizer.token_pronunciations(text).unwrap(), selected);
    assert_eq!(
        normalizer.normalize("関数の値、").unwrap(),
        "関数のアタイ、"
    );
    assert_eq!(
        normalizer.normalize("関数のアタイ。").unwrap(),
        "関数のアタイ。"
    );
}

#[test]
fn contextual_and_symbol_edits_share_original_utf8_positions() {
    let mut normalizer = DictionaryNormalizer::from_dictionary(
        contextual_dictionary("名詞,一般,*,*", "ネ", "ネ"),
        "+",
    );
    let text = "  C++。\t関数の値。\nf(8)の値を求める。(笑)  ";
    assert_eq!(
        normalizer.normalize(text).unwrap(),
        "  シープラスプラス。\t関数のアタイ。\nf(8)のアタイを求める。(ワライ)  "
    );
    // 原文の文脈で決定し、括弧・引数・改行や末尾の注記は別々の座標で保つ。
    assert_eq!(normalizer.normalize("f(8)の値を").unwrap(), "f(8)の値を");
}

#[test]
fn contextual_projection_preserves_numeric_labels_and_nonreceipt_bytes() {
    let mut normalizer = DictionaryNormalizer::from_dictionary(
        contextual_dictionary("名詞,一般,*,*", "ネ", "ネ"),
        "+",
    );
    for suffix in ["0002", "3・14", "70m2"] {
        let text = format!("関数の値は{suffix}");
        assert_eq!(
            normalizer.normalize(&text).unwrap(),
            format!("関数のアタイは{suffix}")
        );
    }
    for text in [
        "魚の値を求める。",
        "「関数」の値。",
        "関数 の値。",
        "f8の値を求める。",
        "関数の値,ね,を求める。",
        "モドリチ",
        "",
        " \t\n",
    ] {
        assert_eq!(normalizer.normalize(text).unwrap(), text);
    }
}

#[test]
fn already_correct_custom_and_name_pronunciations_have_no_projection() {
    for (pos, reading, pronunciation) in [
        ("名詞,一般,*,*", "アタイ", "アタイ"),
        ("名詞,一般,*,*", "チ", "チ"),
        ("名詞,一般,*,*", "ネ", "アタイ"),
        ("名詞,固有名詞,人名,名", "ネ", "ネ"),
    ] {
        let mut normalizer = DictionaryNormalizer::from_dictionary(
            contextual_dictionary(pos, reading, pronunciation),
            "",
        );
        assert_eq!(normalizer.normalize("関数の値。").unwrap(), "関数の値。");
    }
}

#[test]
fn mathematical_contexts_use_shared_corrections_inside_outer_quotes() {
    let mut normalizer = DictionaryNormalizer::from_dictionary(
        contextual_dictionary("名詞,一般,*,*", "ネ", "ネ"),
        "+",
    );
    for text in [
        "「関数の値」関数の値。",
        "「魚。関数の値」関数の値。",
        "「魚。\n関数の値」関数の値。",
        "「関数の値」を買う。",
        "「f(8)の値を求める」",
    ] {
        let expected = text.replace('値', "アタイ");
        assert_eq!(normalizer.normalize(text).unwrap(), expected);
    }
    // 価格や非連続の構文はこの委譲経路で補正されない。従来の字面保持を、
    // 文脈上の発音が「ネ」であるという根拠にはしない。
    for text in [
        "「魚の値」を買う。",
        "「関数 の値を求める」を買う。",
        "「積 分した値を求める」を買う。",
        "関数の値（ね）を読む。",
    ] {
        assert_eq!(normalizer.normalize(text).unwrap(), text);
    }
}

#[test]
fn strong_math_readings_project_only_original_utf8_receipts() {
    let mut analyzer =
        hasami::Analyzer::from_dict(contextual_dictionary("名詞,一般,*,*", "ネ", "ネ"));
    let mut normalizer = DictionaryNormalizer::from_dictionary(
        contextual_dictionary("名詞,一般,*,*", "ネ", "ネ"),
        "+",
    );
    let input = "  C++。😀。式自体が値を返す。\nとりうる最大の値。\t最も頻度が高い値です。  0002/3・14/70m2";
    let receipts = analyzer.contextual_reading_spans(input).unwrap();
    assert_eq!(receipts.len(), 3);
    assert!(receipts.iter().all(
        |span| input.get(span.start..span.end) == Some("値") && span.pronunciation == "アタイ"
    ));
    let selected = normalizer.token_pronunciations(input).unwrap();
    let normalized = normalizer.normalize(input).unwrap();
    assert_eq!(
        normalized,
        "  シープラスプラス。😀。式自体がアタイを返す。\nとりうる最大のアタイ。\t最も頻度が高いアタイです。  0002/3・14/70m2"
    );
    assert_eq!(normalizer.token_pronunciations(input).unwrap(), selected);
    assert_eq!(normalizer.normalize(&normalized).unwrap(), normalized);
}

#[test]
fn strong_math_readings_reach_njd_and_phones_from_actual_analyzer_tokens() {
    let mut analyzer =
        hasami::Analyzer::from_dict(contextual_dictionary("名詞,一般,*,*", "ネ", "ネ"));
    let mut normalizer = DictionaryNormalizer::from_dictionary(
        contextual_dictionary("名詞,一般,*,*", "ネ", "ネ"),
        "",
    );
    let engine = Engine::default();
    for text in [
        "式自体が値を返す。",
        "とりうる最大の値。",
        "最も頻度が高い値です。",
    ] {
        let normalized = normalizer.normalize(text).unwrap();
        let raw = analyzer.try_tokenize(text).unwrap();
        let projected = analyzer.try_tokenize(&normalized).unwrap();
        let raw_value = raw.iter().find(|t| t.surface.as_ref() == "値").unwrap();
        let projected_value = projected
            .iter()
            .find(|t| t.surface.as_ref() == "アタイ")
            .unwrap();
        // 異なる入力の表層・品詞を同値扱いせず、採用された発音と音素を確かめる。
        assert_ne!(raw_value.surface, projected_value.surface);
        for selected in [raw_value, projected_value] {
            let tokens = vec![InputToken::from(selected.clone())];
            let nodes = engine.analyze(&tokens);
            assert_eq!(nodes.len(), 1);
            assert_eq!(nodes[0].pronunciation, "アタイ");
            assert_eq!(nodes[0].mora_count, 3);
            let tones = engine.tokens_to_phone_tones(&tokens);
            let phones: Vec<_> = tones.iter().map(|tone| tone.phone.as_str()).collect();
            assert_eq!(phones, ["sil", "a", "t", "a", "i", "sil"]);
        }
    }
}

#[test]
fn strong_math_readings_preserve_prices_names_notes_and_identifiers() {
    let mut normalizer = DictionaryNormalizer::from_dictionary(
        contextual_dictionary("名詞,一般,*,*", "ネ", "ネ"),
        "",
    );
    for text in [
        "魚の値を返す。",
        "最も高い値です。",
        "頻度が高い値です。",
        "式自体が値（ね）を返す。",
        "とりうる最大の値（ね）。",
        "最も頻度が高い値（ね）です。",
        "SKU-式自体が値を返す。",
        "SKU：とりうる最大の値。",
        "SKU/最も頻度が高い値です。",
        "😀式自体が値を返す。",
        "式 自体が値を返す。",
        "とり うる最大の値。",
        "最も 頻度が高い値です。",
        "「式」自体が値を返す。",
    ] {
        assert_eq!(normalizer.normalize(text).unwrap(), text, "{text}");
    }
    for (pos, reading, pronunciation) in [
        ("名詞,固有名詞,人名,名", "ネ", "ネ"),
        ("名詞,一般,*,*", "アタイ", "アタイ"),
        ("名詞,一般,*,*", "チ", "チ"),
        ("名詞,一般,*,*", "ネ", "アタイ"),
    ] {
        let mut normalizer = DictionaryNormalizer::from_dictionary(
            contextual_dictionary(pos, reading, pronunciation),
            "",
        );
        for text in [
            "式自体が値を返す。",
            "とりうる最大の値。",
            "最も頻度が高い値です。",
        ] {
            assert_eq!(normalizer.normalize(text).unwrap(), text);
        }
    }
}
