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
