use hasami::{DictEntry, Dictionary, dict::DictBuilder};
use kotonoha::normalizer::DictionaryNormalizer;

fn word(surface: &str, pronunciation: &str) -> DictEntry {
    DictEntry {
        surface: surface.into(),
        base_form: surface.into(),
        pos: "名詞,固有名詞,一般,*".into(),
        reading: pronunciation.into(),
        pronunciation: pronunciation.into(),
        ..DictEntry::default()
    }
}

fn dictionary(extra: Vec<DictEntry>) -> Dictionary {
    let mut builder = DictBuilder::new();
    for entry in [
        word("前", "マエ"),
        word("後", "アト"),
        word("River Cloud", "リバークラウド"),
    ]
    .into_iter()
    .chain(extra)
    {
        builder.add_entry(entry);
    }
    builder.build().unwrap()
}

#[test]
fn whole_lexical_occurrences_keep_utf8_ranges_and_selected_pronunciations() {
    let mut normalizer = DictionaryNormalizer::from_dictionary(dictionary(vec![]), "");
    let text = "前River Cloud後River Cloud";
    let selected = normalizer.token_pronunciations(text).unwrap();
    assert_eq!(
        normalizer.lexical_pronunciation_spans(text).unwrap(),
        vec![
            (3, 14, "リバークラウド".to_owned()),
            (17, 28, "リバークラウド".to_owned()),
        ]
    );
    assert_eq!(normalizer.normalize(text).unwrap(), text);
    assert_eq!(normalizer.token_pronunciations(text).unwrap(), selected);
    for text in [
        "xRiver Cloud",
        "River Cloud2",
        "path/River Cloud",
        "River Cloud-Code",
        "",
        "前後",
    ] {
        assert!(
            normalizer
                .lexical_pronunciation_spans(text)
                .unwrap()
                .is_empty()
        );
    }
}

#[test]
fn unselected_heteronyms_and_case_variant_names_block_lexical_admission() {
    let mut heteronym = word("River Cloud", "リバーハウス");
    heteronym.cost = 100;
    let mut person = word("river cloud", "リバークラウド");
    person.pos = "名詞,固有名詞,人名,一般".into();
    let mut alias = word("river cloud", "リバークラウド");
    alias.base_form = "Different Name".into();
    let mut unreadable = word("river cloud", "");
    unreadable.cost = 100;
    for extra in [heteronym, person, alias, unreadable] {
        let mut normalizer = DictionaryNormalizer::from_dictionary(dictionary(vec![extra]), "");
        let selected = vec![(0, 11, "リバークラウド".to_owned())];
        assert_eq!(
            normalizer.token_pronunciations("River Cloud").unwrap(),
            selected
        );
        assert!(
            normalizer
                .lexical_pronunciation_spans("River Cloud")
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            normalizer.token_pronunciations("River Cloud").unwrap(),
            selected
        );
    }
    let mut normalizer = DictionaryNormalizer::from_dictionary(
        dictionary(vec![word("river cloud", "リバアクラウド")]),
        "",
    );
    assert_eq!(
        normalizer
            .lexical_pronunciation_spans("River Cloud")
            .unwrap(),
        vec![(0, 11, "リバークラウド".to_owned())]
    );
}
