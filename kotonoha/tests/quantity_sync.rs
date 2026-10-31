//! hasami の数量補正を実際の辞書読込から NJD・音素・ラベルまで検証する。

use hasami::dict::DictBuilder;
use kotonoha::{Engine, mora, phoneme};
use std::path::Path;

#[test]
fn quantity_readings_reach_njd_phones_and_labels() {
    let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/quantities");
    let mut builder = DictBuilder::new();
    builder.add_csv_dir(&source).unwrap();
    builder.load_matrix(source.join("matrix.def")).unwrap();
    builder.load_unk(source.join("unk.def")).unwrap();
    builder.load_char_def(source.join("char.def")).unwrap();
    let temp = tempfile::tempdir().unwrap();
    let dict = temp.path().join("quantity.hsd");
    builder
        .write_hsd(&dict, &builder.write_options(), |_, _| {})
        .unwrap();
    let mut engine = Engine::default();
    engine.load_dictionary(&dict).unwrap();
    let cases: Vec<[String; 3]> =
        serde_json::from_str(include_str!("fixtures/quantities/cases.json")).unwrap();

    for [text, reading, pronunciation] in cases {
        let mut nodes = engine.text_to_analyze(&text).unwrap();
        assert_eq!(
            nodes.iter().map(|n| n.reading.as_str()).collect::<String>(),
            reading,
            "{text}"
        );
        assert_eq!(
            nodes
                .iter()
                .map(|n| n.pronunciation.as_str())
                .collect::<String>(),
            pronunciation,
            "{text}"
        );
        for node in &nodes {
            assert_eq!(node.mora_count, mora::count_mora(&node.pronunciation));
        }
        let phrases = engine.estimate_accent(&mut nodes);
        assert_eq!(
            phrases
                .iter()
                .map(|phrase| phrase.mora_count)
                .sum::<usize>(),
            mora::count_mora(&pronunciation),
            "{text}"
        );
        let phones = engine.extract_phone_tones(&nodes, &phrases);
        assert_eq!(
            phones[1..phones.len() - 1]
                .iter()
                .map(|phone| phone.phone.as_str())
                .collect::<Vec<_>>(),
            phoneme::katakana_to_phonemes(&pronunciation),
            "{text}"
        );
        assert_eq!(phones.len(), engine.make_label(&nodes, &phrases).len());
    }
}
