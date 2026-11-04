//! 月に続く分割された初日がHSD・InputToken・NJD・音声出力へ届くことを確認する。

use hasami::dict::{DictBuilder, DictEntry};
use kotonoha::njd::{InputToken, Pos};
use kotonoha::{Engine, mora, phoneme};

struct Fixture {
    engine: Engine,
    analyzer: hasami::Analyzer,
    _directory: tempfile::TempDir,
}

fn fixture() -> Fixture {
    let mut builder = DictBuilder::new();
    for (surface, pos, reading, pronunciation) in [
        (
            "11月",
            "名詞,副詞可能,*,*",
            "ジュウイチガツ",
            "ジューイチガツ",
        ),
        ("1", "名詞,数,*,*", "イチ", "イチ"),
        ("一", "名詞,数,*,*", "イチ", "イチ"),
        ("日", "名詞,接尾,助数詞,*", "ニチ", "ニチ"),
        ("日付", "名詞,一般,*,*", "ヒヅケ", "ヒズケ"),
        ("は", "助詞,係助詞,*,*", "ハ", "ワ"),
        ("です", "助動詞,*,*,*", "デス", "デス"),
        ("かかる", "動詞,自立,*,*", "カカル", "カカル"),
        ("ID", "名詞,固有名詞,組織,*", "アイディー", "アイディー"),
        ("四月一日", "名詞,固有名詞,人名,姓", "ワタヌキ", "ワタヌキ"),
        ("さん", "名詞,接尾,人名,*", "サン", "サン"),
        ("。", "記号,句点,*,*", "。", "。"),
        ("「", "記号,括弧開,*,*", "「", "「"),
        ("」", "記号,括弧閉,*,*", "」", "」"),
    ] {
        builder.add_entry(DictEntry {
            surface: surface.into(),
            base_form: surface.into(),
            pos: pos.into(),
            reading: reading.into(),
            pronunciation: pronunciation.into(),
            cost: -1000,
            ..Default::default()
        });
    }
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("first-day.hsd");
    builder
        .write_hsd(&path, &builder.write_options(), |_, _| {})
        .unwrap();
    let mut engine = Engine::default();
    engine.load_dictionary(&path).unwrap();
    Fixture {
        engine,
        analyzer: hasami::Analyzer::load(&path).unwrap(),
        _directory: directory,
    }
}

fn assert_speech(engine: &Engine, text: &str, expected: &str) {
    let mut nodes = engine.text_to_analyze(text).unwrap();
    assert_eq!(
        nodes
            .iter()
            .filter(|n| n.mora_count > 0)
            .map(|n| n.pronunciation.as_str())
            .collect::<String>(),
        expected,
        "{text}: {nodes:?}"
    );
    for node in &nodes {
        assert_eq!(node.mora_count, mora::count_mora(&node.pronunciation));
    }
    let phrases = engine.estimate_accent(&mut nodes);
    assert_eq!(
        phrases.iter().map(|p| p.mora_count).sum::<usize>(),
        mora::count_mora(expected)
    );
    let phones = engine.text_to_phone_tones(text).unwrap();
    assert_eq!(phones, engine.extract_phone_tones(&nodes, &phrases));
    assert!(phones.iter().all(|p| p.tone <= 1));
    assert_eq!(
        phones
            .iter()
            .filter(|p| !matches!(p.phone.as_str(), "sil" | "pau"))
            .map(|p| p.phone.as_str())
            .collect::<Vec<_>>(),
        phoneme::katakana_to_phonemes(expected),
        "{text}"
    );
    assert!(!engine.text_to_prosody_symbols(text).unwrap().is_empty());
    let labels = engine.text_to_labels(text).unwrap();
    // HTSは無声化した母音をI/Uで表す。PhoneToneと照合するときだけ字種をそろえる。
    let label_phones: Vec<_> = labels
        .iter()
        .map(|label| {
            let phone = label.split_once('-').unwrap().1.split_once('+').unwrap().0;
            match phone {
                "I" => "i",
                "U" => "u",
                _ => phone,
            }
        })
        .collect();
    assert_eq!(
        label_phones,
        phones.iter().map(|p| p.phone.as_str()).collect::<Vec<_>>(),
        "{text}"
    );
}

#[test]
fn split_calendar_first_day_reaches_context_input_njd_and_all_speech_outputs() {
    let mut fixture = fixture();
    let text = "日付は11月1日です。";
    let tokens = fixture.analyzer.try_tokenize(text).unwrap();
    let day = tokens.iter().find(|t| t.surface.as_ref() == "1日").unwrap();
    assert_eq!((day.start, day.end), (14, 18));
    assert!(!day.is_known);
    let input = InputToken::from(day.clone());
    assert_eq!(
        (
            input.lemma.as_str(),
            input.reading.as_str(),
            input.pronunciation.as_str()
        ),
        ("1日", "ツイタチ", "ツイタチ")
    );
    assert_eq!(
        (
            input.pos.as_str(),
            input.pos_detail1.as_str(),
            input.ctype.as_str(),
            input.cform.as_str()
        ),
        ("名詞", "副詞可能", "*", "*")
    );
    let spans = fixture.analyzer.contextual_reading_spans(text).unwrap();
    assert_eq!(
        spans,
        [hasami::ContextualReadingSpan {
            start: 14,
            end: 18,
            pronunciation: "ツイタチ".into()
        }]
    );
    assert!(
        fixture
            .analyzer
            .quantity_spans(text)
            .unwrap()
            .iter()
            .all(|s| s.end <= 14 || s.start >= 18)
    );
    let nodes = fixture.engine.text_to_analyze(text).unwrap();
    let day = nodes.iter().find(|n| n.surface == "1日").unwrap();
    assert_eq!(day.pos, Pos::Meishi);
    assert_eq!(day.pos_detail1, "副詞可能");
    assert_eq!(
        (
            day.reading.as_str(),
            day.pronunciation.as_str(),
            day.mora_count
        ),
        ("ツイタチ", "ツイタチ", 4)
    );
    // アクセント辞書を使わないEngineの既存の数詞処理に合わせ、月の長音は読み形で比較する。
    assert_speech(&fixture.engine, text, "ヒズケワジュウイチガツツイタチデス");
}

#[test]
fn duration_name_identifier_and_quote_keep_their_readings_after_calendar_analysis() {
    let mut fixture = fixture();
    for (text, expected) in [
        ("11月1日", "ジュウイチガツツイタチ"),
        ("1日", "イチニチ"),
        ("一日かかる。", "イチニチカカル"),
        ("ID11月1日", "アイディイジュウイチガツイチニチ"),
        ("四月一日さん", "ワタヌキサン"),
        ("「11月1日」", "ジュウイチガツイチニチ"),
        ("11月1日", "ジュウイチガツツイタチ"),
    ] {
        assert_speech(&fixture.engine, text, expected);
        if !text.starts_with("11月") {
            assert!(
                fixture
                    .analyzer
                    .contextual_reading_spans(text)
                    .unwrap()
                    .is_empty(),
                "{text}"
            );
        }
    }
    assert_eq!(
        fixture
            .analyzer
            .quantity_pronunciation("1日")
            .unwrap()
            .as_deref(),
        Some("イチニチ")
    );
}
