//! hasami の文脈上の読み・品詞を、HSD読込から音声処理まで検証する。

use hasami::dict::{DictBuilder, DictEntry, UnkEntry};
use kotonoha::njd::{InputToken, Pos};
use kotonoha::{Engine, mora, phoneme};
use std::collections::HashMap;

struct Fixture {
    engine: Engine,
    analyzer: hasami::Analyzer,
    // Windowsでも辞書のmmapを閉じてから一時ファイルを消す。
    _directory: tempfile::TempDir,
}

fn fixture() -> Fixture {
    let mut builder = DictBuilder::new();
    for (surface, pos, base, reading, pronunciation, kind, form) in [
        (
            "2024",
            "名詞,数,*,*",
            "2024",
            "ニセンニジュウヨン",
            "ニセンニジューヨン",
            "",
            "",
        ),
        ("年", "名詞,接尾,助数詞,*", "年", "ネン", "ネン", "", ""),
        (
            "11月",
            "名詞,副詞可能,*,*",
            "11月",
            "ジュウイチガツ",
            "ジューイチガツ",
            "",
            "",
        ),
        (
            "1日",
            "名詞,副詞可能,*,*",
            "1日",
            "ツイタチ",
            "ツイタチ",
            "",
            "",
        ),
        (
            "75",
            "名詞,数,*,*",
            "75",
            "ナナジュウゴ",
            "ナナジューゴ",
            "",
            "",
        ),
        (
            "出力",
            "名詞,一般,*,*",
            "出力",
            "シュツリョク",
            "シュツリョク",
            "",
            "",
        ),
        ("は", "助詞,係助詞,*,*", "は", "ハ", "ワ", "", ""),
        (
            "ID",
            "名詞,固有名詞,組織,*",
            "ID",
            "アイディー",
            "アイディー",
            "",
            "",
        ),
        (
            "震度7",
            "名詞,固有名詞,一般,*",
            "震度7",
            "シンドセブン",
            "シンドセブン",
            "",
            "",
        ),
        (
            "最大",
            "名詞,一般,*,*",
            "最大",
            "サイダイ",
            "サイダイ",
            "",
            "",
        ),
        ("を", "助詞,格助詞,一般,*", "を", "ヲ", "ヲ", "", ""),
        (
            "行く",
            "動詞,自立,*,*",
            "行く",
            "イク",
            "イク",
            "五段・カ行促音便",
            "基本形",
        ),
        (
            "等",
            "名詞,固有名詞,人名,名",
            "等",
            "ヒトシ",
            "ヒトシ",
            "",
            "",
        ),
        (
            "山田",
            "名詞,固有名詞,人名,姓",
            "山田",
            "ヤマダ",
            "ヤマダ",
            "",
            "",
        ),
        ("さん", "名詞,接尾,人名,*", "さん", "サン", "サン", "", ""),
        (
            "観測",
            "名詞,サ変接続,*,*",
            "観測",
            "カンソク",
            "カンソク",
            "",
            "",
        ),
        (
            "し",
            "動詞,自立,*,*",
            "する",
            "シ",
            "シ",
            "サ変・スル",
            "連用形",
        ),
        ("た", "助動詞,*,*,*", "た", "タ", "タ", "特殊・タ", "基本形"),
        ("前", "名詞,一般,*,*", "前", "マエ", "マエ", "", ""),
        ("。", "記号,句点,*,*", "。", "。", "。", "", ""),
        ("「", "記号,括弧開,*,*", "「", "「", "「", "", ""),
        ("」", "記号,括弧閉,*,*", "」", "」", "」", "", ""),
    ] {
        builder.add_entry(DictEntry {
            surface: surface.into(),
            pos: pos.into(),
            base_form: base.into(),
            reading: reading.into(),
            pronunciation: pronunciation.into(),
            conj_type: kind.into(),
            conj_form: form.into(),
            cost: -1000,
            ..Default::default()
        });
    }
    // kWそのものを辞書に登録せず、未知ALPHAの組織名テンプレートを通す。
    builder.set_unk_entries(HashMap::from([(
        "ALPHA".into(),
        vec![UnkEntry {
            char_class: "ALPHA".into(),
            left_id: 0,
            right_id: 0,
            cost: 9000,
            pos: "名詞,固有名詞,組織,*".into(),
        }],
    )]));
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("revision.hsd");
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

fn input(tokens: &[hasami::Token], surface: &str) -> InputToken {
    InputToken::from(
        tokens
            .iter()
            .find(|t| t.surface.as_ref() == surface)
            .unwrap()
            .clone(),
    )
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
fn calendar_year_terminal_four_reaches_input_njd_and_speech() {
    let mut f = fixture();
    let tokens = f.analyzer.try_tokenize("2024年11月1日").unwrap();
    let year = input(&tokens, "2024");
    assert_eq!(year.reading, "ニセンニジュウヨ");
    assert_eq!(year.pronunciation, "ニセンニジューヨ");
    assert_eq!(
        (year.pos.as_str(), year.pos_detail1.as_str()),
        ("名詞", "数")
    );
    assert_eq!(year.lemma, "2024");
    let nodes = f.engine.text_to_analyze("2024年11月1日").unwrap();
    let year = nodes.iter().find(|n| n.surface == "2024").unwrap();
    assert_eq!(year.reading, "ニセンニジュウヨ");
    assert_eq!(year.pronunciation, "ニセンニジュウヨ");
    assert_eq!(year.mora_count, 7);
    assert_speech(
        &f.engine,
        "2024年11月1日",
        "ニセンニジュウヨネンジュウイチガツツイタチ",
    );
    // IDに埋め込まれた同形の数詞では年読みを新たに認可しない。
    let tokens = f.analyzer.try_tokenize("ID2024年11月1日").unwrap();
    assert_eq!(input(&tokens, "2024").pronunciation, "ニセンニジューヨン");
}

#[test]
fn direct_clause_enumeration_and_person_name_keep_distinct_speech() {
    let mut f = fixture();
    for (text, reading, pos, detail, speech) in [
        ("行く等を", "ナド", Pos::Joshi, "副助詞", "イクナドヲ"),
        (
            "山田等さん",
            "ヒトシ",
            Pos::Meishi,
            "固有名詞",
            "ヤマダヒトシサン",
        ),
    ] {
        let raw = f.analyzer.try_tokenize(text).unwrap();
        let converted = input(&raw, "等");
        assert_eq!(converted.pos_detail1, detail);
        assert_eq!(converted.reading, reading);
        assert_eq!(converted.pronunciation, reading);
        assert_eq!(
            (converted.ctype.as_str(), converted.cform.as_str()),
            ("*", "*")
        );
        let nodes = f.engine.text_to_analyze(text).unwrap();
        let target = nodes.iter().find(|n| n.surface == "等").unwrap();
        assert_eq!(target.pos, pos);
        assert_eq!(target.pos_detail1, detail);
        assert_eq!(target.reading, reading);
        assert_eq!(target.pronunciation, reading);
        assert_eq!(target.mora_count, mora::count_mora(reading));
        assert_speech(&f.engine, text, speech);
    }
}

#[test]
fn unknown_power_unit_reaches_input_and_njd_as_a_counter() {
    let mut f = fixture();
    let tokens = f.analyzer.try_tokenize("75kW").unwrap();
    let raw = tokens.iter().find(|t| t.surface.as_ref() == "kW").unwrap();
    assert!(!raw.is_known);
    assert_eq!((raw.start, raw.end), (2, 4));
    let unit = input(&tokens, "kW");
    assert_eq!(
        (
            unit.pos.as_str(),
            unit.pos_detail1.as_str(),
            unit.pos_detail2.as_str(),
            unit.pos_detail3.as_str()
        ),
        ("名詞", "接尾", "助数詞", "*")
    );
    assert_eq!(
        (
            unit.ctype.as_str(),
            unit.cform.as_str(),
            unit.lemma.as_str()
        ),
        ("*", "*", "kW")
    );
    assert_eq!(
        (unit.reading.as_str(), unit.pronunciation.as_str()),
        ("キロワット", "キロワット")
    );
    let nodes = f.engine.text_to_analyze("75kW").unwrap();
    let unit = nodes.iter().find(|n| n.surface == "kW").unwrap();
    assert_eq!(unit.pos, Pos::Meishi);
    assert_eq!(
        (unit.reading.as_str(), unit.pronunciation.as_str()),
        ("キロワット", "キロワット")
    );
    assert_eq!(
        (unit.pos_detail1.as_str(), unit.pos_detail2.as_str()),
        ("接尾", "助数詞")
    );
    assert_eq!(unit.mora_count, 5);
    assert_speech(&f.engine, "75kW", "ナナジュウゴキロワット");
    assert_speech(
        &f.engine,
        "出力は75kW。",
        "シュツリョクワナナジュウゴキロワット",
    );
    // 裸の略号は数量の単位へ変えない。
    let bare = f.analyzer.try_tokenize("kW").unwrap();
    let bare = input(&bare, "kW");
    assert_eq!(
        (bare.pos_detail1.as_str(), bare.pos_detail2.as_str()),
        ("固有名詞", "組織")
    );
    assert_ne!(bare.pronunciation, "キロワット");
}

#[test]
fn seismic_measurement_and_quoted_alias_keep_distinct_readings_through_speech() {
    let mut f = fixture();
    for (text, reading, speech, mora_count) in [
        (
            "最大震度7を観測した。",
            "シンドナナ",
            "サイダイシンドナナヲカンソクシタ",
            5,
        ),
        ("震度7", "シンドセブン", "シンドセブン", 6),
        (
            "「前。最大震度7を観測した。」",
            "シンドセブン",
            "マエサイダイシンドセブンヲカンソクシタ",
            6,
        ),
        // 引用索引や補正済み読みを、次の解析へ持ち越さない。
        (
            "最大震度7を観測した。",
            "シンドナナ",
            "サイダイシンドナナヲカンソクシタ",
            5,
        ),
    ] {
        let raw = f.analyzer.try_tokenize(text).unwrap();
        let alias = input(&raw, "震度7");
        assert_eq!(
            (alias.reading.as_str(), alias.pronunciation.as_str()),
            (reading, reading)
        );
        assert_eq!(alias.lemma, "震度7");
        assert_eq!(
            (alias.pos_detail1.as_str(), alias.pos_detail2.as_str()),
            ("固有名詞", "一般")
        );
        let nodes = f.engine.text_to_analyze(text).unwrap();
        let alias = nodes.iter().find(|n| n.surface == "震度7").unwrap();
        assert_eq!(alias.reading, reading);
        assert_eq!(alias.pronunciation, reading);
        assert_eq!(alias.mora_count, mora_count);
        assert_speech(&f.engine, text, speech);
    }
}
