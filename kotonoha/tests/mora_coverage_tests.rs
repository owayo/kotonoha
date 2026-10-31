//! カタカナ 1 モーラの音素化が、Style-Bert-VITS2 のモーラ表と一致することを検証する。
//!
//! 音素化とモーラ処理は対応表を共有する。未対応の小書き仮名まで結合すると、
//! 「カャ」の `k`,`a`,`y`,`a` から存在しない子音 `kay` を作ってしまう。
//! 対応表・境界例・カタカナ全組み合わせを突き合わせ、その取りこぼしを検出する。
//!
//! 次の 3 つは意図的に Style-Bert-VITS2 と表記が異なる:
//! - 促音は HTS 表記の `cl`（Style-Bert-VITS2 側が `q` に読み替える）
//! - 「クヮ」「グヮ」は `kw` / `gw` 音素を持たないため「クワ」「グワ」と同じ音素列にする
//!   （どちらの音素セットにも無く、現代語ではこの表記自体がまれ）

use kotonoha::phoneme::katakana_to_phonemes;

/// Style-Bert-VITS2 の `MORA_KATA_TO_MORA_PHONEMES` に対応するモーラと音素列
const MORA_PHONEMES: &[(&str, &[&str])] = &[
    ("ヴォ", &["v", "o"]),
    ("ヴェ", &["v", "e"]),
    ("ヴィ", &["v", "i"]),
    ("ヴァ", &["v", "a"]),
    ("ヴ", &["v", "u"]),
    ("ン", &["N"]),
    ("ワ", &["w", "a"]),
    ("ロ", &["r", "o"]),
    ("レ", &["r", "e"]),
    ("ル", &["r", "u"]),
    ("リョ", &["ry", "o"]),
    ("リュ", &["ry", "u"]),
    ("リャ", &["ry", "a"]),
    ("リェ", &["ry", "e"]),
    ("リ", &["r", "i"]),
    ("ラ", &["r", "a"]),
    ("ヨ", &["y", "o"]),
    ("ユ", &["y", "u"]),
    ("ヤ", &["y", "a"]),
    ("モ", &["m", "o"]),
    ("メ", &["m", "e"]),
    ("ム", &["m", "u"]),
    ("ミョ", &["my", "o"]),
    ("ミュ", &["my", "u"]),
    ("ミャ", &["my", "a"]),
    ("ミェ", &["my", "e"]),
    ("ミ", &["m", "i"]),
    ("マ", &["m", "a"]),
    ("ポ", &["p", "o"]),
    ("ボ", &["b", "o"]),
    ("ホ", &["h", "o"]),
    ("ペ", &["p", "e"]),
    ("ベ", &["b", "e"]),
    ("ヘ", &["h", "e"]),
    ("プ", &["p", "u"]),
    ("ブ", &["b", "u"]),
    ("フォ", &["f", "o"]),
    ("フェ", &["f", "e"]),
    ("フィ", &["f", "i"]),
    ("ファ", &["f", "a"]),
    ("フ", &["f", "u"]),
    ("ピョ", &["py", "o"]),
    ("ピュ", &["py", "u"]),
    ("ピャ", &["py", "a"]),
    ("ピェ", &["py", "e"]),
    ("ピ", &["p", "i"]),
    ("ビョ", &["by", "o"]),
    ("ビュ", &["by", "u"]),
    ("ビャ", &["by", "a"]),
    ("ビェ", &["by", "e"]),
    ("ビ", &["b", "i"]),
    ("ヒョ", &["hy", "o"]),
    ("ヒュ", &["hy", "u"]),
    ("ヒャ", &["hy", "a"]),
    ("ヒェ", &["hy", "e"]),
    ("ヒ", &["h", "i"]),
    ("パ", &["p", "a"]),
    ("バ", &["b", "a"]),
    ("ハ", &["h", "a"]),
    ("ノ", &["n", "o"]),
    ("ネ", &["n", "e"]),
    ("ヌ", &["n", "u"]),
    ("ニョ", &["ny", "o"]),
    ("ニュ", &["ny", "u"]),
    ("ニャ", &["ny", "a"]),
    ("ニェ", &["ny", "e"]),
    ("ニ", &["n", "i"]),
    ("ナ", &["n", "a"]),
    ("ドゥ", &["d", "u"]),
    ("ド", &["d", "o"]),
    ("トゥ", &["t", "u"]),
    ("ト", &["t", "o"]),
    ("デョ", &["dy", "o"]),
    ("デュ", &["dy", "u"]),
    ("デャ", &["dy", "a"]),
    ("ディ", &["d", "i"]),
    ("デ", &["d", "e"]),
    ("テョ", &["ty", "o"]),
    ("テュ", &["ty", "u"]),
    ("テャ", &["ty", "a"]),
    ("ティ", &["t", "i"]),
    ("テ", &["t", "e"]),
    ("ツォ", &["ts", "o"]),
    ("ツェ", &["ts", "e"]),
    ("ツィ", &["ts", "i"]),
    ("ツァ", &["ts", "a"]),
    ("ツ", &["ts", "u"]),
    ("ッ", &["cl"]),
    ("チョ", &["ch", "o"]),
    ("チュ", &["ch", "u"]),
    ("チャ", &["ch", "a"]),
    ("チェ", &["ch", "e"]),
    ("チ", &["ch", "i"]),
    ("ダ", &["d", "a"]),
    ("タ", &["t", "a"]),
    ("ゾ", &["z", "o"]),
    ("ソ", &["s", "o"]),
    ("ゼ", &["z", "e"]),
    ("セ", &["s", "e"]),
    ("ズィ", &["z", "i"]),
    ("ズ", &["z", "u"]),
    ("スィ", &["s", "i"]),
    ("ス", &["s", "u"]),
    ("ジョ", &["j", "o"]),
    ("ジュ", &["j", "u"]),
    ("ジャ", &["j", "a"]),
    ("ジェ", &["j", "e"]),
    ("ジ", &["j", "i"]),
    ("ショ", &["sh", "o"]),
    ("シュ", &["sh", "u"]),
    ("シャ", &["sh", "a"]),
    ("シェ", &["sh", "e"]),
    ("シ", &["sh", "i"]),
    ("ザ", &["z", "a"]),
    ("サ", &["s", "a"]),
    ("ゴ", &["g", "o"]),
    ("コ", &["k", "o"]),
    ("ゲ", &["g", "e"]),
    ("ケ", &["k", "e"]),
    ("グヮ", &["g", "u", "w", "a"]),
    ("グ", &["g", "u"]),
    ("クヮ", &["k", "u", "w", "a"]),
    ("ク", &["k", "u"]),
    ("ギョ", &["gy", "o"]),
    ("ギュ", &["gy", "u"]),
    ("ギャ", &["gy", "a"]),
    ("ギェ", &["gy", "e"]),
    ("ギ", &["g", "i"]),
    ("キョ", &["ky", "o"]),
    ("キュ", &["ky", "u"]),
    ("キャ", &["ky", "a"]),
    ("キェ", &["ky", "e"]),
    ("キ", &["k", "i"]),
    ("ガ", &["g", "a"]),
    ("カ", &["k", "a"]),
    ("オ", &["o"]),
    ("エ", &["e"]),
    ("ウォ", &["w", "o"]),
    ("ウェ", &["w", "e"]),
    ("ウィ", &["w", "i"]),
    ("ウ", &["u"]),
    ("イェ", &["y", "e"]),
    ("イ", &["i"]),
    ("ア", &["a"]),
    ("ヴョ", &["by", "o"]),
    ("ヴュ", &["by", "u"]),
    ("ヴャ", &["by", "a"]),
    ("ヲ", &["o"]),
    ("ヱ", &["e"]),
    ("ヰ", &["i"]),
    ("ヮ", &["w", "a"]),
    ("ョ", &["y", "o"]),
    ("ュ", &["y", "u"]),
    ("ヅ", &["z", "u"]),
    ("ヂ", &["j", "i"]),
    ("ヶ", &["k", "e"]),
    ("ャ", &["y", "a"]),
    ("ォ", &["o"]),
    ("ェ", &["e"]),
    ("ゥ", &["u"]),
    ("ィ", &["i"]),
    ("ァ", &["a"]),
];

#[test]
fn all_moras_map_to_expected_phonemes() {
    let mismatched: Vec<String> = MORA_PHONEMES
        .iter()
        .filter_map(|(mora, expect)| {
            let got = katakana_to_phonemes(mora);
            (got != *expect).then(|| format!("{}: expect={:?} got={:?}", mora, expect, got))
        })
        .collect();
    assert!(
        mismatched.is_empty(),
        "{} / {} moras mismatched:
{}",
        mismatched.len(),
        MORA_PHONEMES.len(),
        mismatched.join(
            "
"
        )
    );
}

#[test]
fn all_moras_preserve_following_long_vowels_and_kana() {
    for &(mora, phonemes) in MORA_PHONEMES {
        let mut expected = phonemes.to_vec();
        if let Some(&last) = phonemes.last()
            && kotonoha::phoneme::is_vowel(last)
        {
            expected.extend([last, last]);
        }
        expected.extend(["k", "a", "N"]);

        let input = format!("{mora}ーーカン");
        assert_eq!(katakana_to_phonemes(&input), expected, "{input}");
    }
}

#[test]
fn small_kana_and_ignored_characters_preserve_conversion_boundaries() {
    let cases: &[(&str, &[&str])] = &[
        ("", &[]),
        ("ーー", &[]),
        ("ンーッー", &["N", "cl"]),
        (
            "イェキャティドゥヴョ",
            &["y", "e", "ky", "a", "t", "i", "d", "u", "by", "o"],
        ),
        ("ビィー", &["b", "i", "i"]),
        ("キャィー", &["ky", "i", "i"]),
        ("アィー", &["a", "i", "i"]),
        ("ンァー", &["N", "a", "a"]),
        ("ァィゥェォ", &["a", "i", "u", "e", "o"]),
        ("カャ", &["k", "a", "y", "a"]),
        ("ヂャ", &["j", "i", "y", "a"]),
        ("キ?ャ", &["k", "i", "y", "a"]),
        ("キ?ィー", &["k", "i", "i"]),
        ("コ!ー", &["k", "o", "o"]),
        ("漢字かな🙂", &[]),
    ];

    for &(input, expected) in cases {
        assert_eq!(katakana_to_phonemes(input), expected, "{input}");
    }
}

#[test]
fn all_mora_phonemes_are_defined() {
    let undefined: Vec<String> = MORA_PHONEMES
        .iter()
        .flat_map(|(mora, _)| {
            katakana_to_phonemes(mora)
                .into_iter()
                .filter(|p| !kotonoha::phoneme::ALL_PHONEMES.contains(&p.as_str()))
                .map(move |p| format!("{}: {}", mora, p))
        })
        .collect();
    assert!(undefined.is_empty(), "undefined phonemes: {:?}", undefined);
}

#[test]
fn mora_boundaries_preserve_vowels_and_reject_nonexistent_consonants() {
    for input in [
        "アィー",
        "ンァー",
        "カャ",
        "ヂャ",
        "キャィー",
        "ンーッー",
        "ヿヽヾ",
        "キ?ィー",
    ] {
        let moras = kotonoha::mora::parse_mora(input);
        let phones: Vec<_> = moras
            .iter()
            .flat_map(|m| m.consonant.iter().chain(std::iter::once(&m.vowel)))
            .cloned()
            .collect();
        assert_eq!(phones, katakana_to_phonemes(input), "{input}");
        assert_eq!(kotonoha::mora::count_mora(input), moras.len(), "{input}");
        assert!(
            moras
                .iter()
                .filter_map(|m| m.consonant.as_deref())
                .all(kotonoha::phoneme::is_consonant),
            "{input}"
        );
        for phone in phones {
            assert!(
                kotonoha::phoneme::ALL_PHONEMES.contains(&phone.as_str()),
                "{input}: {phone}"
            );
        }
    }
}

#[test]
fn all_katakana_pairs_agree_between_mora_and_phoneme_conversion() {
    for first in '\u{30a0}'..='\u{30ff}' {
        for second in '\u{30a0}'..='\u{30ff}' {
            let input = format!("{first}{second}");
            let moras = kotonoha::mora::parse_mora(&input);
            let phones: Vec<_> = moras
                .iter()
                .flat_map(|m| m.consonant.iter().chain(std::iter::once(&m.vowel)))
                .cloned()
                .collect();
            assert_eq!(phones, katakana_to_phonemes(&input), "{input}");
            assert_eq!(kotonoha::mora::count_mora(&input), moras.len(), "{input}");
        }
    }
}

#[test]
fn mora_boundaries_reach_accent_tones_and_labels() {
    use kotonoha::Engine;
    use kotonoha::accent::AccentPhrase;
    use kotonoha::njd::InputToken;

    let engine = Engine::default();
    let cases: &[(&str, usize, &[&str])] = &[
        ("カャ", 2, &["k", "a", "y", "a"]),
        ("アィー", 3, &["a", "i", "i"]),
        ("ンァー", 3, &["N", "a", "a"]),
        ("キャィー", 2, &["ky", "i", "i"]),
        ("ンーッー", 2, &["N", "cl"]),
    ];

    for &(input, expected_count, expected_phones) in cases {
        let mut nodes = engine.analyze(&[
            InputToken::new("境界", "名詞", input, input),
            InputToken::new("後続", "名詞", "ア", "ア"),
        ]);
        assert_eq!(nodes[0].mora_count, expected_count, "{input}");
        let estimated = engine.estimate_accent(&mut nodes);
        assert_eq!(
            estimated
                .iter()
                .map(|phrase| phrase.mora_count)
                .sum::<usize>(),
            expected_count + 1,
            "{input}"
        );

        // 核を最終モーラに置き、後続語へ移る位置でもトーンとラベルを照合する。
        let phrases = [AccentPhrase {
            nodes: vec![0, 1],
            accent_type: expected_count,
            mora_count: expected_count + 1,
            is_interrogative: false,
        }];
        let tones = engine.extract_phone_tones(&nodes, &phrases);
        let mut expected = expected_phones.to_vec();
        expected.push("a");
        assert_eq!(
            tones[1..tones.len() - 1]
                .iter()
                .map(|phone| phone.phone.as_str())
                .collect::<Vec<_>>(),
            expected,
            "{input}"
        );
        assert_eq!(tones[1].tone, 0, "{input}");
        assert_eq!(tones[tones.len() - 3].tone, 1, "{input}");
        assert_eq!(tones[tones.len() - 2].tone, 0, "{input}");

        let labels = engine.make_label(&nodes, &phrases);
        assert_eq!(labels.len(), tones.len(), "{input}");
        for (label, phone) in labels[1..labels.len() - 1].iter().zip(&expected) {
            assert!(label.contains(&format!("-{phone}+")), "{input}: {label}");
            assert!(
                label.contains(&format!("/F:{}_{expected_count}#", expected_count + 1)),
                "{input}: {label}"
            );
        }
        assert!(
            labels[labels.len() - 3].contains(&format!("/A:0+{expected_count}+2")),
            "{input}: {}",
            labels[labels.len() - 3]
        );
        assert!(
            labels[labels.len() - 2].contains(&format!("/A:-1+{}+1", expected_count + 1)),
            "{input}: {}",
            labels[labels.len() - 2]
        );
    }
}
