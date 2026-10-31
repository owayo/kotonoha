//! 配布辞書の読みを実際の解析から音素・モーラ・ラベルまで確認する。

use kotonoha::Engine;

#[test]
#[ignore = "KOTONOHA_TEST_DICT に監査済みの配布辞書を指定する"]
fn distributed_reviewed_readings_reach_phonemes_and_moras() {
    let path = std::env::var("KOTONOHA_TEST_DICT").expect("KOTONOHA_TEST_DICT is required");
    let mut engine = Engine::default();
    engine.load_dictionary(std::path::Path::new(&path)).unwrap();
    for (text, expected) in [
        ("近日中", "キンジツチュー"),
        ("正典", "セイテン"),
        ("章の説明", "ショーノセツメイ"),
        ("COO", "シーオーオー"),
        ("DEC", "ディーイーシー"),
        ("BMW", "ビーエムダブリュー"),
        ("ppm", "ピーピーエム"),
        ("粗利率", "アラリリツ"),
        ("手水舎", "テミズヤ"),
        ("小売店", "コウリテン"),
        ("音信不通", "オンシンフツー"),
        ("センテンス", "センテンス"),
        ("韓信", "カンシン"),
        ("Language", "ランゲージ"),
        ("ＳＮＳ", "エスエヌエス"),
        ("9.2", "キューテンニ"),
        ("十階", "ジュッカイ"),
        ("十冊", "ジュッサツ"),
        ("十足", "ジュッソク"),
        ("十歳", "ジュッサイ"),
        ("十才", "ジュッサイ"),
        ("六冊", "ロクサツ"),
        ("六足", "ロクソク"),
        ("10階", "ジュッカイ"),
        ("10冊", "ジュッサツ"),
        ("10足", "ジュッソク"),
        ("10歳", "ジュッサイ"),
        ("10才", "ジュッサイ"),
    ] {
        let mut nodes = engine.text_to_analyze(text).unwrap();
        for node in &nodes {
            assert_eq!(
                node.mora_count,
                kotonoha::mora::count_mora(&node.pronunciation),
                "{text}"
            );
        }
        let phrases = engine.estimate_accent(&mut nodes);
        assert_eq!(
            phrases.iter().map(|p| p.mora_count).sum::<usize>(),
            kotonoha::mora::count_mora(expected),
            "{text}"
        );
        let phones = engine.extract_phone_tones(&nodes, &phrases);
        assert_eq!(phones.len(), engine.make_label(&nodes, &phrases).len());
        assert_eq!(
            phones
                .iter()
                .filter(|p| !matches!(p.phone.as_str(), "sil" | "pau"))
                .map(|p| p.phone.as_str())
                .collect::<Vec<_>>(),
            kotonoha::phoneme::katakana_to_phonemes(expected),
            "{text}"
        );
        assert!(!engine.text_to_prosody_symbols(text).unwrap().is_empty());
    }
    let nodes = engine.text_to_analyze("ひとは考える。").unwrap();
    let particle = nodes.iter().find(|node| node.surface == "は").unwrap();
    assert_eq!(particle.pronunciation, "ワ");
    let nodes = engine.text_to_analyze("ひとはさんに聞く。").unwrap();
    assert_eq!(nodes[0].surface, "ひとは");
    assert_eq!(nodes[0].pronunciation, "ヒトハ");
}
