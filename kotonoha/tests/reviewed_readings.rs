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

// 辞書の一般語・人名の読み分けと数詞の音便を、意味から決めた期待発音で確認する。
// 例文は独立に作成し、数詞は最小の数量表現を使う。
const LEXICAL_AND_COUNTER_READINGS: &[(&str, &str)] = &[
    ("有機EL", "ユーキイーエル"),
    ("前出", "ゼンシュツ"),
    ("上下", "ジョーゲ"),
    ("この候補が大本命だ。", "コノコーホガダイホンメイダ"),
    ("数本", "スーホン"),
    ("二文字", "フタモジ"),
    ("一言一句", "イチゴンイック"),
    ("手札", "テフダ"),
    ("教示", "キョージ"),
    ("夜中", "ヨナカ"),
    ("高名", "コーメイ"),
    ("三刀流", "サントーリュー"),
    ("1つ", "ヒトツ"),
    ("二口", "フタクチ"),
    ("第一章", "ダイイッショー"),
    ("六曲", "ロッキョク"),
    ("6頭", "ロクトー"),
    ("1週間", "イッシューカン"),
    ("6週間", "ロクシューカン"),
    ("二〇四二年", "ニセンヨンジューニネン"),
    ("六〇〇円", "ロッピャクエン"),
    ("三〇〇本", "サンビャッポン"),
    ("100本", "ヒャッポン"),
    ("百本さん", "ヒャクモトサン"),
    ("大間違い", "オオマチガイ"),
    ("大違い", "オオチガイ"),
    ("グループ会社", "グループガイシャ"),
    ("事業会社", "ジギョーガイシャ"),
    ("経時", "ケイジ"),
    ("氏名", "シメイ"),
    ("一昨日", "オトトイ"),
    ("七五三", "シチゴサン"),
    ("必要十分", "ヒツヨージューブン"),
    ("杉浦健二", "スギウラケンジ"),
    ("数本亮太", "カズモトリョータ"),
    ("二文字氏", "ニモンジシ"),
    ("声優の大本命さん", "セイユーノオオモトミコトサン"),
    ("真に受ける", "マニウケル"),
    ("身を粉にする", "ミヲコニスル"),
    ("得も言われぬ", "エモイワレヌ"),
    ("体よく断る", "テイヨクコトワル"),
    ("間もない", "マモナイ"),
    ("分の悪い", "ブノワルイ"),
    ("1軒", "イッケン"),
    ("3軒", "サンゲン"),
    ("6軒", "ロッケン"),
    ("10軒", "ジュッケン"),
    ("3票", "サンピョー"),
    ("1000票", "センビョー"),
    ("3拍", "サンパク"),
    ("1000拍", "センパク"),
    ("3府", "サンプ"),
    ("1000府", "センプ"),
    ("6キロ", "ロッキロ"),
    ("8キロ", "ハチキロ"),
    ("10号", "ジューゴー"),
    ("1箱", "ヒトハコ"),
    ("2箱", "フタハコ"),
    ("10箱", "ジュッパコ"),
    ("1皿", "ヒトサラ"),
    ("2皿", "フタサラ"),
    ("1束", "ヒトタバ"),
    ("2束", "フタタバ"),
    ("20日", "ハツカ"),
    ("二十日間", "ハツカカン"),
    ("41都府県警", "ヨンジューイチトフケンケイ"),
    ("1府県", "イチフケン"),
    ("一品親王", "イッポンシンノー"),
    ("二品親王", "ニホンシンノー"),
    ("三品親王", "サンボンシンノー"),
    ("四品親王", "シホンシンノー"),
];

fn assert_lexical_and_counter_readings(
    engine: &Engine,
    dictionary: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut analyzer = hasami::Analyzer::load(dictionary)?;
    for &(text, expected) in LEXICAL_AND_COUNTER_READINGS {
        let tokens: Vec<kotonoha::njd::InputToken> = analyzer
            .try_tokenize(text)?
            .into_iter()
            .map(kotonoha::njd::InputToken::from)
            .collect();
        assert!(!tokens.is_empty(), "{text}");
        let mut nodes = engine.analyze(&tokens);
        // NJD の長音展開後も、期待した発音そのものが保たれていることを確かめる。
        let pronunciation: String = nodes
            .iter()
            .filter(|node| node.mora_count > 0)
            .map(|node| node.pronunciation.as_str())
            .collect();
        assert_eq!(
            pronunciation,
            kotonoha::njd::expand_long_vowels(expected),
            "pronunciation: {text}"
        );
        for node in &nodes {
            assert_eq!(
                node.mora_count,
                kotonoha::mora::count_mora(&node.pronunciation),
                "node mora: {text} / {}",
                node.surface
            );
        }
        // NN 推論の失敗を通常 API のフォールバックで隠さない。
        let accents = engine.try_predict_accent_types(&tokens)?;
        assert_eq!(accents.len(), nodes.len(), "NN output length: {text}");
        for (node, accent) in nodes.iter_mut().zip(accents) {
            node.accent_type = accent;
        }
        let phrases = engine.estimate_accent(&mut nodes);
        assert_eq!(
            phrases
                .iter()
                .map(|phrase| phrase.mora_count)
                .sum::<usize>(),
            kotonoha::mora::count_mora(expected),
            "phrase mora: {text}"
        );
        let phones = engine.extract_phone_tones(&nodes, &phrases);
        let labels = engine.make_label(&nodes, &phrases);
        assert_eq!(
            phones
                .iter()
                .filter(|phone| !matches!(phone.phone.as_str(), "sil" | "pau"))
                .map(|phone| phone.phone.as_str())
                .collect::<Vec<_>>(),
            kotonoha::phoneme::katakana_to_phonemes(expected),
            "semantic phones: {text}"
        );
        // ラベル数だけでなく、各 full-context label の中心音素も照合する。
        let label_phones: Vec<_> = labels
            .iter()
            .map(|label| {
                let phone = label
                    .split_once('-')
                    .and_then(|(_, tail)| tail.split_once('+'))
                    .map(|(phone, _)| phone)
                    .expect("full-context label must contain its central phone");
                // HTS の I/U は母音無声化を表す。基底音素との比較でのみ戻す。
                match phone {
                    "I" => "i",
                    "U" => "u",
                    phone => phone,
                }
            })
            .collect();
        assert_eq!(
            label_phones,
            phones
                .iter()
                .map(|phone| phone.phone.as_str())
                .collect::<Vec<_>>(),
            "label phones: {text}"
        );
        // 公開のテキスト API も同じ実辞書・NN を通り、同じ出力になる。
        assert_eq!(
            engine.text_to_phone_tones(text)?,
            phones,
            "text phones: {text}"
        );
        assert_eq!(engine.text_to_labels(text)?, labels, "text labels: {text}");
    }
    Ok(())
}

#[test]
#[ignore = "KOTONOHA_TEST_DICT に監査済みの配布辞書を指定する"]
fn distributed_lexical_and_counter_readings_reach_phonemes_moras_and_labels()
-> Result<(), Box<dyn std::error::Error>> {
    let dictionary = std::path::PathBuf::from(std::env::var("KOTONOHA_TEST_DICT")?);
    let mut engine = Engine::default();
    engine.load_dictionary(&dictionary)?;
    assert_lexical_and_counter_readings(&engine, &dictionary)
}

#[cfg(feature = "neural")]
#[test]
#[ignore = "KOTONOHA_TEST_DICT と KOTONOHA_RELEASE_MODELS に検証済みの辞書・モデルを指定する"]
fn distributed_lexical_and_counter_readings_reach_native_neural_output()
-> Result<(), Box<dyn std::error::Error>> {
    use kotonoha::nn::v66::{V66Bundle, V66Pipeline};

    let dictionary = std::path::PathBuf::from(std::env::var("KOTONOHA_TEST_DICT")?);
    let models = std::path::PathBuf::from(std::env::var("KOTONOHA_RELEASE_MODELS")?);
    let mut engine = Engine::default();
    engine.load_dictionary(&dictionary)?;
    engine
        .set_contextual_accent_predictor(Box::new(V66Pipeline::new(V66Bundle::from_dir(&models)?)));
    assert_lexical_and_counter_readings(&engine, &dictionary)
}
