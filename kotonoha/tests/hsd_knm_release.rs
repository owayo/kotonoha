//! 公開 HSD と生成済み KNM を、失敗を隠さず結合検証する Release 用テスト。

#![cfg(feature = "neural")]

use std::path::PathBuf;

use kotonoha::Engine;
use kotonoha::njd::{InputToken, build_njd_nodes};
use kotonoha::nn::FeatureMorpheme;
use kotonoha::nn::v66::{V66Bundle, V66Pipeline};

#[test]
#[ignore = "requires KOTONOHA_RELEASE_HSD and KOTONOHA_RELEASE_MODELS"]
fn released_hsd_runs_native_pipeline() -> Result<(), Box<dyn std::error::Error>> {
    let dictionary = PathBuf::from(std::env::var("KOTONOHA_RELEASE_HSD")?);
    let models = PathBuf::from(std::env::var("KOTONOHA_RELEASE_MODELS")?);
    let mut analyzer = hasami::Analyzer::load(&dictionary)?;
    let pipeline = V66Pipeline::new(V66Bundle::from_dir(&models)?);
    let texts = ["今日は良い天気です。", "作ろうと思う。", "C++とC#を使う。"];
    for text in texts {
        let tokens: Vec<InputToken> = analyzer
            .try_tokenize(text)?
            .into_iter()
            .map(InputToken::from)
            .collect();
        assert!(!tokens.is_empty(), "HSD produced no tokens");
        let nodes = build_njd_nodes(&tokens);
        let context: Vec<_> = tokens
            .iter()
            .zip(&nodes)
            .map(|(token, node)| FeatureMorpheme { token, node })
            .collect();
        // Engine の推論エラー時フォールバックを経由せず、失敗は Release を止める。
        let accents = pipeline.predict(&context)?;
        assert_eq!(accents.len(), tokens.len());
    }
    let mut engine = Engine::with_default_rules();
    engine.load_dictionary(&dictionary)?;
    engine.set_contextual_accent_predictor(Box::new(pipeline));
    for text in texts {
        assert!(!engine.text_to_phone_tones(text)?.is_empty());
        assert!(!engine.text_to_labels(text)?.is_empty());
    }
    Ok(())
}
