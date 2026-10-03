//! 予測失敗の伝播と、韻律生成での既存アクセントの保持を検査する。

use kotonoha::accent_dict::AccentDict;
use kotonoha::njd::{InputToken, NjdNode};
use kotonoha::nn::{
    AccentPredictor, ContextualAccentError, ContextualAccentPredictor, FeatureMorpheme,
};
use kotonoha::{AccentPredictionError, Engine};

struct FixedPredictor(Vec<u8>);

impl AccentPredictor for FixedPredictor {
    fn predict(&self, _nodes: &[NjdNode]) -> Vec<u8> {
        self.0.clone()
    }
}

impl ContextualAccentPredictor for FixedPredictor {
    fn predict_with_context(
        &self,
        _ctx: &[FeatureMorpheme<'_>],
    ) -> Result<Vec<u8>, ContextualAccentError> {
        Ok(self.0.clone())
    }
}

struct FailingPredictor;

impl AccentPredictor for FailingPredictor {
    fn predict(&self, _nodes: &[NjdNode]) -> Vec<u8> {
        panic!("Engine must use the fallible prediction method");
    }

    fn try_predict(&self, _nodes: &[NjdNode]) -> Result<Vec<u8>, ContextualAccentError> {
        Err(std::io::Error::other("backend unavailable").into())
    }
}

impl ContextualAccentPredictor for FailingPredictor {
    fn predict_with_context(
        &self,
        _ctx: &[FeatureMorpheme<'_>],
    ) -> Result<Vec<u8>, ContextualAccentError> {
        Err(std::io::Error::other("backend unavailable").into())
    }
}

fn engine_and_tokens() -> (Engine, Vec<InputToken>) {
    let mut engine = Engine::default();
    let mut dict = AccentDict::new();
    dict.insert("猫", "ネコ", 1);
    dict.insert("犬", "イヌ", 2);
    engine.set_accent_dict(dict);
    (
        engine,
        vec![
            InputToken::new("猫", "名詞", "ネコ", "ネコ"),
            InputToken::new("犬", "名詞", "イヌ", "イヌ"),
        ],
    )
}

#[test]
fn strict_prediction_uses_dictionary_without_a_predictor() {
    let (engine, tokens) = engine_and_tokens();
    assert_eq!(
        engine.try_predict_accent_types(&tokens).unwrap(),
        vec![1, 2]
    );
    assert!(engine.try_predict_accent_types(&[]).unwrap().is_empty());
}

#[test]
fn strict_prediction_supports_existing_predictors_and_contextual_priority() {
    let (mut engine, tokens) = engine_and_tokens();
    engine.set_accent_predictor(Box::new(FixedPredictor(vec![2, 1])));
    assert_eq!(
        engine.try_predict_accent_types(&tokens).unwrap(),
        vec![2, 1]
    );
    engine.set_accent_predictor(Box::new(FailingPredictor));
    engine.set_contextual_accent_predictor(Box::new(FixedPredictor(vec![0, 1])));
    assert_eq!(
        engine.try_predict_accent_types(&tokens).unwrap(),
        vec![0, 1]
    );
}

#[test]
fn backend_errors_propagate_while_prosody_preserves_dictionary_accents() {
    for contextual in [false, true] {
        let (mut engine, tokens) = engine_and_tokens();
        let labels = engine.tokens_to_labels(&tokens);
        let tones = engine.tokens_to_phone_tones(&tokens);
        if contextual {
            engine.set_contextual_accent_predictor(Box::new(FailingPredictor));
        } else {
            engine.set_accent_predictor(Box::new(FailingPredictor));
        }
        let error = engine.try_predict_accent_types(&tokens).unwrap_err();
        assert!(matches!(&error, AccentPredictionError::Backend(_)));
        assert_eq!(
            std::error::Error::source(&error).unwrap().to_string(),
            "backend unavailable"
        );
        assert_eq!(engine.predict_accent_types(&tokens), vec![1, 2]);
        assert_eq!(engine.tokens_to_labels(&tokens), labels);
        assert_eq!(engine.tokens_to_phone_tones(&tokens), tones);
    }
}

#[test]
fn incomplete_predictions_never_partially_replace_dictionary_accents() {
    for contextual in [false, true] {
        for predicted in [vec![5], vec![5, 4, 3]] {
            let (mut engine, tokens) = engine_and_tokens();
            let labels = engine.tokens_to_labels(&tokens);
            let actual = predicted.len();
            if contextual {
                engine.set_contextual_accent_predictor(Box::new(FixedPredictor(predicted)));
            } else {
                engine.set_accent_predictor(Box::new(FixedPredictor(predicted)));
            }
            assert!(
                matches!(engine.try_predict_accent_types(&tokens), Err(AccentPredictionError::OutputLength { expected: 2, actual: got }) if got == actual)
            );
            assert_eq!(engine.predict_accent_types(&tokens), vec![1, 2]);
            assert_eq!(engine.tokens_to_labels(&tokens), labels);
        }
    }
}
