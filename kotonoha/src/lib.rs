//! kotonoha - Rust製日本語韻律エンジン
//! OpenJTalkの機能を置き換え、HTS Full-Context LabelおよびPhoneToneを生成する

pub mod accent;
pub mod accent_dict;
pub mod accent_rule;
pub mod crf;
pub mod label;
pub mod mora;
pub mod njd;
pub mod nn;
pub mod phoneme;
pub mod prosody;

use accent::AccentPhrase;
use accent_dict::AccentDict;
use accent_rule::AccentRuleTable;
use njd::{InputToken, NjdNode};
use nn::AccentPredictor;
use prosody::PhoneTone;
use std::path::Path;
use std::sync::Mutex;

/// kotonohaエンジン
/// スレッドセーフ: ルールテーブルは不変、アクセント辞書・予測器はオプション
pub struct Engine {
    rule_table: AccentRuleTable,
    accent_dict: Option<AccentDict>,
    accent_predictor: Option<Box<dyn AccentPredictor + Send + Sync>>,
    analyzer: Option<Mutex<hasami::Analyzer>>,
}

impl Engine {
    /// アクセント規則CSVからEngineを構築する
    pub fn new(accent_rule_path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let rule_table = AccentRuleTable::from_csv(accent_rule_path)?;
        Ok(Self {
            rule_table,
            accent_dict: None,
            accent_predictor: None,
            analyzer: None,
        })
    }

    /// デフォルトの組み込み規則でEngineを構築する
    pub fn with_default_rules() -> Self {
        Self {
            rule_table: AccentRuleTable::default_rules(),
            accent_dict: None,
            accent_predictor: None,
            analyzer: None,
        }
    }

    /// 形態素解析辞書（.hsd）を読み込む
    pub fn load_dictionary(
        &mut self,
        path: &Path,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let analyzer = hasami::Analyzer::load(path)?;
        self.analyzer = Some(Mutex::new(analyzer));
        Ok(())
    }

    /// アクセント辞書CSVを読み込んで設定する
    pub fn load_accent_dict(&mut self, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        let dict = AccentDict::from_csv(path)?;
        self.accent_dict = Some(dict);
        Ok(())
    }

    /// アクセント辞書を直接設定する
    pub fn set_accent_dict(&mut self, dict: AccentDict) {
        self.accent_dict = Some(dict);
    }

    /// CRFモデルファイルからアクセント予測器を読み込んで設定する
    pub fn load_crf_model(&mut self, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        let predictor = crf::CrfAccentPredictor::new(path)?;
        self.accent_predictor = Some(Box::new(predictor));
        Ok(())
    }

    /// ニューラルアクセント予測器を設定する
    pub fn set_accent_predictor(
        &mut self,
        predictor: Box<dyn AccentPredictor + Send + Sync>,
    ) {
        self.accent_predictor = Some(predictor);
    }

    /// トークン列からNjdNode列を構築する
    ///
    /// アクセント辞書が設定されている場合、見出し語からアクセント型を自動設定する。
    pub fn analyze(&self, tokens: &[InputToken]) -> Vec<NjdNode> {
        let mut nodes = njd::build_njd_nodes(tokens);
        if let Some(ref dict) = self.accent_dict {
            for node in &mut nodes {
                if let Some(accent) = dict.lookup(&node.lemma, Some(&node.reading)) {
                    node.accent_type = accent;
                }
            }
        }
        nodes
    }

    /// NjdNode列からアクセント句を推定する
    pub fn estimate_accent(&self, nodes: &mut [NjdNode]) -> Vec<AccentPhrase> {
        accent::estimate_accent(nodes, &self.rule_table)
    }

    /// NjdNode列とアクセント句からHTS Full-Context Labelを生成する
    pub fn make_label(&self, nodes: &[NjdNode], phrases: &[AccentPhrase]) -> Vec<String> {
        label::generate_labels(nodes, phrases)
    }

    /// NjdNode列とアクセント句からPhoneTone列を抽出する
    pub fn extract_phone_tones(
        &self,
        nodes: &[NjdNode],
        phrases: &[AccentPhrase],
    ) -> Vec<PhoneTone> {
        prosody::extract_phone_tones(nodes, phrases)
    }

    /// NjdNode列とアクセント句からPhoneTone列を抽出する（句読点を保持）
    pub fn extract_phone_tones_with_punct(
        &self,
        nodes: &[NjdNode],
        phrases: &[AccentPhrase],
    ) -> Vec<PhoneTone> {
        prosody::extract_phone_tones_with_punct(nodes, phrases)
    }

    /// NjdNode列とアクセント句から韻律記号列を抽出する
    pub fn extract_prosody_symbols(
        &self,
        nodes: &[NjdNode],
        phrases: &[AccentPhrase],
    ) -> Vec<String> {
        prosody::extract_prosody_symbols(nodes, phrases)
    }

    /// ニューラル予測器が設定されている場合、ノードのアクセント型を上書きする
    fn apply_predictor(&self, nodes: &mut [NjdNode]) {
        if let Some(ref predictor) = self.accent_predictor {
            let predicted = predictor.predict(nodes);
            for (node, accent) in nodes.iter_mut().zip(predicted) {
                node.accent_type = accent;
            }
        }
    }

    // === 便利メソッド（トークンから直接出力を得る） ===

    /// トークン列からHTS Full-Context Labelを生成する
    pub fn tokens_to_labels(&self, tokens: &[InputToken]) -> Vec<String> {
        let mut nodes = self.analyze(tokens);
        self.apply_predictor(&mut nodes);
        let phrases = self.estimate_accent(&mut nodes);
        self.make_label(&nodes, &phrases)
    }

    /// トークン列からPhoneTone列を抽出する
    pub fn tokens_to_phone_tones(&self, tokens: &[InputToken]) -> Vec<PhoneTone> {
        let mut nodes = self.analyze(tokens);
        self.apply_predictor(&mut nodes);
        let phrases = self.estimate_accent(&mut nodes);
        self.extract_phone_tones(&nodes, &phrases)
    }

    /// トークン列からPhoneTone列を抽出する（句読点を保持）
    pub fn tokens_to_phone_tones_with_punct(&self, tokens: &[InputToken]) -> Vec<PhoneTone> {
        let mut nodes = self.analyze(tokens);
        self.apply_predictor(&mut nodes);
        let phrases = self.estimate_accent(&mut nodes);
        self.extract_phone_tones_with_punct(&nodes, &phrases)
    }

    /// トークン列から韻律記号列を抽出する
    pub fn tokens_to_prosody_symbols(&self, tokens: &[InputToken]) -> Vec<String> {
        let mut nodes = self.analyze(tokens);
        self.apply_predictor(&mut nodes);
        let phrases = self.estimate_accent(&mut nodes);
        self.extract_prosody_symbols(&nodes, &phrases)
    }

    // === テキスト直接解析メソッド（形態素解析含む） ===

    /// テキストを形態素解析してInputTokenに変換する
    fn tokenize_text(
        &self,
        text: &str,
    ) -> Result<Vec<InputToken>, Box<dyn std::error::Error>> {
        let analyzer = self.analyzer.as_ref().ok_or(
            "辞書が読み込まれていません。Engine::load_dictionary()で辞書を設定してください",
        )?;
        let mut analyzer = analyzer
            .lock()
            .map_err(|e| format!("Analyzer lock error: {e}"))?;
        let tokens = analyzer.tokenize(text);
        Ok(tokens.into_iter().map(InputToken::from).collect())
    }

    /// テキストを直接解析してHTS Labelを生成する（形態素解析含む）
    pub fn text_to_labels(
        &self,
        text: &str,
    ) -> Result<Vec<String>, Box<dyn std::error::Error>> {
        let tokens = self.tokenize_text(text)?;
        Ok(self.tokens_to_labels(&tokens))
    }

    /// テキストを直接解析してPhoneToneを生成する
    pub fn text_to_phone_tones(
        &self,
        text: &str,
    ) -> Result<Vec<PhoneTone>, Box<dyn std::error::Error>> {
        let tokens = self.tokenize_text(text)?;
        Ok(self.tokens_to_phone_tones(&tokens))
    }

    /// テキストを直接解析してPhoneToneを生成する（句読点を保持）
    pub fn text_to_phone_tones_with_punct(
        &self,
        text: &str,
    ) -> Result<Vec<PhoneTone>, Box<dyn std::error::Error>> {
        let tokens = self.tokenize_text(text)?;
        Ok(self.tokens_to_phone_tones_with_punct(&tokens))
    }

    /// テキストを直接解析してNjdNode列を返す（形態素解析含む）
    pub fn text_to_analyze(
        &self,
        text: &str,
    ) -> Result<Vec<NjdNode>, Box<dyn std::error::Error>> {
        let tokens = self.tokenize_text(text)?;
        Ok(self.analyze(&tokens))
    }

    /// テキストを直接解析して韻律記号を生成する
    pub fn text_to_prosody_symbols(
        &self,
        text: &str,
    ) -> Result<Vec<String>, Box<dyn std::error::Error>> {
        let tokens = self.tokenize_text(text)?;
        Ok(self.tokens_to_prosody_symbols(&tokens))
    }
}

impl Default for Engine {
    fn default() -> Self {
        Self::with_default_rules()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_default() {
        let engine = Engine::default();
        let tokens = vec![
            InputToken::new("今日", "名詞", "キョウ", "キョー"),
            InputToken::new("は", "助詞", "ワ", "ワ"),
            InputToken::new("良い", "形容詞", "ヨイ", "ヨイ"),
            InputToken::new("天気", "名詞", "テンキ", "テンキ"),
            InputToken::new("です", "助動詞", "デス", "デス"),
        ];

        let labels = engine.tokens_to_labels(&tokens);
        assert!(!labels.is_empty());
        assert!(labels.first().unwrap().contains("sil"));
        assert!(labels.last().unwrap().contains("sil"));

        let phone_tones = engine.tokens_to_phone_tones(&tokens);
        assert!(!phone_tones.is_empty());
        assert_eq!(phone_tones.first().unwrap().phone, "sil");
        assert_eq!(phone_tones.last().unwrap().phone, "sil");

        let symbols = engine.tokens_to_prosody_symbols(&tokens);
        assert!(!symbols.is_empty());
        assert_eq!(symbols.first().unwrap(), "^");
    }

    #[test]
    fn test_engine_single_word() {
        let engine = Engine::default();
        let tokens = vec![
            InputToken::new("猫", "名詞", "ネコ", "ネコ"),
        ];

        let mut nodes = engine.analyze(&tokens);
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0].mora_count, 2);

        // アクセント型を手動設定（辞書未実装のため）
        nodes[0].accent_type = 1;

        let phrases = engine.estimate_accent(&mut nodes);
        assert_eq!(phrases.len(), 1);

        let phone_tones = engine.extract_phone_tones(&nodes, &phrases);
        // sil, n, e, k, o, sil = 6
        assert_eq!(phone_tones.len(), 6);
    }

    #[test]
    fn test_engine_thread_safety() {
        use std::sync::Arc;
        use std::thread;

        let engine = Arc::new(Engine::default());
        let mut handles = vec![];

        for _ in 0..4 {
            let engine = Arc::clone(&engine);
            handles.push(thread::spawn(move || {
                let tokens = vec![InputToken::new("テスト", "名詞", "テスト", "テスト")];
                let labels = engine.tokens_to_labels(&tokens);
                assert!(!labels.is_empty());
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }
    }
}
