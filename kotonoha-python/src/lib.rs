//! kotonoha Python バインディング（PyO3）

use ::kotonoha::njd::{InputToken, NjdNode};
use ::kotonoha::Engine;
use pyo3::prelude::*;
use std::path::PathBuf;

/// Python用エンジンラッパー
#[pyclass]
struct KotonohaEngine {
    inner: Engine,
}

#[pymethods]
impl KotonohaEngine {
    /// デフォルト規則でエンジンを作成
    ///
    /// Args:
    ///     accent_rule_path: アクセント規則CSVファイルパス（省略時はデフォルト規則）
    ///     model_path: ONNXモデルファイルパス（省略時は環境変数 KOTONOHA_MODEL_PATH を参照、
    ///         未設定ならルールベース予測）
    ///     dict_path: 形態素解析辞書（.hsd）ファイルパス（省略時は形態素解析なし）
    #[new]
    #[pyo3(signature = (accent_rule_path=None, model_path=None, dict_path=None))]
    fn new(
        accent_rule_path: Option<String>,
        model_path: Option<String>,
        dict_path: Option<String>,
    ) -> PyResult<Self> {
        let mut inner = match accent_rule_path {
            Some(path) => Engine::new(PathBuf::from(path).as_path())
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?,
            None => Engine::with_default_rules(),
        };

        // model_path 引数 → KOTONOHA_MODEL_PATH 環境変数の優先順で解決
        let resolved_model_path = model_path.or_else(|| std::env::var("KOTONOHA_MODEL_PATH").ok());
        if let Some(ref path) = resolved_model_path {
            Self::try_load_onnx_predictor(&mut inner, path)?;
        }

        if let Some(ref path) = dict_path {
            inner
                .load_dictionary(PathBuf::from(path).as_path())
                .map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                        "Failed to load dictionary: {e}"
                    ))
                })?;
        }

        Ok(Self { inner })
    }

    /// トークン列からHTS Full-Context Labelを生成する
    ///
    /// Args:
    ///     tokens: list of dict with keys:
    ///         surface, pos, pos_detail1, pos_detail2, pos_detail3,
    ///         ctype, cform, lemma, reading, pronunciation
    ///
    /// Returns:
    ///     list[str]: HTS Full-Context Label strings
    fn make_label(&self, tokens: Vec<PyToken>) -> Vec<String> {
        let input_tokens = convert_tokens(&tokens);
        self.inner.tokens_to_labels(&input_tokens)
    }

    /// トークン列からPhoneToneペアを抽出する
    ///
    /// Returns:
    ///     list[tuple[str, int]]: (phone, tone) pairs
    fn phone_tones(&self, tokens: Vec<PyToken>) -> Vec<(String, u8)> {
        let input_tokens = convert_tokens(&tokens);
        self.inner
            .tokens_to_phone_tones(&input_tokens)
            .into_iter()
            .map(|pt| (pt.phone, pt.tone))
            .collect()
    }

    /// トークン列からPhoneToneペアを抽出する（句読点を保持）
    ///
    /// Returns:
    ///     list[tuple[str, int]]: (phone, tone) pairs including punctuation
    fn phone_tones_with_punct(&self, tokens: Vec<PyToken>) -> Vec<(String, u8)> {
        let input_tokens = convert_tokens(&tokens);
        self.inner
            .tokens_to_phone_tones_with_punct(&input_tokens)
            .into_iter()
            .map(|pt| (pt.phone, pt.tone))
            .collect()
    }

    /// トークン列から韻律記号列を抽出する
    ///
    /// Returns:
    ///     list[str]: Prosody symbols
    fn prosody_symbols(&self, tokens: Vec<PyToken>) -> Vec<String> {
        let input_tokens = convert_tokens(&tokens);
        self.inner.tokens_to_prosody_symbols(&input_tokens)
    }

    /// テキストを直接解析してHTS Full-Context Labelを生成する（形態素解析含む）
    ///
    /// Args:
    ///     text: 解析対象テキスト
    ///
    /// Returns:
    ///     list[str]: HTS Full-Context Label strings
    fn text_to_labels(&self, text: &str) -> PyResult<Vec<String>> {
        self.inner.text_to_labels(text).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
        })
    }

    /// テキストを直接解析してPhoneToneペアを生成する
    ///
    /// Args:
    ///     text: 解析対象テキスト
    ///
    /// Returns:
    ///     list[tuple[str, int]]: (phone, tone) pairs
    fn text_to_phone_tones(&self, text: &str) -> PyResult<Vec<(String, u8)>> {
        self.inner
            .text_to_phone_tones(text)
            .map(|pts| pts.into_iter().map(|pt| (pt.phone, pt.tone)).collect())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// テキストを直接解析してPhoneToneペアを生成する（句読点を保持）
    ///
    /// Args:
    ///     text: 解析対象テキスト
    ///
    /// Returns:
    ///     list[tuple[str, int]]: (phone, tone) pairs including punctuation
    fn text_to_phone_tones_with_punct(&self, text: &str) -> PyResult<Vec<(String, u8)>> {
        self.inner
            .text_to_phone_tones_with_punct(text)
            .map(|pts| pts.into_iter().map(|pt| (pt.phone, pt.tone)).collect())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// テキストを直接解析して韻律記号列を生成する
    ///
    /// Args:
    ///     text: 解析対象テキスト
    ///
    /// Returns:
    ///     list[str]: Prosody symbols
    fn text_to_prosody_symbols(&self, text: &str) -> PyResult<Vec<String>> {
        self.inner.text_to_prosody_symbols(text).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
        })
    }

    /// 個別ステップ: トークンからNjdNodeを構築する
    fn analyze(&self, tokens: Vec<PyToken>) -> Vec<PyNjdNode> {
        let input_tokens = convert_tokens(&tokens);
        let nodes = self.inner.analyze(&input_tokens);
        nodes.into_iter().map(PyNjdNode::from).collect()
    }

    /// テキストを直接解析してNjdNodeを返す（形態素解析含む）
    fn text_to_analyze(&self, text: &str) -> PyResult<Vec<PyNjdNode>> {
        self.inner
            .text_to_analyze(text)
            .map(|nodes| nodes.into_iter().map(PyNjdNode::from).collect())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }
}

impl KotonohaEngine {
    #[cfg(feature = "cuda")]
    fn try_load_onnx_predictor(engine: &mut Engine, path: &str) -> PyResult<()> {
        let predictor =
            ::kotonoha::nn::OnnxPredictor::new(PathBuf::from(path).as_path()).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Failed to load ONNX model: {e}"
                ))
            })?;
        engine.set_accent_predictor(Box::new(predictor));
        Ok(())
    }

    #[cfg(not(feature = "cuda"))]
    fn try_load_onnx_predictor(_engine: &mut Engine, _path: &str) -> PyResult<()> {
        Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            "ONNX support requires the 'cuda' feature to be enabled",
        ))
    }
}

/// Python用トークン入力
#[derive(FromPyObject)]
struct PyToken {
    surface: String,
    pos: String,
    #[pyo3(attribute("pos_detail1"))]
    pos_detail1: Option<String>,
    #[pyo3(attribute("pos_detail2"))]
    pos_detail2: Option<String>,
    #[pyo3(attribute("pos_detail3"))]
    pos_detail3: Option<String>,
    ctype: Option<String>,
    cform: Option<String>,
    lemma: Option<String>,
    reading: String,
    pronunciation: Option<String>,
}

/// Python用NjdNode出力
#[pyclass(skip_from_py_object)]
#[derive(Clone)]
struct PyNjdNode {
    #[pyo3(get)]
    surface: String,
    #[pyo3(get)]
    pos: String,
    #[pyo3(get)]
    reading: String,
    #[pyo3(get)]
    pronunciation: String,
    #[pyo3(get)]
    accent_type: u8,
    #[pyo3(get)]
    mora_count: u8,
}

impl From<NjdNode> for PyNjdNode {
    fn from(node: NjdNode) -> Self {
        Self {
            surface: node.surface,
            pos: node.pos.to_label_str().to_string(),
            reading: node.reading,
            pronunciation: node.pronunciation,
            accent_type: node.accent_type,
            mora_count: node.mora_count,
        }
    }
}

fn convert_tokens(tokens: &[PyToken]) -> Vec<InputToken> {
    tokens
        .iter()
        .map(|t| InputToken {
            surface: t.surface.clone(),
            pos: t.pos.clone(),
            pos_detail1: t.pos_detail1.clone().unwrap_or_else(|| "*".to_string()),
            pos_detail2: t.pos_detail2.clone().unwrap_or_else(|| "*".to_string()),
            pos_detail3: t.pos_detail3.clone().unwrap_or_else(|| "*".to_string()),
            ctype: t.ctype.clone().unwrap_or_else(|| "*".to_string()),
            cform: t.cform.clone().unwrap_or_else(|| "*".to_string()),
            lemma: t.lemma.clone().unwrap_or_else(|| t.surface.clone()),
            reading: t.reading.clone(),
            pronunciation: t
                .pronunciation
                .clone()
                .unwrap_or_else(|| t.reading.clone()),
        })
        .collect()
}

/// kotonoha Python module
#[pymodule]
fn kotonoha(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<KotonohaEngine>()?;
    m.add_class::<PyNjdNode>()?;
    Ok(())
}
