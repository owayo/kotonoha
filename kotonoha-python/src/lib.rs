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
    ///     model_path: legacy v8 ONNX モデルファイルパス。`model_bundle` が
    ///         指定されない場合のみ使う（環境変数 KOTONOHA_MODEL_PATH をフォールバック）。
    ///     dict_path: 形態素解析辞書（.hsd）ファイルパス（省略時は形態素解析なし）
    ///     model_bundle: v66 系 12 ONNX を含むディレクトリパス
    ///         （環境変数 KOTONOHA_MODEL_BUNDLE をフォールバック）。
    ///         指定された場合は `model_path` より優先する。
    ///     model_variant: 使用する variant 名。デフォルトは bundle なら
    ///         "v66_split1"、未指定なら "v8"。
    ///     accent_dict_paths: bundle の `accent_dict.csv` を上書きする CSV パスのリスト。
    ///         複数指定時は順にマージし、後の path のエントリが先のエントリを上書きする。
    #[new]
    #[pyo3(signature = (
        accent_rule_path=None,
        model_path=None,
        dict_path=None,
        model_bundle=None,
        model_variant=None,
        accent_dict_paths=None,
    ))]
    fn new(
        accent_rule_path: Option<String>,
        model_path: Option<String>,
        dict_path: Option<String>,
        model_bundle: Option<String>,
        model_variant: Option<String>,
        accent_dict_paths: Option<Vec<String>>,
    ) -> PyResult<Self> {
        let mut inner = match accent_rule_path {
            Some(path) => Engine::new(PathBuf::from(path).as_path())
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?,
            None => Engine::with_default_rules(),
        };

        // 優先順位: model_bundle 引数 → KOTONOHA_MODEL_BUNDLE 環境変数 →
        //          model_path 引数 → KOTONOHA_MODEL_PATH 環境変数
        let resolved_bundle =
            model_bundle.or_else(|| std::env::var("KOTONOHA_MODEL_BUNDLE").ok());
        let resolved_model_path =
            model_path.or_else(|| std::env::var("KOTONOHA_MODEL_PATH").ok());
        let variant = model_variant
            .or_else(|| std::env::var("KOTONOHA_MODEL_VARIANT").ok())
            .unwrap_or_else(|| {
                if resolved_bundle.is_some() {
                    "v66_split1".to_string()
                } else {
                    "v8".to_string()
                }
            });

        if let Some(ref bundle_dir) = resolved_bundle {
            Self::try_load_v66_pipeline(
                &mut inner,
                bundle_dir,
                &variant,
                accent_dict_paths.as_deref(),
            )?;
        } else if let Some(ref path) = resolved_model_path {
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

    /// トークン列に対するアクセント型の予測値だけを返す
    ///
    /// 設定された予測器 (`model_bundle` 経由の v66 系または `model_path` 経由の v8) を
    /// そのまま使う。predictor 未設定の場合はルールベース由来の `accent_type`
    /// (NjdNode の初期値) を返す。
    ///
    /// Returns:
    ///     list[int]: 各トークンの predicted accent_type (0..20)
    fn predict_accent_types(&self, tokens: Vec<PyToken>) -> Vec<u8> {
        let input_tokens = convert_tokens(&tokens);
        self.inner.predict_accent_types(&input_tokens)
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

    #[cfg(feature = "cuda")]
    fn try_load_v66_pipeline(
        engine: &mut Engine,
        bundle_dir: &str,
        variant: &str,
        accent_dict_paths: Option<&[String]>,
    ) -> PyResult<()> {
        if variant != "v66_split1" {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "unknown model_variant '{variant}'; supported: 'v66_split1'"
            )));
        }
        let dir = PathBuf::from(bundle_dir);
        let bundle = if let Some(paths) = accent_dict_paths {
            let path_bufs: Vec<PathBuf> = paths.iter().map(PathBuf::from).collect();
            let path_refs: Vec<&std::path::Path> =
                path_bufs.iter().map(PathBuf::as_path).collect();
            ::kotonoha::nn::v66::V66Bundle::from_paths(&dir, &path_refs)
        } else {
            ::kotonoha::nn::v66::V66Bundle::from_dir(&dir)
        }
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load v66 bundle from {bundle_dir}: {e}"
            ))
        })?;
        let pipeline = ::kotonoha::nn::v66::V66Pipeline::new(bundle);
        engine.set_contextual_accent_predictor(Box::new(pipeline));
        Ok(())
    }

    #[cfg(not(feature = "cuda"))]
    fn try_load_v66_pipeline(
        _engine: &mut Engine,
        _bundle_dir: &str,
        _variant: &str,
        _accent_dict_paths: Option<&[String]>,
    ) -> PyResult<()> {
        Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            "v66 model_bundle requires the 'cuda' feature to be enabled",
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
