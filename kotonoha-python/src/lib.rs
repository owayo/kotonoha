//! kotonoha Python バインディング（PyO3）

use ::hasami::download as hasami_download;
use ::kotonoha::Engine;
use ::kotonoha::njd::{InputToken, NjdNode};
use ::kotonoha::normalizer::DictionaryNormalizer as RustDictionaryNormalizer;
use pyo3::prelude::*;
use std::path::{Path, PathBuf};

/// 辞書にある記号付き語を発音表記へ写す。モデルは読み込まない。
#[pyclass]
struct DictionaryNormalizer {
    inner: RustDictionaryNormalizer,
}

#[pymethods]
impl DictionaryNormalizer {
    #[new]
    fn new(dict_path: PathBuf, symbols: &str) -> PyResult<Self> {
        RustDictionaryNormalizer::load(dict_path, symbols)
            .map(|inner| Self { inner })
            .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))
    }

    /// 単一数量の発音を共有解析器へ委譲する。未対応ならNone。
    fn quantity_pronunciation(&mut self, input: &str) -> PyResult<Option<String>> {
        self.inner
            .quantity_pronunciation(input)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// 全文で認可された数量の(start, end, pronunciation)を返す。位置はUTF-8バイト。
    fn quantity_spans(&mut self, input: &str) -> PyResult<Vec<(usize, usize, String)>> {
        self.inner
            .quantity_spans(input)
            .map(|spans| {
                spans
                    .into_iter()
                    .map(|span| (span.start, span.end, span.pronunciation))
                    .collect()
            })
            .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))
    }

    /// 選択済み既知語の(start, end)を返す。Python文字位置ではなくUTF-8バイト位置。
    fn known_word_spans(&mut self, input: &str) -> PyResult<Vec<(usize, usize)>> {
        self.inner
            .known_word_spans(input)
            .map(|spans| {
                spans
                    .into_iter()
                    .map(|span| (span.start, span.end))
                    .collect()
            })
            .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))
    }

    /// 全選択トークンのUTF-8開始・終了・発音。未知語と空発音も含み、正読の判定はしない。
    fn token_pronunciations(&mut self, input: &str) -> PyResult<Vec<(usize, usize, String)>> {
        self.inner
            .token_pronunciations(input)
            .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))
    }

    /// 選択済み英字語・固定句の辞書発音候補。位置はUTF-8バイト。意味は判定しない。
    fn lexical_pronunciation_spans(
        &mut self,
        input: &str,
    ) -> PyResult<Vec<(usize, usize, String)>> {
        self.inner
            .lexical_pronunciation_spans(input)
            .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))
    }

    /// 完全なカタカナ発音だけを共通処理で音素へ写す。空・不正字はNone。
    #[staticmethod]
    fn pronunciation_phonemes(pronunciation: &str) -> Option<Vec<String>> {
        RustDictionaryNormalizer::pronunciation_phonemes(pronunciation)
    }

    /// 識別子の正規化済み文字列。既知全体は原文、限定した未知ID・郵便電話形は字母・桁読み。
    /// Someを再数値変換せず保持し、未対応のNoneだけ従来変換へ戻す。
    fn normalize_identifier(&mut self, input: &str) -> PyResult<Option<String>> {
        self.inner
            .normalize_identifier(input)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// 既知の記号付き語だけを辞書の発音で置換し、括弧と他の入力を保持する。
    fn normalize(&mut self, text: &str) -> PyResult<String> {
        self.inner
            .normalize(text)
            .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))
    }
}

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
    ///     model_path: 11 次元入力の単一 KNM モデルファイルパス。`model_bundle` が
    ///         指定されない場合のみ使う（環境変数 KOTONOHA_MODEL_PATH をフォールバック）。
    ///     dict_path: 形態素解析辞書（.hsd）ファイルパス（省略時は形態素解析なし）
    ///     model_bundle: 13 KNM を含むディレクトリパス
    ///         （環境変数 KOTONOHA_MODEL_BUNDLE、配置済み共有ディレクトリをフォールバック）。
    ///         指定された場合は `model_path` より優先する。
    ///     model_variant: 使用する variant 名。デフォルトは "default"。
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
        model_bundle: Option<PathBuf>,
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
        let resolved_bundle = model_bundle.or_else(|| {
            std::env::var_os("KOTONOHA_MODEL_BUNDLE")
                .filter(|v| !v.is_empty())
                .map(PathBuf::from)
        });
        let resolved_model_path = model_path.or_else(|| std::env::var("KOTONOHA_MODEL_PATH").ok());
        // legacy モデルの明示指定は共有ディレクトリの自動検出より優先する。
        #[cfg(feature = "neural")]
        let resolved_bundle = resolved_bundle.or_else(|| {
            resolved_model_path
                .is_none()
                .then(::kotonoha::model::installed_model_dir)
                .flatten()
        });
        let variant = model_variant
            .or_else(|| std::env::var("KOTONOHA_MODEL_VARIANT").ok())
            .unwrap_or_else(|| "default".to_string());

        if let Some(ref bundle_dir) = resolved_bundle {
            Self::try_load_v66_pipeline(
                &mut inner,
                bundle_dir,
                &variant,
                accent_dict_paths.as_deref(),
            )?;
        } else if let Some(ref path) = resolved_model_path {
            Self::try_load_native_predictor(&mut inner, path)?;
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
    ///     tokens: 属性アクセス可能なオブジェクトのリスト
    ///         (dict は不可。`types.SimpleNamespace` やdataclass等を使う)。
    ///         必要な属性: surface, pos, reading (必須)、
    ///         pos_detail1, pos_detail2, pos_detail3, ctype, cform,
    ///         lemma, pronunciation (省略可、None でもよい)
    ///
    /// Returns:
    ///     list[str]: HTS Full-Context Label strings
    fn make_label(&self, tokens: Vec<PyToken>) -> Vec<String> {
        let input_tokens = convert_tokens(tokens);
        self.inner.tokens_to_labels(&input_tokens)
    }

    /// トークン列からPhoneToneペアを抽出する
    ///
    /// Returns:
    ///     list[tuple[str, int]]: (phone, tone) pairs
    fn phone_tones(&self, tokens: Vec<PyToken>) -> Vec<(String, u8)> {
        let input_tokens = convert_tokens(tokens);
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
        let input_tokens = convert_tokens(tokens);
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
        let input_tokens = convert_tokens(tokens);
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
        self.inner
            .text_to_labels(text)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
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
        self.inner
            .text_to_prosody_symbols(text)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// 個別ステップ: トークンからNjdNodeを構築する
    fn analyze(&self, tokens: Vec<PyToken>) -> Vec<PyNjdNode> {
        let input_tokens = convert_tokens(tokens);
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
    ///
    /// Raises:
    ///     RuntimeError: 推論が失敗した場合、または予測結果の長さが不正な場合
    fn predict_accent_types<'py>(
        &self,
        py: Python<'py>,
        tokens: Vec<PyToken>,
    ) -> PyResult<Bound<'py, pyo3::types::PyList>> {
        let input_tokens = convert_tokens(tokens);
        let predictions = self
            .inner
            .try_predict_accent_types(&input_tokens)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        // Vec<u8> の既定変換は bytes になるため、整数リストを明示的に構築する。
        pyo3::types::PyList::new(py, predictions)
    }
}

impl KotonohaEngine {
    #[cfg(feature = "neural")]
    fn try_load_native_predictor(engine: &mut Engine, path: &str) -> PyResult<()> {
        let predictor = ::kotonoha::nn::NativePredictor::new(PathBuf::from(path).as_path())
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Failed to load KNM model: {e}"
                ))
            })?;
        engine.set_accent_predictor(Box::new(predictor));
        Ok(())
    }

    #[cfg(not(feature = "neural"))]
    fn try_load_native_predictor(_engine: &mut Engine, _path: &str) -> PyResult<()> {
        Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            "KNM support requires the 'neural' feature to be enabled",
        ))
    }

    #[cfg(feature = "neural")]
    fn try_load_v66_pipeline(
        engine: &mut Engine,
        bundle_dir: &std::path::Path,
        variant: &str,
        accent_dict_paths: Option<&[String]>,
    ) -> PyResult<()> {
        if !matches!(variant, "default" | "v66_split1") {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "unknown model_variant '{variant}'; supported: 'default'"
            )));
        }
        let bundle = if let Some(paths) = accent_dict_paths {
            let path_bufs: Vec<PathBuf> = paths.iter().map(PathBuf::from).collect();
            let path_refs: Vec<&std::path::Path> = path_bufs.iter().map(PathBuf::as_path).collect();
            ::kotonoha::nn::v66::V66Bundle::from_paths(bundle_dir, &path_refs)
        } else {
            ::kotonoha::nn::v66::V66Bundle::from_dir(bundle_dir)
        }
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load KNM bundle from {}: {e}",
                bundle_dir.display()
            ))
        })?;
        let pipeline = ::kotonoha::nn::v66::V66Pipeline::new(bundle);
        engine.set_contextual_accent_predictor(Box::new(pipeline));
        Ok(())
    }

    #[cfg(not(feature = "neural"))]
    fn try_load_v66_pipeline(
        _engine: &mut Engine,
        _bundle_dir: &std::path::Path,
        _variant: &str,
        _accent_dict_paths: Option<&[String]>,
    ) -> PyResult<()> {
        Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            "KNM model_bundle requires the 'neural' feature to be enabled",
        ))
    }
}

/// Python用トークン入力
///
/// `FromPyObject` derive は属性アクセス (`getattr`) で各フィールドを抽出する。
/// dict は受け付けない (`types.SimpleNamespace` やdataclass等を渡すこと)。
/// `Option<String>` のフィールドは `#[pyo3(default)]` により属性自体が無くてもよく、
/// `None` も許容される。
#[derive(FromPyObject)]
struct PyToken {
    surface: String,
    pos: String,
    #[pyo3(attribute("pos_detail1"), default)]
    pos_detail1: Option<String>,
    #[pyo3(attribute("pos_detail2"), default)]
    pos_detail2: Option<String>,
    #[pyo3(attribute("pos_detail3"), default)]
    pos_detail3: Option<String>,
    #[pyo3(default)]
    ctype: Option<String>,
    #[pyo3(default)]
    cform: Option<String>,
    #[pyo3(default)]
    lemma: Option<String>,
    reading: String,
    #[pyo3(default)]
    pronunciation: Option<String>,
}

/// Python用NjdNode出力
#[pyclass]
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
    mora_count: usize,
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

fn convert_tokens(tokens: Vec<PyToken>) -> Vec<InputToken> {
    tokens
        .into_iter()
        .map(|token| {
            let PyToken {
                surface,
                pos,
                pos_detail1,
                pos_detail2,
                pos_detail3,
                ctype,
                cform,
                lemma,
                reading,
                pronunciation,
            } = token;
            let lemma = lemma.unwrap_or_else(|| surface.clone());
            let pronunciation = pronunciation.unwrap_or_else(|| reading.clone());
            InputToken {
                surface,
                pos,
                pos_detail1: pos_detail1.unwrap_or_else(|| "*".to_string()),
                pos_detail2: pos_detail2.unwrap_or_else(|| "*".to_string()),
                pos_detail3: pos_detail3.unwrap_or_else(|| "*".to_string()),
                ctype: ctype.unwrap_or_else(|| "*".to_string()),
                cform: cform.unwrap_or_else(|| "*".to_string()),
                lemma,
                reading,
                pronunciation,
            }
        })
        .collect()
}

/// モデルの共有先を返す。通信やディレクトリの作成は行わない。
#[pyfunction]
fn model_dir() -> PyResult<PathBuf> {
    ::kotonoha::model::default_model_dir().ok_or_else(|| {
        pyo3::exceptions::PyRuntimeError::new_err(
            "Cannot determine model directory; set KOTONOHA_MODEL_BUNDLE",
        )
    })
}

/// 最新 Release の KNM を共有先（または dir）へ取得・更新する。推論時には通信しない。
#[pyfunction]
#[pyo3(signature = (dir=None, *, force=false, base_url=None))]
fn download_models(
    py: Python<'_>,
    dir: Option<PathBuf>,
    force: bool,
    base_url: Option<String>,
) -> PyResult<PathBuf> {
    py.detach(move || {
        ::kotonoha::model::download(
            dir.as_deref(),
            ::kotonoha::model::DownloadOptions { force, base_url },
        )
    })
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

pyo3::create_exception!(
    kotonoha,
    DictionaryError,
    pyo3::exceptions::PyRuntimeError,
    "辞書を取得・検証できなかったときの例外。`code` に種類（format・conflict・corrupt・download）、`path` に対象のファイル（無ければ None）を持つ。"
);

/// 種類と対象のファイルを属性に持つ `DictionaryError` を作る。
fn dictionary_error(py: Python<'_>, code: &str, message: String, path: Option<&Path>) -> PyErr {
    let error = DictionaryError::new_err(message);
    let value = error.value(py);
    // 属性を付けられなくても、メッセージで原因は伝わる
    let _ = value.setattr("code", code);
    let _ = value.setattr("path", path.map(Path::to_path_buf));
    error
}

/// 組み込みの hasami が辞書を読めなかった理由を、利用側が分岐できる例外に写す。
fn dict_error_to_py(py: Python<'_>, error: ::hasami::DictError, path: &Path) -> PyErr {
    use ::hasami::DictError;
    match error {
        DictError::Io(source) => PyErr::from(source),
        DictError::NotFound(_) => pyo3::exceptions::PyFileNotFoundError::new_err(error.to_string()),
        DictError::UnsupportedVersion(_) | DictError::UnsupportedPlatform => {
            dictionary_error(py, "format", error.to_string(), Some(path))
        }
        DictError::NotHsd | DictError::Corrupt(_) | DictError::Invalid(_) => {
            dictionary_error(py, "corrupt", error.to_string(), Some(path))
        }
    }
}

/// 辞書の取得で起きた失敗。GIL を外した処理の中では Python の例外を作らない。
enum DictionaryDownloadFailure {
    /// 目録に無い辞書の名前
    UnknownName(String),
    /// 目録が組み込みの hasami と別の版のもの（ミラーを指定したとき）
    CatalogVersion { found: String },
    /// hasami の取得処理の失敗
    Download(hasami_download::DownloadError),
}

impl From<hasami_download::DownloadError> for DictionaryDownloadFailure {
    fn from(error: hasami_download::DownloadError) -> Self {
        Self::Download(error)
    }
}

/// 取得の失敗を、利用側が分岐できる例外に写す。
fn download_failure_to_py(py: Python<'_>, failure: DictionaryDownloadFailure) -> PyErr {
    use hasami_download::DownloadError;
    let error = match failure {
        DictionaryDownloadFailure::UnknownName(name) => {
            return pyo3::exceptions::PyValueError::new_err(format!(
                "unknown dictionary name: {name}"
            ));
        }
        DictionaryDownloadFailure::CatalogVersion { found } => {
            return dictionary_error(
                py,
                "download",
                format!(
                    "the catalog is for hasami {found}, but this kotonoha embeds hasami {}",
                    hasami_version()
                ),
                None,
            );
        }
        DictionaryDownloadFailure::Download(error) => error,
    };
    let message = error.to_string();
    match &error {
        DownloadError::FormatVersion { .. } => dictionary_error(py, "format", message, None),
        DownloadError::Differs { path, .. } => {
            dictionary_error(py, "conflict", message, Some(path))
        }
        DownloadError::NotDictionary { .. }
        | DownloadError::Checksum { .. }
        | DownloadError::Decompress { .. }
        | DownloadError::Truncated { .. }
        | DownloadError::Oversized { .. }
        | DownloadError::ContentLength { .. } => dictionary_error(py, "corrupt", message, None),
        DownloadError::InvalidDict(_) => pyo3::exceptions::PyValueError::new_err(message),
        DownloadError::CreateDir { path, .. }
        | DownloadError::Inspect { path, .. }
        | DownloadError::TempFile { dir: path, .. }
        | DownloadError::Write { path, .. }
        | DownloadError::Persist { path, .. } => {
            pyo3::exceptions::PyOSError::new_err(format!("{message} ({})", path.display()))
        }
        // 目録・通信・HTTP の設定の失敗と、hasami が今後足す失敗
        _ => dictionary_error(py, "download", message, None),
    }
}

/// 組み込みの hasami の版（例: "26.11.102"）を返す。
#[pyfunction]
fn hasami_version() -> &'static str {
    hasami_download::CURRENT_TAG
        .strip_prefix('v')
        .unwrap_or(hasami_download::CURRENT_TAG)
}

/// 組み込みの hasami が読める辞書（HSD）の形式の版を返す。
#[pyfunction]
fn dictionary_format_version() -> u32 {
    ::hasami::hsd::FORMAT_VERSION
}

/// 辞書の共有保存先を返す。通信やディレクトリの作成は行わない。
#[pyfunction]
fn dictionary_dir() -> PyResult<PathBuf> {
    ::hasami::analyzer::data_dir().ok_or_else(|| {
        pyo3::exceptions::PyRuntimeError::new_err(
            "Cannot determine dictionary directory; set HASAMI_DATA_DIR",
        )
    })
}

/// 辞書を組み込みの hasami で読めるかを確かめる。通信しない。
#[pyfunction]
fn validate_dictionary(py: Python<'_>, path: PathBuf) -> PyResult<()> {
    let result = py.detach(|| ::hasami::Dictionary::load(&path).map(drop));
    result.map_err(|error| dict_error_to_py(py, error, &path))
}

/// 組み込みの hasami と同じ版の配布辞書を、共有保存先（または dir）へ取得する。
///
/// 正しいファイルがすでにあれば辞書は取り直さない（目録だけを確かめる）。
#[pyfunction]
#[pyo3(signature = (dir=None, *, name=None, force=false, base_url=None))]
fn download_dictionary(
    py: Python<'_>,
    dir: Option<PathBuf>,
    name: Option<String>,
    force: bool,
    base_url: Option<String>,
) -> PyResult<PathBuf> {
    let dir = match dir {
        Some(dir) => dir,
        None => dictionary_dir()?,
    };
    let result = py.detach(|| -> Result<PathBuf, DictionaryDownloadFailure> {
        let catalog = match base_url.as_deref() {
            Some(url) => {
                let catalog = hasami_download::catalog_from(url)?;
                // ミラーの目録も、組み込みの hasami と同じ版のものだけを使う
                if catalog.hasami_version != hasami_version() {
                    return Err(DictionaryDownloadFailure::CatalogVersion {
                        found: catalog.hasami_version,
                    });
                }
                catalog
            }
            None => hasami_download::catalog(hasami_download::CURRENT_TAG)?,
        };
        catalog.check_format()?;
        let name = match name.as_deref() {
            Some(name) => name,
            None if catalog.recommended.is_empty() => hasami_download::RECOMMENDED,
            None => catalog.recommended.as_str(),
        };
        let dict = catalog
            .find(name)
            .ok_or_else(|| DictionaryDownloadFailure::UnknownName(name.to_string()))?;
        let options = hasami_download::DownloadOptions {
            base_url: base_url.as_deref(),
            force,
            ..Default::default()
        };
        Ok(hasami_download::download(dict, &dir, options)?
            .path()
            .to_path_buf())
    });
    result.map_err(|failure| download_failure_to_py(py, failure))
}

/// kotonoha Python module
#[pymodule]
fn kotonoha(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(model_dir, m)?)?;
    m.add_function(wrap_pyfunction!(download_models, m)?)?;
    m.add_function(wrap_pyfunction!(hasami_version, m)?)?;
    m.add_function(wrap_pyfunction!(dictionary_format_version, m)?)?;
    m.add_function(wrap_pyfunction!(dictionary_dir, m)?)?;
    m.add_function(wrap_pyfunction!(validate_dictionary, m)?)?;
    m.add_function(wrap_pyfunction!(download_dictionary, m)?)?;
    m.add("DictionaryError", m.py().get_type::<DictionaryError>())?;
    m.add_class::<KotonohaEngine>()?;
    m.add_class::<DictionaryNormalizer>()?;
    m.add_class::<PyNjdNode>()?;
    Ok(())
}
