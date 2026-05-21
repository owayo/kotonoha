//! v66 系モデルバンドル
//!
//! `kotonoha-models/` ディレクトリ配下の 12 ONNX + `accent_dict.csv` を
//! 一括ロードするためのコンテナ。
//!
//! ## 想定ディレクトリ構成
//!
//! ```text
//! <bundle_dir>/
//!   accent_model_v24.onnx
//!   accent_model_v38.onnx
//!   accent_model_v54_split1.onnx
//!   accent_model_v54_split2.onnx
//!   accent_model_v54_split3.onnx
//!   accent_model_v59_fold0.onnx
//!   accent_model_v59_fold1.onnx
//!   accent_model_v59_fold2.onnx
//!   accent_model_v59_fold3.onnx
//!   accent_model_v59_fold4.onnx
//!   accent_model_v61.onnx
//!   accent_model_v63.onnx
//!   accent_model_v66_split1.onnx
//!   accent_dict.csv         (optional; あればロード時に enrich に使う)
//! ```

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use ort::execution_providers::CUDAExecutionProvider;
use ort::session::Session;

use crate::accent_dict::AccentDict;

/// 9 student モデルのファイル名 (固定順)
///
/// 訓練側 `precompute_v66_stacker.py:77-87` の `student_14d_specs` と一致。
pub const STUDENT_FILENAMES: [&str; 9] = [
    "accent_model_v38.onnx",
    "accent_model_v54_split1.onnx",
    "accent_model_v54_split2.onnx",
    "accent_model_v54_split3.onnx",
    "accent_model_v59_fold0.onnx",
    "accent_model_v59_fold1.onnx",
    "accent_model_v59_fold2.onnx",
    "accent_model_v59_fold3.onnx",
    "accent_model_v59_fold4.onnx",
];

/// 各 ONNX セッションを Mutex で保護した束
pub struct V66Models {
    /// teacher (11 dim 入力 → softmax)
    pub v24: Mutex<Session>,
    /// 9 student (14 dim 入力)
    pub students: Vec<Mutex<Session>>,
    /// meta predictor v61 (24 dim 入力)
    pub v61: Mutex<Session>,
    /// meta predictor v63 (24 dim 入力)
    pub v63: Mutex<Session>,
    /// final stacker v66_split1 (103 dim 入力)
    pub v66_split1: Mutex<Session>,
}

/// v66 系の本番ランタイムが必要とするリソース一式
pub struct V66Bundle {
    /// `accent_dict.csv` から構築した辞書 (enrich に使用)
    pub accent_dict: AccentDict,
    /// 12 ONNX セッション
    pub models: V66Models,
    /// 元の bundle ディレクトリパス (デバッグ用)
    pub root: PathBuf,
}

impl V66Bundle {
    /// バンドルディレクトリから 12 ONNX + `accent_dict.csv` を eager load する
    ///
    /// `dir/accent_dict.csv` があればそれを使い、無ければ空辞書 (enrich は no-op)。
    /// 複数 CSV をマージしたい場合は [`Self::from_paths`] を使う。
    ///
    /// CUDA 利用可能なら CUDA、そうでなければ CPU を使う (ort 既定の fallback 動作)。
    pub fn from_dir(dir: &Path) -> Result<Self, BundleError> {
        // 空配列で呼ぶと `from_paths` が `dir/accent_dict.csv` をフォールバックして
        // ロードする (存在しなければ空辞書)。
        Self::from_paths(dir, &[])
    }

    /// 任意の accent_dict CSV 群を指定してバンドルを構築する
    ///
    /// - `models_dir/accent_dict.csv` があればまずベースとしてロードする
    /// - その上に `accent_dict_paths` を順にマージし、後の path のエントリが先の
    ///   エントリを上書きする (Python `_load_accent_dicts` と同じ後勝ち挙動)
    /// - したがって bundle 同梱の `accent_dict.csv` は常に取り込まれ、追加辞書を
    ///   渡しても recall が下がらない
    pub fn from_paths(models_dir: &Path, accent_dict_paths: &[&Path]) -> Result<Self, BundleError> {
        if !models_dir.is_dir() {
            return Err(BundleError::NotADirectory(models_dir.to_path_buf()));
        }

        let v24 = load_session(&models_dir.join("accent_model_v24.onnx"))?;
        let students = STUDENT_FILENAMES
            .iter()
            .map(|name| load_session(&models_dir.join(name)))
            .collect::<Result<Vec<_>, _>>()?;
        let v61 = load_session(&models_dir.join("accent_model_v61.onnx"))?;
        let v63 = load_session(&models_dir.join("accent_model_v63.onnx"))?;
        let v66_split1 = load_session(&models_dir.join("accent_model_v66_split1.onnx"))?;

        let accent_dict = load_accent_dicts(models_dir, accent_dict_paths)?;

        Ok(Self {
            accent_dict,
            models: V66Models {
                v24: Mutex::new(v24),
                students: students.into_iter().map(Mutex::new).collect(),
                v61: Mutex::new(v61),
                v63: Mutex::new(v63),
                v66_split1: Mutex::new(v66_split1),
            },
            root: models_dir.to_path_buf(),
        })
    }
}

/// 複数 accent_dict.csv をマージしてロードする (Python `_load_accent_dicts` 互換)
///
/// 1. `models_dir/accent_dict.csv` があればまずベースとしてロード
/// 2. `paths` を順に後勝ちマージ
///
/// 後勝ち順は呼び出し側に委ねるが、典型的には `accent_dict_jsut.csv` が bundle
/// 同梱されている場合、その上に呼び出し側の追加 CSV を渡す形となる。
fn load_accent_dicts(models_dir: &Path, paths: &[&Path]) -> Result<AccentDict, BundleError> {
    let mut dict = AccentDict::new();
    let bundled = models_dir.join("accent_dict.csv");
    if bundled.exists() {
        let base = AccentDict::from_csv(&bundled)
            .map_err(|e| BundleError::AccentDict(bundled.clone(), e.to_string()))?;
        merge_accent_dict(&mut dict, &base);
    }
    for p in paths {
        let extra = AccentDict::from_csv(p)
            .map_err(|e| BundleError::AccentDict(p.to_path_buf(), e.to_string()))?;
        merge_accent_dict(&mut dict, &extra);
    }
    Ok(dict)
}

/// `extra` のエントリで `base` を `(lemma, reading)` キー単位で上書きする
fn merge_accent_dict(base: &mut AccentDict, extra: &AccentDict) {
    for (lemma, reading, accent) in extra.iter_entries() {
        base.set(lemma, reading, accent);
    }
}

/// 単一 ONNX を CUDA / CPU フォールバックでロードする
fn load_session(path: &Path) -> Result<Session, BundleError> {
    if !path.is_file() {
        return Err(BundleError::MissingModel(path.to_path_buf()));
    }
    let session = Session::builder()
        .map_err(|e| BundleError::SessionBuild(path.to_path_buf(), e.to_string()))?
        .with_execution_providers([CUDAExecutionProvider::default().build()])
        .map_err(|e| BundleError::SessionBuild(path.to_path_buf(), e.to_string()))?
        .commit_from_file(path)
        .map_err(|e| BundleError::SessionBuild(path.to_path_buf(), e.to_string()))?;
    Ok(session)
}

/// バンドルロード時のエラー
#[derive(Debug, thiserror::Error)]
pub enum BundleError {
    /// 指定パスがディレクトリでない
    #[error("bundle path is not a directory: {0}")]
    NotADirectory(PathBuf),
    /// 必須 ONNX ファイルが欠けている
    #[error("missing model file: {0}")]
    MissingModel(PathBuf),
    /// ort セッション構築失敗
    #[error("failed to build ort session for {0}: {1}")]
    SessionBuild(PathBuf, String),
    /// accent_dict.csv のロード失敗
    #[error("failed to load accent_dict at {0}: {1}")]
    AccentDict(PathBuf, String),
}
