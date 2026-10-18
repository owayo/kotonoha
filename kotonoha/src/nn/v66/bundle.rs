//! 役割名で読み込む KNM モデルバンドル
//!
//! `kotonoha-models/` ディレクトリ配下の 13 KNM + `accent_dict.csv` を
//! 一括ロードするためのコンテナ。
//!
//! ## 想定ディレクトリ構成
//!
//! ```text
//! <bundle_dir>/
//!   teacher.knm
//!   student-01.knm
//!   student-02.knm
//!   student-03.knm
//!   student-04.knm
//!   student-05.knm
//!   student-06.knm
//!   student-07.knm
//!   student-08.knm
//!   student-09.knm
//!   meta-01.knm
//!   meta-02.knm
//!   accent.knm
//!   accent_dict.csv         (optional; あればロード時に enrich に使う)
//! ```

use std::path::{Path, PathBuf};

use crate::nn::native::NativeModel;

use crate::accent_dict::AccentDict;

/// 9 student モデルのファイル名 (固定順)
///
/// 訓練側 `precompute_v66_stacker.py:77-87` の `student_14d_specs` と一致。
pub const STUDENT_FILENAMES: [&str; 9] = [
    "student-01.knm",
    "student-02.knm",
    "student-03.knm",
    "student-04.knm",
    "student-05.knm",
    "student-06.knm",
    "student-07.knm",
    "student-08.knm",
    "student-09.knm",
];

/// 不変の専用モデル一式。同時推論にロックは不要。
pub struct V66Models {
    /// teacher (11 dim 入力 → softmax)
    pub v24: NativeModel,
    /// 9 student (14 dim 入力)
    pub students: Vec<NativeModel>,
    /// meta predictor v61 (24 dim 入力)
    pub v61: NativeModel,
    /// meta predictor v63 (24 dim 入力)
    pub v63: NativeModel,
    /// final stacker v66_split1 (103 dim 入力)
    pub v66_split1: NativeModel,
}

/// v66 系の本番ランタイムが必要とするリソース一式
pub struct V66Bundle {
    /// `accent_dict.csv` から構築した辞書 (enrich に使用)
    pub accent_dict: AccentDict,
    /// 13 KNM セッション
    pub models: V66Models,
    /// 元の bundle ディレクトリパス (デバッグ用)
    pub root: PathBuf,
}

impl V66Bundle {
    /// バンドルディレクトリから 13 KNM + `accent_dict.csv` を eager load する
    ///
    /// `dir/accent_dict.csv` があればそれを使い、無ければ空辞書 (enrich は no-op)。
    /// 複数 CSV をマージしたい場合は [`Self::from_paths`] を使う。
    ///
    /// 外部の推論ライブラリを使わず CPU 上で実行する。
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

        let v24 = load_session(&models_dir.join("teacher.knm"), 11)?;
        let students = STUDENT_FILENAMES
            .iter()
            .map(|name| load_session(&models_dir.join(name), 14))
            .collect::<Result<Vec<_>, _>>()?;
        let v61 = load_session(&models_dir.join("meta-01.knm"), 24)?;
        let v63 = load_session(&models_dir.join("meta-02.knm"), 24)?;
        let v66_split1 = load_session(&models_dir.join("accent.knm"), 103)?;

        let accent_dict = load_accent_dicts(models_dir, accent_dict_paths)?;

        Ok(Self {
            accent_dict,
            models: V66Models {
                v24,
                students,
                v61,
                v63,
                v66_split1,
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

/// 単一 KNM を専用 CPU モデルとしてロードする
fn load_session(path: &Path, expected: usize) -> Result<NativeModel, BundleError> {
    if !path.is_file() {
        return Err(BundleError::MissingModel(path.to_path_buf()));
    }
    let model = NativeModel::load(path)
        .map_err(|e| BundleError::SessionBuild(path.to_path_buf(), e.to_string()))?;
    if model.input_dim() != expected {
        return Err(BundleError::SessionBuild(
            path.to_path_buf(),
            "unexpected feature dimension".into(),
        ));
    }
    Ok(model)
}

/// バンドルロード時のエラー
#[derive(Debug, thiserror::Error)]
pub enum BundleError {
    /// 指定パスがディレクトリでない
    #[error("bundle path is not a directory: {0}")]
    NotADirectory(PathBuf),
    /// 必須 KNM ファイルが欠けている
    #[error("missing KNM model: {0}; run kotonoha model convert --dir <bundle-dir>")]
    MissingModel(PathBuf),
    /// 専用モデルのロード失敗
    #[error("failed to load native model for {0}: {1}")]
    SessionBuild(PathBuf, String),
    /// accent_dict.csv のロード失敗
    #[error("failed to load accent_dict at {0}: {1}")]
    AccentDict(PathBuf, String),
}
