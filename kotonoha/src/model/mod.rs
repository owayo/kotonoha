//! v66 モデルの共有ディレクトリと、検証付きダウンロード。
//!
//! CLI と Python は同じ置き場所を使う。通信は `download` feature を有効にして
//! `download()` を明示的に呼んだときだけ行う。

use std::ffi::OsString;
use std::path::{Path, PathBuf};

mod manifest;
pub use manifest::{MODEL_FILES, MODEL_REVISION};

#[cfg(feature = "download")]
mod download;
#[cfg(feature = "download")]
pub use download::{DownloadError, DownloadOptions, download};

/// 配布するモデル・辞書 1 ファイル。サイズと SHA-256 は固定した版の実体に対応する。
#[derive(Debug, Clone, Copy)]
pub struct ModelFile {
    /// バンドル内のファイル名
    pub file: &'static str,
    /// 配布元のファイル名
    pub source: &'static str,
    /// 実体のバイト数（Git LFS ポインタのサイズではない）
    pub size: u64,
    /// 実体の SHA-256
    pub sha256: &'static str,
}

/// モデルの共有先。
///
/// `KOTONOHA_MODEL_BUNDLE` → `XDG_DATA_HOME/kotonoha/models` →
/// Windows の `LOCALAPPDATA/kotonoha/models` → `HOME/.local/share/kotonoha/models`。
/// 空の環境変数は無視する。場所が決められなければ `None`。
pub fn default_model_dir() -> Option<PathBuf> {
    model_dir_from(|key| std::env::var_os(key), cfg!(windows))
}

fn model_dir_from(var: impl Fn(&str) -> Option<OsString>, windows: bool) -> Option<PathBuf> {
    let var = |key: &str| var(key).filter(|v| !v.is_empty()).map(PathBuf::from);
    if let Some(dir) = var("KOTONOHA_MODEL_BUNDLE") {
        return Some(dir);
    }
    if let Some(dir) = var("XDG_DATA_HOME") {
        return Some(dir.join("kotonoha/models"));
    }
    if windows && let Some(dir) = var("LOCALAPPDATA") {
        return Some(dir.join("kotonoha/models"));
    }
    var("HOME").map(|dir| dir.join(".local/share/kotonoha/models"))
}

/// 必須の ONNX がすべてあるかを通信なしで調べる。任意のアクセント辞書は条件に含めない。
/// ダウンロード時のサイズ・ハッシュ検証とは別に、手動配置したバンドルも受け付ける。
pub fn is_model_bundle(dir: &Path) -> bool {
    MODEL_FILES
        .iter()
        .filter(|file| file.file.ends_with(".onnx"))
        .all(|file| dir.join(file.file).is_file())
}

/// 共有先に配置済みのバンドルがあれば返す。ダウンロードはしない。
pub fn installed_model_dir() -> Option<PathBuf> {
    default_model_dir().filter(|dir| is_model_bundle(dir))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_directory_precedence_and_empty_values() {
        let resolve = |values: &[(&str, &str)], windows| {
            model_dir_from(
                |key| {
                    values
                        .iter()
                        .find(|(k, _)| *k == key)
                        .map(|(_, v)| v.into())
                },
                windows,
            )
        };
        let env = [
            ("KOTONOHA_MODEL_BUNDLE", "shared"),
            ("XDG_DATA_HOME", "xdg"),
            ("LOCALAPPDATA", "local"),
            ("HOME", "home"),
        ];
        assert_eq!(resolve(&env, true), Some(PathBuf::from("shared")));
        assert_eq!(
            resolve(&env[1..], true),
            Some(PathBuf::from("xdg/kotonoha/models"))
        );
        assert_eq!(
            resolve(&env[2..], true),
            Some(PathBuf::from("local/kotonoha/models"))
        );
        assert_eq!(
            resolve(&env[2..], false),
            Some(PathBuf::from("home/.local/share/kotonoha/models"))
        );
        assert_eq!(resolve(&[("KOTONOHA_MODEL_BUNDLE", "")], true), None);
    }
}
