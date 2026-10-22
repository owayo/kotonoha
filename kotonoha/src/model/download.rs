//! 最新の KNM Release の取得と、固定版 ONNX のオフライン変換。

use super::{MODEL_FILES, MODEL_REVISION};
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

mod release;
pub use release::download;

/// 最新の KNM 配布物を取得するオプション。
#[derive(Debug, Default, Clone)]
pub struct DownloadOptions {
    /// 再取得し、手動配置・変更した既存ファイルの置き換えも許可する。
    pub force: bool,
    /// KNM アーカイブとチェックサムを置いたミラー。GitHub の認証は送らない。
    pub base_url: Option<String>,
}

/// モデル取得エラー。
#[derive(Debug, thiserror::Error)]
pub enum DownloadError {
    /// 保存先を決められなかった。
    #[error("cannot determine model directory; specify --dir or KOTONOHA_MODEL_BUNDLE")]
    NoDirectory,
    /// ファイル操作に失敗した。
    #[error("model file I/O: {0}")]
    Io(#[from] io::Error),
    /// 取得に失敗した。署名付き URL や認証値はエラーへ含めない。
    #[error("failed to download {0}; check network and repository access")]
    Network(String),
    /// サイズまたは SHA-256 が一致しなかった。
    #[error("model size or SHA-256 mismatch: {0}")]
    Integrity(String),
    /// 既存の手動配置・変更済みファイルを保護した。
    #[error("existing model differs: {0}; use --force to replace it")]
    Existing(PathBuf),
    /// 配置失敗後の復元にも失敗したため、旧モデルを回復用に保持した。
    #[error("could not restore previous models; recovery files are preserved in {0}")]
    Recovery(PathBuf),
    /// 版・アーカイブ・マニフェストが未対応または不正。
    #[error("invalid model release: {0}")]
    Release(String),
    /// KNM の検証または既存形式の変換に失敗した。
    #[error("KNM model: {0}")]
    Conversion(#[from] super::knm::KnmError),
}

/// 固定版の既存 ONNX をその場で KNM に変換する。通信せず、元ファイルは保持する。
pub fn convert_legacy_models(dir: &Path, force: bool) -> Result<PathBuf, DownloadError> {
    convert_legacy_models_to(dir, dir, force)
}

/// 固定版 ONNX を別の出力先へ変換する。通信せず、元ファイルは保持する。
pub fn convert_legacy_models_to(
    dir: &Path,
    output_dir: &Path,
    force: bool,
) -> Result<PathBuf, DownloadError> {
    fs::create_dir_all(output_dir)?;
    for file in MODEL_FILES.iter().filter(|f| f.file.ends_with(".knm")) {
        let native = output_dir.join(file.file);
        if native.exists() && !force {
            match super::knm::Weights::load(&native) {
                Ok(weights) if weights.source_sha256 == file.sha256 => continue,
                _ => return Err(DownloadError::Existing(native)),
            }
        }
        let source = dir.join(file.source);
        if !valid_file(&source, file.size, file.sha256)? {
            return Err(DownloadError::Integrity(file.file.into()));
        }
        let weights = super::legacy::convert(&fs::read(&source)?, file.sha256)?;
        let mut temporary = tempfile::NamedTempFile::new_in(output_dir)?;
        weights.write(&mut temporary, file.sha256)?;
        temporary.as_file().sync_all()?;
        super::knm::Weights::load(temporary.path())?;
        temporary.persist(native).map_err(|e| e.error)?;
    }
    let mut files = Vec::new();
    for file in MODEL_FILES.iter().filter(|f| f.file.ends_with(".knm")) {
        let path = output_dir.join(file.file);
        let bytes = fs::read(&path)?;
        let digest: String = Sha256::digest(&bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        files.push(serde_json::json!({"file":file.file,"source":file.source,"source_sha256":file.sha256,"size":bytes.len(),"sha256":digest}));
    }
    let mut manifest = tempfile::NamedTempFile::new_in(output_dir)?;
    serde_json::to_writer_pretty(
        &mut manifest,
        &serde_json::json!({"format_version":1,"source_revision":MODEL_REVISION,"files":files}),
    )
    .map_err(io::Error::other)?;
    manifest.write_all(b"\n")?;
    manifest.as_file().sync_all()?;
    manifest
        .persist(output_dir.join("knm-manifest.json"))
        .map_err(|e| e.error)?;
    Ok(output_dir.to_path_buf())
}

fn github_token() -> Option<String> {
    for key in ["GH_TOKEN", "GITHUB_TOKEN"] {
        if let Ok(token) = std::env::var(key)
            && !token.trim().is_empty()
        {
            return Some(token.trim().to_string());
        }
    }
    let output = Command::new("gh")
        .args(["auth", "token", "--hostname", "github.com"])
        .stdin(Stdio::null())
        .stderr(Stdio::null())
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8(output.stdout)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

fn file_digest(path: &Path) -> Result<String, io::Error> {
    let mut input = File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = [0; 64 * 1024];
    loop {
        let size = input.read(&mut buffer)?;
        if size == 0 {
            break;
        }
        hash.update(&buffer[..size]);
    }
    Ok(hash
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

fn valid_file(path: &Path, size: u64, sha256: &str) -> Result<bool, io::Error> {
    Ok(path.metadata()?.len() == size && file_digest(path)? == sha256)
}

#[cfg(test)]
mod tests;
