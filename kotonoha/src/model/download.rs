//! 固定したモデルを検証して共有ディレクトリに置く。

use super::{MODEL_FILES, MODEL_REVISION, ModelFile, default_model_dir};
use base64::Engine as _;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;
use ureq::tls::{RootCerts, TlsConfig};

/// 取得時のオプション。ミラーでもサイズと SHA-256 は固定値を使う。
#[derive(Debug, Default, Clone)]
pub struct DownloadOptions {
    /// 検証済みファイルも取り直し、中身の違うファイルも置き換える。
    pub force: bool,
    /// `<URL>/<配布元ファイル名>` を取得するミラー。GitHub の認証は送らない。
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
    #[error(
        "failed to download {0}; check network and repository access (GH_TOKEN, GITHUB_TOKEN, or gh auth login)"
    )]
    Network(String),
    /// サイズまたは SHA-256 が一致しなかった。
    #[error("model size or SHA-256 mismatch: {0}")]
    Integrity(String),
    /// 既存ファイルを保護した。
    #[error("existing model differs: {0}; use --force to replace it")]
    Existing(PathBuf),
}

/// 13 ONNX とアクセント辞書を取得する。`dir` を省くと [`default_model_dir`] を使う。
///
/// 正しい既存ファイルは通信せず再利用する。一時ファイルでサイズと SHA-256 を確かめてから
/// 配置するため、通信失敗で既存ファイルは壊れない。認証は `GH_TOKEN`、`GITHUB_TOKEN`、
/// `gh auth token --hostname github.com` の順に参照する。Git / Git LFS の導入は不要。
pub fn download(dir: Option<&Path>, options: DownloadOptions) -> Result<PathBuf, DownloadError> {
    let dir = dir
        .map(Path::to_path_buf)
        .or_else(default_model_dir)
        .ok_or(DownloadError::NoDirectory)?;
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .user_agent(concat!("kotonoha/", env!("CARGO_PKG_VERSION")))
        .timeout_connect(Some(Duration::from_secs(30)))
        .timeout_recv_response(Some(Duration::from_secs(60)))
        .timeout_global(Some(Duration::from_secs(3600)))
        .tls_config(
            TlsConfig::builder()
                .root_certs(RootCerts::PlatformVerifier)
                .build(),
        )
        .build()
        .into();
    // 全ファイルを再利用できる場合は認証ヘルパーも呼ばない。
    let mut token = None;
    let mut resolved_token = false;
    fs::create_dir_all(&dir)?;
    for file in &MODEL_FILES {
        if reuse_existing(&dir.join(file.file), file, options.force)? {
            continue;
        }
        if options.base_url.is_none() && !resolved_token {
            token = github_token();
            resolved_token = true;
        }
        let mut response = fetch(&agent, file, options.base_url.as_deref(), token.as_deref())?;
        install(
            &mut response.body_mut().as_reader(),
            &dir,
            file,
            options.force,
        )?;
    }
    Ok(dir)
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

#[derive(Deserialize)]
struct LfsBatch {
    objects: Vec<LfsObject>,
}
#[derive(Deserialize)]
struct LfsObject {
    oid: String,
    size: u64,
    #[serde(default)]
    actions: HashMap<String, LfsAction>,
}
#[derive(Deserialize)]
struct LfsAction {
    href: String,
    #[serde(default)]
    header: HashMap<String, String>,
}

fn fetch(
    agent: &ureq::Agent,
    file: &ModelFile,
    base_url: Option<&str>,
    token: Option<&str>,
) -> Result<ureq::http::Response<ureq::Body>, DownloadError> {
    let network = || DownloadError::Network(file.file.to_string());
    if let Some(base) = base_url {
        return agent
            .get(format!("{}/{}", base.trim_end_matches('/'), file.source))
            .call()
            .map_err(|_| network());
    }
    if file.source.ends_with(".onnx")
        && let Some(token) = token
    {
        // 非公開リポジトリの LFS 実体は Batch API が返す署名付き URL から取得する。
        let auth =
            base64::engine::general_purpose::STANDARD.encode(format!("x-access-token:{token}"));
        let mut batch = agent
            .post("https://github.com/owayo/kotonoha-models.git/info/lfs/objects/batch")
            .header("Authorization", format!("Basic {auth}"))
            .header("Accept", "application/vnd.git-lfs+json")
            .header("Content-Type", "application/vnd.git-lfs+json")
            .send_json(serde_json::json!({
                "operation": "download", "transfers": ["basic"],
                "objects": [{"oid": file.sha256, "size": file.size}]
            }))
            .map_err(|_| network())?;
        let batch: LfsBatch = batch
            .body_mut()
            .with_config()
            .limit(1024 * 1024)
            .read_json()
            .map_err(|_| network())?;
        let action = batch
            .objects
            .into_iter()
            .find(|o| o.oid == file.sha256 && o.size == file.size)
            .and_then(|mut o| o.actions.remove("download"))
            .ok_or_else(network)?;
        // 取得 URL に GitHub トークンを転送しない。LFS が指定したヘッダーだけを使う。
        let mut request = agent.get(&action.href);
        for (key, value) in action.header {
            request = request.header(key, value);
        }
        return request.call().map_err(|_| network());
    }
    let url = if file.source.ends_with(".onnx") {
        format!(
            "https://media.githubusercontent.com/media/owayo/kotonoha-models/{MODEL_REVISION}/{}",
            file.source
        )
    } else {
        format!(
            "https://api.github.com/repos/owayo/kotonoha-models/contents/{}?ref={MODEL_REVISION}",
            file.source
        )
    };
    let mut request = agent
        .get(url)
        .header("Accept", "application/vnd.github.raw+json");
    if let Some(token) = token {
        request = request.header("Authorization", format!("Bearer {token}"));
    }
    request.call().map_err(|_| network())
}

fn valid_file(path: &Path, file: &ModelFile) -> Result<bool, io::Error> {
    let mut input = File::open(path)?;
    if input.metadata()?.len() != file.size {
        return Ok(false);
    }
    let mut hash = Sha256::new();
    let mut buffer = [0; 64 * 1024];
    loop {
        let size = input.read(&mut buffer)?;
        if size == 0 {
            break;
        }
        hash.update(&buffer[..size]);
    }
    let hex: String = hash
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    Ok(hex == file.sha256)
}

fn reuse_existing(path: &Path, file: &ModelFile, force: bool) -> Result<bool, DownloadError> {
    if force {
        return Ok(false);
    }
    match valid_file(path, file) {
        Ok(true) => Ok(true),
        Ok(false) => Err(DownloadError::Existing(path.to_path_buf())),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(e) => Err(e.into()),
    }
}

fn install(
    input: &mut impl Read,
    dir: &Path,
    file: &ModelFile,
    force: bool,
) -> Result<(), DownloadError> {
    let mut temp = tempfile::NamedTempFile::new_in(dir)?;
    // 宣言されたサイズを超えてディスクを消費しない。
    let size = io::copy(&mut input.take(file.size + 1), &mut temp)?;
    temp.flush()?;
    if size != file.size || !valid_file(temp.path(), file)? {
        return Err(DownloadError::Integrity(file.file.to_string()));
    }
    temp.as_file().sync_all()?;
    let target = dir.join(file.file);
    if force {
        temp.persist(&target).map_err(|e| e.error)?;
    } else if let Err(e) = temp.persist_noclobber(&target) {
        if e.error.kind() != io::ErrorKind::AlreadyExists {
            return Err(e.error.into());
        }
        // 同時取得した別プロセスが先に同じファイルを置いていても成功とする。
        if !valid_file(&target, file)? {
            return Err(DownloadError::Existing(target));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
