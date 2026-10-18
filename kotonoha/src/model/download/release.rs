//! 最新の公開 KNM アーカイブを検証し、既存の管理済みモデルを更新する。

use super::{DownloadError, DownloadOptions, file_digest, github_token, valid_file};
use crate::model::{MODEL_FILES, default_model_dir, knm::Weights};
use flate2::read::GzDecoder;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::time::Duration;
use ureq::tls::{RootCerts, TlsConfig};

const ARCHIVE: &str = "kotonoha-models.tar.gz";
const CHECKSUM: &str = "kotonoha-models.tar.gz.sha256";
const RECEIPT: &str = ".kotonoha-release.json";
const MAX_ARCHIVE: u64 = 512 * 1024 * 1024;
const MAX_FILE: u64 = 64 * 1024 * 1024;
const MAX_METADATA: u64 = 1024 * 1024;

fn invalid(message: impl Into<String>) -> DownloadError {
    DownloadError::Release(message.into())
}

#[derive(Deserialize)]
struct Release {
    tag_name: String,
    draft: bool,
    prerelease: bool,
    assets: Vec<Asset>,
}

#[derive(Deserialize)]
struct Asset {
    name: String,
    state: String,
    size: u64,
    browser_download_url: String,
    digest: Option<String>,
}

#[derive(Deserialize, Serialize)]
struct BundleFile {
    file: String,
    size: u64,
    sha256: String,
}

#[derive(Deserialize)]
struct Manifest {
    schema_version: u32,
    model_format: String,
    files: Vec<BundleFile>,
}

#[derive(Deserialize, Serialize)]
struct Receipt {
    archive_sha256: String,
    manifest_sha256: String,
}

struct Source {
    archive_url: String,
    checksum_url: String,
    size: Option<u64>,
    digest: Option<String>,
}

/// 最新の安定版 Release から KNM を直接取得する。呼び出すたびに最新版を確認する。
///
/// 正常な管理済み旧版は更新する。手動配置・変更済みファイルの置き換えには `force` が必要。
/// 全ファイルの検証が完了してから配置する。推論時には通信しない。
pub fn download(dir: Option<&Path>, options: DownloadOptions) -> Result<PathBuf, DownloadError> {
    let dir = dir
        .map(Path::to_path_buf)
        .or_else(default_model_dir)
        .ok_or(DownloadError::NoDirectory)?;
    if !options.force && !dir.join("models.json").exists() {
        for file in MODEL_FILES.iter().filter(|f| f.file.ends_with(".knm")) {
            let path = dir.join(file.file);
            if path.exists() && Weights::load(&path).is_err() {
                return Err(DownloadError::Existing(path));
            }
        }
    }
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
    let source = resolve_source(&agent, options.base_url.as_deref())?;
    let bytes = fetch_small(&agent, &source.checksum_url, 1024)?;
    let checksum = parse_checksum(&bytes)?;
    if let Some(digest) = &source.digest
        && digest != &format!("sha256:{checksum}")
    {
        return Err(DownloadError::Integrity(ARCHIVE.into()));
    }
    fs::create_dir_all(&dir)?;
    let lock = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(dir.join(".kotonoha-download.lock"))?;
    lock.try_lock().map_err(io::Error::other)?;
    if !options.force && cached_release(&dir, &checksum)? {
        return Ok(dir);
    }
    let temporary = tempfile::tempdir_in(&dir)?;
    let archive_path = temporary.path().join("bundle.tar.gz");
    let mut response = agent
        .get(&source.archive_url)
        .call()
        .map_err(|_| DownloadError::Network(ARCHIVE.into()))?;
    let mut archive_file = File::create(&archive_path)?;
    let size = io::copy(
        &mut response.body_mut().as_reader().take(MAX_ARCHIVE + 1),
        &mut archive_file,
    )?;
    archive_file.sync_all()?;
    if size == 0
        || size > MAX_ARCHIVE
        || source.size.is_some_and(|expected| expected != size)
        || file_digest(&archive_path)? != checksum
    {
        return Err(DownloadError::Integrity(ARCHIVE.into()));
    }
    drop(archive_file);
    let staged = temporary.path().join("files");
    fs::create_dir(&staged)?;
    extract_archive(File::open(&archive_path)?, &staged)?;
    let manifest = validate_bundle(&staged)?;
    let receipt = Receipt {
        archive_sha256: checksum,
        manifest_sha256: file_digest(&staged.join("models.json"))?,
    };
    fs::write(
        staged.join(RECEIPT),
        serde_json::to_vec(&receipt).map_err(io::Error::other)?,
    )?;
    if let Err(error) = install_bundle(&staged, &dir, &manifest, options.force) {
        if matches!(error, DownloadError::Recovery(_)) {
            // 復元できなかった旧モデルを一時領域の自動削除に巻き込まない。
            let _ = temporary.keep();
        }
        return Err(error);
    }
    Ok(dir)
}

fn resolve_source(agent: &ureq::Agent, mirror: Option<&str>) -> Result<Source, DownloadError> {
    if let Some(base) = mirror {
        let base = base.trim_end_matches('/');
        return Ok(Source {
            archive_url: format!("{base}/{ARCHIVE}"),
            checksum_url: format!("{base}/{CHECKSUM}"),
            size: None,
            digest: None,
        });
    }
    let mut request = agent
        .get("https://api.github.com/repos/owayo/kotonoha/releases/latest")
        .header("Accept", "application/vnd.github+json");
    if let Some(token) = github_token() {
        request = request.header("Authorization", format!("Bearer {token}"));
    }
    let mut response = request
        .call()
        .map_err(|_| DownloadError::Network("latest KNM release".into()))?;
    let release: Release = response
        .body_mut()
        .with_config()
        .limit(MAX_METADATA)
        .read_json()
        .map_err(|_| invalid("invalid latest release metadata"))?;
    source_from_release(release)
}

fn source_from_release(release: Release) -> Result<Source, DownloadError> {
    if release.draft
        || release.prerelease
        || release.tag_name.is_empty()
        || !release
            .tag_name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
    {
        return Err(invalid("expected a published stable release"));
    }
    let asset = |name| {
        release
            .assets
            .iter()
            .find(|a| a.name == name && a.state == "uploaded")
            .ok_or_else(|| {
                invalid(format!(
                    "latest release {} has no {name}; publish a KNM release first",
                    release.tag_name
                ))
            })
    };
    let archive = asset(ARCHIVE)?;
    let checksum = asset(CHECKSUM)?;
    let prefix = format!(
        "https://github.com/owayo/kotonoha/releases/download/{}/",
        release.tag_name
    );
    if !(1..=MAX_ARCHIVE).contains(&archive.size)
        || !(1..=1024).contains(&checksum.size)
        || archive.browser_download_url != format!("{prefix}{ARCHIVE}")
        || checksum.browser_download_url != format!("{prefix}{CHECKSUM}")
    {
        return Err(invalid("invalid release asset location or size"));
    }
    // 添付の取得へ GitHub トークンを渡さない。公開済みの固定タグの URL を使う。
    Ok(Source {
        archive_url: archive.browser_download_url.clone(),
        checksum_url: checksum.browser_download_url.clone(),
        size: Some(archive.size),
        digest: archive.digest.clone(),
    })
}

fn fetch_small(agent: &ureq::Agent, url: &str, limit: u64) -> Result<Vec<u8>, DownloadError> {
    let mut response = agent
        .get(url)
        .call()
        .map_err(|_| DownloadError::Network(CHECKSUM.into()))?;
    let mut bytes = Vec::new();
    response
        .body_mut()
        .as_reader()
        .take(limit + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() as u64 > limit {
        return Err(invalid("metadata exceeds size limit"));
    }
    Ok(bytes)
}

fn valid_sha(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn parse_checksum(bytes: &[u8]) -> Result<String, DownloadError> {
    let text = std::str::from_utf8(bytes).map_err(|_| invalid("invalid archive checksum"))?;
    let parts: Vec<_> = text.split_whitespace().collect();
    if parts.len() != 2 || !valid_sha(parts[0]) || parts[1] != ARCHIVE {
        return Err(invalid("invalid archive checksum"));
    }
    Ok(parts[0].into())
}

fn allowed_name(name: &str) -> bool {
    MODEL_FILES.iter().any(|f| f.file == name)
        || matches!(
            name,
            "accent_dict_jsut.csv" | "models.json" | "hasami-dictionaries.json"
        )
}

fn extract_archive(input: impl Read, output: &Path) -> Result<(), DownloadError> {
    let mut archive = tar::Archive::new(GzDecoder::new(input).take(1024 * 1024 * 1024));
    let mut seen = HashSet::new();
    for entry in archive.entries()? {
        let mut entry = entry?;
        let path = entry.path()?;
        let path = path
            .to_str()
            .ok_or_else(|| invalid("non-UTF-8 archive filename"))?;
        if entry.header().entry_type().is_dir() && path.trim_end_matches('/') == "kotonoha-models" {
            continue;
        }
        let name = path
            .strip_prefix("kotonoha-models/")
            .filter(|n| allowed_name(n))
            .ok_or_else(|| invalid("unexpected archive path"))?
            .to_string();
        let size = entry.size();
        let limit = if name.ends_with(".json") {
            MAX_METADATA
        } else {
            MAX_FILE
        };
        if !entry.header().entry_type().is_file() || size > limit || !seen.insert(name.clone()) {
            return Err(invalid("unsupported, oversized or duplicate archive entry"));
        }
        let mut file = File::create(output.join(&name))?;
        if io::copy(&mut entry, &mut file)? != size {
            return Err(invalid("truncated archive entry"));
        }
    }
    // TAR の終端以降も読み、gzip の終端・CRC と展開量の上限を検証する。
    let mut remaining = archive.into_inner();
    io::copy(&mut remaining, &mut io::sink())?;
    if remaining.limit() == 0 {
        return Err(invalid("archive exceeds decompression limit"));
    }
    Ok(())
}

fn read_manifest(dir: &Path) -> Result<Manifest, DownloadError> {
    let path = dir.join("models.json");
    if path.metadata()?.len() > MAX_METADATA {
        return Err(invalid("oversized model manifest"));
    }
    let manifest: Manifest = serde_json::from_reader(File::open(path)?)
        .map_err(|_| invalid("invalid model manifest"))?;
    let mut names = HashSet::new();
    if manifest.schema_version != 1
        || manifest.model_format != "knm-v1"
        || manifest.files.len() > 16
    {
        return Err(invalid("unsupported model manifest"));
    }
    for file in &manifest.files {
        if file.file == "models.json"
            || !allowed_name(&file.file)
            || file.size > MAX_FILE
            || !valid_sha(&file.sha256)
            || !names.insert(file.file.as_str())
        {
            return Err(invalid("invalid or duplicate manifest entry"));
        }
    }
    if !MODEL_FILES.iter().all(|f| names.contains(f.file)) {
        return Err(invalid("model manifest is incomplete"));
    }
    Ok(manifest)
}

fn validate_bundle(dir: &Path) -> Result<Manifest, DownloadError> {
    let manifest = read_manifest(dir)?;
    if fs::read_dir(dir)?.count() != manifest.files.len() + 1 {
        return Err(invalid("archive and manifest files differ"));
    }
    for file in &manifest.files {
        let path = dir.join(&file.file);
        if !valid_file(&path, file.size, &file.sha256)? {
            return Err(DownloadError::Integrity(file.file.clone()));
        }
        if file.file.ends_with(".knm") {
            let expected = match file.file.as_str() {
                "teacher.knm" => 11,
                "accent.knm" => 103,
                "meta-01.knm" | "meta-02.knm" => 24,
                _ => 14,
            };
            if Weights::load(&path)?.input_dim != expected {
                return Err(invalid("unexpected model input dimension"));
            }
        } else if file.file.ends_with(".csv") {
            crate::accent_dict::AccentDict::from_csv(&path)
                .map_err(|_| invalid("invalid accent dictionary"))?;
        }
    }
    Ok(manifest)
}

fn cached_release(dir: &Path, checksum: &str) -> Result<bool, DownloadError> {
    let path = dir.join(RECEIPT);
    if !path.is_file() || path.metadata()?.len() > MAX_METADATA {
        return Ok(false);
    }
    let Ok(data) = fs::read(path) else {
        return Ok(false);
    };
    let Ok(receipt) = serde_json::from_slice::<Receipt>(&data) else {
        return Ok(false);
    };
    if receipt.archive_sha256 != checksum {
        return Ok(false);
    }
    let manifest_path = dir.join("models.json");
    if !manifest_path.is_file() || file_digest(&manifest_path)? != receipt.manifest_sha256 {
        return Ok(false);
    }
    let manifest = read_manifest(dir)?;
    for file in &manifest.files {
        if !dir.join(&file.file).is_file()
            || !valid_file(&dir.join(&file.file), file.size, &file.sha256)?
        {
            return Ok(false);
        }
    }
    Ok(true)
}

fn install_bundle(
    staged: &Path,
    dir: &Path,
    manifest: &Manifest,
    force: bool,
) -> Result<(), DownloadError> {
    let old = if dir.join("models.json").exists() {
        match read_manifest(dir) {
            Ok(manifest) => Some(manifest),
            Err(_) if force => None,
            Err(_) => return Err(DownloadError::Existing(dir.join("models.json"))),
        }
    } else {
        None
    };
    let mut updates = Vec::new();
    for name in manifest
        .files
        .iter()
        .map(|f| f.file.as_str())
        .chain(["models.json", RECEIPT])
    {
        let target = dir.join(name);
        if target.exists() {
            if !target.is_file() {
                return Err(DownloadError::Existing(target));
            }
            if !force && file_digest(&target)? == file_digest(&staged.join(name))? {
                continue;
            }
            let managed = if matches!(name, "models.json" | RECEIPT) {
                old.is_some()
            } else {
                old.as_ref()
                    .and_then(|m| m.files.iter().find(|f| f.file == name))
                    .is_some_and(|f| valid_file(&target, f.size, &f.sha256).unwrap_or(false))
            };
            if !force && !managed {
                return Err(DownloadError::Existing(target));
            }
        }
        updates.push(name);
    }
    let backup = staged.join("backup");
    fs::create_dir(&backup)?;
    let mut changed = Vec::new();
    let result: Result<(), io::Error> = (|| {
        for name in updates {
            let target = dir.join(name);
            let existed = target.exists();
            if existed {
                fs::rename(&target, backup.join(name))?;
            }
            changed.push((name, existed));
            fs::rename(staged.join(name), target)?;
        }
        Ok(())
    })();
    if let Err(error) = result {
        let mut restored = true;
        for (name, existed) in changed.into_iter().rev() {
            let target = dir.join(name);
            if target.exists() && fs::remove_file(&target).is_err() {
                restored = false;
                continue;
            }
            if existed && fs::rename(backup.join(name), target).is_err() {
                restored = false;
            }
        }
        if !restored {
            return Err(DownloadError::Recovery(backup));
        }
        return Err(error.into());
    }
    Ok(())
}

#[cfg(test)]
mod tests;
