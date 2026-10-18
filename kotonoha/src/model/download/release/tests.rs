//! 最新版の確認・キャッシュ・更新・破損時の保護をローカル HTTP と実形式で検査する。

use super::*;
use flate2::Compression;
use flate2::write::GzEncoder;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::io::{Cursor, Write};
use std::net::TcpListener;
use std::thread;

fn hash(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn archive(files: &[(String, Vec<u8>)]) -> Vec<u8> {
    let mut builder = tar::Builder::new(GzEncoder::new(Vec::new(), Compression::fast()));
    for (name, bytes) in files {
        let mut header = tar::Header::new_gnu();
        header.set_size(bytes.len() as u64);
        header.set_mode(0o644);
        header.set_cksum();
        builder
            .append_data(&mut header, format!("kotonoha-models/{name}"), &bytes[..])
            .unwrap();
    }
    builder.into_inner().unwrap().finish().unwrap()
}

fn model_files(accent: u8) -> Vec<(String, Vec<u8>)> {
    let mut templates = HashMap::new();
    MODEL_FILES
        .iter()
        .map(|file| {
            let bytes = if file.file.ends_with(".knm") {
                let dim = match file.file {
                    "teacher.knm" => 11,
                    "accent.knm" => 103,
                    "meta-01.knm" | "meta-02.knm" => 24,
                    _ => 14,
                };
                templates
                    .entry(dim)
                    .or_insert_with(|| {
                        let tensors = crate::model::knm::tensor_shapes(dim)
                            .unwrap()
                            .into_iter()
                            .map(|(name, shape)| (name, vec![0.0; shape.iter().product()]))
                            .collect();
                        let weights = Weights {
                            input_dim: dim,
                            source_sha256: "9".repeat(64),
                            tensors,
                        };
                        let mut data = Vec::new();
                        weights.write(&mut data, &weights.source_sha256).unwrap();
                        data
                    })
                    .clone()
            } else {
                format!("lemma,reading,accent_type\n猫,ネコ,{accent}\n").into_bytes()
            };
            (file.file.to_string(), bytes)
        })
        .collect()
}

fn with_manifest(mut files: Vec<(String, Vec<u8>)>) -> Vec<(String, Vec<u8>)> {
    let entries: Vec<_> = files.iter().map(|(name, bytes)| serde_json::json!({"file":name,"size":bytes.len(),"sha256":hash(bytes)})).collect();
    files.push((
        "models.json".into(),
        serde_json::to_vec(
            &serde_json::json!({"schema_version":1,"model_format":"knm-v1","files":entries}),
        )
        .unwrap(),
    ));
    files
}

fn serve(responses: Vec<(&str, Vec<u8>)>) -> (String, thread::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let responses: Vec<_> = responses
        .into_iter()
        .map(|(name, bytes)| (name.to_string(), bytes))
        .collect();
    let server = thread::spawn(move || {
        for (name, bytes) in responses {
            let (mut stream, _) = listener.accept().unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(10)))
                .unwrap();
            let mut request = Vec::new();
            let mut byte = [0];
            while !request.ends_with(b"\r\n\r\n") {
                stream.read_exact(&mut byte).unwrap();
                request.push(byte[0]);
            }
            let text = String::from_utf8(request).unwrap();
            assert!(text.starts_with(&format!("GET /{name} HTTP/1.1")), "{text}");
            assert!(!text.to_lowercase().contains("authorization"));
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                bytes.len()
            )
            .unwrap();
            stream.write_all(&bytes).unwrap();
        }
    });
    (url, server)
}

fn from_mirror(
    dir: &Path,
    archive: Vec<u8>,
    force: bool,
    cached: bool,
) -> Result<PathBuf, DownloadError> {
    let checksum = format!("{}  {ARCHIVE}\n", hash(&archive)).into_bytes();
    let mut responses = vec![(CHECKSUM, checksum)];
    if !cached {
        responses.push((ARCHIVE, archive));
    }
    let (base, server) = serve(responses);
    let result = download(
        Some(dir),
        DownloadOptions {
            force,
            base_url: Some(base),
        },
    );
    server.join().unwrap();
    result
}

#[test]
fn latest_native_bundle_download_cache_updates_and_manual_change_protection() {
    let dir = tempfile::tempdir().unwrap();
    let mut files = model_files(0);
    let first = archive(&with_manifest(files.clone()));
    from_mirror(dir.path(), first.clone(), false, false).unwrap();
    assert!(crate::model::is_model_bundle(dir.path()));
    assert!(!dir.path().join("teacher.onnx").exists());
    from_mirror(dir.path(), first, false, true).unwrap();
    files
        .iter_mut()
        .find(|(n, _)| n == "accent_dict.csv")
        .unwrap()
        .1 = b"lemma,reading,accent_type\ncat,cat,1\n".to_vec();
    let latest = archive(&with_manifest(files));
    from_mirror(dir.path(), latest.clone(), false, false).unwrap();
    assert!(
        fs::read_to_string(dir.path().join("accent_dict.csv"))
            .unwrap()
            .contains("cat,cat,1")
    );
    fs::write(dir.path().join("teacher.knm"), b"manual changes").unwrap();
    assert!(matches!(
        from_mirror(dir.path(), latest.clone(), false, false),
        Err(DownloadError::Existing(_))
    ));
    assert_eq!(
        fs::read(dir.path().join("teacher.knm")).unwrap(),
        b"manual changes"
    );
    from_mirror(dir.path(), latest, true, false).unwrap();
    assert!(Weights::load(&dir.path().join("teacher.knm")).is_ok());
}

#[test]
fn checksum_parser_and_release_selection_reject_legacy_or_unpublished_assets() {
    let sha = "a".repeat(64);
    assert_eq!(
        parse_checksum(format!("{sha}  {ARCHIVE}\n").as_bytes()).unwrap(),
        sha
    );
    for input in [
        "broken",
        "a other.tar.gz",
        &format!("{} ../../file", "a".repeat(64)),
    ] {
        assert!(parse_checksum(input.as_bytes()).is_err());
    }
    let release = Release {
        tag_name: "v26.10.100".into(),
        draft: false,
        prerelease: false,
        assets: vec![],
    };
    assert!(
        source_from_release(release)
            .err()
            .unwrap()
            .to_string()
            .contains("publish a KNM release")
    );
}

#[test]
fn archive_links_unknown_paths_and_duplicate_names_are_rejected() {
    let dir = tempfile::tempdir().unwrap();
    assert!(
        extract_archive(
            Cursor::new(archive(&[("unlisted.knm".into(), vec![])])),
            dir.path()
        )
        .is_err()
    );
    let duplicate = archive(&[
        ("teacher.knm".into(), b"one".to_vec()),
        ("teacher.knm".into(), b"two".to_vec()),
    ]);
    assert!(extract_archive(Cursor::new(duplicate), dir.path()).is_err());
    let mut builder = tar::Builder::new(GzEncoder::new(Vec::new(), Compression::fast()));
    let mut header = tar::Header::new_gnu();
    header.set_size(0);
    header.set_mode(0o644);
    header.set_entry_type(tar::EntryType::Symlink);
    header.set_link_name("../../outside").unwrap();
    header.set_cksum();
    builder
        .append_data(&mut header, "kotonoha-models/teacher.knm", io::empty())
        .unwrap();
    let data = builder.into_inner().unwrap().finish().unwrap();
    assert!(extract_archive(Cursor::new(data), dir.path()).is_err());
}

#[test]
fn archive_corruption_preserves_existing_models() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("teacher.knm"), b"original").unwrap();
    assert!(from_mirror(dir.path(), b"corrupt gzip".to_vec(), true, false).is_err());
    assert_eq!(
        fs::read(dir.path().join("teacher.knm")).unwrap(),
        b"original"
    );
}

#[test]
fn wrong_archive_checksum_and_truncated_gzip_are_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let checksum = format!("{}  {ARCHIVE}\n", "a".repeat(64)).into_bytes();
    let (base, server) = serve(vec![
        (CHECKSUM, checksum),
        (ARCHIVE, b"different content".to_vec()),
    ]);
    assert!(matches!(
        download(
            Some(dir.path()),
            DownloadOptions {
                force: false,
                base_url: Some(base)
            }
        ),
        Err(DownloadError::Integrity(_))
    ));
    server.join().unwrap();
    assert!(!dir.path().join("models.json").exists());
    let mut data = archive(&[("teacher.knm".into(), b"content".to_vec())]);
    data.truncate(data.len() - 5);
    assert!(extract_archive(Cursor::new(data), dir.path()).is_err());
}

#[test]
fn tar_path_traversal_is_rejected_without_writing_outside_stage() {
    let dir = tempfile::tempdir().unwrap();
    let mut builder = tar::Builder::new(GzEncoder::new(Vec::new(), Compression::fast()));
    let mut header = tar::Header::new_gnu();
    header.set_size(3);
    header.set_mode(0o644);
    let path = b"kotonoha-models/../../outside";
    header.as_mut_bytes()[..path.len()].copy_from_slice(path);
    header.set_cksum();
    builder.append(&header, &b"bad"[..]).unwrap();
    let bytes = builder.into_inner().unwrap().finish().unwrap();
    assert!(extract_archive(Cursor::new(bytes), dir.path()).is_err());
    assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 0);
}

#[test]
fn interrupted_install_rolls_back_every_replaced_file() {
    let dir = tempfile::tempdir().unwrap();
    let stage = tempfile::tempdir().unwrap();
    let files: Vec<_> = MODEL_FILES
        .iter()
        .map(|f| (f.file.to_string(), b"old".to_vec()))
        .collect();
    for (name, bytes) in with_manifest(files.clone()) {
        fs::write(dir.path().join(name), bytes).unwrap();
    }
    let mut latest: Vec<_> = files
        .into_iter()
        .map(|(n, _)| (n, b"new".to_vec()))
        .collect();
    latest.push(("hasami-dictionaries.json".into(), b"{}".to_vec()));
    for (name, bytes) in with_manifest(latest) {
        fs::write(stage.path().join(name), bytes).unwrap();
    }
    fs::write(stage.path().join(RECEIPT), b"{}").unwrap();
    let manifest = read_manifest(stage.path()).unwrap();
    fs::remove_file(stage.path().join("hasami-dictionaries.json")).unwrap();
    assert!(install_bundle(stage.path(), dir.path(), &manifest, false).is_err());
    for file in MODEL_FILES {
        assert_eq!(fs::read(dir.path().join(file.file)).unwrap(), b"old");
    }
    assert!(!dir.path().join("hasami-dictionaries.json").exists());
}
