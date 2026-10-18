//! 通信・整合性検証・既存ファイル保護を小さい実体で検証する。

use super::*;
use std::net::TcpListener;
use std::thread;

const SAMPLE: ModelFile = ModelFile {
    file: "model.onnx",
    source: "source.onnx",
    size: 3,
    sha256: "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
};

#[test]
fn verified_install_reuse_and_corrupt_file_protection() {
    let dir = tempfile::tempdir().unwrap();
    let target = dir.path().join(SAMPLE.file);
    install(&mut &b"abc"[..], dir.path(), &SAMPLE, false).unwrap();
    assert!(reuse_existing(&target, &SAMPLE, false).unwrap());
    assert!(!reuse_existing(&target, &SAMPLE, true).unwrap());
    // 同時取得は同じ中身なら成功する。
    install(&mut &b"abc"[..], dir.path(), &SAMPLE, false).unwrap();
    fs::write(&target, b"old").unwrap();
    assert!(matches!(
        reuse_existing(&target, &SAMPLE, false),
        Err(DownloadError::Existing(_))
    ));
    install(&mut &b"abc"[..], dir.path(), &SAMPLE, true).unwrap();
    assert_eq!(fs::read(&target).unwrap(), b"abc");
}

#[test]
fn failed_or_oversized_download_preserves_existing_file_and_cleans_temp() {
    let dir = tempfile::tempdir().unwrap();
    let target = dir.path().join(SAMPLE.file);
    fs::write(&target, b"old").unwrap();
    for data in [
        &b"ab"[..],
        &b"abd"[..],
        &b"abcdef"[..],
        &b"version https://git-lfs.github.com/spec/v1"[..],
    ] {
        assert!(matches!(
            install(&mut &data[..], dir.path(), &SAMPLE, true),
            Err(DownloadError::Integrity(_))
        ));
        assert_eq!(fs::read(&target).unwrap(), b"old");
        assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 1);
    }
    struct Broken;
    impl Read for Broken {
        fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
            Err(io::Error::other("connection lost"))
        }
    }
    assert!(install(&mut Broken, dir.path(), &SAMPLE, true).is_err());
    assert_eq!(fs::read(&target).unwrap(), b"old");
    assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 1);
}

#[test]
fn mirror_fetch_uses_source_name_without_credentials() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let server = thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        let mut request = Vec::new();
        let mut byte = [0];
        while !request.ends_with(b"\r\n\r\n") {
            stream.read_exact(&mut byte).unwrap();
            request.push(byte[0]);
        }
        let request = String::from_utf8(request).unwrap();
        assert!(request.starts_with("GET /source.onnx HTTP/1.1"));
        assert!(!request.to_lowercase().contains("authorization"));
        stream
            .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 3\r\nConnection: close\r\n\r\nabc")
            .unwrap();
    });
    let mut response = fetch(
        &ureq::Agent::new_with_defaults(),
        &SAMPLE,
        Some(&base),
        Some("unused-secret"),
    )
    .unwrap();
    let dir = tempfile::tempdir().unwrap();
    install(
        &mut response.body_mut().as_reader(),
        dir.path(),
        &SAMPLE,
        false,
    )
    .unwrap();
    assert_eq!(fs::read(dir.path().join(SAMPLE.file)).unwrap(), b"abc");
    server.join().unwrap();
}

#[test]
fn manifest_covers_runtime_bundle() {
    assert_eq!(
        MODEL_FILES
            .iter()
            .filter(|f| f.file.ends_with(".knm"))
            .count(),
        13
    );
    assert!(MODEL_FILES.iter().any(|f| f.file == "accent_dict.csv"));
    // CLI の取得と Release の梱包が同じ役割名・変換元を使うことを検査する。
    let lock: serde_json::Value =
        serde_json::from_str(include_str!("../../../../.github/model-bundle.json")).unwrap();
    for file in MODEL_FILES.iter().filter(|f| f.file.ends_with(".knm")) {
        let entry = lock["files"]
            .as_array()
            .unwrap()
            .iter()
            .find(|entry| entry["file"] == file.source)
            .unwrap();
        assert_eq!(entry["native_file"], file.file);
        assert_eq!(entry["sha256"], file.sha256);
        assert_eq!(entry["size"], file.size);
    }
    #[cfg(feature = "neural")]
    for student in crate::nn::v66::bundle::STUDENT_FILENAMES {
        assert!(MODEL_FILES.iter().any(|f| f.file == student));
    }
    let dir = tempfile::tempdir().unwrap();
    assert!(!super::super::is_model_bundle(dir.path()));
    for file in MODEL_FILES.iter().filter(|f| f.file.ends_with(".knm")) {
        File::create(dir.path().join(file.file)).unwrap();
    }
    assert!(super::super::is_model_bundle(dir.path()));
}
