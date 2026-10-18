//! 共有モデルの CLI。通信が不要なパス解決と既存ファイル保護を検証する。

use std::process::Command;

fn cli() -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_kotonoha"));
    cmd.env_remove("KOTONOHA_MODEL_BUNDLE");
    cmd
}

#[test]
fn path_uses_shared_directory_and_explicit_override_without_creating_it() {
    let dir = tempfile::tempdir().unwrap();
    let shared = dir.path().join("shared");
    let explicit = dir.path().join("explicit");
    let output = cli()
        .args(["model", "path"])
        .env("KOTONOHA_MODEL_BUNDLE", &shared)
        .output()
        .unwrap();
    assert!(output.status.success());
    assert_eq!(
        String::from_utf8(output.stdout).unwrap().trim(),
        shared.to_str().unwrap()
    );
    let output = cli()
        .args(["model", "path", "--dir"])
        .arg(&explicit)
        .env("KOTONOHA_MODEL_BUNDLE", &shared)
        .output()
        .unwrap();
    assert!(output.status.success());
    assert_eq!(
        String::from_utf8(output.stdout).unwrap().trim(),
        explicit.to_str().unwrap()
    );
    assert!(!shared.exists());
    assert!(!explicit.exists());
}

#[test]
fn download_refuses_to_overwrite_existing_different_model_without_network() {
    let dir = tempfile::tempdir().unwrap();
    let model = dir.path().join("accent_model_v24.onnx");
    std::fs::write(&model, b"existing model").unwrap();
    let output = cli()
        .args(["model", "download", "--dir"])
        .arg(dir.path())
        .args(["--base-url", "http://127.0.0.1:1"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("--force"));
    assert_eq!(std::fs::read(model).unwrap(), b"existing model");
    assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
}

#[test]
fn label_reports_explicit_model_loading_errors() {
    let dir = tempfile::tempdir().unwrap();
    let missing = dir.path().join("missing");
    let output = cli()
        .args(["label", "[]", "--model-bundle"])
        .arg(&missing)
        .env("KOTONOHA_MODEL_BUNDLE", dir.path().join("other"))
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    #[cfg(feature = "neural")]
    assert!(stderr.contains(missing.to_str().unwrap()), "{stderr}");
    #[cfg(not(feature = "neural"))]
    assert!(stderr.contains("--features neural"), "{stderr}");
}
