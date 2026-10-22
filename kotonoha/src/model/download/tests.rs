//! 固定した変換元と本番ファイル名・Release 入力の対応を検査する。

use super::*;

#[test]
fn separate_conversion_checks_source_and_protects_output() {
    let root = tempfile::tempdir().unwrap();
    let source = root.path().join("onnx");
    let output = root.path().join("bundles/base");
    fs::create_dir(&source).unwrap();
    let model = &MODEL_FILES[0];
    let source_path = source.join(model.source);
    fs::write(&source_path, b"invalid ONNX").unwrap();
    assert!(matches!(
        convert_legacy_models_to(&source, &output, false),
        Err(DownloadError::Integrity(_))
    ));
    assert!(output.is_dir());
    assert_eq!(fs::read_dir(&output).unwrap().count(), 0);

    let native = output.join(model.file);
    fs::write(&native, b"user model").unwrap();
    assert!(matches!(
        convert_legacy_models_to(&source, &output, false),
        Err(DownloadError::Existing(path)) if path == native
    ));
    assert_eq!(fs::read(&native).unwrap(), b"user model");
    assert_eq!(fs::read(&source_path).unwrap(), b"invalid ONNX");
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
