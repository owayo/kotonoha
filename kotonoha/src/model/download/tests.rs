//! 固定した変換元と本番ファイル名・Release 入力の対応を検査する。

use super::*;

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
