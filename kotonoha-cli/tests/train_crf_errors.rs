//! CRF 学習 CLI の不正入力を検査する。

use std::process::Command;

#[test]
fn invalid_training_rows_do_not_create_a_model() {
    let dir = std::env::temp_dir().join(format!("kotonoha-crf-cli-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let data = dir.join("training.csv");
    let model = dir.join("model.bin");

    for (row, expected_error) in [
        ("猫,名詞,ネコ,不正\n", "不正なアクセント型"),
        ("猫,名詞,ネコ\n", "4列または9列"),
        ("猫,名詞,ネコ,200\n", "範囲外"),
    ] {
        std::fs::write(&data, row).unwrap();
        let output = Command::new(env!("CARGO_BIN_EXE_kotonoha"))
            .args(["train-crf", "--data"])
            .arg(&data)
            .arg("--output")
            .arg(&model)
            .output()
            .unwrap();
        assert!(!output.status.success());
        assert!(
            String::from_utf8_lossy(&output.stderr).contains(expected_error),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(!model.exists());
    }

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn invalid_training_parameters_do_not_create_a_model() {
    let dir = std::env::temp_dir().join(format!("kotonoha-crf-params-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let data = dir.join("training.csv");
    let model = dir.join("model.bin");
    std::fs::write(&data, "猫,名詞,ネコ,1\n").unwrap();

    for (args, expected_error) in [
        (["--epochs", "0"], "エポック数"),
        (["--lr", "NaN"], "学習率"),
        (["--lr", "inf"], "学習率"),
        (["--lr", "0"], "学習率"),
        (["--l2-reg", "NaN"], "L2正則化係数"),
        (["--l2-reg", "10"], "積は1未満"),
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_kotonoha"))
            .args(["train-crf", "--data"])
            .arg(&data)
            .arg("--output")
            .arg(&model)
            .args(args)
            .output()
            .unwrap();
        assert!(!output.status.success());
        assert!(
            String::from_utf8_lossy(&output.stderr).contains(expected_error),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(!model.exists());
    }
    std::fs::remove_dir_all(&dir).unwrap();
}
