//! analyze の形態素表示と解析結果の出力先を検証する。

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use hasami::dict::{DictBuilder, DictEntry};

fn cli(data_dir: &Path) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_kotonoha"));
    command
        .env_remove("KOTONOHA_MODEL_BUNDLE")
        .env_remove("KOTONOHA_MODEL_PATH")
        .env_remove("KOTONOHA_MODEL_VARIANT")
        .env("XDG_DATA_HOME", data_dir);
    command
}

fn dictionary(dir: &Path) -> PathBuf {
    let mut builder = DictBuilder::new();
    for (surface, pos, reading, conj_type, conj_form) in [
        ("猫", "名詞,一般,*,*", "ネコ", "", ""),
        ("が", "助詞,格助詞,一般,*", "ガ", "", ""),
        (
            "歩く",
            "動詞,自立,*,*",
            "アルク",
            "五段・カ行イ音便",
            "基本形",
        ),
        ("。", "記号,句点,*,*", "。", "", ""),
    ] {
        builder.add_entry(DictEntry {
            surface: surface.into(),
            base_form: surface.into(),
            pos: pos.into(),
            reading: reading.into(),
            pronunciation: reading.into(),
            conj_type: conj_type.into(),
            conj_form: conj_form.into(),
            cost: -10000,
            ..Default::default()
        });
    }
    let path = dir.join("test.hsd");
    builder
        .write_hsd(&path, &builder.write_options(), |_, _| {})
        .unwrap();
    path
}

fn successful(output: Output) -> Output {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    output
}

#[test]
fn tokens_go_to_stderr_and_preserve_every_analysis_format() {
    let dir = tempfile::tempdir().unwrap();
    let dict = dictionary(dir.path());
    let text = "猫 が歩く。";
    let mecab = successful(
        cli(dir.path())
            .args(["tokenize", text, "--dict"])
            .arg(&dict)
            .args(["--format", "mecab"])
            .output()
            .unwrap(),
    );
    let displayed = String::from_utf8_lossy(&mecab.stdout);
    assert!(displayed.contains("猫\t名詞,一般,*,*,*,*,猫,ネコ,ネコ\n"));
    assert!(displayed.contains("歩く\t動詞,自立,*,*,五段・カ行イ音便,基本形,歩く,アルク,アルク\n"));
    assert!(displayed.ends_with("EOS\n"));

    let mut engine = kotonoha::Engine::default();
    engine.load_dictionary(&dict).unwrap();
    for format in ["labels", "phone-tones", "prosody", "json"] {
        let plain = successful(
            cli(dir.path())
                .args(["analyze", text, "--dict"])
                .arg(&dict)
                .args(["--format", format])
                .output()
                .unwrap(),
        );
        assert!(plain.stderr.is_empty());
        let with_tokens = successful(
            cli(dir.path())
                .args(["analyze", text, "--dict"])
                .arg(&dict)
                .args(["--format", format, "--tokens"])
                .output()
                .unwrap(),
        );
        assert_eq!(with_tokens.stdout, plain.stdout, "{format}");
        assert_eq!(with_tokens.stderr, mecab.stdout, "{format}");

        let expected = match format {
            "labels" => format!("{}\n", engine.text_to_labels(text).unwrap().join("\n")),
            "phone-tones" => engine
                .text_to_phone_tones(text)
                .unwrap()
                .iter()
                .map(|pt| format!("{}\t{}\n", pt.phone, pt.tone))
                .collect(),
            "prosody" => format!(
                "{}\n",
                engine.text_to_prosody_symbols(text).unwrap().join("")
            ),
            "json" => {
                let labels: Vec<String> = serde_json::from_slice(&with_tokens.stdout).unwrap();
                assert_eq!(labels, engine.text_to_labels(text).unwrap());
                format!("{}\n", serde_json::to_string_pretty(&labels).unwrap())
            }
            _ => unreachable!(),
        };
        assert_eq!(with_tokens.stdout, expected.as_bytes(), "{format}");
    }
}

#[test]
fn empty_input_has_an_empty_token_listing_and_valid_json() {
    let dir = tempfile::tempdir().unwrap();
    let dict = dictionary(dir.path());
    let output = successful(
        cli(dir.path())
            .args(["analyze", "", "--dict"])
            .arg(&dict)
            .args(["--tokens", "--format", "json"])
            .output()
            .unwrap(),
    );
    assert_eq!(output.stderr, b"EOS\n");
    let labels: Vec<String> = serde_json::from_slice(&output.stdout).unwrap();
    let mut engine = kotonoha::Engine::default();
    engine.load_dictionary(&dict).unwrap();
    assert_eq!(labels, engine.text_to_labels("").unwrap());
    let plain = successful(
        cli(dir.path())
            .args(["analyze", "", "--dict"])
            .arg(&dict)
            .args(["--format", "json"])
            .output()
            .unwrap(),
    );
    assert!(plain.stderr.is_empty());
    assert_eq!(plain.stdout, output.stdout);
}

#[test]
fn missing_or_invalid_dictionary_fails_without_analysis_output() {
    let dir = tempfile::tempdir().unwrap();
    let invalid = dir.path().join("invalid.hsd");
    std::fs::write(&invalid, b"invalid dictionary").unwrap();
    for dict in [None, Some(dir.path().join("missing.hsd")), Some(invalid)] {
        let mut command = cli(dir.path());
        command.args(["analyze", "猫", "--tokens", "--format", "json"]);
        let expected = if let Some(path) = dict {
            command.arg("--dict").arg(path);
            "Error loading dictionary:"
        } else {
            "No dictionary specified"
        };
        let output = command.output().unwrap();
        assert!(!output.status.success());
        assert!(output.stdout.is_empty());
        assert!(String::from_utf8_lossy(&output.stderr).contains(expected));
    }
}
