//! 固定 SHA-256 の既存 ONNX 配布元を KNM へ移すオフライン変換。
//!
//! 未公開の KNM 配布版を参照せず、検証済みの現行配布物から移行できるようにする。
//! グラフは実行しない。既知の export の重みだけを取り出す。

use super::knm::{KnmError, Weights};
use sha2::{Digest, Sha256};
use std::collections::HashMap;

fn error() -> KnmError {
    KnmError::Invalid("unsupported or corrupt legacy ONNX export".into())
}

struct Field<'a> {
    number: u64,
    bytes: &'a [u8],
    value: u64,
}

fn varint(bytes: &mut &[u8]) -> Result<u64, KnmError> {
    let mut value = 0;
    for shift in (0..70).step_by(7) {
        let (&b, rest) = bytes.split_first().ok_or_else(error)?;
        *bytes = rest;
        if shift == 63 && b > 1 {
            return Err(error());
        }
        value |= u64::from(b & 127) << shift;
        if b < 128 {
            return Ok(value);
        }
    }
    Err(error())
}

fn fields(mut bytes: &[u8]) -> Result<Vec<Field<'_>>, KnmError> {
    let mut result = Vec::new();
    while !bytes.is_empty() {
        let key = varint(&mut bytes)?;
        let (value, len) = match key & 7 {
            0 => (varint(&mut bytes)?, 0),
            1 => (0, 8),
            2 => (
                0,
                usize::try_from(varint(&mut bytes)?).map_err(|_| error())?,
            ),
            5 => (0, 4),
            _ => return Err(error()),
        };
        if key >> 3 == 0 || len > bytes.len() {
            return Err(error());
        }
        result.push(Field {
            number: key >> 3,
            bytes: &bytes[..len],
            value,
        });
        bytes = &bytes[len..];
    }
    Ok(result)
}

/// 配布マニフェスト内の既知の SHA-256 のモデルだけを受け付ける。
pub(super) fn convert(bytes: &[u8], sha256: &str) -> Result<Weights, KnmError> {
    let digest: String = Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    if digest != sha256
        || !super::MODEL_FILES
            .iter()
            .any(|f| f.source.ends_with(".onnx") && f.sha256 == sha256)
    {
        return Err(error());
    }
    let root = fields(bytes)?;
    let graph = root.iter().find(|f| f.number == 7).ok_or_else(error)?;
    let mut tensors = HashMap::new();
    let mut matrices = Vec::new();
    let mut lstms = Vec::new();
    for field in fields(graph.bytes)?.into_iter().filter(|f| f.number == 5) {
        let tensor = fields(field.bytes)?;
        if tensor.iter().find(|f| f.number == 2).map(|f| f.value) != Some(1) {
            return Err(error());
        }
        let name = std::str::from_utf8(
            tensor
                .iter()
                .find(|f| f.number == 8)
                .ok_or_else(error)?
                .bytes,
        )
        .map_err(|_| error())?;
        let raw = tensor
            .iter()
            .find(|f| f.number == 9)
            .ok_or_else(error)?
            .bytes;
        if raw.len() % 4 != 0 {
            return Err(error());
        }
        let data: Vec<f32> = raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        if name.starts_with("onnx::MatMul_") {
            matrices.push(data);
        } else if name.starts_with("onnx::LSTM_") {
            lstms.push(data);
        } else if let Some(name) = name.strip_prefix("model.") {
            tensors.insert(name.to_string(), data);
        } else {
            return Err(error());
        }
    }
    if matrices.len() != 8 || lstms.len() != 9 {
        return Err(error());
    }
    let dim = matrices[1].len() / 16 + 5;
    for (name, data) in [
        "frontend.kana_proj",
        "frontend.cont_proj",
        "frontend.input_proj",
        "lstm_proj",
        "self_attn.qkv_proj",
        "self_attn.out_proj",
        "classifier.0",
        "classifier.3",
    ]
    .into_iter()
    .zip(matrices)
    {
        tensors.insert(format!("{name}.weight"), data);
    }
    for (layer, chunk) in lstms.chunks(3).enumerate() {
        tensors.insert(format!("lstm.{layer}.bias"), chunk[0].clone());
        for (kind, data, inner) in [
            ("input", &chunk[1], if layer == 0 { 256 } else { 512 }),
            ("recurrent", &chunk[2], 256),
        ] {
            if data.len() != 2 * 1024 * inner {
                return Err(error());
            }
            let mut packed = Vec::with_capacity(data.len());
            for direction in 0..2 {
                for block in 0..32 {
                    for k in 0..inner {
                        for lane in 0..32 {
                            packed.push(data[(direction * 1024 + block * 32 + lane) * inner + k]);
                        }
                    }
                }
            }
            tensors.insert(format!("lstm.{layer}.{kind}"), packed);
        }
    }
    Ok(Weights {
        input_dim: dim,
        source_sha256: sha256.into(),
        tensors,
    })
}
