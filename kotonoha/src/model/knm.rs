//! KNM v1: 推論用に配置した FP32 重み。形式仕様は kotonoha-models/docs/knm-format.md。

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const MAGIC: &[u8; 8] = b"KOTONNM\0";
const MAX_FILE: u64 = 64 * 1024 * 1024;
const MAX_METADATA: usize = 64 * 1024;
const COMPRESSION_LEVEL: i32 = 19;

/// 推論前に形状・チェックサムを検証したテンソル。
pub struct Weights {
    /// 入力特徴量の次元。
    pub input_dim: usize,
    /// 元モデルの SHA-256。移行後の同一モデルを確認する。
    pub source_sha256: String,
    /// 行優先 FP32。LSTM 再帰行列だけは 32 出力単位のタイル配置。
    pub tensors: HashMap<String, Vec<f32>>,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Metadata {
    architecture: String,
    input_dim: usize,
    source_sha256: String,
    tensors: Vec<Entry>,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Entry {
    name: String,
    shape: Vec<usize>,
    offset: usize,
    stored_len: usize,
    codec: u8,
    sha256: String,
}

/// ファイル操作・形式・整合性エラー。
#[derive(Debug, thiserror::Error)]
pub enum KnmError {
    /// ファイル操作失敗。
    #[error("KNM I/O: {0}")]
    Io(#[from] std::io::Error),
    /// バージョン・構造・重みの不整合。
    #[error("invalid KNM: {0}")]
    Invalid(String),
}

fn invalid(message: impl Into<String>) -> KnmError {
    KnmError::Invalid(message.into())
}

/// 固定アーキテクチャで必要な重みの形状。未知の構造を推測して受け付けない。
pub fn tensor_shapes(dim: usize) -> Result<HashMap<String, Vec<usize>>, KnmError> {
    if ![11, 13, 14, 24, 103].contains(&dim) {
        return Err(invalid("unsupported feature dimension"));
    }
    let mut shapes = HashMap::new();
    for (name, rows, cols) in [
        ("pos", 15, 16),
        ("pd1", 21, 16),
        ("pd2", 11, 8),
        ("ct", 9, 8),
        ("cf", 10, 8),
    ] {
        shapes.insert(format!("frontend.{name}_emb.weight"), vec![rows, cols]);
    }
    for (name, input, output) in [
        ("frontend.kana_proj", 16, 16),
        ("frontend.cont_proj", dim - 5, 16),
        ("frontend.input_proj", 88, 256),
        ("lstm_proj", 512, 256),
        ("self_attn.qkv_proj", 256, 768),
        ("self_attn.out_proj", 256, 256),
        ("classifier.0", 256, 256),
        ("classifier.3", 256, 21),
    ] {
        shapes.insert(format!("{name}.weight"), vec![input, output]);
        shapes.insert(format!("{name}.bias"), vec![output]);
    }
    for (name, width) in [
        ("frontend.input_norm", 88),
        ("lstm_norm", 256),
        ("self_attn.norm", 256),
    ] {
        shapes.insert(format!("{name}.weight"), vec![width]);
        shapes.insert(format!("{name}.bias"), vec![width]);
    }
    for layer in 0..3 {
        shapes.insert(
            format!("lstm.{layer}.input"),
            vec![2, 32, if layer == 0 { 256 } else { 512 }, 32],
        );
        shapes.insert(format!("lstm.{layer}.recurrent"), vec![2, 32, 256, 32]);
        shapes.insert(format!("lstm.{layer}.bias"), vec![2, 2048]);
    }
    Ok(shapes)
}

impl Weights {
    /// 外部ランタイムや計算グラフを使わず、検証した重みだけを読む。
    pub fn load(path: &Path) -> Result<Self, KnmError> {
        let mut file = File::open(path)?;
        let file_len = file.metadata()?.len();
        if !(32..=MAX_FILE).contains(&file_len) {
            return Err(invalid("file length outside supported range"));
        }
        let mut header = [0u8; 32];
        file.read_exact(&mut header)?;
        if &header[..8] != MAGIC
            || u32::from_le_bytes(header[8..12].try_into().unwrap()) != 1
            || header[24..] != [0; 8]
        {
            return Err(invalid("magic, version or reserved header"));
        }
        let meta_len = u32::from_le_bytes(header[12..16].try_into().unwrap()) as usize;
        let payload_len = u64::from_le_bytes(header[16..24].try_into().unwrap());
        if meta_len > MAX_METADATA
            || meta_len as u64 > file_len - 32
            || payload_len != file_len - 32 - meta_len as u64
        {
            return Err(invalid("metadata or payload length"));
        }
        let mut json = vec![0; meta_len];
        file.read_exact(&mut json)?;
        let meta: Metadata = serde_json::from_slice(&json).map_err(|e| invalid(e.to_string()))?;
        if meta.architecture != "bilstm-attention-v1" || !is_digest(&meta.source_sha256) {
            return Err(invalid("architecture or source digest"));
        }
        let mut expected = tensor_shapes(meta.input_dim)?;
        if meta.tensors.len() != expected.len() {
            return Err(invalid("tensor count"));
        }
        let mut offset = 0usize;
        // 配列を確保する前に全形状と全範囲を検証する。
        for entry in &meta.tensors {
            let shape = expected
                .remove(&entry.name)
                .ok_or_else(|| invalid("unknown or duplicate tensor"))?;
            let raw_len = shape.iter().product::<usize>() * 4;
            if entry.shape != shape
                || entry.offset != offset
                || entry.codec > 2
                || !is_digest(&entry.sha256)
                || entry.stored_len == 0
                || entry.stored_len > raw_len
                || (entry.codec == 0 && entry.stored_len != raw_len)
                || (entry.codec == 2 && entry.stored_len <= raw_len / 4 * 3)
            {
                return Err(invalid(format!("tensor metadata: {}", entry.name)));
            }
            offset = offset
                .checked_add(entry.stored_len)
                .ok_or_else(|| invalid("offset overflow"))?;
            if offset as u64 > payload_len {
                return Err(invalid("tensor outside payload"));
            }
        }
        if offset as u64 != payload_len {
            return Err(invalid("trailing payload"));
        }
        if let Some((_, expected)) = super::fingerprints::WEIGHT_FINGERPRINTS
            .iter()
            .find(|(source, _)| *source == meta.source_sha256)
        {
            let mut entries: Vec<_> = meta.tensors.iter().collect();
            entries.sort_by(|a, b| a.name.cmp(&b.name));
            let mut hash = Sha256::new();
            hash.update((meta.input_dim as u64).to_le_bytes());
            for entry in entries {
                hash.update(entry.name.as_bytes());
                hash.update([0]);
                hash.update(entry.sha256.as_bytes());
                hash.update([0]);
            }
            let actual: String = hash.finalize().iter().map(|b| format!("{b:02x}")).collect();
            if actual != *expected {
                return Err(invalid("weights do not match pinned source"));
            }
        }
        let mut tensors = HashMap::new();
        for entry in meta.tensors {
            let count = entry.shape.iter().product::<usize>();
            let mut stored = vec![0; entry.stored_len];
            file.read_exact(&mut stored)?;
            let raw = if entry.codec == 2 {
                let high = zstd::bulk::decompress(&stored[count * 3..], count)?;
                if high.len() != count {
                    return Err(invalid("decoded high-byte length"));
                }
                let mut raw = vec![0; count * 4];
                for ((word, low), high) in raw
                    .as_chunks_mut::<4>()
                    .0
                    .iter_mut()
                    .zip(stored[..count * 3].as_chunks::<3>().0)
                    .zip(high)
                {
                    word[..3].copy_from_slice(low);
                    word[3] = high;
                }
                raw
            } else if entry.codec == 1 {
                let shuffled = zstd::bulk::decompress(&stored, count * 4)?;
                if shuffled.len() != count * 4 {
                    return Err(invalid("decoded tensor length"));
                }
                let mut raw = vec![0; count * 4];
                unshuffle(&shuffled, &mut raw);
                raw
            } else {
                stored
            };
            if digest(&raw) != entry.sha256 {
                return Err(invalid(format!("checksum: {}", entry.name)));
            }
            let values: Vec<f32> = raw
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect();
            if !values.iter().all(|v| v.is_finite()) {
                return Err(invalid("non-finite weight"));
            }
            tensors.insert(entry.name, values);
        }
        Ok(Self {
            input_dim: meta.input_dim,
            source_sha256: meta.source_sha256,
            tensors,
        })
    }

    /// 参照変換器と同じ KNM を書く。既存ファイルの置換は呼び出し側で行う。
    pub fn write(&self, output: &mut impl Write, source_sha256: &str) -> Result<(), KnmError> {
        let mut shapes = tensor_shapes(self.input_dim)?;
        if self.tensors.len() != shapes.len() || !is_digest(source_sha256) {
            return Err(invalid("tensor count or source checksum"));
        }
        let mut names: Vec<_> = self.tensors.keys().collect();
        names.sort();
        let mut entries = Vec::new();
        let mut chunks = Vec::new();
        let mut offset = 0;
        for name in names {
            let shape = shapes
                .remove(name)
                .ok_or_else(|| invalid("unknown tensor"))?;
            let values = &self.tensors[name];
            if values.len() != shape.iter().product::<usize>()
                || !values.iter().all(|v| v.is_finite())
            {
                return Err(invalid("tensor values"));
            }
            let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            let mut shuffled = vec![0; raw.len()];
            for (i, word) in raw.as_chunks::<4>().0.iter().enumerate() {
                for byte in 0..4 {
                    shuffled[byte * values.len() + i] = word[byte];
                }
            }
            let compressed = zstd::bulk::compress(&shuffled, COMPRESSION_LEVEL)?;
            let high: Vec<_> = raw.as_chunks::<4>().0.iter().map(|word| word[3]).collect();
            let mut split: Vec<_> = raw
                .as_chunks::<4>()
                .0
                .iter()
                .flat_map(|word| word[..3].iter().copied())
                .collect();
            split.extend_from_slice(&zstd::bulk::compress(&high, COMPRESSION_LEVEL)?);
            let (compressed, kind) =
                if raw.len() >= 4096 && split.len() * 1000 <= compressed.len() * 1005 {
                    (split, 2)
                } else {
                    (compressed, 1)
                };
            let (chunk, codec) = if compressed.len() < raw.len() {
                (compressed, kind)
            } else {
                (raw.clone(), 0)
            };
            entries.push(Entry {
                name: name.clone(),
                shape,
                offset,
                stored_len: chunk.len(),
                codec,
                sha256: digest(&raw),
            });
            offset += chunk.len();
            chunks.push(chunk);
        }
        let metadata = Metadata {
            architecture: "bilstm-attention-v1".into(),
            input_dim: self.input_dim,
            source_sha256: source_sha256.into(),
            tensors: entries,
        };
        let json = serde_json::to_vec(&metadata).map_err(|e| invalid(e.to_string()))?;
        output.write_all(MAGIC)?;
        output.write_all(&1u32.to_le_bytes())?;
        output.write_all(&(json.len() as u32).to_le_bytes())?;
        output.write_all(&(offset as u64).to_le_bytes())?;
        output.write_all(&0u64.to_le_bytes())?;
        output.write_all(&json)?;
        for chunk in chunks {
            output.write_all(&chunk)?;
        }
        Ok(())
    }
}

fn unshuffle(input: &[u8], output: &mut [u8]) {
    let count = output.len() / 4;
    for (i, word) in output.as_chunks_mut::<4>().0.iter_mut().enumerate() {
        word.copy_from_slice(&[
            input[i],
            input[count + i],
            input[2 * count + i],
            input[3 * count + i],
        ]);
    }
}

fn is_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn digest(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Vec<u8> {
        let tensors = tensor_shapes(11)
            .unwrap()
            .into_iter()
            .map(|(name, shape)| {
                let mut values = vec![0.; shape.iter().product()];
                for (slot, value) in values.iter_mut().zip([
                    -0.,
                    f32::MIN_POSITIVE,
                    f32::MAX,
                    -f32::MAX,
                    1e-30,
                    0.1234567,
                ]) {
                    *slot = value;
                }
                (name, values)
            })
            .collect();
        let weights = Weights {
            input_dim: 11,
            source_sha256: "a".repeat(64),
            tensors,
        };
        let mut bytes = Vec::new();
        weights.write(&mut bytes, &weights.source_sha256).unwrap();
        bytes
    }

    fn load(bytes: &[u8]) -> Result<Weights, KnmError> {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(bytes).unwrap();
        Weights::load(file.path())
    }

    #[test]
    fn lossless_weights_roundtrip_and_digest() {
        let bytes = fixture();
        let weights = load(&bytes).unwrap();
        assert_eq!(weights.source_sha256, "a".repeat(64));
        for tensor in weights.tensors.values() {
            assert_eq!(tensor[0].to_bits(), (-0f32).to_bits());
            assert_eq!(tensor[2], f32::MAX);
        }
        let mut corrupt = bytes.clone();
        *corrupt.last_mut().unwrap() ^= 1;
        assert!(load(&corrupt).is_err());
        assert!(load(&bytes[..bytes.len() - 1]).is_err());
    }

    #[test]
    fn malformed_metadata_is_rejected_before_decoding() {
        let original = fixture();
        let meta_len = u32::from_le_bytes(original[12..16].try_into().unwrap()) as usize;
        let meta: serde_json::Value = serde_json::from_slice(&original[32..32 + meta_len]).unwrap();
        for kind in [
            "version",
            "shape",
            "offset",
            "codec",
            "duplicate",
            "architecture",
            "length",
            "source",
        ] {
            let mut current = meta.clone();
            match kind {
                "shape" => current["tensors"][0]["shape"] = serde_json::json!([u64::MAX]),
                "offset" => current["tensors"][0]["offset"] = serde_json::json!(u64::MAX),
                "codec" => current["tensors"][0]["codec"] = serde_json::json!(3),
                "duplicate" => current["tensors"][1] = current["tensors"][0].clone(),
                "architecture" => current["architecture"] = serde_json::json!("unknown"),
                "length" => current["tensors"][0]["stored_len"] = serde_json::json!(u64::MAX),
                "source" => {
                    current["source_sha256"] =
                        serde_json::json!(crate::model::MODEL_FILES[0].sha256)
                }
                _ => (),
            }
            let json = serde_json::to_vec(&current).unwrap();
            let mut bytes = original[..32].to_vec();
            bytes[12..16].copy_from_slice(&(json.len() as u32).to_le_bytes());
            if kind == "version" {
                bytes[8] = 2;
            }
            bytes.extend_from_slice(&json);
            bytes.extend_from_slice(&original[32 + meta_len..]);
            assert!(load(&bytes).is_err(), "{kind}");
        }
    }

    #[test]
    fn raw_mantissa_codec_preserves_all_float_bits() {
        let mut weights = load(&fixture()).unwrap();
        let data = weights
            .tensors
            .get_mut("frontend.input_proj.weight")
            .unwrap();
        let mut seed = 7u32;
        for value in data.iter_mut() {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            *value = f32::from_bits(0x3e000000 | (seed & 0x01ffffff));
        }
        let expected = data.clone();
        let mut bytes = Vec::new();
        weights.write(&mut bytes, &weights.source_sha256).unwrap();
        let meta_len = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
        let meta: Metadata = serde_json::from_slice(&bytes[32..32 + meta_len]).unwrap();
        assert_eq!(
            meta.tensors
                .iter()
                .find(|e| e.name == "frontend.input_proj.weight")
                .unwrap()
                .codec,
            2
        );
        let restored = load(&bytes).unwrap();
        assert_eq!(restored.tensors["frontend.input_proj.weight"], expected);
    }
}
