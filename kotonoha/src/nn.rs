//! ニューラルネットワーク推論モジュール
//!
//! アクセント予測のためのトレイトと実装を提供する。
//! `cuda` featureを有効にすると、ONNX Runtimeを使ったGPU推論が利用可能になる。

#[cfg(any(feature = "cuda", test))]
use crate::njd::Pos;
use crate::njd::NjdNode;

/// アクセント予測トレイト
pub trait AccentPredictor {
    /// NjdNode列からアクセント型を予測する
    fn predict(&self, nodes: &[NjdNode]) -> Vec<u8>;
}

/// ルールベースのアクセント予測器（既存ロジックへの委譲）
///
/// NjdNodeに既に設定されているaccent_typeをそのまま返す。
pub struct RuleBasedPredictor;

impl AccentPredictor for RuleBasedPredictor {
    fn predict(&self, nodes: &[NjdNode]) -> Vec<u8> {
        nodes.iter().map(|n| n.accent_type).collect()
    }
}

/// 品詞を数値IDにエンコードする（v2モデル用）
///
/// Python側のPOS_VOCABと一致させること:
///   <unk>=0, 名詞=1, 助詞=2, 動詞=3, 助動詞=4, 接尾辞=5,
///   形容詞=6, 代名詞=7, 副詞=8, 形状詞=9, 連体詞=10,
///   接頭辞=11, 接続詞=12, 感動詞=13, 記号=14
#[cfg(any(feature = "cuda", test))]
fn pos_to_id(pos: &Pos) -> u32 {
    match pos {
        Pos::Meishi => 1,
        Pos::Joshi => 2,
        Pos::Doushi => 3,
        Pos::Jodoushi => 4,
        Pos::Keiyoushi => 6,
        Pos::Fukushi => 8,
        Pos::Rentaishi => 10,
        Pos::Settoushi => 11,
        Pos::Setsuzokushi => 12,
        Pos::Kandoushi => 13,
        Pos::Kigou => 14,
        Pos::Filler => 0,  // <unk>
        Pos::Sonota => 0,  // <unk>
    }
}

/// pos_detail1を数値IDにエンコードする
#[cfg(any(feature = "cuda", test))]
fn pos_detail1_to_id(detail: &str) -> u32 {
    match detail {
        "*" => 1,
        "普通名詞" => 2,
        "格助詞" => 3,
        "非自立可能" => 4,
        "固有名詞" => 5,
        "接続助詞" => 6,
        "係助詞" => 7,
        "副助詞" => 8,
        "一般" => 9,
        "終助詞" => 10,
        "準体助詞" => 11,
        "数詞" => 12,
        "助動詞語幹" => 13,
        "タリ" => 14,
        "フィラー" => 15,
        "動詞的" => 16,
        "名詞的" => 17,
        "形容詞的" => 18,
        "形状詞的" => 19,
        "文字" => 20,
        _ => 0, // <unk>
    }
}

/// pos_detail2を数値IDにエンコードする
#[cfg(any(feature = "cuda", test))]
fn pos_detail2_to_id(detail: &str) -> u32 {
    match detail {
        "*" => 1,
        "一般" => 2,
        "サ変可能" => 3,
        "サ変形状詞可能" => 4,
        "副詞可能" => 5,
        "形状詞可能" => 6,
        "人名" => 7,
        "地名" => 8,
        "助数詞" => 9,
        "助数詞可能" => 10,
        _ => 0, // <unk>
    }
}

/// 活用型をグループIDにエンコードする
#[cfg(any(feature = "cuda", test))]
fn conj_type_to_id(ctype: &str) -> u32 {
    if ctype == "*" {
        return 0;
    }
    if ctype.starts_with("五段") { return 1; }
    if ctype.starts_with("上一段") { return 2; }
    if ctype.starts_with("下一段") { return 3; }
    if ctype.starts_with("カ行変格") { return 4; }
    if ctype.starts_with("サ行変格") { return 5; }
    if ctype.starts_with("形容詞") { return 6; }
    if ctype.starts_with("助動詞") { return 7; }
    if ctype.starts_with("文語") { return 8; }
    0 // unknown
}

/// 活用形をグループIDにエンコードする
#[cfg(any(feature = "cuda", test))]
fn conj_form_to_id(cform: &str) -> u32 {
    if cform == "*" {
        return 0;
    }
    if cform.starts_with("未然形") { return 1; }
    if cform.starts_with("連用形") { return 2; }
    if cform.starts_with("終止形") { return 3; }
    if cform.starts_with("連体形") { return 4; }
    if cform.starts_with("仮定形") { return 5; }
    if cform.starts_with("命令形") { return 6; }
    if cform.starts_with("已然形") { return 7; }
    if cform.starts_with("意志推量形") { return 8; }
    if cform.starts_with("語幹") { return 9; }
    0 // unknown
}

/// 読みの簡易ハッシュ（特徴量として使用、正規化済み）
#[cfg(any(feature = "cuda", test))]
fn reading_hash(reading: &str) -> f32 {
    let mut hash: u32 = 5381;
    for c in reading.chars() {
        hash = hash.wrapping_mul(33).wrapping_add(c as u32);
    }
    (hash % 10000) as f32 / 10000.0
}

/// 文字の簡易ハッシュ（正規化済み）
#[cfg(any(feature = "cuda", test))]
fn char_hash(ch: char) -> f32 {
    let mut hash: u32 = 5381;
    hash = hash.wrapping_mul(33).wrapping_add(ch as u32);
    (hash % 10000) as f32 / 10000.0
}

/// v2特徴量の次元数
#[cfg(any(feature = "cuda", test))]
const FEATURE_DIM: usize = 11;

/// NjdNodeからv2特徴量ベクトルを抽出する
///
/// 特徴量順序: [pos_id, pd1_id, pd2_id, ct_id, cf_id,
///              mora_count, reading_hash, first_char_hash,
///              last_char_hash, position, dict_accent_type]
#[cfg(any(feature = "cuda", test))]
fn extract_features_v2(node: &NjdNode, position: f32) -> [f32; FEATURE_DIM] {
    let first_ch = node.surface.chars().next().map_or(0.0, char_hash);
    let last_ch = node.surface.chars().last().map_or(0.0, char_hash);

    [
        pos_to_id(&node.pos) as f32,
        pos_detail1_to_id(&node.pos_detail1) as f32,
        pos_detail2_to_id(&node.pos_detail2) as f32,
        conj_type_to_id(&node.ctype) as f32,
        conj_form_to_id(&node.cform) as f32,
        f32::from(node.mora_count) / 10.0,
        reading_hash(&node.reading),
        first_ch,
        last_ch,
        position,
        if node.accent_type > 0 { (node.accent_type as f32 + 1.0) / 8.0 } else { 0.0 },
    ]
}

/// ONNX Runtimeを使ったアクセント予測器（CUDA対応）
#[cfg(feature = "cuda")]
pub struct OnnxPredictor {
    session: std::sync::Mutex<ort::session::Session>,
}

#[cfg(feature = "cuda")]
impl OnnxPredictor {
    /// ONNXモデルファイルからOnnxPredictorを構築する
    ///
    /// CUDAが利用可能な場合はGPU実行プロバイダを使用し、
    /// 利用不可の場合はCPUにフォールバックする。
    pub fn new(model_path: &std::path::Path) -> Result<Self, Box<dyn std::error::Error>> {
        let session = ort::session::Session::builder()?
            .with_execution_providers([
                ort::execution_providers::CUDAExecutionProvider::default().build(),
            ])?
            .commit_from_file(model_path)?;

        Ok(Self {
            session: std::sync::Mutex::new(session),
        })
    }

    /// 特徴量テンソルを構築し推論を実行する（v2: 11次元特徴量）
    fn run_inference(&self, nodes: &[NjdNode]) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let node_count = nodes.len();
        if node_count == 0 {
            return Ok(Vec::new());
        }

        let mut input_data = vec![0.0f32; node_count * FEATURE_DIM];

        for (i, node) in nodes.iter().enumerate() {
            let position = if node_count > 1 {
                i as f32 / (node_count - 1) as f32
            } else {
                0.0
            };
            let features = extract_features_v2(node, position);
            let offset = i * FEATURE_DIM;
            input_data[offset..offset + FEATURE_DIM].copy_from_slice(&features);
        }

        let input_tensor = ort::value::Tensor::from_array((
            vec![node_count as i64, FEATURE_DIM as i64],
            input_data,
        ))?;

        let mut session = self.session.lock().map_err(|e| e.to_string())?;
        let outputs = session.run(ort::inputs!["input" => input_tensor])?;

        let (_shape, output_data) = outputs[0].try_extract_tensor::<f32>()?;

        // 各ノードの出力からアクセント型を取得（argmaxまたは丸め）
        let mut accent_types = Vec::with_capacity(node_count);
        let num_classes = output_data.len() / node_count;

        if num_classes > 1 {
            // 出力が[node_count, num_classes]の場合はargmax
            for i in 0..node_count {
                let offset = i * num_classes;
                let mut max_idx = 0;
                let mut max_val = f32::NEG_INFINITY;
                for j in 0..num_classes {
                    let val = output_data[offset + j];
                    if val > max_val {
                        max_val = val;
                        max_idx = j;
                    }
                }
                accent_types.push(max_idx as u8);
            }
        } else {
            // 出力が[node_count]の場合は丸め
            for i in 0..node_count {
                let val = output_data[i];
                accent_types.push(val.round().max(0.0) as u8);
            }
        }

        Ok(accent_types)
    }
}

#[cfg(feature = "cuda")]
impl AccentPredictor for OnnxPredictor {
    fn predict(&self, nodes: &[NjdNode]) -> Vec<u8> {
        self.run_inference(nodes).unwrap_or_else(|_| {
            // 推論失敗時はルールベースにフォールバック
            RuleBasedPredictor.predict(nodes)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::njd::InputToken;

    #[test]
    fn test_rule_based_predictor() {
        let token = InputToken::new("猫", "名詞", "ネコ", "ネコ");
        let mut node = NjdNode::from_token(&token);
        node.accent_type = 1;

        let predictor = RuleBasedPredictor;
        let result = predictor.predict(&[node]);
        assert_eq!(result, vec![1]);
    }

    #[test]
    fn test_rule_based_predictor_empty() {
        let predictor = RuleBasedPredictor;
        let result = predictor.predict(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_pos_to_id() {
        assert_eq!(pos_to_id(&Pos::Meishi), 1);
        assert_eq!(pos_to_id(&Pos::Doushi), 3);
        assert_eq!(pos_to_id(&Pos::Sonota), 0); // <unk>
    }

    #[test]
    fn test_reading_hash_deterministic() {
        let h1 = reading_hash("ネコ");
        let h2 = reading_hash("ネコ");
        let h3 = reading_hash("イヌ");
        assert_eq!(h1, h2);
        assert_ne!(h1, h3);
        // Should be normalized to [0, 1)
        assert!(h1 >= 0.0 && h1 < 1.0);
    }

    #[test]
    fn test_extract_features_v2() {
        let token = InputToken::new("猫", "名詞", "ネコ", "ネコ");
        let node = NjdNode::from_token(&token);
        let features = extract_features_v2(&node, 0.5);
        assert_eq!(features.len(), FEATURE_DIM);
        assert_eq!(features[0], 1.0); // Meishi = 1
        assert_eq!(features[1], 1.0); // pos_detail1 "*" = 1
        assert_eq!(features[9], 0.5); // position
    }
}
