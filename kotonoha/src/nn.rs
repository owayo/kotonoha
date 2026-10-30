//! ニューラルネットワーク推論モジュール
//!
//! アクセント予測のためのトレイトと実装を提供する。
//! `neural` feature で外部ランタイム不要の専用 CPU 推論が利用可能になる。

#[cfg(any(feature = "neural", test))]
use crate::njd::Pos;
use crate::njd::{InputToken, NjdNode};
#[cfg(any(feature = "neural", test))]
use v66::features::{char_hash, reading_hash};
#[cfg(any(feature = "neural", test))]
use v66::vocab::{conj_form_group, conj_type_group, pos_detail1_to_id, pos_detail2_to_id};

#[cfg(feature = "neural")]
pub mod native;
pub mod v66;

/// アクセント予測トレイト（NjdNodeのみ）
pub trait AccentPredictor {
    /// NjdNode列からアクセント型を予測する
    fn predict(&self, nodes: &[NjdNode]) -> Vec<u8>;

    /// 推論失敗を呼び出し側へ返す。失敗しうる予測器はこのメソッドを実装する。
    ///
    /// 既存の予測器は `predict` の実装だけでも利用できる。
    fn try_predict(&self, nodes: &[NjdNode]) -> Result<Vec<u8>, ContextualAccentError> {
        Ok(self.predict(nodes))
    }
}

/// 特徴量抽出に必要な、生 InputToken と NjdNode のペア
///
/// v66 系 predictor は UniDic 系の生 POS 文字列 (`代名詞` / `接尾辞` / `形状詞` 等)
/// を必要とするため、`NjdNode` を経由せず元の `InputToken` を保持する必要がある。
pub struct FeatureMorpheme<'a> {
    /// 生の入力トークン（UniDic 由来 POS 文字列を保持）
    pub token: &'a InputToken,
    /// 派生済み NjdNode（mora_count・pronunciation・accent_type 初期値などを参照）
    pub node: &'a NjdNode,
}

/// アクセント予測器が返すエラー型
pub type ContextualAccentError = Box<dyn std::error::Error + Send + Sync>;

/// `(InputToken, NjdNode)` ペア列からアクセント型を予測するトレイト
///
/// `AccentPredictor` と異なり、UniDic 文字列を含む `InputToken` も渡される。
///
/// 推論失敗 (ORT エラー、bundle 不整合、shape 不正など) は `Err` で返す。
/// 呼び出し側は失敗を観測し、フォールバックの可否を判断できる。
pub trait ContextualAccentPredictor: Send + Sync {
    /// FeatureMorpheme列からアクセント型を予測する
    ///
    /// 成功時は `ctx.len()` と同じ長さの `Vec<u8>` を返す。
    fn predict_with_context(
        &self,
        ctx: &[FeatureMorpheme<'_>],
    ) -> Result<Vec<u8>, ContextualAccentError>;
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
#[cfg(any(feature = "neural", test))]
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
        Pos::Filler => 0, // <unk>
        Pos::Sonota => 0, // <unk>
    }
}

/// v2特徴量の次元数
#[cfg(any(feature = "neural", test))]
const FEATURE_DIM: usize = 11;

/// NjdNodeからv2特徴量ベクトルを抽出する
///
/// 特徴量順序: [pos_id, pd1_id, pd2_id, ct_id, cf_id,
///              mora_count, reading_hash, first_char_hash,
///              last_char_hash, position, dict_accent_type]
#[cfg(any(feature = "neural", test))]
fn extract_features_v2(node: &NjdNode, position: f32) -> [f32; FEATURE_DIM] {
    let first_ch = node.surface.chars().next().map_or(0.0, char_hash);
    let last_ch = node.surface.chars().last().map_or(0.0, char_hash);

    [
        pos_to_id(&node.pos) as f32,
        pos_detail1_to_id(&node.pos_detail1) as f32,
        pos_detail2_to_id(&node.pos_detail2) as f32,
        conj_type_group(&node.ctype) as f32,
        conj_form_group(&node.cform) as f32,
        node.mora_count as f32 / 10.0,
        reading_hash(&node.reading),
        first_ch,
        last_ch,
        position,
        if node.accent_type > 0 {
            (node.accent_type as f32 + 1.0) / 8.0
        } else {
            0.0
        },
    ]
}

/// KNM の単一モデルを使ったアクセント予測器。
#[cfg(feature = "neural")]
pub struct NativePredictor {
    model: native::NativeModel,
}

#[cfg(feature = "neural")]
impl NativePredictor {
    /// legacy v8 と同じ 11 次元入力の KNM モデルをロードする。
    pub fn new(model_path: &std::path::Path) -> Result<Self, Box<dyn std::error::Error>> {
        let model = native::NativeModel::load(model_path)?;
        if model.input_dim() != FEATURE_DIM {
            return Err("single model requires 11 input features".into());
        }
        Ok(Self { model })
    }

    /// 特徴量テンソルを構築し推論を実行する（v2: 11次元特徴量）
    fn run_inference(&self, nodes: &[NjdNode]) -> Result<Vec<u8>, ContextualAccentError> {
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

        let output_data = self.model.predict(&input_data, node_count)?;

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
            for &val in output_data.iter().take(node_count) {
                accent_types.push(val.round().max(0.0) as u8);
            }
        }

        Ok(accent_types)
    }
}

#[cfg(feature = "neural")]
impl AccentPredictor for NativePredictor {
    fn predict(&self, nodes: &[NjdNode]) -> Vec<u8> {
        self.try_predict(nodes).unwrap_or_else(|e| {
            eprintln!("kotonoha: KNM predictor failed ({e}); keeping existing accent_type");
            // 推論失敗時はルールベースにフォールバック
            RuleBasedPredictor.predict(nodes)
        })
    }

    fn try_predict(&self, nodes: &[NjdNode]) -> Result<Vec<u8>, ContextualAccentError> {
        self.run_inference(nodes)
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
