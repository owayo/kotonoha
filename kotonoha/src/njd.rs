//! NJD (Nihongo Jisho Data) プロセッサ
//! 形態素解析トークンを中間表現 NjdNode に変換する

use crate::mora;
use serde::{Deserialize, Serialize};

/// 品詞の分類
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Pos {
    Meishi,       // 名詞
    Doushi,       // 動詞
    Keiyoushi,    // 形容詞
    Fukushi,      // 副詞
    Joshi,        // 助詞
    Jodoushi,     // 助動詞
    Rentaishi,    // 連体詞
    Setsuzokushi, // 接続詞
    Kandoushi,    // 感動詞
    Settoushi,    // 接頭詞
    Kigou,        // 記号
    Filler,       // フィラー
    Sonota,       // その他
}

impl Pos {
    /// 品詞文字列からPosを生成
    ///
    /// IPAdic形式とUniDic形式の両方に対応する。
    /// UniDic固有の品詞（接尾辞→名詞、代名詞→名詞、形状詞→名詞）は
    /// IPAdic相当にマッピングする。
    pub fn parse(s: &str) -> Self {
        match s {
            s if s.starts_with("名詞") => Pos::Meishi,
            s if s.starts_with("動詞") => Pos::Doushi,
            s if s.starts_with("形容詞") => Pos::Keiyoushi,
            s if s.starts_with("副詞") => Pos::Fukushi,
            s if s.starts_with("助詞") => Pos::Joshi,
            s if s.starts_with("助動詞") => Pos::Jodoushi,
            s if s.starts_with("連体詞") => Pos::Rentaishi,
            s if s.starts_with("接続詞") => Pos::Setsuzokushi,
            s if s.starts_with("感動詞") => Pos::Kandoushi,
            s if s.starts_with("接頭詞") || s.starts_with("接頭辞") => Pos::Settoushi,
            s if s.starts_with("記号") => Pos::Kigou,
            s if s.starts_with("フィラー") => Pos::Filler,
            // UniDic固有の品詞をIPAdic相当にマッピング
            s if s.starts_with("接尾辞") => Pos::Meishi,   // 接尾辞 → 名詞（接尾として扱う）
            s if s.starts_with("代名詞") => Pos::Meishi,   // 代名詞 → 名詞
            s if s.starts_with("形状詞") => Pos::Meishi,   // 形状詞 → 名詞（形容動詞語幹として扱う）
            _ => Pos::Sonota,
        }
    }

    /// 内容語（アクセント句の核となりうる語）かどうか
    pub fn is_content_word(&self) -> bool {
        matches!(
            self,
            Pos::Meishi
                | Pos::Doushi
                | Pos::Keiyoushi
                | Pos::Fukushi
                | Pos::Rentaishi
                | Pos::Setsuzokushi
                | Pos::Kandoushi
        )
    }

    /// 機能語（前の語に接続しやすい語）かどうか
    pub fn is_function_word(&self) -> bool {
        matches!(self, Pos::Joshi | Pos::Jodoushi)
    }

    /// HTS Labelで使用するPOS文字列を返す
    pub fn to_label_str(&self) -> &'static str {
        match self {
            Pos::Meishi => "名詞",
            Pos::Doushi => "動詞",
            Pos::Keiyoushi => "形容詞",
            Pos::Fukushi => "副詞",
            Pos::Joshi => "助詞",
            Pos::Jodoushi => "助動詞",
            Pos::Rentaishi => "連体詞",
            Pos::Setsuzokushi => "接続詞",
            Pos::Kandoushi => "感動詞",
            Pos::Settoushi => "接頭詞",
            Pos::Kigou => "記号",
            Pos::Filler => "フィラー",
            Pos::Sonota => "その他",
        }
    }
}

/// 入力トークン（hasami等の形態素解析器からの出力）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputToken {
    pub surface: String,
    pub pos: String,
    pub pos_detail1: String,
    pub pos_detail2: String,
    pub pos_detail3: String,
    pub ctype: String,
    pub cform: String,
    pub lemma: String,
    pub reading: String,
    pub pronunciation: String,
}

impl InputToken {
    /// 簡易コンストラクタ
    pub fn new(
        surface: &str,
        pos: &str,
        reading: &str,
        pronunciation: &str,
    ) -> Self {
        Self {
            surface: surface.to_string(),
            pos: pos.to_string(),
            pos_detail1: "*".to_string(),
            pos_detail2: "*".to_string(),
            pos_detail3: "*".to_string(),
            ctype: "*".to_string(),
            cform: "*".to_string(),
            lemma: surface.to_string(),
            reading: reading.to_string(),
            pronunciation: pronunciation.to_string(),
        }
    }
}

/// NJDノード（中間表現）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NjdNode {
    pub surface: String,
    pub pos: Pos,
    pub pos_detail1: String,
    pub pos_detail2: String,
    pub pos_detail3: String,
    pub ctype: String,
    pub cform: String,
    pub lemma: String,
    pub reading: String,
    pub pronunciation: String,
    pub accent_type: u8,
    pub mora_count: u8,
    pub chain_rule: String,
    pub chain_flag: i8, // -1: 未決定, 0: 接続しない, 1: 接続する
}

impl NjdNode {
    /// InputTokenからNjdNodeを構築する
    ///
    /// UniDic固有の品詞カテゴリについて、pos_detail1をIPAdic互換に変換する。
    /// - 接尾辞 → 名詞 + 接尾（detail1に"接尾"を設定）
    /// - 代名詞 → 名詞 + 代名詞（detail1に"代名詞"を設定）
    /// - 形状詞 → 名詞 + 形容動詞語幹（detail1に"形容動詞語幹"を設定）
    pub fn from_token(token: &InputToken) -> Self {
        let pos = Pos::parse(&token.pos);
        // pronunciationがカタカナでない場合（辞書の不備）はreadingにフォールバック
        let raw_pron = if is_katakana_str(&token.pronunciation) {
            &token.pronunciation
        } else {
            &token.reading
        };
        let pron = expand_long_vowels(raw_pron);
        let mora_count = mora::count_mora(&token.reading);

        // UniDic品詞をIPAdic互換のpos_detail1にマッピング
        let pos_detail1 = map_unidic_detail(&token.pos, &token.pos_detail1);

        Self {
            surface: token.surface.clone(),
            pos,
            pos_detail1,
            pos_detail2: token.pos_detail2.clone(),
            pos_detail3: token.pos_detail3.clone(),
            ctype: token.ctype.clone(),
            cform: token.cform.clone(),
            lemma: token.lemma.clone(),
            reading: token.reading.clone(),
            pronunciation: pron,
            accent_type: 0, // 後のフェーズで設定
            mora_count,
            chain_rule: String::new(),
            chain_flag: -1,
        }
    }
}

/// UniDic品詞をIPAdic互換のpos_detail1にマッピングする
///
/// UniDic固有のPOSカテゴリ（接尾辞、代名詞、形状詞）は
/// Pos::parseで名詞にマッピングされるため、detail1もIPAdic互換に変換する。
fn map_unidic_detail(pos_str: &str, detail1: &str) -> String {
    match pos_str {
        s if s.starts_with("接尾辞") => {
            // 接尾辞,名詞的 → 接尾
            // 接尾辞,形状詞的 → 接尾,形容動詞語幹
            // 接尾辞,動詞的 → 接尾
            // 接尾辞,形容詞的 → 接尾
            if detail1.contains("形状詞") {
                "接尾,形容動詞語幹".to_string()
            } else {
                "接尾".to_string()
            }
        }
        s if s.starts_with("代名詞") => "代名詞,一般".to_string(),
        s if s.starts_with("形状詞") => {
            // 形状詞,助動詞語幹 → 形容動詞語幹
            // 形状詞,一般 → 形容動詞語幹
            // 形状詞,タリ → 形容動詞語幹
            "形容動詞語幹".to_string()
        }
        s if s.starts_with("名詞") => {
            // UniDicの名詞,普通名詞 → IPAdic 一般
            // UniDicの名詞,数詞 → IPAdic 数
            match detail1 {
                "普通名詞" => "一般".to_string(),
                "数詞" => "数".to_string(),
                "助動詞語幹" => "形容動詞語幹".to_string(),
                _ => detail1.to_string(),
            }
        }
        s if s.starts_with("動詞") => {
            // UniDicの動詞,非自立可能 → IPAdic 非自立
            match detail1 {
                "非自立可能" => "非自立".to_string(),
                "一般" => "自立".to_string(),
                _ => detail1.to_string(),
            }
        }
        s if s.starts_with("形容詞") => {
            match detail1 {
                "非自立可能" => "非自立".to_string(),
                "一般" => "自立".to_string(),
                _ => detail1.to_string(),
            }
        }
        _ => detail1.to_string(),
    }
}

/// 文字列がカタカナ（長音記号ー含む）のみで構成されているかを判定する
fn is_katakana_str(s: &str) -> bool {
    !s.is_empty()
        && s.chars()
            .all(|c| ('\u{30A0}'..='\u{30FF}').contains(&c))
}

/// カタカナの長音記号をモーラ展開する
/// "コーヒー" → "コオヒイ"
pub fn expand_long_vowels(pron: &str) -> String {
    let chars: Vec<char> = pron.chars().collect();
    let mut result = String::with_capacity(pron.len());

    for (i, &ch) in chars.iter().enumerate() {
        if ch == 'ー' && i > 0 {
            // 直前の文字の母音を取得
            if let Some(vowel) = last_vowel_of_kana(chars[i - 1]) {
                result.push(vowel);
            } else {
                result.push(ch);
            }
        } else {
            result.push(ch);
        }
    }
    result
}

/// カタカナ文字の母音部分を返す
fn last_vowel_of_kana(c: char) -> Option<char> {
    // ア段=ア, イ段=イ, ウ段=ウ, エ段=エ, オ段=オ
    match c {
        'ア' | 'カ' | 'サ' | 'タ' | 'ナ' | 'ハ' | 'マ' | 'ヤ' | 'ラ' | 'ワ'
        | 'ガ' | 'ザ' | 'ダ' | 'バ' | 'パ' | 'ァ' | 'ャ' => Some('ア'),
        'イ' | 'キ' | 'シ' | 'チ' | 'ニ' | 'ヒ' | 'ミ' | 'リ'
        | 'ギ' | 'ジ' | 'ヂ' | 'ビ' | 'ピ' | 'ィ' => Some('イ'),
        'ウ' | 'ク' | 'ス' | 'ツ' | 'ヌ' | 'フ' | 'ム' | 'ユ' | 'ル'
        | 'グ' | 'ズ' | 'ヅ' | 'ブ' | 'プ' | 'ヴ' | 'ゥ' | 'ュ' => Some('ウ'),
        'エ' | 'ケ' | 'セ' | 'テ' | 'ネ' | 'ヘ' | 'メ' | 'レ'
        | 'ゲ' | 'ゼ' | 'デ' | 'ベ' | 'ペ' | 'ェ' => Some('エ'),
        'オ' | 'コ' | 'ソ' | 'ト' | 'ノ' | 'ホ' | 'モ' | 'ヨ' | 'ロ' | 'ヲ'
        | 'ゴ' | 'ゾ' | 'ド' | 'ボ' | 'ポ' | 'ォ' | 'ョ' => Some('オ'),
        _ => None,
    }
}

/// hasami::Token から InputToken への変換
impl From<hasami::Token> for InputToken {
    fn from(token: hasami::Token) -> Self {
        // 品詞情報をカンマで分割
        let pos_parts: Vec<&str> = token.pos.splitn(4, ',').collect();
        let pos = pos_parts.first().unwrap_or(&"*").to_string();
        let pos_detail1 = pos_parts.get(1).unwrap_or(&"*").to_string();
        let pos_detail2 = pos_parts.get(2).unwrap_or(&"*").to_string();
        let pos_detail3 = pos_parts.get(3).unwrap_or(&"*").to_string();

        Self {
            surface: token.surface.to_string(),
            pos,
            pos_detail1,
            pos_detail2,
            pos_detail3,
            ctype: "*".to_string(),
            cform: "*".to_string(),
            lemma: token.base_form.to_string(),
            reading: token.reading.to_string(),
            pronunciation: token.pronunciation.to_string(),
        }
    }
}

/// トークン列からNjdNode列を構築する
pub fn build_njd_nodes(tokens: &[InputToken]) -> Vec<NjdNode> {
    tokens.iter().map(NjdNode::from_token).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pos_from_str() {
        assert_eq!(Pos::parse("名詞"), Pos::Meishi);
        assert_eq!(Pos::parse("動詞"), Pos::Doushi);
        assert_eq!(Pos::parse("助詞"), Pos::Joshi);
        assert_eq!(Pos::parse("助動詞"), Pos::Jodoushi);
    }

    #[test]
    fn test_expand_long_vowels() {
        assert_eq!(expand_long_vowels("コーヒー"), "コオヒイ");
        assert_eq!(expand_long_vowels("トーキョー"), "トオキョオ");
        assert_eq!(expand_long_vowels("カタカナ"), "カタカナ"); // 長音なし
    }

    #[test]
    fn test_build_njd_node() {
        let token = InputToken::new("東京", "名詞", "トウキョウ", "トーキョー");
        let node = NjdNode::from_token(&token);
        assert_eq!(node.surface, "東京");
        assert_eq!(node.pos, Pos::Meishi);
        assert_eq!(node.pronunciation, "トオキョオ");
        assert_eq!(node.mora_count, 4); // ト・ウ・キョ・ウ
    }

    #[test]
    fn test_is_content_word() {
        assert!(Pos::Meishi.is_content_word());
        assert!(Pos::Doushi.is_content_word());
        assert!(!Pos::Joshi.is_content_word());
        assert!(!Pos::Jodoushi.is_content_word());
    }

    #[test]
    fn test_unidic_pos_mapping() {
        // UniDic固有のPOSがIPAdic相当にマッピングされることを確認
        assert_eq!(Pos::parse("接尾辞"), Pos::Meishi);
        assert_eq!(Pos::parse("代名詞"), Pos::Meishi);
        assert_eq!(Pos::parse("形状詞"), Pos::Meishi);
    }

    #[test]
    fn test_unidic_detail_mapping() {
        // 接尾辞のdetail1マッピング
        assert_eq!(map_unidic_detail("接尾辞", "名詞的"), "接尾");
        assert_eq!(map_unidic_detail("接尾辞", "形状詞的"), "接尾,形容動詞語幹");
        assert_eq!(map_unidic_detail("接尾辞", "動詞的"), "接尾");

        // 代名詞のdetail1マッピング
        assert_eq!(map_unidic_detail("代名詞", "*"), "代名詞,一般");

        // 形状詞のdetail1マッピング
        assert_eq!(map_unidic_detail("形状詞", "一般"), "形容動詞語幹");
        assert_eq!(map_unidic_detail("形状詞", "助動詞語幹"), "形容動詞語幹");

        // 名詞のdetail1マッピング
        assert_eq!(map_unidic_detail("名詞", "普通名詞"), "一般");
        assert_eq!(map_unidic_detail("名詞", "数詞"), "数");
        assert_eq!(map_unidic_detail("名詞", "助動詞語幹"), "形容動詞語幹");
        assert_eq!(map_unidic_detail("名詞", "固有名詞"), "固有名詞");

        // 動詞のdetail1マッピング
        assert_eq!(map_unidic_detail("動詞", "非自立可能"), "非自立");
        assert_eq!(map_unidic_detail("動詞", "一般"), "自立");

        // 形容詞のdetail1マッピング
        assert_eq!(map_unidic_detail("形容詞", "非自立可能"), "非自立");
        assert_eq!(map_unidic_detail("形容詞", "一般"), "自立");

        // IPAdic品詞はそのまま通過
        assert_eq!(map_unidic_detail("助詞", "格助詞"), "格助詞");
        assert_eq!(map_unidic_detail("助動詞", "*"), "*");
    }
}
