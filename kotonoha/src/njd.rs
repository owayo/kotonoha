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
            s if s.starts_with("接尾辞") => Pos::Meishi, // 接尾辞 → 名詞（接尾として扱う）
            s if s.starts_with("代名詞") => Pos::Meishi, // 代名詞 → 名詞
            s if s.starts_with("形状詞") => Pos::Meishi, // 形状詞 → 名詞（形容動詞語幹として扱う）
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
    pub fn new(surface: &str, pos: &str, reading: &str, pronunciation: &str) -> Self {
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
    pub mora_count: usize,
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
        let pron = expand_long_vowels(select_pronunciation_source(token, &pos));
        // 発音とモーラ数がずれるとアクセント句の構築が崩れるので、
        // reading ではなく実際に採用した発音から数える
        let mora_count = mora::count_mora(&pron);

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

/// 発音として採用する文字列を選ぶ。
///
/// 優先順位:
/// 1. 表層形をそのまま読むべきカタカナ語なら表層形
/// 2. 辞書の発音 (カタカナとして妥当な場合)
/// 3. 辞書の読み
fn select_pronunciation_source<'a>(token: &'a InputToken, pos: &Pos) -> &'a str {
    if should_use_surface_pronunciation(token, pos) {
        return &token.surface;
    }
    // pronunciationがカタカナでない場合（辞書の不備）はreadingにフォールバック
    if is_katakana_str(&token.pronunciation) {
        &token.pronunciation
    } else {
        &token.reading
    }
}

/// 表層形をそのまま発音として使うべきカタカナ語かどうかを判定する。
///
/// カタカナ表記は表音的なので、書かれた通りに読むのが原則。しかし辞書の発音
/// フィールドには「マネジメント→マネージメント」「チャネル→チャンネル」のような
/// 表記と食い違う値が多数入っており、そのまま使うと表記と違う音声になる。
///
/// 誤爆を避けるため、次の語は対象外とする:
/// - 名詞・副詞・感動詞以外 (カタカナ表記の助詞「ハ」→「ワ」等を巻き込まないため)
/// - 人名・組織・地域の固有名詞 (「キヤノン」→「キャノン」、「グローヴ」→「グローブ」の
///   ような意図的な表記や原音に基づく読みがあるため)
/// - 1 文字の語 (品詞判定をすり抜けた助詞への二重防御)
/// - 音素化できない文字を含む語
///
/// 「固有名詞,一般」は除外しない。NEologd が「ダイバーシティ」「ティール」
/// 「インタラクション」のような普通名詞をこの品詞で大量に登録しており、除外すると
/// 辞書の発音が使われて「ティール」が「チュール」になる。
fn should_use_surface_pronunciation(token: &InputToken, pos: &Pos) -> bool {
    if !matches!(pos, Pos::Meishi | Pos::Fukushi | Pos::Kandoushi) {
        return false;
    }
    if token.pos_detail1.contains("固有名詞") && token.pos_detail2 != "一般" {
        return false;
    }
    if token.surface.chars().count() < 2 {
        return false;
    }
    // 原形と読みが同じ標準カタカナ語へ正規化された表記ゆれは辞書の発音を使う。
    // 発音だけが異なるエントリや、語末の長音符だけの表記差では表層を優先する。
    if token.lemma != token.surface
        && token.reading == token.lemma
        && is_phonemizable_katakana(&token.lemma)
        && is_katakana_str(&token.pronunciation)
        && token.surface.trim_end_matches('ー') != token.lemma.trim_end_matches('ー')
    {
        return false;
    }
    is_phonemizable_katakana(&token.surface)
}

/// 全ての文字が音素に変換できるカタカナかどうかを判定する。
///
/// 「・」「ヽ」「ヰ」「ヱ」「ヵ」「ヶ」など音素化できない文字を含む語や、
/// 長音・小書き仮名・促音で始まる語は false を返す。
fn is_phonemizable_katakana(s: &str) -> bool {
    let Some(first) = s.chars().next() else {
        return false;
    };
    if matches!(
        first,
        'ー' | 'ァ' | 'ィ' | 'ゥ' | 'ェ' | 'ォ' | 'ャ' | 'ュ' | 'ョ' | 'ッ'
    ) {
        return false;
    }
    s.chars()
        .all(|c| matches!(c, 'ン' | 'ッ' | 'ー') || last_vowel_of_kana(c).is_some())
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
        s if s.starts_with("形容詞") => match detail1 {
            "非自立可能" => "非自立".to_string(),
            "一般" => "自立".to_string(),
            _ => detail1.to_string(),
        },
        _ => detail1.to_string(),
    }
}

/// 文字列がカタカナ（長音記号ー含む）のみで構成されているかを判定する
fn is_katakana_str(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| ('\u{30A0}'..='\u{30FF}').contains(&c))
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
        'ア' | 'カ' | 'サ' | 'タ' | 'ナ' | 'ハ' | 'マ' | 'ヤ' | 'ラ' | 'ワ' | 'ガ' | 'ザ'
        | 'ダ' | 'バ' | 'パ' | 'ァ' | 'ャ' => Some('ア'),
        'イ' | 'キ' | 'シ' | 'チ' | 'ニ' | 'ヒ' | 'ミ' | 'リ' | 'ギ' | 'ジ' | 'ヂ' | 'ビ'
        | 'ピ' | 'ィ' => Some('イ'),
        'ウ' | 'ク' | 'ス' | 'ツ' | 'ヌ' | 'フ' | 'ム' | 'ユ' | 'ル' | 'グ' | 'ズ' | 'ヅ'
        | 'ブ' | 'プ' | 'ヴ' | 'ゥ' | 'ュ' => Some('ウ'),
        'エ' | 'ケ' | 'セ' | 'テ' | 'ネ' | 'ヘ' | 'メ' | 'レ' | 'ゲ' | 'ゼ' | 'デ' | 'ベ'
        | 'ペ' | 'ェ' => Some('エ'),
        'オ' | 'コ' | 'ソ' | 'ト' | 'ノ' | 'ホ' | 'モ' | 'ヨ' | 'ロ' | 'ヲ' | 'ゴ' | 'ゾ'
        | 'ド' | 'ボ' | 'ポ' | 'ォ' | 'ョ' => Some('オ'),
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
            ctype: if token.conj_type.is_empty() {
                "*".to_string()
            } else {
                token.conj_type.to_string()
            },
            cform: if token.conj_form.is_empty() {
                "*".to_string()
            } else {
                token.conj_form.to_string()
            },
            lemma: token.base_form.to_string(),
            reading: token.reading.to_string(),
            pronunciation: token.pronunciation.to_string(),
        }
    }
}

/// トークン列からNjdNode列を構築する
pub fn build_njd_nodes(tokens: &[InputToken]) -> Vec<NjdNode> {
    let mut nodes: Vec<NjdNode> = tokens.iter().map(NjdNode::from_token).collect();
    apply_volitional_long_vowel(&mut nodes);
    nodes
}

/// 意志・推量の「う」を直前のオ段と融合させて長音にする。
///
/// 「作ろう」は tsukuroo、「〜しましょう」は shimashoo で、tsukurou / shimashou では
/// ない。辞書の発音は「ツクロウ」「マショ」+「ウ」のように仮名遣いのままなので、
/// ここで長音に直さないと「オ段 + ウ」の 2 モーラとして読まれて不自然になる。
///
/// 五段動詞の終止形の「う」は融合しない（「思う」は omou、「迷う」は mayou）。
/// 終止形は表層形と原形が一致するので、それを見分けに使う。
///
/// 「ましょ」+「う」のように助動詞「う」が独立したトークンになる場合と、
/// 「作ろう」「見よう」「だろう」のように 1 トークンになる場合の両方を扱う。
/// 1 トークンの表層形が意志形かどうかを判定する。
///
/// 辞書に活用形があれば、それを優先する。基本形の「映ろう」や、UniDic で原形が
/// 漢字になった「まよう（迷う）」を意志形と誤認しない。
/// 活用形が無いトークンでは表層形と原形の一致を手掛かりにする。ただし
/// NEologd には活用形をそのまま原形として登録したエントリ（「作ろう」の原形が
/// 「作ろう」）があり、それでは見分けられない。
///
/// そこで語幹の文字種も見る。五段動詞の終止形は語幹の直後に「う」が 1 文字だけ付く形
/// （思う、問う、背負う）で、平仮名の「オ段 + う」が漢字・カタカナに続く形にはならない。
/// 逆に「作ろう」「探そう」「ウケよう」は必ずこの形になる。
/// 語幹まで平仮名の動詞（つくろう、まよう、かよう）は終止形と区別できないので触らない。
///
/// 活用形の無い入力では「映ろう（うつろう）」のような基本形を見分けられない。
/// hasami の辞書にある活用形を保持することで、その入力では表記の推測を避ける。
///
/// 活用形が無い助動詞は語幹の文字種を見ずに意志形として扱う。「だろう」「でしょう」
/// 「たろう」「ましょう」はすべて推量・意志で、IPAdic で「オ段 + う」で終わる
/// 助動詞は他に「とう（たい）」「のう（ない）」だけ。どちらも発音が既に長音なので
/// この規則では何も変わらない。
fn is_volitional_surface(node: &NjdNode) -> bool {
    if node.cform != "*" && !node.cform.is_empty() {
        return is_volitional_cform(&node.cform);
    }
    let surface = &node.surface;
    let lemma = &node.lemma;
    let pos = &node.pos;
    if surface != lemma || matches!(pos, Pos::Jodoushi) {
        return true;
    }
    // 末尾が「〈漢字またはカタカナ〉+〈オ段の平仮名〉+ う」なら意志形
    let mut chars = surface.chars().rev();
    if chars.next() != Some('う') {
        return false;
    }
    let Some(stem_end) = chars.next() else {
        return false;
    };
    if !matches!(
        stem_end,
        'お' | 'こ'
            | 'そ'
            | 'と'
            | 'の'
            | 'ほ'
            | 'も'
            | 'よ'
            | 'ろ'
            | 'ご'
            | 'ぞ'
            | 'ど'
            | 'ぼ'
            | 'ぽ'
    ) {
        return false;
    }
    chars.next().is_some_and(|c| !matches!(c, 'ぁ'..='ゖ'))
}

/// IPAdic の未然ウ接続と UniDic の意志推量形を見分ける。
fn is_volitional_cform(cform: &str) -> bool {
    cform == "未然ウ接続" || cform.starts_with("意志推量形")
}

fn apply_volitional_long_vowel(nodes: &mut [NjdNode]) {
    for i in 0..nodes.len() {
        if !matches!(nodes[i].pos, Pos::Doushi | Pos::Jodoushi) {
            continue;
        }
        if !nodes[i].surface.ends_with('う') {
            continue;
        }
        // 独立した助動詞「う」は直前のトークンの末尾の母音を見る
        let preceding_vowel = if nodes[i].surface == "う" {
            if i == 0 || nodes[i].pos != Pos::Jodoushi {
                continue;
            }
            let prev = &nodes[i - 1];
            if !matches!(prev.pos, Pos::Doushi | Pos::Jodoushi)
                || (prev.cform != "*"
                    && !prev.cform.is_empty()
                    && !is_volitional_cform(&prev.cform))
            {
                continue;
            }
            nodes[i - 1]
                .pronunciation
                .chars()
                .next_back()
                .and_then(last_vowel_of_kana)
        } else {
            if !is_volitional_surface(&nodes[i]) {
                continue;
            }
            let mut chars = nodes[i].pronunciation.chars().rev();
            if chars.next() != Some('ウ') {
                continue;
            }
            chars.next().and_then(last_vowel_of_kana)
        };
        if preceding_vowel != Some('オ') {
            continue;
        }
        if !nodes[i].pronunciation.ends_with('ウ') {
            continue;
        }
        let len = nodes[i].pronunciation.len() - 'ウ'.len_utf8();
        nodes[i].pronunciation.truncate(len);
        nodes[i].pronunciation.push('オ');
    }
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
    fn test_surface_katakana_overrides_dictionary_pronunciation() {
        // 辞書の発音が表記と食い違うカタカナ語は、表記通りに読む
        let token = InputToken::new("マネジメント", "名詞", "マネジメント", "マネージメント");
        let node = NjdNode::from_token(&token);
        assert_eq!(node.pronunciation, "マネジメント");
        assert_eq!(node.mora_count, 6);

        let token = InputToken::new("チャネル", "名詞", "チャネル", "チャンネル");
        assert_eq!(NjdNode::from_token(&token).pronunciation, "チャネル");
    }

    #[test]
    fn test_surface_katakana_long_vowel_is_expanded() {
        // 表記を採用する場合も長音は展開する
        let token = InputToken::new("ユーザ", "名詞", "ユーザ", "ユーザー");
        let node = NjdNode::from_token(&token);
        assert_eq!(node.pronunciation, "ユウザ");
        assert_eq!(node.mora_count, 3);
    }

    #[test]
    fn test_katakana_particle_keeps_dictionary_pronunciation() {
        // カタカナ表記の助詞は表記通りに読むと誤る
        let token = InputToken::new("ハ", "助詞", "ワ", "ワ");
        assert_eq!(NjdNode::from_token(&token).pronunciation, "ワ");
    }

    #[test]
    fn test_proper_noun_keeps_dictionary_pronunciation() {
        // 「キヤノン」→「キャノン」のような意図的な表記は辞書を尊重する
        let mut token = InputToken::new("キヤノン", "名詞", "キヤノン", "キャノン");
        token.pos_detail1 = "固有名詞".to_string();
        token.pos_detail2 = "組織".to_string();
        assert_eq!(NjdNode::from_token(&token).pronunciation, "キャノン");

        // 人名の読みも原音に基づくので辞書を尊重する
        let mut token = InputToken::new("グローヴ", "名詞", "グローヴ", "グローブ");
        token.pos_detail1 = "固有名詞".to_string();
        token.pos_detail2 = "人名".to_string();
        assert_eq!(NjdNode::from_token(&token).pronunciation, "グロオブ");
    }

    #[test]
    fn test_general_proper_noun_uses_surface() {
        // NEologd は普通名詞も「固有名詞,一般」で登録している。ここまで辞書に
        // 委ねると「ティール」が「チュール」、「ダイバーシティ」が
        // 「ダイバーシティー」になるので、表記通りに読む
        let mut token = InputToken::new("ティール", "名詞", "ティール", "チュール");
        token.pos_detail1 = "固有名詞".to_string();
        token.pos_detail2 = "一般".to_string();
        assert_eq!(NjdNode::from_token(&token).pronunciation, "ティイル");

        let mut token = InputToken::new(
            "ダイバーシティ",
            "名詞",
            "ダイバーシティ",
            "ダイバーシティー",
        );
        token.pos_detail1 = "固有名詞".to_string();
        token.pos_detail2 = "一般".to_string();
        assert_eq!(NjdNode::from_token(&token).pronunciation, "ダイバアシティ");
    }

    #[test]
    fn test_non_katakana_surface_keeps_dictionary_pronunciation() {
        // 漢字表記は表層をそのまま読めないので辞書の発音を使う
        let token = InputToken::new("東京", "名詞", "トウキョウ", "トーキョー");
        assert_eq!(NjdNode::from_token(&token).pronunciation, "トオキョオ");
    }

    #[test]
    fn test_unphonemizable_katakana_keeps_dictionary_pronunciation() {
        // 中黒を含む語は音素化できないので辞書の発音を使う
        let token = InputToken::new(
            "ジョン・カバット・ジン",
            "名詞",
            "ジョンカバットジン",
            "ジョンカバットジン",
        );
        assert_eq!(
            NjdNode::from_token(&token).pronunciation,
            "ジョンカバットジン"
        );
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

    /// 原形を指定できる InputToken を作る
    fn token_with_lemma(
        surface: &str,
        pos: &str,
        lemma: &str,
        reading: &str,
        pronunciation: &str,
    ) -> InputToken {
        let mut token = InputToken::new(surface, pos, reading, pronunciation);
        token.lemma = lemma.to_string();
        token
    }

    fn pronunciations(tokens: &[InputToken]) -> Vec<String> {
        build_njd_nodes(tokens)
            .into_iter()
            .map(|node| node.pronunciation)
            .collect()
    }

    #[test]
    fn test_volitional_u_after_o_row_becomes_long_vowel() {
        // 「〜ましょう」は「ましょ」+「う」に分かれる。融合させないと mashou になる
        let tokens = vec![
            token_with_lemma("確認", "名詞,サ変接続", "確認", "カクニン", "カクニン"),
            token_with_lemma("し", "動詞,自立", "する", "シ", "シ"),
            token_with_lemma("ましょ", "助動詞", "ます", "マショ", "マショ"),
            token_with_lemma("う", "助動詞", "う", "ウ", "ウ"),
        ];
        assert_eq!(pronunciations(&tokens), ["カクニン", "シ", "マショ", "オ"]);
    }

    #[test]
    fn test_volitional_u_in_single_token_becomes_long_vowel() {
        // 「作ろう」「見よう」「だろう」は 1 トークン。原形と表層形が違う
        for (surface, pos, lemma, pron, expected) in [
            ("作ろう", "動詞,自立", "作る", "ツクロウ", "ツクロオ"),
            ("言おう", "動詞,自立", "言う", "イオウ", "イオオ"),
            ("見よう", "動詞,自立", "見る", "ミヨウ", "ミヨオ"),
            ("だろう", "助動詞", "だ", "ダロウ", "ダロオ"),
        ] {
            let tokens = vec![token_with_lemma(surface, pos, lemma, pron, pron)];
            assert_eq!(pronunciations(&tokens), [expected], "{surface}");
        }
    }

    #[test]
    fn test_volitional_with_broken_lemma_is_merged() {
        // NEologd には活用形をそのまま原形にしたエントリがある。
        // 語幹が漢字・カタカナなら、平仮名の「オ段 + う」は終止形にはならない
        for (surface, pron, expected) in [
            ("作ろう", "ツクロウ", "ツクロオ"),
            ("探そう", "サガソウ", "サガソオ"),
            ("雇おう", "ヤトオウ", "ヤトオオ"),
            ("ウケよう", "ウケヨウ", "ウケヨオ"),
        ] {
            let tokens = vec![token_with_lemma(surface, "動詞,自立", surface, pron, pron)];
            assert_eq!(pronunciations(&tokens), [expected], "{surface}");
        }
    }

    #[test]
    fn test_dictionary_cform_precedes_volitional_surface_heuristics() {
        for (surface, lemma, cform, pron, expected) in [
            ("映ろう", "映ろう", "基本形", "ウツロウ", "ウツロウ"),
            ("まよう", "迷う", "終止形-一般", "マヨウ", "マヨウ"),
            ("まよう", "迷う", "連体形-一般", "マヨウ", "マヨウ"),
            ("つくろう", "つくろう", "未然ウ接続", "ツクロウ", "ツクロオ"),
            ("つくろう", "つくろう", "意志推量形", "ツクロウ", "ツクロオ"),
            ("みよう", "見る", "意志推量形-一般", "ミヨウ", "ミヨオ"),
        ] {
            let mut token = token_with_lemma(surface, "動詞", lemma, pron, pron);
            token.cform = cform.to_string();
            let nodes = build_njd_nodes(&[token]);
            assert_eq!(nodes[0].pronunciation, expected, "{surface}: {cform}");
            assert_eq!(nodes[0].mora_count, mora::count_mora(expected));
        }
    }

    #[test]
    fn test_separate_u_requires_a_volitional_preceding_word() {
        for (pos, cform, expected) in [
            ("動詞", "未然ウ接続", "オ"),
            ("動詞", "意志推量形-一般", "オ"),
            ("動詞", "基本形", "ウ"),
            ("名詞", "*", "ウ"),
        ] {
            let mut prev = token_with_lemma("つくろ", pos, "つくる", "ツクロ", "ツクロ");
            prev.cform = cform.to_string();
            let tokens = [prev, InputToken::new("う", "助動詞", "ウ", "ウ")];
            assert_eq!(
                pronunciations(&tokens),
                ["ツクロ", expected],
                "{pos}: {cform}"
            );
        }
    }

    #[test]
    fn test_auxiliary_verb_is_always_volitional() {
        // 「だろう」「たろう」は原形が壊れていても推量。語幹が平仮名でも長音にする
        for (surface, pron, expected) in [
            ("だろう", "ダロウ", "ダロオ"),
            ("たろう", "タロウ", "タロオ"),
            ("でしょう", "デショウ", "デショオ"),
        ] {
            let tokens = vec![token_with_lemma(surface, "助動詞", surface, pron, pron)];
            assert_eq!(pronunciations(&tokens), [expected], "{surface}");
        }
    }

    #[test]
    fn test_auxiliary_verb_with_long_vowel_pronunciation_is_untouched() {
        // 「とう(たい)」「のう(ない)」は発音が既に長音なので何も変わらない
        // (期待値は expand_long_vowels() が長音記号を母音に展開したあとの形)
        for (surface, pron, expected) in [("とう", "トー", "トオ"), ("のう", "ノー", "ノオ")]
        {
            let tokens = vec![token_with_lemma(surface, "助動詞", "たい", pron, pron)];
            assert_eq!(pronunciations(&tokens), [expected], "{surface}");
        }
    }

    #[test]
    fn test_all_kana_verb_is_not_merged() {
        // 語幹まで平仮名の動詞は終止形と区別できないので触らない
        // (つくろう=繕う、まよう=迷う、かよう=通う はいずれも終止形)
        for (surface, pron) in [("つくろう", "ツクロウ"), ("かよう", "カヨウ")] {
            let tokens = vec![token_with_lemma(surface, "動詞,自立", surface, pron, pron)];
            assert_eq!(pronunciations(&tokens), [pron], "{surface}");
        }
    }

    #[test]
    fn test_dictionary_form_u_is_not_merged() {
        // 五段動詞の終止形の「う」は融合しない (思う=omou、迷う=mayou)
        for (surface, lemma, pron) in [
            ("思う", "思う", "オモウ"),
            ("迷う", "迷う", "マヨウ"),
            ("問う", "問う", "トウ"),
        ] {
            let tokens = vec![token_with_lemma(surface, "動詞,自立", lemma, pron, pron)];
            assert_eq!(pronunciations(&tokens), [pron], "{surface}");
        }
    }

    #[test]
    fn test_u_after_non_o_row_is_not_merged() {
        // オ段以外の後ろの「う」は長音にならない (「言う」は iu)
        let tokens = vec![
            token_with_lemma("言", "動詞,自立", "言う", "イ", "イ"),
            token_with_lemma("う", "助動詞", "う", "ウ", "ウ"),
        ];
        assert_eq!(pronunciations(&tokens), ["イ", "ウ"]);
    }

    #[test]
    fn test_noun_ending_with_u_is_not_merged() {
        // 名詞の「オ段 + ウ」には当てない (「方法」は辞書側で長音化される)
        let tokens = vec![token_with_lemma(
            "ノウハウ",
            "名詞,一般",
            "ノウハウ",
            "ノウハウ",
            "ノウハウ",
        )];
        assert_eq!(pronunciations(&tokens), ["ノウハウ"]);
    }
}
