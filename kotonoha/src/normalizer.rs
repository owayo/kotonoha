//! 辞書の記号付き語と解析器の文脈補正を正規化し、発音候補を照会する。
//!
//! 有限な表記・素性の合意だけを調べ、語義一意やアクセント同一は保証しない。

mod lexical;

use std::path::Path;

use hasami::{Analyzer, DictError, Dictionary, Token};

/// 既知の記号付き語と、解析器が実際に文脈補正した範囲だけを発音表記へ写す。
///
/// 読みは辞書から取得する。数字・字幅・句読点の正規化は行わず、
/// 括弧を含まない原形へ正規化された記号注記では、外側の括弧を保持する。
/// ONNX モデルは不要。
pub struct DictionaryNormalizer {
    analyzer: Analyzer,
    symbols: String,
}

impl DictionaryNormalizer {
    /// `.hsd` 辞書を読み込む。
    pub fn load(path: impl AsRef<Path>, symbols: &str) -> Result<Self, DictError> {
        Ok(Self {
            analyzer: Analyzer::load(path)?,
            symbols: symbols.to_owned(),
        })
    }

    /// メモリ上の辞書を使う。
    pub fn from_dictionary(dictionary: Dictionary, symbols: &str) -> Self {
        Self {
            analyzer: Analyzer::from_dict(dictionary),
            symbols: symbols.to_owned(),
        }
    }

    /// 明示された単一数量の発音を共有解析器へ委譲する。モデルは読み込まない。
    pub fn quantity_pronunciation(&mut self, input: &str) -> Result<Option<String>, DictError> {
        self.analyzer.quantity_pronunciation(input)
    }

    /// 全文で認可された数量のUTF-8バイト範囲と発音を返す。モデルは読み込まない。
    pub fn quantity_spans(&mut self, input: &str) -> Result<Vec<hasami::QuantitySpan>, DictError> {
        self.analyzer.quantity_spans(input)
    }

    /// 全文で選択された既知トークンのUTF-8バイト範囲。モデルを読み込まない。
    pub fn known_word_spans(
        &mut self,
        input: &str,
    ) -> Result<Vec<std::ops::Range<usize>>, DictError> {
        self.analyzer.known_word_spans(input)
    }

    /// 全文の選択済みトークンの開始・終了・発音を返す。位置はUTF-8バイト。
    ///
    /// 未知語と空発音もそのまま含め、発音の正しさは判定しない。モデルは読み込まず、
    /// 入力や通常のトークンを書き換えない。前処理による範囲外の読みの変化を照合できる。
    pub fn token_pronunciations(
        &mut self,
        input: &str,
    ) -> Result<Vec<(usize, usize, String)>, DictError> {
        Ok(self
            .analyzer
            .try_tokenize(input)?
            .into_iter()
            .map(|token| (token.start, token.end, token.pronunciation.to_string()))
            .collect())
    }

    /// 全文で選択された、辞書発音が合意する英字語・固定句のUTF-8範囲と発音。
    ///
    /// 数字・識別子内部・活用語・異音候補等は辞退する。語義の正しさは判定せず、
    /// 入力や通常のトークンを書き換えない。モデルは読み込まない。
    pub fn lexical_pronunciation_spans(
        &mut self,
        input: &str,
    ) -> Result<Vec<(usize, usize, String)>, DictError> {
        lexical::collect(&mut self.analyzer, input)
    }

    /// 完全なカタカナ発音だけを既存の共通処理で音素へ写す。
    ///
    /// 空・未対応文字・発音できない長音記号を含む場合はNone。意味の正解や
    /// アクセントは判定せず、未知文字を消した結果を同音確認には使わない。
    pub fn pronunciation_phonemes(pronunciation: &str) -> Option<Vec<String>> {
        lexical::pronunciation_phonemes(pronunciation)
    }

    /// ASCII識別子の既知全体は原文、限定した未知ID・郵便電話形は字母・桁読みで返す。
    /// Someを再び数量変換せず保持し、Noneだけ呼出側の従来変換へ戻す。
    pub fn normalize_identifier(&mut self, input: &str) -> Result<Option<String>, DictError> {
        self.analyzer.normalize_identifier(input)
    }

    /// 記号付き既知語と共有解析器が実際に補正した発音だけを置換する。
    ///
    /// 英字・下線の続く識別子内部や同じ記号の連続には当てない。
    /// 規格番号のように語の直後へ数字が続く形は許容する。
    /// 他の入力と空白を保持し、語義や発音を独自に判断しない。
    /// 置換範囲はすべて同じ原文のUTF-8バイト位置。文脈補正が記号付き語の置換と
    /// 重なる場合は従来の記号付き語を優先し、置換後の仮名を再解析して読みを選ばない。
    pub fn normalize(&mut self, text: &str) -> Result<String, DictError> {
        let contextual = self.analyzer.contextual_reading_spans(text)?;
        let has_symbol = text.chars().any(|character| {
            self.symbols.contains(character) || matches!(character, '(' | '（' | '[' | '［')
        });
        if !has_symbol && contextual.is_empty() {
            return Ok(text.to_owned());
        }
        let mut replacements = Vec::new();
        if has_symbol {
            let tokens = self.analyzer.try_tokenize(text)?;
            for token in &tokens {
                let Some(pronunciation) = symbol_pronunciation(text, token, &self.symbols) else {
                    continue;
                };
                let pronunciation =
                    if let Some((opening, closing)) = outer_parentheses(&token.surface) {
                        format!("{opening}{pronunciation}{closing}")
                    } else {
                        pronunciation.to_owned()
                    };
                replacements.push((token.start, token.end, pronunciation));
            }
        }
        // 記号付き語の範囲は原文順。既存の置換を優先し、重なる文脈補正は加えない。
        let symbol_count = replacements.len();
        for span in contextual {
            let following =
                replacements[..symbol_count].partition_point(|(_, end, _)| *end <= span.start);
            if following < symbol_count && replacements[following].0 < span.end {
                continue;
            }
            replacements.push((span.start, span.end, span.pronunciation));
        }
        replacements.sort_by_key(|(start, _, _)| *start);
        let mut output = String::with_capacity(text.len());
        let mut copied_end = 0;
        for (start, end, pronunciation) in replacements {
            output.push_str(&text[copied_end..start]);
            output.push_str(&pronunciation);
            copied_end = end;
        }
        output.push_str(&text[copied_end..]);
        Ok(output)
    }
}

fn symbol_pronunciation<'a>(text: &str, token: &'a Token, symbols: &str) -> Option<&'a str> {
    let surface = &token.surface;
    if !token.is_known
        || surface.chars().next().is_some_and(char::is_numeric)
        || !surface.chars().any(char::is_alphabetic)
        || !(surface.chars().any(|character| symbols.contains(character))
            || token.pos.starts_with("記号,")
                && outer_parentheses(surface).is_some()
                && token.base_form != token.surface
                && outer_parentheses(&token.base_form).is_none())
    {
        return None;
    }
    if outer_parentheses(surface).is_none() {
        let previous = text[..token.start].chars().next_back();
        let following = text[token.end..].chars().next();
        if previous.is_some_and(|character| {
            character.is_ascii_alphanumeric()
                || character == '_'
                || character.is_ascii_punctuation() && surface.contains(character)
        }) || following.is_some_and(|character| {
            character.is_ascii_alphabetic()
                || character == '_'
                || character.is_ascii_punctuation() && surface.contains(character)
        }) {
            return None;
        }
    }
    [&*token.pronunciation, &*token.reading]
        .into_iter()
        .find(|reading| {
            !reading.is_empty()
                && reading
                    .chars()
                    .all(|character| matches!(character, 'ァ'..='ヺ' | 'ー'))
        })
}

fn outer_parentheses(surface: &str) -> Option<(char, char)> {
    let opening = surface.chars().next()?;
    let closing = match opening {
        '(' => ')',
        '（' => '）',
        '[' => ']',
        '［' => '］',
        _ => return None,
    };
    surface.ends_with(closing).then_some((opening, closing))
}
