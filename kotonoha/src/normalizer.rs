//! 辞書にある記号付きの語を、破壊的な記号変換より前に発音表記へ写す。

use std::path::Path;

use hasami::{Analyzer, DictError, Dictionary, Token};

/// 数字で始まる語・数式・識別子を残し、既知の記号付き語だけを発音表記へ写す。
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

    /// 既知語の発音だけを置換し、それ以外の入力と空白を保持する。
    ///
    /// 英字・下線の続く識別子内部や同じ記号の連続には当てない。
    /// 規格番号のように語の直後へ数字が続く形は許容する。
    pub fn normalize(&mut self, text: &str) -> Result<String, DictError> {
        if !text.chars().any(|character| {
            self.symbols.contains(character) || matches!(character, '(' | '（' | '[' | '［')
        }) {
            return Ok(text.to_owned());
        }
        let tokens = self.analyzer.try_tokenize(text)?;
        let mut output = String::with_capacity(text.len());
        let mut copied_end = 0;
        for token in &tokens {
            let Some(pronunciation) = symbol_pronunciation(text, token, &self.symbols) else {
                continue;
            };
            output.push_str(&text[copied_end..token.start]);
            if let Some((opening, closing)) = outer_parentheses(&token.surface) {
                output.push(opening);
                output.push_str(pronunciation);
                output.push(closing);
            } else {
                output.push_str(pronunciation);
            }
            copied_end = token.end;
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
