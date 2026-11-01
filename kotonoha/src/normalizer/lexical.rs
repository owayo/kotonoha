//! 選択済み英字語の辞書内合意を調べる。英語の語義や発音規則は追加しない。

use std::collections::HashMap;

use hasami::{Analyzer, DictEntry, DictError, Dictionary};

use crate::phoneme::{is_vowel, katakana_to_phonemes, single_kana_phonemes};

/// 完全に処理できる発音だけを、既存の音素処理へ渡す。
pub(super) fn pronunciation_phonemes(input: &str) -> Option<Vec<String>> {
    if input.is_empty() {
        return None;
    }
    let mut preceding_vowel = false;
    for character in input.chars() {
        if character == 'ー' {
            if !preceding_vowel {
                return None;
            }
        } else {
            let phones = single_kana_phonemes(character);
            preceding_vowel = is_vowel(phones.last()?);
        }
    }
    let phones = katakana_to_phonemes(input);
    (!phones.is_empty()).then_some(phones)
}

fn lexical_surface(input: &str) -> bool {
    // 1～2字の裸の英字は略語/字母読みの既存経路へ残す。
    // 記号や空白を含む既知wholeはこの短字数判定へ混ぜない。
    if input.len() <= 2 && input.bytes().all(|byte| byte.is_ascii_alphabetic()) {
        return false;
    }
    let mut previous_separator = true;
    let mut letters = 0;
    for byte in input.bytes() {
        if byte.is_ascii_alphabetic() {
            letters += 1;
            previous_separator = false;
        } else if matches!(byte, b' ' | b'&') && !previous_separator {
            previous_separator = true;
        } else {
            return false;
        }
    }
    letters >= 2 && !previous_separator
}

fn nominal_metadata(
    pos: &str,
    conj_type: &str,
    conj_form: &str,
    base: &str,
    surface: &str,
) -> bool {
    if !matches!(conj_type, "" | "*") || !matches!(conj_form, "" | "*") || base != surface {
        return false;
    }
    let parts: Vec<_> = pos.split(',').collect();
    matches!(
        parts.as_slice(),
        ["名詞", "一般" | "サ変接続", "*", "*"] | ["名詞", "固有名詞", "一般" | "組織", "*"]
    )
}

fn entry_phones(entry: &DictEntry, surface: &str) -> Option<Vec<String>> {
    if entry.surface.as_ref() != surface
        || !nominal_metadata(
            &entry.pos,
            &entry.conj_type,
            &entry.conj_form,
            &entry.base_form,
            surface,
        )
    {
        return None;
    }
    // 読みの欠落や特殊文字を、健全な別候補だけで隠さない。
    pronunciation_phonemes(&entry.reading)?;
    pronunciation_phonemes(&entry.pronunciation)
}

fn case_variants(surface: &str) -> Vec<String> {
    let lower = surface.to_ascii_lowercase();
    let mut initial = true;
    let title: String = lower
        .bytes()
        .map(|byte| {
            let result = if initial {
                byte.to_ascii_uppercase()
            } else {
                byte
            };
            initial = matches!(byte, b' ' | b'&');
            char::from(result)
        })
        .collect();
    let mut variants = vec![
        surface.to_owned(),
        lower,
        surface.to_ascii_uppercase(),
        title,
    ];
    variants.sort_unstable();
    variants.dedup();
    variants
}

fn dictionary_consensus(
    dictionary: &Dictionary,
    surface: &str,
) -> Result<Option<Vec<String>>, DictError> {
    let mut consensus = None;
    let mut exact_seen = false;
    for variant in case_variants(surface) {
        for (length, entries) in dictionary.lookup(&variant)? {
            if length != variant.len() {
                continue;
            }
            for entry in entries {
                exact_seen |= variant == surface;
                let Some(phones) = entry_phones(&entry, &variant) else {
                    return Ok(None);
                };
                if let Some(previous) = &consensus {
                    if previous != &phones {
                        return Ok(None);
                    }
                } else {
                    consensus = Some(phones);
                }
            }
        }
    }
    Ok(exact_seen.then_some(consensus).flatten())
}

fn lexical_boundary(input: &str, start: usize, end: usize) -> bool {
    let surface = &input[start..end];
    let forbidden = |character: char| {
        character.is_ascii_alphanumeric()
            || character == '_'
            || character.is_ascii_punctuation() && surface.contains(character)
    };
    let before = &input[..start];
    let after = &input[end..];
    if before.chars().next_back().is_some_and(forbidden)
        || after.chars().next().is_some_and(forbidden)
    {
        return false;
    }
    // path・ドメイン・ハイフン等の連結を、語単位の候補へ昇格しない。
    // 括弧、引用符、comma、文末記号だけは通常の区切りとして残す。
    let ordinary_delimiter = |character: char| {
        matches!(
            character,
            '(' | ')' | '[' | ']' | '{' | '}' | ',' | '!' | '?' | '\'' | '"'
        )
    };
    if before
        .chars()
        .next_back()
        .is_some_and(|character| character.is_ascii_punctuation() && !ordinary_delimiter(character))
    {
        return false;
    }
    if let Some(character) = after.chars().next() {
        if character == '.' {
            if after
                .trim_start_matches('.')
                .chars()
                .next()
                .is_some_and(|next| next.is_ascii_alphanumeric() || next == '_')
            {
                return false;
            }
        } else if character.is_ascii_punctuation() && !ordinary_delimiter(character) {
            return false;
        }
    }
    // 英語の所有格等を、引用符の単純な境界と取り違えない。
    if before.ends_with('\'')
        && before[..before.len() - 1]
            .chars()
            .next_back()
            .is_some_and(|character| character.is_ascii_alphanumeric())
        || after.starts_with('\'')
            && after[1..]
                .chars()
                .next()
                .is_some_and(|character| character.is_ascii_alphanumeric())
    {
        return false;
    }
    // 空白を挟んだ数量/版等を、単位だけの英語語彙へ昇格しない。
    let numeric_or_underscore = |character: char| character.is_ascii_digit() || character == '_';
    !before
        .trim_end_matches([' ', '\t'])
        .chars()
        .next_back()
        .is_some_and(numeric_or_underscore)
        && !after
            .trim_start_matches([' ', '\t'])
            .chars()
            .next()
            .is_some_and(numeric_or_underscore)
}

/// 全文の選択を保ち、語の内部や曖昧候補は辞退する。
pub(super) fn collect(
    analyzer: &mut Analyzer,
    input: &str,
) -> Result<Vec<(usize, usize, String)>, DictError> {
    if !input.bytes().any(|byte| byte.is_ascii_alphabetic()) {
        return Ok(Vec::new());
    }
    let tokens = analyzer.try_tokenize(input)?;
    let mut cache = HashMap::new();
    let mut output = Vec::new();
    for token in tokens {
        if !token.is_known
            || !lexical_surface(&token.surface)
            || !nominal_metadata(
                &token.pos,
                &token.conj_type,
                &token.conj_form,
                &token.base_form,
                &token.surface,
            )
            || !lexical_boundary(input, token.start, token.end)
        {
            continue;
        }
        let Some(selected) = pronunciation_phonemes(&token.pronunciation) else {
            continue;
        };
        let consensus = if let Some(found) = cache.get(token.surface.as_ref()) {
            found
        } else {
            let found = dictionary_consensus(analyzer.dictionary(), &token.surface)?;
            cache.entry(token.surface.to_string()).or_insert(found)
        };
        if consensus.as_ref() == Some(&selected) {
            output.push((token.start, token.end, token.pronunciation.to_string()));
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry() -> DictEntry {
        DictEntry {
            surface: "River Cloud".into(),
            pos: "名詞,固有名詞,一般,*".into(),
            conj_type: "*".into(),
            conj_form: "*".into(),
            base_form: "River Cloud".into(),
            reading: "リバークラウド".into(),
            pronunciation: "リバークラウド".into(),
            ..DictEntry::default()
        }
    }

    #[test]
    fn whole_ascii_grammar() {
        for value in ["River Cloud", "R&D", "Software as a Service", "Stripe"] {
            assert!(lexical_surface(value), "{value}");
        }
        for value in [
            "A",
            " River",
            "River ",
            "River  Cloud",
            "R&&D",
            "R & D",
            "River2",
            "SIP4D",
            "3mg/dL",
            "River_Cloud",
            "River-Cloud",
            "（株）",
        ] {
            assert!(!lexical_surface(value), "{value}");
        }
    }

    #[test]
    fn whole_boundaries() {
        for (value, start, end) in [
            ("River Cloudを確認", 0, 11),
            ("前River Cloud後", 3, 14),
            ("(River Cloud)", 1, 12),
        ] {
            assert!(lexical_boundary(value, start, end), "{value}");
        }
        for (value, start, end) in [
            ("xRiver Cloud", 1, 12),
            ("River Cloud2", 0, 11),
            ("River CloudX", 0, 11),
            ("_River Cloud", 1, 12),
            ("2 River Cloud", 2, 13),
            ("River Cloud 2", 0, 11),
            ("R&D&D", 0, 3),
        ] {
            assert!(!lexical_boundary(value, start, end), "{value}");
        }
    }

    #[test]
    fn custom_metadata_is_not_inferred() {
        let reference = entry();
        assert!(entry_phones(&reference, "River Cloud").is_some());
        for pos in [
            "名詞,固有名詞,人名,一般",
            "名詞,固有名詞,地域,一般",
            "名詞,接尾,助数詞,*",
            "動詞,自立,*,*",
        ] {
            let mut changed = reference.clone();
            changed.pos = pos.into();
            assert!(entry_phones(&changed, "River Cloud").is_none());
        }
        let mut changed = reference.clone();
        changed.base_form = "Other Alias".into();
        assert!(entry_phones(&changed, "River Cloud").is_none());
        changed = reference.clone();
        changed.conj_type = "特殊".into();
        assert!(entry_phones(&changed, "River Cloud").is_none());
        changed = reference.clone();
        changed.conj_form = "連用形".into();
        assert!(entry_phones(&changed, "River Cloud").is_none());
        changed = reference.clone();
        changed.pronunciation = "".into();
        assert!(entry_phones(&changed, "River Cloud").is_none());
        changed = reference.clone();
        changed.reading = "River Cloud".into();
        assert!(entry_phones(&changed, "River Cloud").is_none());
        changed = reference;
        changed.base_form = "river cloud".into();
        assert!(entry_phones(&changed, "River Cloud").is_none());
    }

    #[test]
    fn no_silent_loss_or_vowel_length_collapse() {
        for value in ["", "ーア", "ンー", "ッー", "abc", "ア〇", "アヺ"] {
            assert!(pronunciation_phonemes(value).is_none(), "{value}");
        }
        assert_eq!(
            pronunciation_phonemes("アール"),
            pronunciation_phonemes("アアル")
        );
        assert_ne!(
            pronunciation_phonemes("アサナ"),
            pronunciation_phonemes("アーサナ")
        );
        assert_ne!(
            pronunciation_phonemes("ストライプ"),
            pronunciation_phonemes("ストゥライプ")
        );
    }

    #[test]
    fn finite_case_variants_not_component_lookup() {
        assert_eq!(
            case_variants("River Cloud"),
            vec!["RIVER CLOUD", "River Cloud", "river cloud"]
        );
        assert_eq!(
            case_variants("Software as a Service"),
            vec![
                "SOFTWARE AS A SERVICE",
                "Software As A Service",
                "Software as a Service",
                "software as a service"
            ]
        );
        assert_eq!(case_variants("R&D"), vec!["R&D", "r&d"]);
    }

    #[test]
    fn punctuation_is_not_automatically_a_word_boundary() {
        for text in [
            "path/Stripe",
            "host.Stripe",
            "Stripe-Code",
            "Stripe&Code",
            "Stripe.Code",
            "Stripe..Code",
            "Stripe\\Code",
            "Stripe:Code",
            "Stripe@Code",
            "Stripe's",
        ] {
            let start = text.find("Stripe").unwrap();
            assert!(!lexical_boundary(text, start, start + 6), "{text}");
        }
        for text in [
            "(Stripe)",
            "Stripe,",
            "Stripe.",
            "Stripe..",
            "Stripe!",
            "'Stripe'",
            "Stripe.次",
            "前Stripe後",
        ] {
            let start = text.find("Stripe").unwrap();
            assert!(lexical_boundary(text, start, start + 6), "{text}");
        }
    }

    #[test]
    fn linear_validation_preserves_supported_kana_and_long_vowels() {
        for text in [
            "アーー",
            "キェー",
            "トゥー",
            "フィー",
            "ビィー",
            "ヵー",
            "ヶー",
            "ヴォー",
            "ヰー",
            "ヱー",
        ] {
            assert_eq!(
                pronunciation_phonemes(text),
                Some(katakana_to_phonemes(text)),
                "{text}"
            );
        }
        for text in [
            "ー",
            "ーー",
            "ンー",
            "ッー",
            "アッー",
            "アヾ",
            "ア・",
            "アヺ",
        ] {
            assert!(pronunciation_phonemes(text).is_none(), "{text}");
        }
    }

    #[test]
    fn short_bare_letter_sequences_stay_with_existing_abbreviation_route() {
        for text in ["de", "De", "DE", "GO", "AI", "IT", "Ox", "R"] {
            assert!(!lexical_surface(text), "{text}");
        }
        for text in ["R&D", "JRE GO", "R D", "River"] {
            assert!(lexical_surface(text), "{text}");
        }
    }
}
