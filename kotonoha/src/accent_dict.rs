//! アクセント辞書モジュール
//! 見出し語からアクセント型を引くためのHashMap辞書を提供する
//!
//! CSVファイルから辞書をロードし、NjdNode構築時にアクセント型を設定するために使用する。
//! フォーマット: lemma,reading,accent_type

use std::collections::HashMap;
use std::path::Path;

/// アクセント辞書エントリ
#[derive(Debug, Clone)]
pub struct AccentDictEntry {
    /// カタカナ読み
    pub reading: String,
    /// アクセント型（0=平板）
    pub accent_type: u8,
}

/// アクセント辞書
#[derive(Debug, Clone)]
pub struct AccentDict {
    /// 見出し語 → エントリのマップ
    entries: HashMap<String, Vec<AccentDictEntry>>,
}

impl AccentDict {
    /// 空の辞書を作成する
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// CSVファイルから辞書を読み込む
    pub fn from_csv(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let mut dict = Self::new();

        let mut reader = csv::ReaderBuilder::new()
            .has_headers(true)
            .comment(Some(b'#'))
            .from_path(path)?;

        for result in reader.records() {
            let record = result?;
            if record.len() < 3 {
                continue;
            }
            let lemma = record[0].trim().to_string();
            let reading = record[1].trim().to_string();
            let accent_type: u8 = record[2].trim().parse().unwrap_or(0);

            dict.entries
                .entry(lemma)
                .or_default()
                .push(AccentDictEntry {
                    reading,
                    accent_type,
                });
        }

        Ok(dict)
    }

    /// 見出し語からアクセント型を検索する
    ///
    /// 読みが一致するエントリを優先する。読みが指定されない場合は最初のエントリを返す。
    pub fn lookup(&self, lemma: &str, reading: Option<&str>) -> Option<u8> {
        let entries = self.entries.get(lemma)?;

        if let Some(reading) = reading {
            // 読みが一致するエントリを優先
            if let Some(entry) = entries.iter().find(|e| e.reading == reading) {
                return Some(entry.accent_type);
            }
        }

        // 読みが一致しない場合、最初のエントリを返す
        entries.first().map(|e| e.accent_type)
    }

    /// エントリを追加する
    pub fn insert(&mut self, lemma: &str, reading: &str, accent_type: u8) {
        self.entries
            .entry(lemma.to_string())
            .or_default()
            .push(AccentDictEntry {
                reading: reading.to_string(),
                accent_type,
            });
    }

    /// 辞書のエントリ数を返す
    pub fn len(&self) -> usize {
        self.entries.values().map(|v| v.len()).sum()
    }

    /// 辞書が空かどうかを返す
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl Default for AccentDict {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_dict() {
        let dict = AccentDict::new();
        assert!(dict.is_empty());
        assert_eq!(dict.len(), 0);
        assert_eq!(dict.lookup("猫", None), None);
    }

    #[test]
    fn test_insert_and_lookup() {
        let mut dict = AccentDict::new();
        dict.insert("猫", "ネコ", 1);
        dict.insert("犬", "イヌ", 2);

        assert_eq!(dict.lookup("猫", None), Some(1));
        assert_eq!(dict.lookup("犬", None), Some(2));
        assert_eq!(dict.lookup("鳥", None), None);
    }

    #[test]
    fn test_lookup_with_reading() {
        let mut dict = AccentDict::new();
        dict.insert("本", "ホン", 1);
        dict.insert("本", "モト", 2);

        // 読み一致を優先
        assert_eq!(dict.lookup("本", Some("モト")), Some(2));
        assert_eq!(dict.lookup("本", Some("ホン")), Some(1));
        // 読みなしは最初のエントリ
        assert_eq!(dict.lookup("本", None), Some(1));
    }

    #[test]
    fn test_from_csv() {
        let csv_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("data/accent_dict.csv");

        if csv_path.exists() {
            let dict = AccentDict::from_csv(&csv_path).unwrap();
            assert!(!dict.is_empty());
            // 辞書に猫があるはず
            assert!(dict.lookup("猫", Some("ネコ")).is_some());
        }
    }

    #[test]
    fn test_default() {
        let dict = AccentDict::default();
        assert!(dict.is_empty());
    }
}
