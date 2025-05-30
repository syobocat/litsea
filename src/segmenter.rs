use crate::adaboost::AdaBoost;
use regex::Regex;
use std::collections::HashSet;

/// 形態素セグメンター
pub struct Segmenter {
    patterns: Vec<(Regex, &'static str)>,
    pub learner: AdaBoost,
}

impl Segmenter {
    /// 新規作成（学習済みモデル or 空で初期化）
    pub fn new(learner: Option<AdaBoost>) -> Self {
        let patterns = vec![
            (
                Regex::new(r"[一二三四五六七八九十百千万億兆]").unwrap(),
                "M",
            ),
            (Regex::new(r"[一-龠々〆ヵヶ]").unwrap(), "H"),
            (Regex::new(r"[ぁ-ん]").unwrap(), "I"),
            (Regex::new(r"[ァ-ヴーｱ-ﾝﾞｰ]").unwrap(), "K"),
            (Regex::new(r"[a-zA-Zａ-ｚＡ-Ｚ]").unwrap(), "A"),
            (Regex::new(r"[0-9０-９]").unwrap(), "N"),
        ];
        Segmenter {
            patterns,
            learner: learner.unwrap_or_else(|| AdaBoost::new(0.01, 100, 1)),
        }
    }

    /// 文字タイプ判定
    pub fn get_type(&self, ch: &str) -> &'static str {
        for (pattern, s_type) in &self.patterns {
            if pattern.is_match(ch) {
                return s_type;
            }
        }
        "O"
    }

    /// 特徴抽出だけする（writerでラベル＋特徴量を出力）
    pub fn add_sentence_with_writer<F>(&mut self, sentence: &str, mut writer: F)
    where
        F: FnMut(HashSet<String>, i8),
    {
        if sentence.is_empty() {
            return;
        }
        let mut tags = vec!["U".to_string(); 3];
        let mut chars = vec!["B3".to_string(), "B2".to_string(), "B1".to_string()];
        let mut types = vec!["O".to_string(); 3];

        for word in sentence.split(' ') {
            if word.is_empty() {
                continue;
            }
            tags.push("B".to_string());
            for _ in 1..word.chars().count() {
                tags.push("O".to_string());
            }
            for ch in word.chars() {
                let s = ch.to_string();
                chars.push(s.clone());
                types.push(self.get_type(&s).to_string());
            }
        }
        if tags.len() < 4 {
            return;
        }
        tags[3] = "U".to_string();

        chars.extend_from_slice(&["E1".into(), "E2".into(), "E3".into()]);
        types.extend_from_slice(&["O".into(), "O".into(), "O".into()]);

        for i in 4..(chars.len() - 3) {
            let label = if tags[i] == "B" { 1 } else { -1 };
            let attrs = self.get_attributes(i, &tags, &chars, &types);
            writer(attrs, label);
        }
    }

    /// コーパスから特徴量＋ラベルをAdaBoostに投入（学習用）
    pub fn add_sentence(&mut self, sentence: &str) {
        if sentence.is_empty() {
            return;
        }
        let mut tags = vec!["U".to_string(); 3];
        let mut chars = vec!["B3".to_string(), "B2".to_string(), "B1".to_string()];
        let mut types = vec!["O".to_string(); 3];

        for word in sentence.split(' ') {
            if word.is_empty() {
                continue;
            }
            tags.push("B".to_string());
            for _ in 1..word.chars().count() {
                tags.push("O".to_string());
            }
            for ch in word.chars() {
                let s = ch.to_string();
                chars.push(s.clone());
                types.push(self.get_type(&s).to_string());
            }
        }
        if tags.len() < 4 {
            return;
        }
        tags[3] = "U".to_string();

        chars.extend_from_slice(&["E1".into(), "E2".into(), "E3".into()]);
        types.extend_from_slice(&["O".into(), "O".into(), "O".into()]);

        for i in 4..(chars.len() - 3) {
            let label = if tags[i] == "B" { 1 } else { -1 };
            let attrs = self.get_attributes(i, &tags, &chars, &types);
            // ★ ここで毎回 self.learner を呼ぶことで借用がぶつからない！
            self.learner.add_instance(attrs, label);
        }
    }

    /// 推論
    pub fn parse(&self, sentence: &str) -> Vec<String> {
        if sentence.is_empty() {
            return Vec::new();
        }
        let learner = &self.learner;
        let mut tags = vec!["U".to_string(); 4];
        let mut chars = vec!["B3".to_string(), "B2".to_string(), "B1".to_string()];
        let mut types = vec!["O".to_string(); 3];

        for ch in sentence.chars() {
            let s = ch.to_string();
            chars.push(s.clone());
            types.push(self.get_type(&s).to_string());
        }
        chars.extend_from_slice(&["E1".into(), "E2".into(), "E3".into()]);
        types.extend_from_slice(&["O".into(), "O".into(), "O".into()]);

        let mut result = Vec::new();
        let mut word = chars[3].clone();
        for i in 4..(chars.len() - 3) {
            let label = learner.predict(self.get_attributes(i, &tags, &chars, &types));
            if label >= 0 {
                result.push(word.clone());
                word.clear();
                tags.push("B".to_string());
            } else {
                tags.push("O".to_string());
            }
            word += &chars[i];
        }
        result.push(word);
        result
    }

    /// 属性集合（特徴量セット）を生成
    fn get_attributes(
        &self,
        i: usize,
        tags: &[String],
        chars: &[String],
        types: &[String],
    ) -> HashSet<String> {
        let w1 = &chars[i - 3];
        let w2 = &chars[i - 2];
        let w3 = &chars[i - 1];
        let w4 = &chars[i];
        let w5 = &chars[i + 1];
        let w6 = &chars[i + 2];
        let c1 = &types[i - 3];
        let c2 = &types[i - 2];
        let c3 = &types[i - 1];
        let c4 = &types[i];
        let c5 = &types[i + 1];
        let c6 = &types[i + 2];
        let p1 = &tags[i - 3];
        let p2 = &tags[i - 2];
        let p3 = &tags[i - 1];

        [
            format!("UP1:{}", p1),
            format!("UP2:{}", p2),
            format!("UP3:{}", p3),
            format!("BP1:{}{}", p1, p2),
            format!("BP2:{}{}", p2, p3),
            format!("UW1:{}", w1),
            format!("UW2:{}", w2),
            format!("UW3:{}", w3),
            format!("UW4:{}", w4),
            format!("UW5:{}", w5),
            format!("UW6:{}", w6),
            format!("BW1:{}{}", w2, w3),
            format!("BW2:{}{}", w3, w4),
            format!("BW3:{}{}", w4, w5),
            format!("TW1:{}{}{}", w1, w2, w3),
            format!("TW2:{}{}{}", w2, w3, w4),
            format!("TW3:{}{}{}", w3, w4, w5),
            format!("TW4:{}{}{}", w4, w5, w6),
            format!("UC1:{}", c1),
            format!("UC2:{}", c2),
            format!("UC3:{}", c3),
            format!("UC4:{}", c4),
            format!("UC5:{}", c5),
            format!("UC6:{}", c6),
            format!("BC1:{}{}", c2, c3),
            format!("BC2:{}{}", c3, c4),
            format!("BC3:{}{}", c4, c5),
            format!("TC1:{}{}{}", c1, c2, c3),
            format!("TC2:{}{}{}", c2, c3, c4),
            format!("TC3:{}{}{}", c3, c4, c5),
            format!("TC4:{}{}{}", c4, c5, c6),
            format!("UQ1:{}{}", p1, c1),
            format!("UQ2:{}{}", p2, c2),
            format!("UQ3:{}{}", p3, c3),
            format!("BQ1:{}{}{}", p2, c2, c3),
            format!("BQ2:{}{}{}", p2, c3, c4),
            format!("BQ3:{}{}{}", p3, c2, c3),
            format!("BQ4:{}{}{}", p3, c3, c4),
            format!("TQ1:{}{}{}{}", p2, c1, c2, c3),
            format!("TQ2:{}{}{}{}", p2, c2, c3, c4),
            format!("TQ3:{}{}{}{}", p3, c1, c2, c3),
            format!("TQ4:{}{}{}{}", p3, c2, c3, c4),
        ]
        .iter()
        .cloned()
        .collect()
    }
}
