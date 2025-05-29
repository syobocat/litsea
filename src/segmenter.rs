use regex::Regex;
use std::collections::HashSet;

use crate::adaboost::AdaBoost;

pub struct Segmenter {
    learner: AdaBoost,
    patterns: Vec<(Regex, &'static str)>,
}

impl Segmenter {
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
            learner: learner.unwrap_or_else(|| AdaBoost::new(0.01, 100, 1)),
            patterns,
        }
    }

    fn get_type(&self, ch: &str) -> &str {
        for (re, t) in &self.patterns {
            if re.is_match(ch) {
                return t;
            }
        }
        "O"
    }

    pub fn add_sentence_with_writer(
        &mut self,
        sentence: &str,
        learner: &mut impl FnMut(HashSet<String>, i8),
    ) {
        if sentence.is_empty() {
            return;
        }

        let mut tags = vec!["U", "U", "U"];
        let mut chars = vec!["B3".to_string(), "B2".to_string(), "B1".to_string()];
        let mut types = vec!["O", "O", "O"];

        for word in sentence.split_whitespace() {
            tags.push("B");
            for (i, ch) in word.chars().enumerate() {
                if i > 0 {
                    tags.push("O");
                }
                let ch_str = ch.to_string();
                chars.push(ch_str.clone());
                types.push(self.get_type(&ch_str));
            }
        }

        if tags.len() < 4 {
            return;
        }

        tags[3] = "U";
        chars.extend(vec!["E1".to_string(), "E2".to_string(), "E3".to_string()]);
        types.extend(vec!["O", "O", "O"]);

        for i in 4..(chars.len() - 3) {
            let label = if tags[i] == "B" { 1 } else { -1 };
            let attributes = self.get_attributes(i, &tags, &chars, &types);
            learner(attributes, label);
        }
    }

    pub fn add_sentence(&mut self, sentence: &str) {
        if sentence.is_empty() {
            return;
        }
        let mut tags = vec!["U", "U", "U"];
        let mut chars = vec!["B3".to_string(), "B2".to_string(), "B1".to_string()];
        let mut types = vec!["O", "O", "O"];

        for word in sentence.split_whitespace() {
            tags.push("B");
            for (i, ch) in word.chars().enumerate() {
                if i > 0 {
                    tags.push("O");
                }
                let ch_str = ch.to_string();
                chars.push(ch_str.clone());
                types.push(self.get_type(&ch_str));
            }
        }

        if tags.len() < 4 {
            return;
        }

        tags[3] = "U";
        chars.extend(vec!["E1".to_string(), "E2".to_string(), "E3".to_string()]);
        types.extend(vec!["O", "O", "O"]);

        // まずはすべてのインスタンス情報をローカル変数にまとめる
        let new_instances: Vec<(std::collections::HashSet<String>, i8)> = (4..(chars.len() - 3))
            .map(|i| {
                let label = if tags[i] == "B" { 1 } else { -1 };
                let attributes = self.get_attributes(i, &tags, &chars, &types);
                (attributes, label)
            })
            .collect();

        // その後、ローカル変数から self.learner に add_instance を呼び出す
        for (attributes, label) in new_instances {
            self.learner.add_instance(attributes, label);
        }
    }

    pub fn parse(&self, sentence: &str) -> Vec<String> {
        if sentence.is_empty() {
            return vec![];
        }

        let mut tags = vec!["U", "U", "U", "U"];
        let mut chars = vec!["B3".to_string(), "B2".to_string(), "B1".to_string()];
        let mut types = vec!["O", "O", "O"];

        for ch in sentence.chars() {
            let ch_str = ch.to_string();
            chars.push(ch_str.clone());
            types.push(self.get_type(&ch_str));
        }

        chars.extend(vec!["E1".to_string(), "E2".to_string(), "E3".to_string()]);
        types.extend(vec!["O", "O", "O"]);

        let mut result = vec![];
        let mut word = chars[3].clone();

        for i in 4..(chars.len() - 3) {
            let attrs = self.get_attributes(i, &tags, &chars, &types);
            let label = self.learner.predict(attrs);

            if label >= 0 {
                result.push(word);
                word = String::new();
                tags.push("B");
            } else {
                tags.push("O");
            }

            word += &chars[i];
        }

        result.push(word);
        result
    }

    fn get_attributes<'a>(
        &self,
        i: usize,
        tags: &[&'a str],
        chars: &[String],
        types: &[&'a str],
    ) -> HashSet<String> {
        let (w1, w2, w3, w4, w5, w6) = (
            &chars[i - 3],
            &chars[i - 2],
            &chars[i - 1],
            &chars[i],
            &chars[i + 1],
            &chars[i + 2],
        );
        let (c1, c2, c3, c4, c5, c6) = (
            types[i - 3],
            types[i - 2],
            types[i - 1],
            types[i],
            types[i + 1],
            types[i + 2],
        );
        let (p1, p2, p3) = (tags[i - 3], tags[i - 2], tags[i - 1]);

        let mut attributes = HashSet::new();
        attributes.insert(format!("UP1:{}", p1));
        attributes.insert(format!("UP2:{}", p2));
        attributes.insert(format!("UP3:{}", p3));
        attributes.insert(format!("BP1:{}{}", p1, p2));
        attributes.insert(format!("BP2:{}{}", p2, p3));

        attributes.insert(format!("UW1:{}", w1));
        attributes.insert(format!("UW2:{}", w2));
        attributes.insert(format!("UW3:{}", w3));
        attributes.insert(format!("UW4:{}", w4));
        attributes.insert(format!("UW5:{}", w5));
        attributes.insert(format!("UW6:{}", w6));

        attributes.insert(format!("BW1:{}{}", w2, w3));
        attributes.insert(format!("BW2:{}{}", w3, w4));
        attributes.insert(format!("BW3:{}{}", w4, w5));

        attributes.insert(format!("TW1:{}{}{}", w1, w2, w3));
        attributes.insert(format!("TW2:{}{}{}", w2, w3, w4));
        attributes.insert(format!("TW3:{}{}{}", w3, w4, w5));
        attributes.insert(format!("TW4:{}{}{}", w4, w5, w6));

        attributes.insert(format!("UC1:{}", c1));
        attributes.insert(format!("UC2:{}", c2));
        attributes.insert(format!("UC3:{}", c3));
        attributes.insert(format!("UC4:{}", c4));
        attributes.insert(format!("UC5:{}", c5));
        attributes.insert(format!("UC6:{}", c6));

        attributes.insert(format!("BC1:{}{}", c2, c3));
        attributes.insert(format!("BC2:{}{}", c3, c4));
        attributes.insert(format!("BC3:{}{}", c4, c5));

        attributes.insert(format!("TC1:{}{}{}", c1, c2, c3));
        attributes.insert(format!("TC2:{}{}{}", c2, c3, c4));
        attributes.insert(format!("TC3:{}{}{}", c3, c4, c5));
        attributes.insert(format!("TC4:{}{}{}", c4, c5, c6));

        attributes.insert(format!("UQ1:{}{}", p1, c1));
        attributes.insert(format!("UQ2:{}{}", p2, c2));
        attributes.insert(format!("UQ3:{}{}", p3, c3));

        attributes.insert(format!("BQ1:{}{}{}", p2, c2, c3));
        attributes.insert(format!("BQ2:{}{}{}", p2, c3, c4));
        attributes.insert(format!("BQ3:{}{}{}", p3, c2, c3));
        attributes.insert(format!("BQ4:{}{}{}", p3, c3, c4));

        attributes.insert(format!("TQ1:{}{}{}{}", p2, c1, c2, c3));
        attributes.insert(format!("TQ2:{}{}{}{}", p2, c2, c3, c4));
        attributes.insert(format!("TQ3:{}{}{}{}", p3, c1, c2, c3));
        attributes.insert(format!("TQ4:{}{}{}{}", p3, c2, c3, c4));

        attributes
    }
}
