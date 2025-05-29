use crate::segmenter::Segmenter;
use std::collections::HashSet;
use std::io::{self, BufRead, Write};

/// Dummy Learner that writes features to stdout (like the Python version)
struct Learner;

impl Learner {
    pub fn add_instance(&mut self, attributes: HashSet<String>, label: i8) {
        let mut out = io::stdout().lock();
        let mut attrs: Vec<String> = attributes.into_iter().collect();
        attrs.sort(); // optional, but helpful for reproducibility
        let mut line = vec![label.to_string()];
        line.extend(attrs);
        writeln!(out, "{}", line.join("\t")).unwrap();
    }
}

fn main() {
    let stdin = io::stdin();
    let mut learner = Learner;
    let mut segmenter = Segmenter::new(None);

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l.trim().to_string(),
            Err(_) => continue,
        };
        if line.is_empty() {
            continue;
        }
        segmenter.add_sentence_with_writer(&line, &mut learner);
    }
}
