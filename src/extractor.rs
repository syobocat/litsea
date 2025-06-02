use std::collections::HashSet;
use std::error::Error;
use std::fs::File;
use std::io::{self, BufRead, Write};
use std::path::Path;

use crate::segmenter::Segmenter;

pub struct Extractor {
    segmenter: Segmenter,
}

impl Default for Extractor {
    fn default() -> Self {
        Self::new()
    }
}

impl Extractor {
    pub fn new() -> Self {
        Extractor {
            segmenter: Segmenter::new(None),
        }
    }

    pub fn extract(
        &mut self,
        corpus_path: &Path,
        features_path: &Path,
    ) -> Result<(), Box<dyn Error>> {
        // Read sentences from stdin
        // Each line is treated as a separate sentence
        let corpus_file = File::open(corpus_path)?;
        let corpus = io::BufReader::new(corpus_file);

        // Create a file to write the features
        let features_file = File::create(features_path)?;
        let mut features = io::BufWriter::new(features_file);

        // learner function to write features
        // This function will be called for each word in the input sentences
        // It takes a set of attributes and a label, and writes them to stdout
        let mut learner = |attributes: HashSet<String>, label: i8| {
            let mut attrs: Vec<String> = attributes.into_iter().collect();
            attrs.sort();
            let mut line = vec![label.to_string()];
            line.extend(attrs);
            writeln!(features, "{}", line.join("\t")).expect("Failed to write features");
        };

        for line in corpus.lines() {
            match line {
                Ok(line) => {
                    let line = line.trim();
                    if !line.is_empty() {
                        self.segmenter.add_sentence_with_writer(line, &mut learner);
                    }
                }
                Err(err) => {
                    eprintln!("Error reading input: {}", err);
                }
            }
        }

        Ok(())
    }
}
