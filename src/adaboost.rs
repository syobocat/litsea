use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use rayon::prelude::*;

type Label = i8;
type FeatureIndex = usize;

#[derive(Debug)]
pub struct AdaBoost {
    threshold: f64,
    num_iterations: usize,
    num_threads: usize,
    D: Vec<f64>,
    model: Vec<f64>,
    features: Vec<String>,
    labels: Vec<Label>,
    instances_buf: Vec<usize>,
    instances: Vec<(usize, usize)>, // start, end index to instances_buf
    num_instances: usize,
}

impl AdaBoost {
    pub fn new(threshold: f64, num_iterations: usize, num_threads: usize) -> Self {
        AdaBoost {
            threshold,
            num_iterations,
            num_threads,
            D: vec![],
            model: vec![],
            features: vec![],
            labels: vec![],
            instances_buf: vec![],
            instances: vec![],
            num_instances: 0,
        }
    }

    pub fn initialize_features(&mut self, filename: &str) -> std::io::Result<()> {
        let file = File::open(filename)?;
        let reader = BufReader::new(file);
        let mut map = BTreeMap::new(); // BTreeMap to preserve order

        let mut buf_size = 0;
        self.num_instances = 0;

        for line in reader.lines() {
            let line = line?;
            let mut parts = line.split_whitespace();
            let _label = parts.next();
            for h in parts {
                map.entry(h.to_string()).or_insert(0.0);
                buf_size += 1;
            }
            self.num_instances += 1;
            if self.num_instances % 1000 == 0 {
                eprint!(
                    "\rfinding instances...: {} instances found",
                    self.num_instances
                );
            }
        }
        eprintln!(
            "\rfinding instances...: {} instances found",
            self.num_instances
        );
        map.insert("".to_string(), 0.0);

        self.features = map.keys().cloned().collect();
        self.model = map.values().cloned().collect();

        self.D.reserve(self.num_instances);
        self.labels.reserve(self.num_instances);
        self.instances.reserve(self.num_instances);
        self.instances_buf.reserve(buf_size);

        Ok(())
    }

    pub fn initialize_instances(&mut self, filename: &str) -> std::io::Result<()> {
        let file = File::open(filename)?;
        let reader = BufReader::new(file);
        let bias = self.get_bias();

        for line in reader.lines() {
            let line = line?;
            let mut parts = line.split_whitespace();
            let label: Label = parts.next().unwrap().parse().unwrap();
            self.labels.push(label);

            let start = self.instances_buf.len();
            let mut score = bias;

            for h in parts {
                if let Ok(pos) = self.features.binary_search(&h.to_string()) {
                    self.instances_buf.push(pos);
                    score += self.model[pos];
                }
            }

            let end = self.instances_buf.len();
            self.instances.push((start, end));
            self.D.push((-2.0 * label as f64 * score).exp());

            if self.D.len() % 1000 == 0 {
                eprint!(
                    "\rloading instances...: {}/{} instances loaded",
                    self.D.len(),
                    self.num_instances
                );
            }
        }
        eprintln!();
        Ok(())
    }

    pub fn train(&mut self, running: Arc<AtomicBool>) {
        let num_features = self.features.len();
        let mut h_best = 0;
        let mut e_best = 0.5;
        let mut a = 0.0;
        let mut a_exp = 1.0;

        for t in 0..self.num_iterations {
            if !running.load(Ordering::SeqCst) {
                break;
            }

            let mut errors = vec![0.0f64; num_features];
            let mut D_sum = 0.0;
            let mut D_sum_plus = 0.0;

            (0..self.num_instances).into_par_iter().for_each(|i| {
                let label = self.labels[i];
                let (start, end) = self.instances[i];
                let hs = &self.instances_buf[start..end];
                let pred = if hs.binary_search(&h_best).is_ok() {
                    1
                } else {
                    -1
                };

                if label * pred < 0 {
                    self.D[i] *= a_exp;
                } else {
                    self.D[i] /= a_exp;
                }
            });

            // Normalization & error calculation (serial to maintain determinism)
            for i in 0..self.num_instances {
                let d = self.D[i];
                D_sum += d;
                if self.labels[i] > 0 {
                    D_sum_plus += d;
                }
                let delta = d * self.labels[i] as f64;
                for &h in &self.instances_buf[self.instances[i].0..self.instances[i].1] {
                    errors[h] -= delta;
                }
            }

            e_best = D_sum_plus / D_sum;
            h_best = 0;

            for h in 1..num_features {
                let mut e = errors[h] + D_sum_plus;
                e /= D_sum;
                if (0.5 - e).abs() > (0.5 - e_best).abs() {
                    h_best = h;
                    e_best = e;
                }
            }

            eprint!("\rIteration {} - margin: {}", t, (0.5 - e_best).abs());

            if (0.5 - e_best).abs() < self.threshold {
                break;
            }

            e_best = e_best.clamp(1e-10, 1.0 - 1e-10);
            a = 0.5 * ((1.0 - e_best) / e_best).ln();
            a_exp = a.exp();
            self.model[h_best] += a;

            // Normalize D
            for d in &mut self.D {
                *d /= D_sum;
            }
        }

        eprintln!();
    }

    pub fn save_model(&self, filename: &str) -> std::io::Result<()> {
        let mut file = File::create(filename)?;
        let mut bias = -self.model[0];
        for (h, &w) in self.features.iter().zip(self.model.iter()).skip(1) {
            if w != 0.0 {
                writeln!(file, "{}\t{}", h, w)?;
                bias -= w;
            }
        }
        writeln!(file, "{}", bias / 2.0)?;
        Ok(())
    }

    pub fn load_model(&mut self, filename: &str) -> std::io::Result<()> {
        let file = File::open(filename)?;
        let reader = BufReader::new(file);
        let mut m: HashMap<String, f64> = HashMap::new();
        let mut bias = 0.0;

        for line in reader.lines() {
            let line = line?;
            let mut parts = line.split_whitespace();
            let h = parts.next().unwrap();
            if let Some(v) = parts.next() {
                let value: f64 = v.parse().unwrap();
                m.insert(h.to_string(), value);
                bias += value;
            } else {
                let b: f64 = h.parse().unwrap();
                m.insert("".to_string(), -b * 2.0 - bias);
            }
        }

        let mut sorted: BTreeMap<_, _> = m.into_iter().collect();
        self.features = sorted.keys().cloned().collect();
        self.model = sorted.values().cloned().collect();
        Ok(())
    }

    pub fn get_bias(&self) -> f64 {
        -self.model.iter().sum::<f64>() / 2.0
    }

    pub fn show_result(&self) {
        let bias = self.get_bias();
        let mut pp = 0;
        let mut pn = 0;
        let mut np = 0;
        let mut nn = 0;

        for i in 0..self.num_instances {
            let label = self.labels[i];
            let (start, end) = self.instances[i];
            let mut score = bias;
            for &h in &self.instances_buf[start..end] {
                score += self.model[h];
            }

            if score >= 0.0 {
                if label > 0 {
                    pp += 1
                } else {
                    pn += 1
                }
            } else {
                if label > 0 {
                    np += 1
                } else {
                    nn += 1
                }
            }
        }

        let acc = (pp + nn) as f64 / self.num_instances as f64 * 100.0;
        let prec = pp as f64 / (pp + pn).max(1) as f64 * 100.0;
        let recall = pp as f64 / (pp + np).max(1) as f64 * 100.0;

        eprintln!("Result:");
        eprintln!(
            "Accuracy: {:.2}% ({} / {})",
            acc,
            pp + nn,
            self.num_instances
        );
        eprintln!("Precision: {:.2}% ({} / {})", prec, pp, pp + pn);
        eprintln!("Recall: {:.2}% ({} / {})", recall, pp, pp + np);
        eprintln!(
            "Confusion Matrix: TP: {}, FP: {}, FN: {}, TN: {}",
            pp, pn, np, nn
        );
    }
}
