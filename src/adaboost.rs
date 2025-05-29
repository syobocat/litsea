use std::collections::{BTreeMap, HashMap, HashSet};
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
    instance_weights: Vec<f64>,
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
            instance_weights: vec![],
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

        self.instance_weights.reserve(self.num_instances);
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
            self.instance_weights
                .push((-2.0 * label as f64 * score).exp());

            if self.instance_weights.len() % 1000 == 0 {
                eprint!(
                    "\rloading instances...: {}/{} instances loaded",
                    self.instance_weights.len(),
                    self.num_instances
                );
            }
        }
        eprintln!();
        Ok(())
    }

    pub fn train(&mut self, running: Arc<AtomicBool>) {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(self.num_threads)
            .build()
            .expect("Failed to build thread pool");

        let num_features = self.features.len();
        let mut alpha_exp = 1.0;

        for t in 0..self.num_iterations {
            if !running.load(Ordering::SeqCst) {
                break;
            }

            let mut errors = vec![0.0f64; num_features];
            let mut instance_weight_sum = 0.0;
            let mut positive_weight_sum = 0.0;

            let updated_instance_weights: Vec<f64> = pool.install(|| {
                (0..self.num_instances)
                    .into_par_iter()
                    .map(|i| {
                        let label = self.labels[i];
                        let (start, end) = self.instances[i];
                        let hs = &self.instances_buf[start..end];
                        let pred = if hs.binary_search(&0).is_ok() { 1 } else { -1 };

                        if label * pred < 0 {
                            self.instance_weights[i] * alpha_exp
                        } else {
                            self.instance_weights[i] / alpha_exp
                        }
                    })
                    .collect()
            });

            for (i, d) in updated_instance_weights.into_iter().enumerate() {
                self.instance_weights[i] = d;
            }

            for i in 0..self.num_instances {
                let d = self.instance_weights[i];
                instance_weight_sum += d;
                if self.labels[i] > 0 {
                    positive_weight_sum += d;
                }
                let delta = d * self.labels[i] as f64;
                for &h in &self.instances_buf[self.instances[i].0..self.instances[i].1] {
                    errors[h] -= delta;
                }
            }

            // 各ループで初期化
            let mut best_error_rate = positive_weight_sum / instance_weight_sum;
            let mut best_feature_index: FeatureIndex = 0;

            for h in 1..num_features {
                let mut e = errors[h] + positive_weight_sum;
                e /= instance_weight_sum;
                if (0.5 - e).abs() > (0.5 - best_error_rate).abs() {
                    best_feature_index = h;
                    best_error_rate = e;
                }
            }

            eprint!(
                "\rIteration {} - margin: {}",
                t,
                (0.5 - best_error_rate).abs()
            );

            if (0.5 - best_error_rate).abs() < self.threshold {
                break;
            }

            best_error_rate = best_error_rate.clamp(1e-10, 1.0 - 1e-10);

            let alpha = 0.5 * ((1.0 - best_error_rate) / best_error_rate).ln();
            self.model[best_feature_index] += alpha;
            alpha_exp = alpha.exp();

            // instance_weights の正規化
            for d in &mut self.instance_weights {
                *d /= instance_weight_sum;
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

        let sorted: BTreeMap<_, _> = m.into_iter().collect();
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

    pub fn add_instance(&mut self, attributes: HashSet<String>, label: i8) {
        // 現在のインスタンスの属性開始位置
        let start = self.instances_buf.len();
        // HashSet の順序は不定のためソートして安定化する
        let mut attrs: Vec<String> = attributes.into_iter().collect();
        attrs.sort();
        for attr in attrs.iter() {
            // すでに存在する属性ならそのインデックスを取得、なければ追加して新たなインデックスを取得
            let feature_index = if let Some(pos) = self.features.iter().position(|f| f == attr) {
                pos
            } else {
                self.features.push(attr.clone());
                self.model.push(0.0);
                self.features.len() - 1
            };
            self.instances_buf.push(feature_index);
        }
        // 終了インデックス
        let end = self.instances_buf.len();
        // インスタンスごとの属性インデックスの範囲を記録
        self.instances.push((start, end));
        // 対応するラベルを登録
        self.labels.push(label);
        // 初期のインスタンス重みを 1.0 とする
        self.instance_weights.push(1.0);
        // インスタンス数の更新
        self.num_instances += 1;
    }

    /// 属性集合に基づいて予測を行い、ラベル (1 または -1) を返す
    pub fn predict(&self, attributes: HashSet<String>) -> i8 {
        let mut score = 0.0;
        for attr in attributes {
            if let Some(idx) = self.features.iter().position(|f| f == &attr) {
                score += self.model[idx];
            }
        }
        if score >= 0.0 {
            1
        } else {
            -1
        }
    }
}
