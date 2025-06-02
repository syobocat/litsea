use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use crate::adaboost::AdaBoost;

pub struct Trainer {
    learner: AdaBoost,
}

impl Trainer {
    pub fn new(
        threshold: f64,
        num_iterations: usize,
        num_threads: usize,
        features_path: &Path,
    ) -> Self {
        let mut learner = AdaBoost::new(threshold, num_iterations, num_threads);

        learner
            .initialize_features(features_path)
            .expect("Failed to initialize features");
        learner
            .initialize_instances(features_path)
            .expect("Failed to initialize instances");

        Trainer { learner }
    }

    pub fn load_model(&mut self, model_path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        // Load the model from the specified file
        Ok(self.learner.load_model(model_path)?)
    }

    pub fn train(
        &mut self,
        running: Arc<AtomicBool>,
        model_path: &Path,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.learner.train(running.clone());

        // Save the trained model to the specified file
        self.learner.save_model(model_path)?;
        self.learner.show_result();

        Ok(())
    }
}
