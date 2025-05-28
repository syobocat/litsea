mod adaboost;
use adaboost::AdaBoost;
use clap::Parser;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// CLI args
#[derive(Parser, Debug)]
#[command(version, about = "AdaBoost Trainer in Rust")]
struct Args {
    #[arg(short, long, default_value = "0.01")]
    threshold: f64,

    #[arg(short = 'n', long, default_value = "100")]
    num_iterations: usize,

    #[arg(short = 'm', long, default_value = "1")]
    num_threads: usize,

    #[arg(short = 'M', long)]
    load_model: Option<String>,

    instances_file: String,
    model_file: String,
}

fn main() {
    let args = Args::parse();

    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();

    ctrlc::set_handler(move || {
        if r.load(Ordering::SeqCst) {
            r.store(false, Ordering::SeqCst);
        } else {
            std::process::exit(0);
        }
    })
    .expect("Error setting Ctrl-C handler");

    let mut boost = AdaBoost::new(args.threshold, args.num_iterations, args.num_threads);

    if let Some(model_path) = args.load_model.as_ref() {
        boost.load_model(model_path).unwrap();
    }

    boost.initialize_features(&args.instances_file).unwrap();
    boost.initialize_instances(&args.instances_file).unwrap();

    boost.train(running.clone());
    boost.save_model(&args.model_file).unwrap();
    boost.show_result();
}
