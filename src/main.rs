use std::error::Error;
use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use clap::{Args, Parser, Subcommand};

use litsea::adaboost::AdaBoost;
use litsea::extractor::Extractor;
use litsea::get_version;
use litsea::segmenter::Segmenter;
use litsea::trainer::Trainer;

#[derive(Debug, Args)]
#[clap(
    author,
    about = "Extract features from a corpus",
    version = get_version(),
)]
struct ExtractArgs {
    corpus_file: PathBuf,
    features_file: PathBuf,
}

#[derive(Debug, Args)]
#[clap(author,
    about = "Train a segmenter",
    version = get_version(),
)]
struct TrainArgs {
    #[arg(short, long, default_value = "0.01")]
    threshold: f64,

    #[arg(short = 'i', long, default_value = "100")]
    num_iterations: usize,

    #[arg(short = 'n', long, default_value = "1")]
    num_threads: usize,

    #[arg(short = 'm', long)]
    load_model_file: Option<PathBuf>,

    features_file: PathBuf,
    model_file: PathBuf,
}

#[derive(Debug, Args)]
#[clap(author,
    about = "Segment a sentence",
    version = get_version(),
)]
struct SegmentArgs {
    model_file: PathBuf,
}

#[derive(Debug, Subcommand)]
enum Commands {
    Extract(ExtractArgs),
    Train(TrainArgs),
    Segment(SegmentArgs),
}

#[derive(Debug, Parser)]
#[clap(
    name = "litsea",
    author,
    about = "A morphological analysis command line interface",
    version = get_version(),
)]
struct CommandArgs {
    #[clap(subcommand)]
    command: Commands,
}

fn extract(args: ExtractArgs) -> Result<(), Box<dyn Error>> {
    let mut extractor = Extractor::new();

    extractor.extract(args.corpus_file.as_path(), args.features_file.as_path())?;

    println!("Feature extraction completed successfully.");
    Ok(())
}

fn train(args: TrainArgs) -> Result<(), Box<dyn Error>> {
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

    let mut trainer = Trainer::new(
        args.threshold,
        args.num_iterations,
        args.num_threads,
        args.features_file.as_path(),
    );

    if let Some(model_path) = &args.load_model_file {
        trainer.load_model(model_path.as_path())?;
    }

    trainer.train(running, args.model_file.as_path())?;

    println!("Training completed successfully.");
    Ok(())
}

fn segment(args: SegmentArgs) -> Result<(), Box<dyn Error>> {
    let mut leaner = AdaBoost::new(0.01, 100, 1);
    leaner.load_model(args.model_file.as_path())?;

    let segmenter = Segmenter::new(Some(leaner));
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut writer = io::BufWriter::new(stdout.lock());

    for line in stdin.lock().lines() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let tokens = segmenter.parse(line);
        writeln!(writer, "{}", tokens.join(" "))?;
    }

    Ok(())
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = CommandArgs::parse();

    match args.command {
        Commands::Extract(args) => extract(args),
        Commands::Train(args) => train(args),
        Commands::Segment(args) => segment(args),
    }
}

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}
