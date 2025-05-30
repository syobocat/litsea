use std::collections::HashSet;
use std::error::Error;
use std::io::{self, BufRead, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use clap::{Args, Parser, Subcommand};

use litsea::adaboost::AdaBoost;
use litsea::get_version;
use litsea::segmenter::Segmenter;

#[derive(Debug, Args)]
#[clap(
    author,
    about = "Extract features from a corpus",
    version = get_version(),
)]
struct ExtractArgs {}

#[derive(Debug, Args)]
#[clap(author,
    about = "Train a segmenter",
    version = get_version(),
)]
struct TrainArgs {
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

#[derive(Debug, Args)]
#[clap(author,
    about = "Segment a sentence",
    version = get_version(),
)]
struct SegmentArgs {
    model_file: String,
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

fn extract(_args: ExtractArgs) -> Result<(), Box<dyn Error>> {
    let mut stdout = io::BufWriter::new(io::stdout());
    let mut segmenter = Segmenter::new(None);

    // learner function to write features
    // This function will be called for each word in the input sentences
    // It takes a set of attributes and a label, and writes them to stdout
    let mut learner = |attributes: HashSet<String>, label: i8| {
        let mut attrs: Vec<String> = attributes.into_iter().collect();
        attrs.sort(); // 再現性のためにソート
        let mut line = vec![label.to_string()];
        line.extend(attrs);
        writeln!(stdout, "{}", line.join("\t")).expect("Failed to write features");
    };

    // Read sentences from stdin
    // Each line is treated as a separate sentence
    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        match line {
            Ok(line) => {
                let line = line.trim();
                if !line.is_empty() {
                    segmenter.add_sentence_with_writer(line, &mut learner);
                }
            }
            Err(err) => {
                eprintln!("Error reading input: {}", err);
            }
        }
    }

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

    let mut boost = AdaBoost::new(args.threshold, args.num_iterations, args.num_threads);

    if let Some(model_path) = args.load_model.as_ref() {
        boost.load_model(model_path).unwrap();
    }

    boost.initialize_features(&args.instances_file).unwrap();
    boost.initialize_instances(&args.instances_file).unwrap();

    boost.train(running.clone());
    boost.save_model(&args.model_file).unwrap();
    boost.show_result();

    Ok(())
}

fn segment(args: SegmentArgs) -> Result<(), Box<dyn Error>> {
    let model_path = &args.model_file;

    let mut model = AdaBoost::new(0.01, 100, 1);
    if let Err(e) = model.load_model(model_path) {
        eprintln!("Failed to load model: {}", e);
        std::process::exit(1);
    }

    let segmenter = Segmenter::new(Some(model));
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut writer = io::BufWriter::new(stdout.lock());

    for line in stdin.lock().lines() {
        match line {
            Ok(line) => {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                let tokens = segmenter.parse(line);
                writeln!(writer, "{}", tokens.join(" ")).expect("write failed");
            }
            Err(err) => {
                eprintln!("Error reading input: {}", err);
            }
        }
    }

    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = CommandArgs::parse();

    match args.command {
        Commands::Extract(args) => extract(args),
        Commands::Train(args) => train(args),
        Commands::Segment(args) => segment(args),
    }
}
