pub mod adaboost;
pub mod extractor;
pub mod segmenter;
pub mod trainer;

const VERERSION: &str = env!("CARGO_PKG_VERSION");

pub fn get_version() -> &'static str {
    VERERSION
}
