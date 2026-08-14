//! Does the real model, loaded the way Murmur loads it, actually transcribe?
//!
//! Skipped when the model is not on disk, because a checkout should not require
//! a 640 MB download to run `cargo test`. When it *is* present this is the only
//! test in the workspace that exercises ONNX Runtime, the int8 weights and the
//! feature extractor together, against audio with a transcript known in advance.

#![cfg(feature = "parakeet")]

use murmur_asr::{Parakeet, Transcriber};
use murmur_core::config::{Accelerator, AsrConfig, Precision};
use std::path::{Path, PathBuf};
use std::time::Duration;

/// John F. Kennedy, 1961. The canonical ASR smoke-test clip.
const SAMPLE: &str = "../../assets/jfk.wav";

const EXPECTED_WORDS: &[&str] = &[
    "fellow",
    "americans",
    "ask",
    "not",
    "what",
    "your",
    "country",
    "can",
    "do",
    "for",
    "you",
];

fn model_dir() -> PathBuf {
    let configured = AsrConfig::default().model_dir;

    configured.strip_prefix("~/").map_or_else(
        || PathBuf::from(&configured),
        |rest| PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(rest),
    )
}

fn read_wav(path: &Path) -> Vec<f32> {
    let mut reader = hound::WavReader::open(path).expect("opening the sample");
    let spec = reader.spec();
    assert_eq!(
        spec.sample_rate, 16_000,
        "fixture must already be at the target rate"
    );
    assert_eq!(spec.channels, 1, "fixture must be mono");
    let scale = f32::from(i16::MAX);
    reader
        .samples::<i16>()
        .map(|s| f32::from(s.expect("sample")) / scale)
        .collect()
}

/// `Some(model)` when a model is installed, `None` when the test should skip.
///
/// Pinned to the CPU: these assert transcription, not throughput, and must give
/// the same answer on a machine with no GPU.
fn load() -> Option<Parakeet> {
    let root = model_dir();
    if murmur_asr::models::discover(&root).is_empty() {
        eprintln!("skipping: no model under {}", root.display());
        return None;
    }
    Some(Parakeet::open(&root, Precision::Auto, Accelerator::Cpu).expect("loading the model"))
}

#[test]
fn the_model_transcribes_known_speech_correctly() {
    let Some(mut model) = load() else { return };
    let samples = read_wav(Path::new(SAMPLE));

    let transcript = model.transcribe(&samples).expect("transcribing");
    let lowered = transcript.text.to_lowercase();

    for word in EXPECTED_WORDS {
        assert!(
            lowered.contains(word),
            "{word:?} missing from {:?}",
            transcript.text
        );
    }
}

#[test]
fn the_model_punctuates_and_capitalises_without_help_from_the_formatter() {
    let Some(mut model) = load() else { return };
    let samples = read_wav(Path::new(SAMPLE));

    let transcript = model.transcribe(&samples).expect("transcribing");
    assert!(
        transcript.text.contains(',') || transcript.text.contains('.'),
        "expected punctuation from the model: {:?}",
        transcript.text
    );
    assert!(
        transcript
            .text
            .chars()
            .next()
            .is_some_and(char::is_uppercase),
        "expected a capitalised opening: {:?}",
        transcript.text
    );
}

#[test]
fn transcription_keeps_up_with_speech_by_a_wide_margin() {
    let Some(mut model) = load() else { return };
    let samples = read_wav(Path::new(SAMPLE));
    let audio = Duration::from_secs_f32(samples.len() as f32 / 16_000.0);

    let transcript = model.transcribe(&samples).expect("transcribing");
    let rtf = transcript.realtime_factor(audio);
    assert!(
        rtf > 3.0,
        "only {rtf:.1}x realtime; dictation needs headroom over the audio it is given"
    );
}

#[test]
fn silence_transcribes_to_nothing_rather_than_hallucinating() {
    let Some(mut model) = load() else { return };
    let silence = vec![0.0f32; 16_000 * 3];

    let transcript = model.transcribe(&silence).expect("transcribing silence");
    assert!(
        transcript.text.trim().is_empty(),
        "three seconds of silence produced {:?}",
        transcript.text
    );
}

#[test]
fn a_missing_model_directory_is_reported_as_missing_not_as_a_load_failure() {
    let error = Parakeet::open(
        Path::new("/nonexistent/murmur/model"),
        Precision::Auto,
        Accelerator::Cpu,
    )
    .expect_err("should fail");
    assert!(
        matches!(error, murmur_asr::AsrError::ModelMissing(_)),
        "got {error:?}, which does not tell the user to fetch the model"
    );
}
