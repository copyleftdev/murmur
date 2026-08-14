//! The streaming model, driven the way the daemon drives it.
//!
//! Skipped when no streaming weights are installed. What these assert that the
//! batch tests cannot is the consequence of carrying state between calls: an
//! utterance must not be decoded in the acoustic context of the one before it,
//! and the tail must be flushed or the last word of every dictation is lost.

#![cfg(feature = "parakeet")]

use murmur_asr::streaming::transcribe_all;
use murmur_asr::{NemotronStream, StreamingTranscriber};
use murmur_core::config::{Accelerator, AsrConfig, Precision};
use std::path::{Path, PathBuf};

const SAMPLE: &str = "../../assets/jfk.wav";

fn model_root() -> PathBuf {
    let configured = AsrConfig::default().model_dir;
    configured.strip_prefix("~/").map_or_else(
        || PathBuf::from(&configured),
        |rest| PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(rest),
    )
}

fn read_wav(path: &Path) -> Vec<f32> {
    let mut reader = hound::WavReader::open(path).expect("opening the sample");
    let scale = f32::from(i16::MAX);
    reader
        .samples::<i16>()
        .map(|s| f32::from(s.expect("sample")) / scale)
        .collect()
}

fn load() -> Option<NemotronStream> {
    let root = model_root();
    let has_streaming = murmur_asr::models::discover(&root)
        .iter()
        .any(|v| v.family == murmur_asr::models::Family::NemotronStreaming);
    if !has_streaming {
        eprintln!("skipping: no streaming model under {}", root.display());
        return None;
    }
    Some(
        NemotronStream::open(&root, Precision::Auto, Accelerator::Cpu, Some("en-US"))
            .expect("loading the streaming model"),
    )
}

#[test]
fn chunked_audio_transcribes_the_known_speech() {
    let Some(mut model) = load() else { return };
    let samples = read_wav(Path::new(SAMPLE));

    let (transcript, _) = transcribe_all(&mut model, &samples).expect("transcribing");
    let lowered = transcript.text.to_lowercase();
    for word in ["fellow", "americans", "country"] {
        assert!(
            lowered.contains(word),
            "{word:?} missing from {:?}",
            transcript.text
        );
    }
}

/// Pins a limitation, not a feature.
///
/// The streaming model decodes only on chunk boundaries and never emits the
/// final partial chunk — up to 560 ms of speech. There is no flush in its API,
/// and padding the tail does not force one: neither digital silence nor a noise
/// floor produces any token, because the decoder emits on acoustic evidence
/// rather than on elapsed frames.
///
/// This matters because dictation ends exactly where the risk is highest: the
/// user releases the key just after their last word. Measured on a clip cut
/// mid-utterance, the batch model recovers "…ask not what" while the streaming
/// model stops at "…ask not".
///
/// If this test ever fails, the limitation has been fixed upstream and the
/// streaming model can become the authoritative transcriber instead of only a
/// source of live partials.
#[test]
fn the_streaming_model_cannot_flush_its_final_partial_chunk() {
    let Some(mut model) = load() else { return };

    let full = read_wav(Path::new(SAMPLE));
    let truncated = &full[..full.len() / 2];

    model.reset();
    for piece in truncated.chunks(model.chunk_samples()) {
        model.feed(piece).expect("feeding");
    }

    assert!(
        model.finish().expect("flushing").trim().is_empty(),
        "the tail flushed after all -- streaming can now be trusted for final text, \
         so revisit the hybrid in murmur-engine and delete this test"
    );
}

#[test]
fn resetting_between_utterances_gives_the_same_answer_every_time() {
    let Some(mut model) = load() else { return };
    let samples = read_wav(Path::new(SAMPLE));

    let (first, _) = transcribe_all(&mut model, &samples).expect("first");
    let (second, _) = transcribe_all(&mut model, &samples).expect("second");

    assert_eq!(
        first.text, second.text,
        "the same audio decoded differently the second time: state leaked between utterances"
    );
}

#[test]
fn without_a_reset_state_leaks_and_the_transcript_grows() {
    let Some(mut model) = load() else { return };
    let samples = read_wav(Path::new(SAMPLE));

    let (clean, _) = transcribe_all(&mut model, &samples).expect("clean run");

    // Deliberately skipping reset, which is what `transcribe_all` does for us.
    let mut contaminated = String::new();
    for piece in samples.chunks(model.chunk_samples()) {
        contaminated.push_str(&model.feed(piece).expect("feeding"));
    }
    contaminated.push_str(&model.finish().expect("flushing"));

    assert_ne!(
        clean.text.trim(),
        contaminated.trim(),
        "reset appears to be a no-op; if so the daemon's per-utterance reset is not \
         actually protecting anything and this test should be deleted"
    );
}

#[test]
fn silence_streams_to_nothing_rather_than_hallucinating() {
    let Some(mut model) = load() else { return };
    let silence = vec![0.0f32; 16_000 * 3];

    let (transcript, _) = transcribe_all(&mut model, &silence).expect("transcribing silence");
    assert!(
        transcript.text.trim().is_empty(),
        "three seconds of silence produced {:?}",
        transcript.text
    );
}

#[test]
fn a_missing_streaming_model_is_reported_as_missing() {
    let error = NemotronStream::open(
        Path::new("/nonexistent/murmur/model"),
        Precision::Auto,
        Accelerator::Cpu,
        None,
    )
    .expect_err("should fail");
    assert!(
        matches!(error, murmur_asr::AsrError::ModelMissing(_)),
        "got {error:?}, which does not tell the user to fetch the model"
    );
}
