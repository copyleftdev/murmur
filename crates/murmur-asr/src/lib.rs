//! Speech to text.
//!
//! One trait, so the rest of Murmur never learns which engine is behind it. That
//! matters more than it sounds: the whole dictation loop — trigger edges,
//! pre-roll, formatting, injection, latency accounting — can be exercised end to
//! end against [`Mock`], on a machine with no model, no ONNX runtime and no GPU.

pub mod mock;
#[cfg(feature = "parakeet")]
pub mod parakeet;

pub use mock::Mock;
#[cfg(feature = "parakeet")]
pub use parakeet::Parakeet;

use std::time::Duration;

#[derive(Debug, thiserror::Error)]
pub enum AsrError {
    #[error("model not found at {0}; run `murmur models pull` to fetch it")]
    ModelMissing(String),
    #[error("loading model: {0}")]
    Load(String),
    #[error("transcribing: {0}")]
    Inference(String),
}

type Result<T> = std::result::Result<T, AsrError>;

/// What a transcriber returns, with enough detail to explain a slow dictation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Transcript {
    pub text: String,
    /// Wall-clock time spent inside the model.
    pub elapsed: Duration,
}

impl Transcript {
    #[must_use]
    pub fn new(text: impl Into<String>, elapsed: Duration) -> Self {
        Self { text: text.into(), elapsed }
    }

    /// Audio seconds processed per second of compute.
    ///
    /// Below 1.0 the engine cannot keep up with speech; Parakeet on a modern GPU
    /// is in the hundreds, which is what makes the release-to-text budget viable.
    #[must_use]
    pub fn realtime_factor(&self, audio: Duration) -> f32 {
        let seconds = self.elapsed.as_secs_f32();
        if seconds <= 0.0 { f32::INFINITY } else { audio.as_secs_f32() / seconds }
    }
}

/// Turns 16 kHz mono audio into text.
pub trait Transcriber: Send {
    /// Human-readable engine and model, for `murmur doctor`.
    fn name(&self) -> String;

    /// # Errors
    /// Fails if the model rejects the audio or inference fails.
    fn transcribe(&mut self, samples: &[f32]) -> Result<Transcript>;
}
