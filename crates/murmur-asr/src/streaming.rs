//! Transcribing while the user is still speaking.
//!
//! A batch transcriber cannot start until the key is released, so every
//! millisecond of inference lands squarely inside the delay the user feels. A
//! cache-aware streaming model instead consumes the utterance as it happens, and
//! by the time the key comes up only the final chunk is outstanding. The work
//! does not get cheaper; it gets moved off the critical path.
//!
//! The trade is that state now persists between calls, which makes [`reset`] a
//! correctness requirement rather than housekeeping: without it the second
//! dictation is decoded in the acoustic context of the first.
//!
//! [`reset`]: StreamingTranscriber::reset

use crate::models::{self, Family};
use crate::{AsrError, Transcript};
use murmur_core::config::{Accelerator, Precision};
use parakeet_rs::{ExecutionConfig, ExecutionProvider, Nemotron, NemotronMode};
use std::path::Path;
use std::time::{Duration, Instant};

type Result<T> = std::result::Result<T, AsrError>;

/// A transcriber that consumes audio incrementally.
pub trait StreamingTranscriber: Send {
    fn name(&self) -> String;

    /// Samples the model wants per step. Feeding other sizes is allowed —
    /// implementations buffer — but this is the granularity at which new text
    /// can appear.
    fn chunk_samples(&self) -> usize;

    /// Discard all state. Must be called before each new utterance.
    fn reset(&mut self);

    /// Feed audio and return whatever new text it produced, which is usually
    /// nothing until a chunk boundary is crossed.
    ///
    /// # Errors
    /// Fails if inference fails.
    fn feed(&mut self, samples: &[f32]) -> Result<String>;

    /// Flush buffered audio and return the last of the text.
    ///
    /// # Errors
    /// Fails if inference fails.
    fn finish(&mut self) -> Result<String>;
}

/// NVIDIA Nemotron cache-aware streaming ASR.
pub struct NemotronStream {
    model: Nemotron,
    label: String,
    fed: usize,
}

impl std::fmt::Debug for NemotronStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NemotronStream").field("label", &self.label).finish_non_exhaustive()
    }
}

impl NemotronStream {
    /// Find and load streaming weights under `root`.
    ///
    /// # Errors
    /// Fails if no streaming model is installed, or it cannot be loaded.
    pub fn open(
        root: &Path,
        precision: Precision,
        accelerator: Accelerator,
        language: Option<&str>,
    ) -> Result<Self> {
        let gpu_usable = crate::parakeet::gpu_usable(accelerator);
        let variants = models::discover(root);
        let chosen = models::choose(&variants, gpu_usable, precision, Family::NemotronStreaming)
            .ok_or_else(|| AsrError::ModelMissing(root.display().to_string()))?;

        let provider = if gpu_usable && !matches!(accelerator, Accelerator::Cpu) {
            #[cfg(feature = "cuda")]
            {
                crate::cuda::ensure_runtime();
                ExecutionProvider::Cuda
            }
            #[cfg(not(feature = "cuda"))]
            {
                ExecutionProvider::Cpu
            }
        } else {
            ExecutionProvider::Cpu
        };

        let started = Instant::now();
        let mut model = Nemotron::from_pretrained(
            &chosen.dir,
            Some(ExecutionConfig::new().with_execution_provider(provider)),
        )
        .map_err(|e| AsrError::Load(e.to_string()))?;

        // The multilingual variant defaults to detecting the language per
        // utterance. Naming it is strictly more accurate when it is known.
        if model.mode() == NemotronMode::Multilingual
            && let Some(language) = language
        {
            model.set_target_lang(language).map_err(|e| AsrError::Load(e.to_string()))?;
        }

        let label = format!(
            "nemotron-streaming-0.6b ({}) on {}",
            match model.mode() {
                NemotronMode::Multilingual => "multilingual",
                NemotronMode::EnglishOnly => "en",
            },
            if matches!(provider, ExecutionProvider::Cpu) { "cpu" } else { "cuda" }
        );
        tracing::info!(
            model = %label,
            chunk_ms = model.chunk_samples() * 1000 / 16_000,
            load_ms = started.elapsed().as_millis(),
            "streaming model ready"
        );

        Ok(Self { model, label, fed: 0 })
    }
}

impl StreamingTranscriber for NemotronStream {
    fn name(&self) -> String {
        self.label.clone()
    }

    fn chunk_samples(&self) -> usize {
        self.model.chunk_samples()
    }

    fn reset(&mut self) {
        self.model.reset();
        self.fed = 0;
    }

    fn feed(&mut self, samples: &[f32]) -> Result<String> {
        if samples.is_empty() {
            return Ok(String::new());
        }
        self.fed += samples.len();
        self.model.transcribe_chunk(samples).map_err(|e| AsrError::Inference(e.to_string()))
    }

    fn finish(&mut self) -> Result<String> {
        // The model only decodes on a full chunk boundary, and there is no flush
        // in its API. Padding with silence to the next boundary is what forces
        // the tail out — without it the last word of every dictation is lost.
        let chunk = self.chunk_samples();
        if chunk == 0 {
            return Ok(String::new());
        }
        let remainder = self.fed % chunk;
        let padding = if remainder == 0 { chunk } else { chunk - remainder + chunk };
        self.feed(&vec![0.0f32; padding])
    }
}

/// Transcribe a whole recording through a streaming model, as the daemon does.
///
/// Reports the tail separately: the time from the last real audio arriving to
/// the text being complete. That, not the total, is what a user waits through.
///
/// # Errors
/// Fails if inference fails.
pub fn transcribe_all(
    model: &mut dyn StreamingTranscriber,
    samples: &[f32],
) -> Result<(Transcript, Duration)> {
    model.reset();
    let chunk = model.chunk_samples().max(1);
    let started = Instant::now();

    let mut text = String::new();
    for piece in samples.chunks(chunk) {
        text.push_str(&model.feed(piece)?);
    }
    let before_tail = Instant::now();
    text.push_str(&model.finish()?);
    let tail = before_tail.elapsed();

    Ok((Transcript::new(text.trim(), started.elapsed()), tail))
}
