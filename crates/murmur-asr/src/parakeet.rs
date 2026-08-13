use crate::{AsrError, Transcriber, Transcript};
use murmur_core::config::{Accelerator, TARGET_SAMPLE_RATE};
use parakeet_rs::{ExecutionConfig, ExecutionProvider, ParakeetTDT, Transcriber as _};
use std::path::Path;
use std::time::Instant;

/// NVIDIA Parakeet TDT via ONNX Runtime.
///
/// The model is a FastConformer transducer that emits punctuation and casing of
/// its own, which is why Murmur's formatter deliberately does not try to add
/// either — it only applies the user's dictionary and spoken commands on top.
pub struct Parakeet {
    model: ParakeetTDT,
    label: String,
}

impl std::fmt::Debug for Parakeet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Parakeet").field("label", &self.label).finish_non_exhaustive()
    }
}

impl Parakeet {
    /// Load the model in `dir`, which must hold the encoder, decoder and vocab.
    ///
    /// # Errors
    /// Fails if the directory is missing or the model cannot be loaded.
    pub fn load(dir: &Path, accelerator: Accelerator) -> Result<Self, AsrError> {
        if !dir.is_dir() {
            return Err(AsrError::ModelMissing(dir.display().to_string()));
        }

        let (provider, requested) = provider_for(accelerator);
        let config = ExecutionConfig::new().with_execution_provider(provider);
        let requested = requested.to_owned();

        let started = Instant::now();
        let model = ParakeetTDT::from_pretrained(dir, Some(config))
            .map_err(|e| AsrError::Load(e.to_string()))?;

        let quantised = dir.join("encoder-model.int8.onnx").exists()
            && !dir.join("encoder-model.onnx").exists();
        let label = format!(
            "parakeet-tdt-0.6b-v3{} on {requested}",
            if quantised { " (int8)" } else { "" }
        );
        tracing::info!(model = %label, load_ms = started.elapsed().as_millis(), "model ready");

        Ok(Self { model, label })
    }
}

/// Can ONNX Runtime's CUDA provider actually be loaded on this machine?
///
/// ONNX Runtime registers execution providers by `dlopen`-ing a shared library
/// and *logs* the failure rather than refusing to build the session, so a model
/// asked to run on the GPU will quietly run on the CPU instead. Doing the same
/// `dlopen` ourselves, up front, turns that silent downgrade into a fact we can
/// report — and the loader's own error message is exactly the diagnostic the
/// user needs ("libcublasLt.so.13: cannot open shared object file").
///
/// # Errors
/// Returns the dynamic loader's reason the provider is unusable.
#[cfg(feature = "cuda")]
pub fn cuda_availability() -> Result<(), String> {
    const PROVIDER: &str = "libonnxruntime_providers_cuda.so";

    let beside_binary = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|dir| dir.join(PROVIDER)));
    let candidates =
        [beside_binary, Some(std::path::PathBuf::from(PROVIDER))].into_iter().flatten();

    let mut reason = String::from("provider library not found");
    let mut found_the_library = false;
    for candidate in candidates {
        // SAFETY: loading a library runs its initialisers. This one ships with
        // ONNX Runtime and is about to be loaded by ORT itself regardless.
        match unsafe { libloading::Library::new(&candidate) } {
            Ok(_) => return Ok(()),
            Err(error) => {
                // A candidate that exists explains the *real* problem — a missing
                // CUDA dependency — while a candidate that does not merely says
                // so. Never let the second overwrite the first.
                if !found_the_library {
                    reason = error.to_string();
                    found_the_library = candidate.exists();
                }
            }
        }
    }
    Err(reason)
}

/// Map the configured accelerator to a provider this binary was actually built
/// with *and* can actually load.
fn provider_for(accelerator: Accelerator) -> (ExecutionProvider, String) {
    match accelerator {
        Accelerator::Cpu => (ExecutionProvider::Cpu, "cpu".into()),
        Accelerator::Cuda => {
            #[cfg(feature = "cuda")]
            {
                match cuda_availability() {
                    Ok(()) => (ExecutionProvider::Cuda, "cuda".into()),
                    Err(why) => {
                        tracing::warn!(
                            %why,
                            "CUDA provider cannot be loaded; running on CPU instead"
                        );
                        (ExecutionProvider::Cpu, "cpu (cuda unavailable)".into())
                    }
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                tracing::warn!(
                    "config asks for CUDA but this build has no CUDA execution provider; \
                     rebuild with `--features cuda`. Falling back to CPU."
                );
                (ExecutionProvider::Cpu, "cpu (cuda not built in)".into())
            }
        }
        Accelerator::TensorRt => {
            #[cfg(feature = "tensorrt")]
            {
                (ExecutionProvider::TensorRT, "tensorrt".into())
            }
            #[cfg(not(feature = "tensorrt"))]
            {
                tracing::warn!(
                    "config asks for TensorRT but this build has no TensorRT execution \
                     provider; rebuild with `--features tensorrt`. Falling back to CPU."
                );
                (ExecutionProvider::Cpu, "cpu (tensorrt not built in)".into())
            }
        }
    }
}

impl Transcriber for Parakeet {
    fn name(&self) -> String {
        self.label.clone()
    }

    fn transcribe(&mut self, samples: &[f32]) -> Result<Transcript, AsrError> {
        let started = Instant::now();
        let result = self
            .model
            .transcribe_samples(samples.to_vec(), TARGET_SAMPLE_RATE, 1, None)
            .map_err(|e| AsrError::Inference(e.to_string()))?;
        Ok(Transcript::new(result.text.trim(), started.elapsed()))
    }
}
