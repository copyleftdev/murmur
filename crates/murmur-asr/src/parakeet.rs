use crate::{AsrError, Transcriber, Transcript};
use crate::models;
use murmur_core::config::{Accelerator, Precision, TARGET_SAMPLE_RATE};
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
    /// Find the best weights under `root` and load them.
    ///
    /// `root` may be a directory of models or a single model directory. The
    /// choice between precisions is made from what the hardware can actually
    /// do — see [`models::choose`] — rather than from what the config asked for,
    /// because loading fp32 onto a machine with no working GPU costs 1.9 GB of
    /// memory for nothing.
    ///
    /// # Errors
    /// Fails if no model is found under `root`, or it cannot be loaded.
    pub fn open(
        root: &Path,
        precision: Precision,
        accelerator: Accelerator,
    ) -> Result<Self, AsrError> {
        let gpu_usable = gpu_usable(accelerator);
        let variants = models::discover(root);
        let chosen = models::choose(&variants, gpu_usable, precision)
            .ok_or_else(|| AsrError::ModelMissing(root.display().to_string()))?;

        tracing::info!(
            dir = %chosen.dir.display(),
            precision = ?chosen.kind,
            gpu_usable,
            candidates = variants.len(),
            "selected weights"
        );
        // Without a usable GPU there is nothing for the accelerator to do, and
        // asking for one only produces a warning the user cannot act on.
        let effective = if gpu_usable { accelerator } else { Accelerator::Cpu };
        Self::load(&chosen.dir, effective)
    }

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

        #[cfg(feature = "cuda")]
        let failures_before = crate::cuda::registration_failures();

        let started = Instant::now();
        let model = ParakeetTDT::from_pretrained(dir, Some(config))
            .map_err(|e| AsrError::Load(e.to_string()))?;

        // ORT logs a failed provider registration and silently continues on the
        // CPU, so the device is reported from its verdict, never from our request.
        #[cfg(feature = "cuda")]
        let requested = if crate::cuda::registration_failures() > failures_before {
            tracing::warn!(
                "ONNX Runtime could not register the requested execution provider; \
                 this model is running on the CPU"
            );
            format!("cpu ({requested} registration failed)")
        } else {
            requested
        };

        let quantised = dir.join("encoder-model.int8.onnx").exists()
            && !dir.join("encoder-model.onnx").exists();
        if quantised && requested.starts_with("cuda") {
            tracing::warn!(
                "int8 weights run poorly on the CUDA provider: unsupported ops fall back \
                 to the CPU and the graph is copied across the bus repeatedly. Use the \
                 fp32 model on a GPU."
            );
        }
        let label = format!(
            "parakeet-tdt-0.6b-v3{} on {requested}",
            if quantised { " (int8)" } else { "" }
        );
        tracing::info!(model = %label, load_ms = started.elapsed().as_millis(), "model ready");

        Ok(Self { model, label })
    }
}

/// Can this machine actually run a model on the GPU right now?
fn gpu_usable(accelerator: Accelerator) -> bool {
    if matches!(accelerator, Accelerator::Cpu) {
        return false;
    }
    #[cfg(feature = "cuda")]
    {
        crate::cuda::is_usable()
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

/// Map the configured accelerator to a provider this binary was actually built
/// with *and* can actually load.
fn provider_for(accelerator: Accelerator) -> (ExecutionProvider, String) {
    match accelerator {
        Accelerator::Cpu => (ExecutionProvider::Cpu, "cpu".into()),
        Accelerator::Cuda => {
            #[cfg(feature = "cuda")]
            {
                crate::cuda::ensure_runtime();
                (ExecutionProvider::Cuda, "cuda".into())
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
