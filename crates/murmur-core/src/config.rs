use crate::session::Tuning;
use crate::text::FormatConfig;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub trigger: TriggerConfig,
    pub audio: AudioConfig,
    pub asr: AsrConfig,
    pub polish: PolishConfig,
    pub inject: InjectConfig,
    pub format: FormatConfig,
    pub tuning: Tuning,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TriggerConfig {
    /// Linux input key name, as reported by `murmur keys`. Right Ctrl is the
    /// default because no desktop binds it alone and it is reachable one-handed.
    pub key: String,
    /// Allow double-tap to enter hands-free capture.
    pub hands_free: bool,
}

impl Default for TriggerConfig {
    fn default() -> Self {
        Self {
            key: "RIGHTCTRL".into(),
            hands_free: true,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AudioConfig {
    /// Substring of the input device name; `None` uses the system default.
    pub device: Option<String>,
    /// Audio kept from before the key went down, so a fast talker is never clipped.
    pub preroll_ms: u32,
    /// Trim silence at the tail before sending to the transcriber.
    pub vad: bool,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            device: None,
            preroll_ms: 300,
            vad: true,
        }
    }
}

/// Sample rate every transcriber in this project expects.
pub const TARGET_SAMPLE_RATE: u32 = 16_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AsrEngine {
    /// NVIDIA Parakeet TDT via ONNX Runtime: one call per finished recording.
    Parakeet,
    /// NVIDIA Nemotron cache-aware streaming ASR: transcribes while you speak.
    Nemotron,
    /// whisper.cpp, as a portability fallback.
    Whisper,
    /// Echoes fixture text; used by the simulator and by `murmur doctor`.
    Mock,
}

/// Which weights to load.
///
/// `Auto` picks fp32 when the GPU can actually be used and int8 otherwise. That
/// is not a speed trade-off — the two are within noise of each other on a CPU —
/// but a footprint one: int8 is a quarter of the size, and fp32 is the only
/// precision the CUDA provider has kernels for.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Precision {
    #[default]
    Auto,
    Fp32,
    Int8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Accelerator {
    Cuda,
    TensorRt,
    Cpu,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AsrConfig {
    pub engine: AsrEngine,
    /// A directory of models, or a single model directory.
    pub model_dir: String,
    pub precision: Precision,
    pub accelerator: Accelerator,
    /// `None` lets multilingual models detect the language per utterance.
    pub language: Option<String>,
}

impl Default for AsrConfig {
    fn default() -> Self {
        Self {
            engine: AsrEngine::Parakeet,
            model_dir: "~/.local/share/murmur/models".into(),
            precision: Precision::Auto,
            accelerator: Accelerator::Cuda,
            language: None,
        }
    }
}

/// Optional LLM pass over the transcript.
///
/// Off by default, and deliberately so: it buys punctuation and tone at the cost
/// of the one thing the product is judged on, the delay between releasing the key
/// and seeing text. Any OpenAI-compatible endpoint works — Ollama, llama.cpp,
/// vLLM — so the model is a config value, not a build dependency.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PolishConfig {
    pub enabled: bool,
    pub endpoint: String,
    pub model: String,
    /// Past this, the raw transcript is injected instead. Never leave the user waiting.
    pub deadline_ms: u32,
    pub instructions: String,
}

impl Default for PolishConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            endpoint: "http://127.0.0.1:11434/v1".into(),
            model: "nemotron-3.5-lightning".into(),
            deadline_ms: 700,
            instructions: "Fix punctuation, capitalisation and obvious transcription \
                errors. Preserve the speaker's wording and register. Never answer, \
                explain, or add content. Reply with the corrected text only."
                .into(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum InjectBackend {
    /// Probe every backend and use the best one available on this desktop.
    Auto,
    /// XDG `RemoteDesktop` portal: full Unicode on Wayland, one consent prompt.
    Portal,
    /// A kernel-level virtual keyboard via `/dev/uinput`. No consent, no compositor
    /// cooperation, but limited to what the active layout can type.
    Uinput,
    /// Set the clipboard and synthesise a paste. Fastest for long text.
    Clipboard,
    /// XTEST, for X11 and `XWayland` sessions.
    X11,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct InjectConfig {
    pub backend: InjectBackend,
    /// Above this many characters, paste instead of typing.
    pub paste_threshold: usize,
    /// Gap between synthesised keystrokes. Zero drops characters in some toolkits.
    pub keystroke_delay_us: u64,
    /// Put the clipboard back the way it was after a paste.
    pub restore_clipboard: bool,
}

impl Default for InjectConfig {
    fn default() -> Self {
        Self {
            backend: InjectBackend::Auto,
            paste_threshold: 80,
            keystroke_delay_us: 1_200,
            restore_clipboard: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_serde_round_trippable() {
        let config = Config::default();
        let json = serde_json::to_string(&config).expect("serialise");
        let back: Config = serde_json::from_str(&json).expect("deserialise");
        assert_eq!(config, back);
    }

    #[test]
    fn an_empty_document_yields_the_defaults() {
        let back: Config = serde_json::from_str("{}").expect("deserialise");
        assert_eq!(back, Config::default());
    }

    #[test]
    fn a_partial_document_overrides_only_what_it_names() {
        let back: Config =
            serde_json::from_str(r#"{"trigger":{"key":"CAPSLOCK"}}"#).expect("deserialise");
        assert_eq!(back.trigger.key, "CAPSLOCK");
        assert!(back.trigger.hands_free);
        assert_eq!(back.audio, AudioConfig::default());
    }

    #[test]
    fn an_unknown_key_is_rejected_rather_than_silently_ignored() {
        let err = serde_json::from_str::<Config>(r#"{"trigger":{"keyy":"CAPSLOCK"}}"#);
        assert!(err.is_err(), "typo in config was accepted");
    }

    #[test]
    fn precision_defaults_to_auto_so_hardware_decides() {
        assert_eq!(AsrConfig::default().precision, Precision::Auto);
    }

    #[test]
    fn polish_is_off_by_default() {
        assert!(
            !Config::default().polish.enabled,
            "latency budget must be opt-out, not opt-in"
        );
    }
}
