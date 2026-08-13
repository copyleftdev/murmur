//! Pure dictation logic: no threads, no clocks, no IO.
//!
//! Everything that decides *what* Murmur does lives here; everything that
//! decides *how* lives in the backend crates. The daemon feeds this crate
//! timestamped [`Event`]s and executes the [`Command`]s it returns, which means
//! a full session — key presses, transcripts, injections, timeouts — is a pure
//! function of its event log and can be replayed exactly.

pub mod config;
pub mod session;
pub mod text;
pub mod time;

pub use config::{
    Accelerator, AsrConfig, AsrEngine, AudioConfig, Config, InjectBackend, InjectConfig,
    PolishConfig, Precision, TARGET_SAMPLE_RATE, TriggerConfig,
};
pub use session::{Command, Event, Hud, Latency, Mode, Phase, Session, Stage, Tuning, UtteranceId};
pub use text::{DictEntry, EmitContext, FormatConfig, Formatter};
pub use time::Millis;
