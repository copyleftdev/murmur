//! The dictation loop: trigger edges in, text at the cursor out.
//!
//! All the decisions live in [`murmur_core::Session`], which is pure. This crate
//! only performs the [`Command`]s that come back and feeds the resulting facts
//! in as [`Event`]s. Keeping the split honest is what makes the timing numbers
//! trustworthy — the engine measures, the core decides.

pub mod recorder;
pub mod stats;

use murmur_asr::Transcriber;
use murmur_core::{Command, Event, Hud, Millis, Session, Stage};
use murmur_hotkey::{Edge, TriggerEvent};
use murmur_inject::TextSink;
use std::collections::VecDeque;
use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::time::{Duration, Instant};

pub use recorder::{Fixture, Recorded, Recorder};
pub use stats::Stats;

/// How often the session is ticked when no key is moving.
///
/// This is the resolution of the utterance cap and the transcription timeout,
/// and it is also the HUD's meter refresh, so it wants to be smooth to the eye
/// rather than merely correct.
const TICK: Duration = Duration::from_millis(50);

#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error(transparent)]
    Audio(#[from] murmur_audio::AudioError),
    #[error(transparent)]
    Inject(#[from] murmur_inject::InjectError),
}

type Result<T> = std::result::Result<T, EngineError>;

/// Where the user is told what is happening.
pub trait Surface {
    fn show(&mut self, hud: &Hud);
    /// Input level in `0.0..=1.0`, pushed while capturing.
    fn level(&mut self, _level: f32) {}
    /// Text that actually reached the cursor.
    fn emitted(&mut self, _text: &str) {}
    /// One completed utterance, for the running latency report.
    fn completed(&mut self, _stats: &Stats) {}
}

pub struct Engine {
    session: Session,
    recorder: Box<dyn Recorder>,
    transcriber: Box<dyn Transcriber>,
    sink: Box<dyn TextSink>,
    surface: Box<dyn Surface>,
    stats: Stats,
    origin: Instant,
}

impl Engine {
    pub fn new(
        session: Session,
        recorder: Box<dyn Recorder>,
        transcriber: Box<dyn Transcriber>,
        sink: Box<dyn TextSink>,
        surface: Box<dyn Surface>,
    ) -> Self {
        Self {
            session,
            recorder,
            transcriber,
            sink,
            surface,
            stats: Stats::default(),
            origin: Instant::now(),
        }
    }

    #[must_use]
    pub fn stats(&self) -> &Stats {
        &self.stats
    }

    fn now(&self) -> Millis {
        Millis(u64::try_from(self.origin.elapsed().as_millis()).unwrap_or(u64::MAX))
    }

    /// Run until the trigger source goes away.
    ///
    /// # Errors
    /// Propagates unrecoverable device failures. Failures of a single utterance
    /// are reported to the session instead, which is what lets one bad
    /// transcription not end the session.
    pub fn run(&mut self, triggers: &Receiver<TriggerEvent>) -> Result<()> {
        loop {
            match triggers.recv_timeout(TICK) {
                Ok(TriggerEvent { edge: Edge::Down, .. }) => {
                    self.pump(Event::TriggerDown(self.now()))?;
                }
                Ok(TriggerEvent { edge: Edge::Up, .. }) => {
                    self.pump(Event::TriggerUp(self.now()))?;
                }
                Err(RecvTimeoutError::Timeout) => {
                    if self.session.is_capturing() {
                        self.surface.level(self.recorder.level());
                    }
                    self.pump(Event::Tick(self.now()))?;
                }
                Err(RecvTimeoutError::Disconnected) => return Ok(()),
            }
        }
    }

    /// Feed one event in and run the resulting work to completion.
    ///
    /// Commands can produce further events — a finished transcription becomes a
    /// `Final`, which becomes an `Inject`, which becomes an `Injected` — so this
    /// drains a queue rather than recursing.
    fn pump(&mut self, event: Event) -> Result<()> {
        let mut queue = VecDeque::from([event]);
        while let Some(event) = queue.pop_front() {
            for command in self.session.handle(event) {
                if let Some(next) = self.execute(command)? {
                    queue.push_back(next);
                }
            }
        }
        Ok(())
    }

    fn execute(&mut self, command: Command) -> Result<Option<Event>> {
        Ok(match command {
            Command::StartCapture { .. } => {
                self.recorder.begin();
                None
            }
            Command::Discard { .. } => {
                self.recorder.discard();
                None
            }
            Command::StopCapture { id } => {
                let at = self.now();
                match self.transcribe() {
                    Ok(text) => Some(Event::Final { at: self.now(), id, text }),
                    Err(message) => Some(Event::Failed {
                        at,
                        id,
                        stage: Stage::Transcribe,
                        message,
                    }),
                }
            }
            Command::Inject { id, text } => {
                let at = self.now();
                match self.sink.inject(&text) {
                    Ok(()) => {
                        self.stats.record_emission(text.chars().count());
                        self.surface.emitted(&text);
                        Some(Event::Injected { at: self.now(), id })
                    }
                    Err(error) => Some(Event::Failed {
                        at,
                        id,
                        stage: Stage::Inject,
                        message: error.to_string(),
                    }),
                }
            }
            Command::Hud(hud) => {
                self.surface.show(&hud);
                if matches!(hud, Hud::Hidden) {
                    if let Some(latency) = self.session.last_latency() {
                        self.stats.record_latency(latency);
                        self.surface.completed(&self.stats);
                    }
                }
                None
            }
        })
    }

    /// Take the recording and transcribe it, reporting failure as a message
    /// rather than an error so one bad utterance cannot end the session.
    fn transcribe(&mut self) -> std::result::Result<String, String> {
        let capture = self.recorder.finish()?;
        if capture.is_empty() {
            return Ok(String::new());
        }
        let transcript =
            self.transcriber.transcribe(&capture.samples).map_err(|e| e.to_string())?;
        tracing::info!(
            audio_ms = capture.duration.as_millis(),
            trimmed = capture.trimmed,
            inference_ms = transcript.elapsed.as_millis(),
            rtf = transcript.realtime_factor(capture.duration),
            "transcribed"
        );
        self.stats.record_audio(capture.duration, transcript.elapsed);
        Ok(transcript.text)
    }
}
