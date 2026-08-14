//! The dictation loop: trigger edges in, text at the cursor out.
//!
//! All the decisions live in [`murmur_core::Session`], which is pure. This crate
//! only performs the [`Command`]s that come back and feeds the resulting facts
//! in as [`Event`]s. Keeping the split honest is what makes the timing numbers
//! trustworthy — the engine measures, the core decides.

pub mod partials;
pub mod recorder;
pub mod stats;

use murmur_asr::Transcriber;
use murmur_core::{Command, Event, Hud, Millis, Phase, Session, Stage};
use murmur_hotkey::{Edge, TriggerEvent};
use murmur_inject::TextSink;
use std::collections::VecDeque;
use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub use partials::{Partials, SharedTranscriber};
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

/// Boxed devices cannot be `Debug`, so this reports what it is rather than
/// what it holds.
pub struct Engine {
    session: Session,
    recorder: Box<dyn Recorder>,
    transcriber: SharedTranscriber,
    partials: Partials,
    sink: Box<dyn TextSink>,
    surface: Box<dyn Surface>,
    stats: Stats,
    origin: Instant,
}

impl std::fmt::Debug for Engine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Engine")
            .field("phase", &self.session.phase())
            .field("utterances", &self.stats.utterances())
            .finish_non_exhaustive()
    }
}

impl Engine {
    #[must_use]
    pub fn new(
        session: Session,
        recorder: Box<dyn Recorder>,
        transcriber: Box<dyn Transcriber>,
        sink: Box<dyn TextSink>,
        surface: Box<dyn Surface>,
    ) -> Self {
        // One loaded model serves both the live partials and the final pass, so
        // what the user watches appear is produced by exactly what types it.
        let transcriber: SharedTranscriber = Arc::new(Mutex::new(transcriber));
        Self {
            session,
            recorder,
            partials: Partials::spawn(Arc::clone(&transcriber)),
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
                Ok(TriggerEvent {
                    edge: Edge::Down, ..
                }) => {
                    self.pump(Event::TriggerDown(self.now()));
                }
                Ok(TriggerEvent { edge: Edge::Up, .. }) => {
                    self.pump(Event::TriggerUp(self.now()));
                }
                Err(RecvTimeoutError::Timeout) => {
                    if self.session.is_capturing() {
                        self.surface.level(self.recorder.level());
                        self.request_partial();
                    }
                    self.deliver_partials();
                    self.pump(Event::Tick(self.now()));
                }
                Err(RecvTimeoutError::Disconnected) => return Ok(()),
            }
        }
    }

    /// Hand the worker the audio recorded so far, if it is ready for more.
    fn request_partial(&mut self) {
        let Phase::Capturing { id, .. } = self.session.phase() else {
            return;
        };
        if let Some(recorded) = self.recorder.snapshot() {
            self.partials.offer(id, recorded.samples, Instant::now());
        }
    }

    /// Show whatever the worker finished. Stale text is dropped by the session,
    /// which knows which utterance is current and this thread does not.
    fn deliver_partials(&mut self) {
        for reply in self.partials.collect() {
            tracing::debug!(id = reply.id, took_ms = reply.took.as_millis(), "partial");
            self.pump(Event::Partial {
                at: self.now(),
                id: reply.id,
                text: reply.text,
            });
        }
    }

    /// Feed one event in and run the resulting work to completion.
    ///
    /// Commands can produce further events — a finished transcription becomes a
    /// `Final`, which becomes an `Inject`, which becomes an `Injected` — so this
    /// drains a queue rather than recursing.
    fn pump(&mut self, event: Event) {
        let mut queue = VecDeque::from([event]);
        while let Some(event) = queue.pop_front() {
            for command in self.session.handle(event) {
                if let Some(next) = self.execute(command) {
                    queue.push_back(next);
                }
            }
        }
    }

    /// Carry out one command, and report whatever fact it produced.
    ///
    /// Infallible on purpose: a device failure is an [`Event::Failed`] for the
    /// session to act on, not an error that ends the loop.
    fn execute(&mut self, command: Command) -> Option<Event> {
        match command {
            Command::StartCapture { .. } => {
                self.partials.reset();
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
                    Ok(text) => Some(Event::Final {
                        at: self.now(),
                        id,
                        text,
                    }),
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
        }
    }

    /// Take the recording and transcribe it, reporting failure as a message
    /// rather than an error so one bad utterance cannot end the session.
    fn transcribe(&mut self) -> std::result::Result<String, String> {
        let capture = self.recorder.finish()?;
        if capture.is_empty() {
            return Ok(String::new());
        }
        // Blocks until any partial in flight finishes, which is the price of
        // sharing one loaded model. Bounded by a single pass, and the alternative
        // is a second copy of the weights on the GPU.
        let transcript = self
            .transcriber
            .lock()
            .map_err(|_| "transcriber lock poisoned".to_owned())?
            .transcribe(&capture.samples)
            .map_err(|e| e.to_string())?;
        tracing::info!(
            audio_ms = capture.duration.as_millis(),
            trimmed = capture.trimmed,
            inference_ms = transcript.elapsed.as_millis(),
            rtf = transcript.realtime_factor(capture.duration),
            "transcribed"
        );
        self.stats
            .record_audio(capture.duration, transcript.elapsed);
        Ok(transcript.text)
    }
}
