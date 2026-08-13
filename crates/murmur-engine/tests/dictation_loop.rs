//! The whole loop, driven from fixtures.
//!
//! These run the real [`Engine`] against the real [`Session`] with every device
//! replaced: a scripted recorder, a scripted transcriber, and a sink that keeps
//! what it was given instead of typing it. That makes the dictation path — press,
//! record, transcribe, format, inject, recover — assertable on any machine, with
//! no microphone, no window to type into, and no model on disk.

use murmur_asr::Mock;
use murmur_core::{FormatConfig, Formatter, Hud, Millis, Session, Tuning};
use murmur_engine::{Engine, Fixture, Recorder, Stats, Surface};
use murmur_hotkey::{Edge, TriggerEvent};
use murmur_inject::{InjectError, TextSink};
use std::sync::mpsc::channel;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// A sink that records what it was asked to type.
#[derive(Clone, Default)]
struct Spy {
    emitted: Arc<Mutex<Vec<String>>>,
    fail: bool,
}

impl Spy {
    fn failing() -> Self {
        Self { fail: true, ..Self::default() }
    }

    fn emitted(&self) -> Vec<String> {
        self.emitted.lock().unwrap().clone()
    }
}

impl TextSink for Spy {
    fn name(&self) -> &'static str {
        "spy"
    }

    fn inject(&mut self, text: &str) -> Result<(), InjectError> {
        if self.fail {
            return Err(InjectError::Unavailable("spy"));
        }
        self.emitted.lock().unwrap().push(text.to_owned());
        Ok(())
    }
}

#[derive(Clone, Default)]
struct Watcher {
    huds: Arc<Mutex<Vec<String>>>,
}

impl Watcher {
    fn saw_error(&self) -> bool {
        self.huds.lock().unwrap().iter().any(|h| h.starts_with("error:"))
    }
}

impl Surface for Watcher {
    fn show(&mut self, hud: &Hud) {
        let label = match hud {
            Hud::Hidden => "hidden".to_owned(),
            Hud::Listening { .. } => "listening".to_owned(),
            Hud::Partial { .. } => "partial".to_owned(),
            Hud::Thinking => "thinking".to_owned(),
            Hud::Error { message } => format!("error: {message}"),
        };
        self.huds.lock().unwrap().push(label);
    }
}

/// Tuning that treats any measurable hold as speech.
///
/// The engine reads a real clock, so tests hold the key for a few milliseconds
/// rather than pretending time does not pass.
fn tuning() -> Tuning {
    Tuning { tap_max: Millis(0), ..Tuning::default() }
}

fn engine(recorder: Box<dyn Recorder>, asr: Mock, sink: Spy, watcher: Watcher) -> Engine {
    Engine::new(
        Session::new(tuning(), Formatter::new(FormatConfig::default())),
        recorder,
        Box::new(asr),
        Box::new(sink),
        Box::new(watcher),
    )
}

/// Press and release the trigger `holds` times, then close the channel.
fn drive(engine: &mut Engine, holds: &[Duration]) {
    let (tx, rx) = channel();
    let holds = holds.to_vec();
    std::thread::spawn(move || {
        for hold in holds {
            if tx.send(TriggerEvent { edge: Edge::Down, at: Instant::now() }).is_err() {
                return;
            }
            std::thread::sleep(hold);
            if tx.send(TriggerEvent { edge: Edge::Up, at: Instant::now() }).is_err() {
                return;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
    });
    engine.run(&rx).expect("engine loop");
}

const HOLD: Duration = Duration::from_millis(40);

#[test]
fn a_hold_puts_the_transcript_at_the_cursor() {
    let sink = Spy::default();
    let mut engine =
        engine(Box::new(Fixture::speaking()), Mock::new(["Hello world."]), sink.clone(), Watcher::default());
    drive(&mut engine, &[HOLD]);

    assert_eq!(sink.emitted(), vec!["Hello world. "]);
    assert_eq!(engine.stats().utterances(), 1);
}

#[test]
fn a_tap_shorter_than_the_threshold_types_nothing() {
    let sink = Spy::default();
    let mut engine = Engine::new(
        // A realistic tap threshold, and a hold well under it.
        Session::new(Tuning::default(), Formatter::new(FormatConfig::default())),
        Box::new(Fixture::speaking()),
        Box::new(Mock::default()),
        Box::new(sink.clone()),
        Box::new(Watcher::default()),
    );
    drive(&mut engine, &[Duration::from_millis(30)]);

    assert!(sink.emitted().is_empty(), "a stray tap typed {:?}", sink.emitted());
    assert_eq!(engine.stats().utterances(), 0);
}

#[test]
fn silence_types_nothing_but_still_completes_cleanly() {
    let sink = Spy::default();
    let watcher = Watcher::default();
    let mut engine =
        engine(Box::new(Fixture::silent()), Mock::default(), sink.clone(), watcher.clone());
    drive(&mut engine, &[HOLD]);

    assert!(sink.emitted().is_empty());
    assert!(!watcher.saw_error(), "silence must not be reported as a failure");
}

#[test]
fn consecutive_dictations_flow_into_one_another() {
    let sink = Spy::default();
    let mut engine = engine(
        Box::new(Fixture::speaking()),
        Mock::new(["First one.", "Second one."]),
        sink.clone(),
        Watcher::default(),
    );
    drive(&mut engine, &[HOLD, HOLD]);

    assert_eq!(sink.emitted(), vec!["First one. ", " Second one. "]);
}

#[test]
fn a_failing_transcriber_reports_and_the_next_dictation_still_works() {
    let sink = Spy::default();
    let watcher = Watcher::default();
    let mut engine =
        engine(Box::new(Fixture::speaking()), Mock::failing(), sink.clone(), watcher.clone());
    drive(&mut engine, &[HOLD, HOLD]);

    assert!(sink.emitted().is_empty());
    assert!(watcher.saw_error(), "a failed transcription must be surfaced, not swallowed");
}

#[test]
fn a_failing_sink_does_not_wedge_the_session() {
    let watcher = Watcher::default();
    let mut engine = engine(
        Box::new(Fixture::speaking()),
        Mock::default(),
        Spy::failing(),
        watcher.clone(),
    );
    drive(&mut engine, &[HOLD, HOLD]);

    assert!(watcher.saw_error());
    assert_eq!(engine.stats().utterances(), 0, "nothing reached the cursor");
}

#[test]
fn a_failing_recorder_is_reported_rather_than_crashing_the_loop() {
    let sink = Spy::default();
    let watcher = Watcher::default();
    let mut engine = engine(
        Box::new(Fixture::failing("microphone went away")),
        Mock::default(),
        sink.clone(),
        watcher.clone(),
    );
    drive(&mut engine, &[HOLD]);

    assert!(sink.emitted().is_empty());
    assert!(watcher.saw_error());
}

#[test]
fn recording_starts_on_press_and_never_outlives_the_utterance() {
    let fixture = Fixture::speaking();
    let mut engine =
        engine(Box::new(fixture), Mock::default(), Spy::default(), Watcher::default());
    drive(&mut engine, &[HOLD, HOLD, HOLD]);

    assert_eq!(engine.stats().utterances(), 3);
}

#[test]
fn the_dictionary_and_spoken_commands_reach_the_cursor() {
    use murmur_core::DictEntry;

    let sink = Spy::default();
    let format = FormatConfig {
        dictionary: vec![DictEntry::new("murmur", "Murmur")],
        ..FormatConfig::default()
    };
    let mut engine = Engine::new(
        Session::new(tuning(), Formatter::new(format)),
        Box::new(Fixture::speaking()),
        Box::new(Mock::new(["murmur is listening new line and typing"])),
        Box::new(sink.clone()),
        Box::new(Watcher::default()),
    );
    drive(&mut engine, &[HOLD]);

    assert_eq!(sink.emitted(), vec!["Murmur is listening\nand typing "]);
}

#[test]
fn latency_is_measured_from_release_to_text() {
    let sink = Spy::default();
    let mut engine = engine(
        Box::new(Fixture::speaking()),
        Mock::default().with_delay(Duration::from_millis(60)),
        sink.clone(),
        Watcher::default(),
    );
    drive(&mut engine, &[HOLD]);

    let stats: &Stats = engine.stats();
    let release_to_text = stats.release_to_text(100.0).expect("one sample");
    let transcribe = stats.transcribe(100.0).expect("one sample");
    assert!(
        transcribe >= Millis(55),
        "the scripted 60ms of inference should dominate, got {transcribe}"
    );
    assert!(
        release_to_text >= transcribe,
        "release-to-text ({release_to_text}) must include transcription ({transcribe})"
    );
    assert!(
        release_to_text < Millis(2_000),
        "the loop added unexplained latency: {release_to_text}"
    );
}
