//! The dictation engine, running beside the interface rather than inside it.
//!
//! iced owns the main thread, and the microphone cannot leave the thread that
//! opened it, so the engine runs on its own thread and speaks to the interface
//! only in messages. That split is why the HUD cannot stall dictation: the worst
//! a slow frame can do is show stale text.

use crate::Message;
use futures::channel::mpsc::Sender;
use murmur_core::{Config, Formatter, Hud, Millis, Session};
use murmur_engine::{Engine, Stats, Surface};

/// Sends what the engine is doing to the interface, dropping messages rather
/// than blocking.
///
/// A full channel means the interface is behind, and the newest state is the
/// only one worth showing — waiting for room would put interface latency on the
/// dictation path, which is exactly backwards.
pub struct ChannelSurface {
    to_interface: Sender<Message>,
}

impl ChannelSurface {
    pub fn new(to_interface: Sender<Message>) -> Self {
        Self { to_interface }
    }

    fn send(&mut self, message: Message) {
        let _ = self.to_interface.try_send(message);
    }
}

impl Surface for ChannelSurface {
    fn show(&mut self, hud: &Hud) {
        self.send(Message::Hud(hud.clone()));
    }

    fn level(&mut self, level: f32) {
        self.send(Message::Level(level));
    }

    fn emitted(&mut self, text: &str) {
        self.send(Message::Emitted(text.to_owned()));
    }

    fn completed(&mut self, stats: &Stats) {
        if let Some(latency) = stats.release_to_text(100.0) {
            self.send(Message::Completed(latency));
        }
    }
}

/// Build everything the engine needs and run it until the trigger goes away.
///
/// # Errors
/// Fails if a device, model or backend cannot be opened.
pub fn run(config: &Config, mut to_interface: Sender<Message>) -> anyhow::Result<()> {
    let key = murmur_hotkey::key_by_name(&config.trigger.key)?;
    let triggers = murmur_hotkey::watch(key)?;

    let microphone = murmur_audio::Microphone::open(&config.audio)?;
    let transcriber = crate::transcriber(config)?;
    let sink = murmur_inject::Injector::open(config.inject)?;

    let _ = to_interface.try_send(Message::Ready {
        trigger: config.trigger.key.clone(),
        microphone: microphone.name().to_owned(),
        transcriber: transcriber.name(),
    });

    let session = Session::new(config.tuning, Formatter::new(config.format.clone()));
    let mut engine = Engine::new(
        session,
        Box::new(microphone),
        transcriber,
        Box::new(sink),
        Box::new(ChannelSurface::new(to_interface)),
    );
    engine.run(&triggers)?;
    Ok(())
}

/// Everything the interface needs to know before the first dictation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ready {
    pub trigger: String,
    pub microphone: String,
    pub transcriber: String,
}

/// The interface's own view of what the engine is doing.
///
/// Deliberately not [`murmur_core::Hud`]: the HUD keeps a level, live text and
/// the last result at the same time, where the core emits them as separate
/// events. Translating once here keeps the view a pure function of one value.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum Phase {
    #[default]
    Starting,
    Idle,
    Listening {
        locked: bool,
    },
    Thinking,
    Failed(String),
}

impl Phase {
    #[must_use]
    pub fn from_hud(hud: &Hud, previous: &Self) -> Self {
        match hud {
            Hud::Hidden => Self::Idle,
            Hud::Listening { mode } => {
                Self::Listening { locked: matches!(mode, murmur_core::Mode::Locked) }
            }
            // Live text arrives while listening and must not change the phase,
            // or the meter would vanish the moment the first word appeared.
            Hud::Partial { .. } => previous.clone(),
            Hud::Thinking => Self::Thinking,
            Hud::Error { message } => Self::Failed(message.clone()),
        }
    }
}

/// The most recent completed dictation, shown briefly after it lands.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Landed {
    pub text: String,
    pub release_to_text: Option<Millis>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use murmur_core::Mode;

    #[test]
    fn live_text_does_not_change_the_phase() {
        let listening = Phase::Listening { locked: false };
        let after =
            Phase::from_hud(&Hud::Partial { text: "hello".into() }, &listening);
        assert_eq!(after, listening, "the meter would vanish on the first word");
    }

    #[test]
    fn every_hud_state_maps_somewhere_sensible() {
        let idle = Phase::Idle;
        assert_eq!(Phase::from_hud(&Hud::Hidden, &idle), Phase::Idle);
        assert_eq!(Phase::from_hud(&Hud::Thinking, &idle), Phase::Thinking);
        assert_eq!(
            Phase::from_hud(&Hud::Listening { mode: Mode::Locked }, &idle),
            Phase::Listening { locked: true }
        );
        assert!(matches!(
            Phase::from_hud(&Hud::Error { message: "x".into() }, &idle),
            Phase::Failed(_)
        ));
    }
}
