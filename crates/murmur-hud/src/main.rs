//! Murmur's overlay: a single bar that says what is happening, while it happens.
//!
//! The design constraint that shapes everything here is that this window must
//! never take keyboard focus. Text is injected into whichever window the
//! compositor considers focused, so a HUD that steals focus would type into
//! itself. Wayland offers no way for a client to refuse focus, and GNOME
//! implements no layer-shell protocol, so the window is created once at
//! start-up, never mapped or unmapped, and simply changes what it draws.

mod theme;
mod worker;

use anyhow::Context as _;
use iced::widget::{container, row};
use iced::{Element, Length, Subscription, Task};
use murmur_core::{Config, Hud, Millis};
use worker::{Landed, Phase, Ready};

/// Everything that can change what the overlay shows.
#[derive(Debug, Clone)]
pub enum Message {
    Ready { trigger: String, microphone: String, transcriber: String },
    Hud(Hud),
    Level(f32),
    Emitted(String),
    Completed(Millis),
    Fatal(String),
}

#[derive(Default)]
struct Murmur {
    phase: Phase,
    ready: Option<Ready>,
    level: f32,
    partial: String,
    landed: Option<Landed>,
}

impl Murmur {
    fn update(&mut self, message: Message) -> Task<Message> {
        match message {
            Message::Ready { trigger, microphone, transcriber } => {
                self.phase = Phase::Idle;
                self.ready = Some(Ready { trigger, microphone, transcriber });
            }
            Message::Hud(hud) => {
                if let Hud::Partial { text } = &hud {
                    self.partial.clone_from(text);
                } else {
                    self.partial.clear();
                }
                if matches!(hud, Hud::Listening { .. }) {
                    // A new utterance replaces the last result rather than
                    // sitting beside it.
                    self.landed = None;
                }
                self.phase = Phase::from_hud(&hud, &self.phase);
            }
            Message::Level(level) => self.level = level,
            Message::Emitted(text) => {
                self.landed = Some(Landed { text, release_to_text: None });
            }
            Message::Completed(latency) => {
                if let Some(landed) = &mut self.landed {
                    landed.release_to_text = Some(latency);
                }
            }
            Message::Fatal(reason) => self.phase = Phase::Failed(reason),
        }
        Task::none()
    }

    fn view(&self) -> Element<'_, Message> {
        let body: Element<'_, Message> = match &self.phase {
            Phase::Starting => theme::muted("starting\u{2026}").into(),
            Phase::Failed(reason) => theme::failed(reason).into(),
            Phase::Thinking => row![theme::dot(theme::THINKING), theme::muted("transcribing\u{2026}")]
                .spacing(12)
                .align_y(iced::Center)
                .into(),
            Phase::Listening { locked } => row![
                theme::dot(theme::LISTENING),
                theme::meter(self.level),
                self.speech_or_hint(*locked),
            ]
            .spacing(12)
            .align_y(iced::Center)
            .into(),
            Phase::Idle => self.idle_view(),
        };

        container(body)
            .style(theme::pill)
            .padding([14, 20])
            .width(Length::Fill)
            .height(Length::Fill)
            .align_y(iced::Center)
            .into()
    }

    /// While listening: the words so far, or what to do if there are none yet.
    fn speech_or_hint(&self, locked: bool) -> Element<'_, Message> {
        if self.partial.is_empty() {
            let hint = if locked { "hands-free \u{2014} press again to stop" } else { "speak\u{2026}" };
            theme::muted(hint).into()
        } else {
            theme::speech(self.partial.clone()).into()
        }
    }

    /// While idle: the last result if there is one, otherwise how to start.
    fn idle_view(&self) -> Element<'_, Message> {
        match &self.landed {
            Some(landed) => {
                let timing = landed
                    .release_to_text
                    .map_or_else(String::new, |latency| format!("{latency}"));
                row![
                    theme::dot(theme::DONE),
                    theme::speech(landed.text.trim().to_owned()),
                    theme::muted(timing),
                ]
                .spacing(12)
                .align_y(iced::Center)
                .into()
            }
            None => {
                let hint = self.ready.as_ref().map_or_else(
                    || "waiting for the engine".to_owned(),
                    |ready| format!("hold {} and speak", ready.trigger),
                );
                row![theme::dot(theme::IDLE), theme::muted(hint)]
                    .spacing(12)
                    .align_y(iced::Center)
                    .into()
            }
        }
    }

    fn subscription(&self) -> Subscription<Message> {
        Subscription::run(engine)
    }

    fn title(&self) -> String {
        "Murmur".to_owned()
    }
}

/// Run the engine on its own thread, forwarding what it does to the interface.
fn engine() -> impl futures::Stream<Item = Message> {
    iced::stream::channel(256, async |sender| {
        let mut fatal = sender.clone();
        std::thread::Builder::new()
            .name("murmur-engine".into())
            .spawn(move || {
                let config = load_config();
                if let Err(error) = worker::run(&config, sender) {
                    let _ = fatal.try_send(Message::Fatal(format!("{error:#}")));
                }
            })
            .ok();
    })
}

fn load_config() -> Config {
    let path = std::env::var_os("XDG_CONFIG_HOME")
        .map(std::path::PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| std::path::PathBuf::from(h).join(".config")))
        .unwrap_or_default()
        .join("murmur/config.toml");

    std::fs::read_to_string(&path)
        .ok()
        .and_then(|text| toml::from_str(&text).ok())
        .unwrap_or_default()
}

/// Load the transcriber named by the config.
///
/// # Errors
/// Fails if the engine is unavailable or its model cannot be loaded.
pub fn transcriber(config: &Config) -> anyhow::Result<Box<dyn murmur_asr::Transcriber>> {
    use murmur_core::AsrEngine;
    match config.asr.engine {
        AsrEngine::Mock => Ok(Box::new(murmur_asr::Mock::default())),
        AsrEngine::Parakeet => {
            let dir = expand_home(&config.asr.model_dir);
            let model =
                murmur_asr::Parakeet::open(&dir, config.asr.precision, config.asr.accelerator)
                    .with_context(|| format!("loading a model from {}", dir.display()))?;
            Ok(Box::new(model))
        }
        other => anyhow::bail!("the {other:?} engine cannot be used for dictation yet"),
    }
}

fn expand_home(input: &str) -> std::path::PathBuf {
    input.strip_prefix("~/").map_or_else(
        || std::path::PathBuf::from(input),
        |rest| {
            std::env::var_os("HOME")
                .map_or_else(|| std::path::PathBuf::from(input), |home| std::path::PathBuf::from(home).join(rest))
        },
    )
}

/// The overlay's size. Wide enough for a sentence, short enough to ignore.
const WINDOW: iced::Size = iced::Size::new(760.0, 78.0);

fn main() -> iced::Result {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_env("MURMUR_LOG")
                .unwrap_or_else(|_| "warn".into()),
        )
        .with_target(false)
        .with_writer(std::io::stderr)
        .init();

    iced::application(Murmur::default, Murmur::update, Murmur::view)
        .title(Murmur::title)
        .subscription(Murmur::subscription)
        .window_size(WINDOW)
        .decorations(false)
        .transparent(true)
        .resizable(false)
        // Always on top so the overlay is visible over the window being dictated
        // into. It never takes focus, because it is never re-mapped.
        .level(iced::window::Level::AlwaysOnTop)
        .position(iced::window::Position::Centered)
        // The window itself paints nothing: the rounded bar is the only thing
        // drawn, so the corners outside its radius stay genuinely transparent.
        .style(|_state, _theme| iced::theme::Style {
            background_color: iced::Color::TRANSPARENT,
            text_color: theme::TEXT,
        })
        .run()
}

#[cfg(test)]
mod tests {
    use super::*;
    use murmur_core::Mode;

    fn ready() -> Murmur {
        let mut app = Murmur::default();
        let _ = app.update(Message::Ready {
            trigger: "RIGHTCTRL".into(),
            microphone: "mic".into(),
            transcriber: "parakeet".into(),
        });
        app
    }

    #[test]
    fn live_text_is_kept_while_listening_and_dropped_afterwards() {
        let mut app = ready();
        let _ = app.update(Message::Hud(Hud::Listening { mode: Mode::Hold }));
        let _ = app.update(Message::Hud(Hud::Partial { text: "hello there".into() }));
        assert_eq!(app.partial, "hello there");

        let _ = app.update(Message::Hud(Hud::Thinking));
        assert!(app.partial.is_empty(), "live text outlived the utterance");
    }

    #[test]
    fn a_new_utterance_clears_the_previous_result() {
        let mut app = ready();
        let _ = app.update(Message::Emitted("first".into()));
        assert!(app.landed.is_some());

        let _ = app.update(Message::Hud(Hud::Listening { mode: Mode::Hold }));
        assert!(app.landed.is_none(), "the last result lingered into the next dictation");
    }

    #[test]
    fn timing_attaches_to_the_text_it_belongs_to() {
        let mut app = ready();
        let _ = app.update(Message::Emitted("done".into()));
        let _ = app.update(Message::Completed(Millis(120)));

        let landed = app.landed.expect("a result");
        assert_eq!(landed.text, "done");
        assert_eq!(landed.release_to_text, Some(Millis(120)));
    }

    #[test]
    fn timing_without_text_is_ignored_rather_than_shown_alone() {
        let mut app = ready();
        let _ = app.update(Message::Completed(Millis(120)));
        assert!(app.landed.is_none());
    }

    #[test]
    fn a_fatal_error_replaces_whatever_was_showing() {
        let mut app = ready();
        let _ = app.update(Message::Hud(Hud::Listening { mode: Mode::Hold }));
        let _ = app.update(Message::Fatal("no microphone".into()));
        assert!(matches!(app.phase, Phase::Failed(_)));
    }

    /// Render a state and compare it with the image checked in beside this file.
    ///
    /// The first run writes the image; later runs fail if a pixel moves. That is
    /// a blunt instrument, and exactly the right one for a surface whose whole
    /// job is to be glanceable — a layout regression is invisible to every other
    /// kind of test here.
    fn snapshot(app: &Murmur, name: &str) {
        // The real window size, so a snapshot shows what a user would see
        // rather than the same widgets adrift in a much larger frame.
        let mut simulator = iced_test::Simulator::with_size(iced::Settings::default(), WINDOW, app.view());
        let snapshot = simulator.snapshot(&iced::Theme::Dark).expect("rendering");
        assert!(
            snapshot.matches_image(format!("snapshots/{name}")).expect("comparing"),
            "{name} no longer looks the way it did; \
             delete crates/murmur-hud/snapshots/{name}.png to accept the new design"
        );
    }

    #[test]
    fn the_idle_bar_looks_the_way_it_did() {
        snapshot(&ready(), "idle");
    }

    #[test]
    fn the_listening_bar_looks_the_way_it_did() {
        let mut app = ready();
        let _ = app.update(Message::Hud(Hud::Listening { mode: Mode::Hold }));
        let _ = app.update(Message::Level(0.62));
        snapshot(&app, "listening");
    }

    #[test]
    fn the_live_text_bar_looks_the_way_it_did() {
        let mut app = ready();
        let _ = app.update(Message::Hud(Hud::Listening { mode: Mode::Hold }));
        let _ = app.update(Message::Level(0.41));
        let _ = app.update(Message::Hud(Hud::Partial {
            text: "testing one two three".into(),
        }));
        snapshot(&app, "live-text");
    }

    #[test]
    fn the_finished_bar_looks_the_way_it_did() {
        let mut app = ready();
        let _ = app.update(Message::Emitted("Testing one two three.".into()));
        let _ = app.update(Message::Completed(Millis(169)));
        snapshot(&app, "landed");
    }

    #[test]
    fn the_failure_bar_looks_the_way_it_did() {
        let mut app = ready();
        let _ = app.update(Message::Fatal("no microphone found".into()));
        snapshot(&app, "failed");
    }

    #[test]
    fn every_phase_renders_without_panicking() {
        let mut app = Murmur::default();
        let _ = app.view();

        for message in [
            Message::Ready {
                trigger: "RIGHTCTRL".into(),
                microphone: "m".into(),
                transcriber: "t".into(),
            },
            Message::Hud(Hud::Listening { mode: Mode::Hold }),
            Message::Level(0.4),
            Message::Hud(Hud::Partial { text: "words".into() }),
            Message::Hud(Hud::Listening { mode: Mode::Locked }),
            Message::Hud(Hud::Thinking),
            Message::Emitted("done".into()),
            Message::Completed(Millis(99)),
            Message::Hud(Hud::Hidden),
            Message::Hud(Hud::Error { message: "boom".into() }),
        ] {
            let _ = app.update(message);
            let _ = app.view();
        }
    }
}
