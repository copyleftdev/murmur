//! Murmur's overlay: a single bar that says what is happening, while it happens.
//!
//! The design constraint that shapes everything here is that this window must
//! never take keyboard focus. Text is injected into whichever window the
//! compositor considers focused, so a HUD that steals focus would type into
//! itself. Wayland offers no way for a client to refuse focus, and GNOME
//! implements no layer-shell protocol, so the window is created once at
//! start-up, never mapped or unmapped, and simply changes what it draws.

mod icon;
mod theme;
mod tray;
mod worker;

use anyhow::Context as _;
use iced::widget::{Space, button, container, mouse_area, row, text};
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
    /// Move the overlay by dragging the bar.
    Drag,
    /// Put the overlay away; it stays reachable from the panel icon.
    Hide,
    Show,
    Quit,
}

#[derive(Default)]
struct Murmur {
    phase: Phase,
    ready: Option<Ready>,
    level: f32,
    partial: String,
    landed: Option<Landed>,
    hidden: bool,
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
            Message::Drag => {
                return iced::window::latest().and_then(iced::window::drag);
            }
            Message::Hide => {
                self.hidden = true;
                return iced::window::latest()
                    .and_then(|id| iced::window::set_mode(id, iced::window::Mode::Hidden));
            }
            Message::Show => {
                self.hidden = false;
                return iced::window::latest()
                    .and_then(|id| iced::window::set_mode(id, iced::window::Mode::Windowed));
            }
            Message::Quit => return iced::exit(),
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

        // Dragging the bar moves the window. An undecorated overlay has no
        // titlebar to grab, so without this it can only ever sit where it
        // started -- which is not acceptable for something that floats over
        // whatever you are working in.
        let draggable = mouse_area(row![body, Space::new().width(Length::Fill)].align_y(iced::Center))
            .on_press(Message::Drag);

        container(
            row![draggable, hide_button(), close_button()].spacing(4).align_y(iced::Center),
        )
            .style(theme::pill)
            .padding([14, 16])
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
        // Escape closes it too, for the moment after start-up when the overlay
        // still holds focus -- which is the moment a user most wants it gone.
        let escape = iced::event::listen_with(|event, _status, _window| {
            use iced::keyboard::{Event, Key, key::Named};
            match event {
                iced::Event::Keyboard(Event::KeyPressed {
                    key: Key::Named(Named::Escape),
                    ..
                }) => Some(Message::Quit),
                _ => None,
            }
        });

        Subscription::batch([Subscription::run(engine), Subscription::run(panel), escape])
    }

    fn title(&self) -> String {
        "Murmur".to_owned()
    }
}

/// Put the overlay away without ending the session.
///
/// Distinct from closing: dictation keeps working while it is hidden, and the
/// panel icon brings it back. Without a panel icon this would be a trapdoor,
/// which is why the two were built together.
fn hide_button() -> iced::widget::Button<'static, Message> {
    button(text("\u{2013}").size(18).center())
        .on_press(Message::Hide)
        .padding([2, 8])
        .style(theme::close)
}

/// The one control the overlay needs: a way to make it go away.
fn close_button() -> iced::widget::Button<'static, Message> {
    button(text("\u{00d7}").size(18).center())
        .on_press(Message::Quit)
        .padding([2, 8])
        .style(theme::close)
}

/// Publish the panel icon and report what the user does with it.
fn panel() -> impl futures::Stream<Item = Message> {
    iced::stream::channel(16, async |sender| {
        tray::publish(sender);
    })
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

/// Where the overlay sits.
///
/// A plain `fn` because that is what iced's placement hook takes — no captures —
/// which is why the knobs are environment variables rather than config fields.
/// Reading them here, at placement time, costs nothing and keeps the hook pure
/// with respect to its arguments.
///
/// The default is bottom-centre, out of the way of what is being worked on.
/// `MURMUR_HUD_ANCHOR` moves it to a corner instead, which is what you want when
/// several monitors are exposed as one wide screen and "centre" lands on the seam.
fn place(window: iced::Size, screen: iced::Size) -> iced::Point {
    let margin = std::env::var("MURMUR_HUD_MARGIN")
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .unwrap_or(96.0);

    let anchor = std::env::var("MURMUR_HUD_ANCHOR").unwrap_or_default();
    let monitor = monitor_width(screen);
    let x = match anchor.as_str() {
        "bottom-left" | "left" => margin,
        "bottom-right" | "right" => screen.width - window.width - margin,
        // Centred on the first monitor, not on the whole desktop: on a
        // multi-monitor setup those are different places, and the second one is
        // the gap between two screens.
        _ => (monitor - window.width) / 2.0,
    };

    tracing::debug!(?window, ?screen, %anchor, "placing the overlay");
    iced::Point::new(x.max(0.0), (screen.height - window.height - margin).max(0.0))
}

/// The width of one monitor, when several are exposed as a single wide screen.
///
/// X11 reports a dual 1920x1080 setup as one 3840x1080 screen, so centring on it
/// puts the overlay exactly on the bezel. There is no monitor list available
/// here, but the count can be inferred: assume the panels are a conventional
/// aspect and see how many fit. An ultrawide stays one monitor, because 3440x1440
/// is not close to twice 16:9.
fn monitor_width(screen: iced::Size) -> f32 {
    if screen.height <= 0.0 {
        return screen.width;
    }
    let panels = (screen.width / (screen.height * 16.0 / 9.0)).round().max(1.0);
    screen.width / panels
}

/// Prefer XWayland unless told otherwise.
///
/// Wayland gives a client no way to place its own window: `xdg-shell` has no
/// concept of a position, so the compositor decides — and GNOME decides on the
/// middle of the screen, which is the one place an overlay must not be. Going
/// through XWayland restores absolute placement, and nothing else Murmur does
/// touches Wayland: injection is `uinput`, the trigger is evdev, and audio is
/// ALSA. Set `MURMUR_HUD_WAYLAND=1` to keep the native surface and place it by
/// dragging instead.
fn prefer_positionable_backend() {
    if std::env::var_os("MURMUR_HUD_WAYLAND").is_some() {
        return;
    }
    if std::env::var_os("WAYLAND_DISPLAY").is_some() && std::env::var_os("DISPLAY").is_some() {
        // SAFETY: called before any window, thread or event loop exists.
        unsafe { std::env::remove_var("WAYLAND_DISPLAY") };
        tracing::debug!("using XWayland so the overlay can place itself");
    }
}

/// Write a desktop entry and icon, so Murmur is an application rather than a
/// binary you have to remember the path of.
///
/// Everything lands under the user's own data directory: no root, and removing
/// the two files removes every trace.
fn install() -> std::io::Result<()> {
    let home = std::env::var_os("HOME").map(std::path::PathBuf::from).unwrap_or_default();
    let data = std::env::var_os("XDG_DATA_HOME")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| home.join(".local/share"));

    let icons = data.join("icons/hicolor/scalable/apps");
    std::fs::create_dir_all(&icons)?;
    let icon_path = icons.join("murmur.svg");
    std::fs::write(&icon_path, icon::svg())?;

    let applications = data.join("applications");
    std::fs::create_dir_all(&applications)?;
    let exe = std::env::current_exe()?;
    let entry = applications.join("murmur.desktop");
    std::fs::write(
        &entry,
        format!(
            "[Desktop Entry]\n\
             Type=Application\n\
             Name=Murmur\n\
             Comment=Local-first voice typing that never leaves your machine\n\
             Exec={}\n\
             Icon=murmur\n\
             Terminal=false\n\
             Categories=Utility;AudioVideo;Accessibility;\n\
             Keywords=dictation;speech;voice;transcription;\n\
             StartupWMClass=murmur\n",
            exe.display()
        ),
    )?;

    println!("installed:\n  {}\n  {}", icon_path.display(), entry.display());
    println!("\nMurmur now appears in your applications, and in the panel while running.");
    Ok(())
}

fn main() -> iced::Result {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_env("MURMUR_LOG")
                .unwrap_or_else(|_| "warn".into()),
        )
        .with_target(false)
        .with_writer(std::io::stderr)
        .init();

    if std::env::args().any(|arg| arg == "--install") {
        if let Err(error) = install() {
            eprintln!("could not install: {error}");
            std::process::exit(1);
        }
        return Ok(());
    }

    prefer_positionable_backend();

    iced::application(Murmur::default, Murmur::update, Murmur::view)
        .window(iced::window::Settings {
            icon: iced::window::icon::from_rgba(icon::rgba(256, true), 256, 256).ok(),
            // Matches the basename of the desktop entry, which is how the shell
            // ties a window to its name and icon. Without it the window reports
            // an empty WM_CLASS and shows up as an anonymous box.
            platform_specific: iced::window::settings::PlatformSpecific {
                application_id: "murmur".to_owned(),
                ..iced::window::settings::PlatformSpecific::default()
            },
            ..iced::window::Settings::default()
        })
        .title(Murmur::title)
        .subscription(Murmur::subscription)
        .window_size(WINDOW)
        .decorations(false)
        .transparent(true)
        .resizable(false)
        // Always on top so the overlay is visible over the window being dictated
        // into. It never takes focus, because it is never re-mapped.
        .level(iced::window::Level::AlwaysOnTop)
        // Bottom-centre, clear of the middle of the screen where the user is
        // actually working. Centred is where a dialog belongs, not an overlay.
        .position(iced::window::Position::SpecificWith(place))
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
    fn one_monitor_is_reported_whole() {
        for screen in [(1920.0, 1080.0), (2560.0, 1440.0), (1366.0, 768.0)] {
            let size = iced::Size::new(screen.0, screen.1);
            assert!((monitor_width(size) - screen.0).abs() < 1.0, "{screen:?}");
        }
    }

    #[test]
    fn a_wide_desktop_is_split_into_its_panels() {
        assert!((monitor_width(iced::Size::new(3840.0, 1080.0)) - 1920.0).abs() < 1.0);
        assert!((monitor_width(iced::Size::new(5120.0, 1440.0)) - 2560.0).abs() < 1.0);
        assert!((monitor_width(iced::Size::new(5760.0, 1080.0)) - 1920.0).abs() < 1.0);
    }

    #[test]
    fn an_ultrawide_is_not_mistaken_for_two_screens() {
        for screen in [(3440.0, 1440.0), (2560.0, 1080.0), (3840.0, 1600.0)] {
            let size = iced::Size::new(screen.0, screen.1);
            assert!(
                (monitor_width(size) - screen.0).abs() < 1.0,
                "{screen:?} was split, so the overlay would sit off-centre"
            );
        }
    }

    #[test]
    fn placement_keeps_the_whole_bar_on_screen() {
        for screen in [(1920.0, 1080.0), (3840.0, 1080.0), (1366.0, 768.0)] {
            let screen = iced::Size::new(screen.0, screen.1);
            let point = place(WINDOW, screen);
            assert!(point.x >= 0.0 && point.y >= 0.0, "{screen:?} -> {point:?}");
            assert!(point.x + WINDOW.width <= screen.width, "{screen:?} -> {point:?}");
            assert!(point.y + WINDOW.height <= screen.height, "{screen:?} -> {point:?}");
        }
    }

    #[test]
    fn a_screen_smaller_than_the_bar_still_places_it_somewhere_visible() {
        let point = place(WINDOW, iced::Size::new(400.0, 200.0));
        assert!(point.x >= 0.0 && point.y >= 0.0, "{point:?}");
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
