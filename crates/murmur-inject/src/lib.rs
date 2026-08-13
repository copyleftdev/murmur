//! Put text at the cursor of whatever application has focus.
//!
//! This is the layer that decides whether Murmur is usable, and it is entirely a
//! desktop-integration problem rather than a machine-learning one. Wayland gives
//! an unfocused background process no sanctioned way to type, so we reach below
//! it: a `/dev/uinput` virtual keyboard is indistinguishable from real hardware
//! to every compositor, and the clipboard covers what a scancode cannot express.

pub mod clipboard;
pub mod keymap;
pub mod uinput;

use murmur_core::config::{InjectBackend, InjectConfig};
use std::time::Duration;
use uinput::{UINPUT_NODE, UinputKeyboard};

pub type Result<T> = std::result::Result<T, InjectError>;

/// How long the target application gets to read the clipboard before we hand it
/// back to the user. Too short and the paste arrives empty.
const CLIPBOARD_HANDOVER: Duration = Duration::from_millis(120);

#[derive(Debug, thiserror::Error)]
pub enum InjectError {
    #[error(
        "{UINPUT_NODE} is not writable. Add yourself to the 'input' group \
         (`sudo usermod -aG input $USER`, then log out and back in), or run `murmur doctor`."
    )]
    UinputPermission,
    #[error("virtual keyboard: {0}")]
    Uinput(String),
    #[error("cannot type {ch:?} with a scancode; this session has no working clipboard to fall back on")]
    Untypable { ch: char },
    #[error("clipboard: {0}")]
    Clipboard(String),
    #[error("the {0} backend is not available in this session")]
    Unavailable(&'static str),
}

impl InjectError {
    fn from_uinput(error: std::io::Error) -> Self {
        if error.kind() == std::io::ErrorKind::PermissionDenied {
            Self::UinputPermission
        } else {
            Self::Uinput(error.to_string())
        }
    }
}

/// Somewhere text can be delivered.
pub trait TextSink: Send {
    fn name(&self) -> &'static str;

    /// # Errors
    /// Fails if the underlying desktop mechanism rejects the text.
    fn inject(&mut self, text: &str) -> Result<()>;
}

/// Types short ASCII and pastes everything else.
///
/// Typing looks natural and leaves the clipboard alone, but costs a keystroke
/// per character and cannot express anything outside the active layout. Pasting
/// is constant-time and Unicode-exact, but borrows the user's clipboard. The
/// split is a config value because the right answer depends on the application.
#[derive(Debug)]
pub struct Injector {
    keyboard: UinputKeyboard,
    config: InjectConfig,
    clipboard_available: bool,
}

impl Injector {
    /// Open the desktop integration described by `config`.
    ///
    /// # Errors
    /// Fails if the chosen backend cannot be initialised at all.
    pub fn open(config: InjectConfig) -> Result<Self> {
        match config.backend {
            InjectBackend::Portal => return Err(InjectError::Unavailable("portal")),
            InjectBackend::X11 => return Err(InjectError::Unavailable("x11")),
            _ => {}
        }
        let keyboard = UinputKeyboard::open(config.keystroke_delay_us)?;
        let clipboard_available = clipboard::is_available();
        tracing::info!(clipboard = clipboard_available, "injection ready");
        Ok(Self { keyboard, config, clipboard_available })
    }

    #[must_use]
    pub fn clipboard_available(&self) -> bool {
        self.clipboard_available
    }

    fn paste(&mut self, text: &str) -> Result<()> {
        let saved = self.config.restore_clipboard.then(clipboard::snapshot).flatten();
        clipboard::offer_once(text)?;
        self.keyboard.chord(&[evdev::KeyCode::KEY_LEFTCTRL], evdev::KeyCode::KEY_V)?;
        if let Some(saved) = saved {
            std::thread::sleep(CLIPBOARD_HANDOVER);
            clipboard::restore(&saved)?;
        }
        Ok(())
    }
}

impl TextSink for Injector {
    fn name(&self) -> &'static str {
        "auto"
    }

    fn inject(&mut self, text: &str) -> Result<()> {
        if text.is_empty() {
            return Ok(());
        }
        if routes_to_clipboard(&self.config, text) && self.clipboard_available {
            match self.paste(text) {
                Ok(()) => return Ok(()),
                // Typing is slower and cannot do Unicode, but a degraded emission
                // beats a lost one: the user has already spoken the words.
                Err(error) if keymap::is_typable(text) => {
                    tracing::warn!(%error, "paste failed, falling back to typing");
                }
                Err(error) => return Err(error),
            }
        }
        self.keyboard.type_text(text)
    }
}

/// Should this text go via the clipboard rather than the keyboard?
///
/// Pure, so the routing policy can be tested without a desktop attached.
#[must_use]
pub fn routes_to_clipboard(config: &InjectConfig, text: &str) -> bool {
    match config.backend {
        InjectBackend::Uinput => false,
        InjectBackend::Clipboard => true,
        _ => !keymap::is_typable(text) || text.chars().count() > config.paste_threshold,
    }
}

/// One thing `murmur doctor` checks, and how to fix it.
#[derive(Debug, Clone)]
pub struct Capability {
    pub name: &'static str,
    pub available: bool,
    pub detail: String,
    pub remedy: Option<String>,
}

/// Inspect what this session can actually do, without changing anything.
#[must_use]
pub fn probe() -> Vec<Capability> {
    let uinput = std::fs::OpenOptions::new().write(true).open(UINPUT_NODE);
    let clipboard_status = clipboard::availability();
    let clipboard = clipboard_status.is_ok();
    let session = std::env::var("XDG_SESSION_TYPE").unwrap_or_else(|_| "unknown".into());

    vec![
        Capability {
            name: "uinput",
            available: uinput.is_ok(),
            detail: match &uinput {
                Ok(_) => format!("{UINPUT_NODE} is writable"),
                Err(e) => format!("{UINPUT_NODE}: {e}"),
            },
            remedy: uinput.is_err().then(|| {
                "sudo usermod -aG input $USER   # then log out and back in".to_owned()
            }),
        },
        Capability {
            name: "clipboard",
            available: clipboard,
            detail: match &clipboard_status {
                Ok(()) => "data-control protocol present; paste injection available".into(),
                Err(why) => format!("unavailable ({why}); long or non-ASCII text cannot be pasted"),
            },
            remedy: (!clipboard).then(|| {
                "set inject.backend = \"uinput\" to type everything as scancodes".to_owned()
            }),
        },
        Capability {
            name: "session",
            available: session != "unknown",
            detail: format!("{session} session"),
            remedy: None,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn injector_config(backend: InjectBackend, threshold: usize) -> InjectConfig {
        InjectConfig { backend, paste_threshold: threshold, ..InjectConfig::default() }
    }

    use super::routes_to_clipboard as should_paste;

    #[test]
    fn short_ascii_is_typed_and_leaves_the_clipboard_alone() {
        assert!(!should_paste(&injector_config(InjectBackend::Auto, 80), "hello there "));
    }

    #[test]
    fn long_text_is_pasted_because_typing_is_linear_in_length() {
        let long = "word ".repeat(40);
        assert!(should_paste(&injector_config(InjectBackend::Auto, 80), &long));
    }

    #[test]
    fn non_ascii_is_pasted_however_short_it_is() {
        assert!(should_paste(&injector_config(InjectBackend::Auto, 80), "café"));
        assert!(should_paste(&injector_config(InjectBackend::Auto, 80), "🎙"));
    }

    #[test]
    fn an_explicit_backend_overrides_the_length_heuristic() {
        let long = "word ".repeat(40);
        assert!(!should_paste(&injector_config(InjectBackend::Uinput, 80), &long));
        assert!(should_paste(&injector_config(InjectBackend::Clipboard, 80), "hi"));
    }

    #[test]
    fn the_threshold_counts_characters_not_bytes() {
        let text = "é".repeat(10);
        assert_eq!(text.len(), 20, "precondition: multi-byte");
        assert!(should_paste(&injector_config(InjectBackend::Auto, 15), &text));
    }

    #[test]
    fn probe_reports_every_capability_without_side_effects() {
        let report = probe();
        assert!(report.iter().any(|c| c.name == "uinput"));
        assert!(report.iter().all(|c| c.available || c.remedy.is_some() || c.name == "session"));
    }
}
