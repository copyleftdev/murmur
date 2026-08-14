//! Global push-to-talk, read straight from the kernel.
//!
//! Wayland has no portable global-hotkey API, and the one GNOME does offer
//! (`org.freedesktop.portal.GlobalShortcuts`) reports activation, not the press
//! and release edges that push-to-talk is built on. Reading evdev sidesteps both
//! problems: we see every key on every keyboard, in any session, with press and
//! release distinguished, and it costs one blocking read per device.
//!
//! The trade-off is that evdev cannot *swallow* a key. `EVIOCGRAB` is
//! all-or-nothing per device, so suppressing the trigger would suppress the
//! whole keyboard. The trigger key must therefore be one that does nothing on
//! its own — which is exactly why the default is Right Ctrl.

use std::sync::mpsc::{Receiver, Sender, channel};
use std::time::Instant;

#[derive(Debug, thiserror::Error)]
pub enum HotkeyError {
    #[error("unknown key name {0:?}; run `murmur keys` and press the key you want")]
    UnknownKey(String),
    #[error(
        "no readable keyboard reports {0}. Add yourself to the 'input' group \
         (`sudo usermod -aG input $USER`), then log out and back in."
    )]
    NoDevice(String),
}

type Result<T> = std::result::Result<T, HotkeyError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Edge {
    Down,
    Up,
}

#[derive(Debug, Clone, Copy)]
pub struct TriggerEvent {
    pub edge: Edge,
    pub at: Instant,
}

/// Our own virtual keyboard, which must never be treated as a trigger source.
const SELF_DEVICE: &str = "Murmur virtual keyboard";

/// Resolve a key name such as `RIGHTCTRL` or `KEY_F13` to its code.
///
/// # Errors
/// Fails if no key in the evdev table has that name.
pub fn key_by_name(name: &str) -> Result<evdev::KeyCode> {
    let wanted = name.trim().to_ascii_uppercase();
    let wanted = wanted.strip_prefix("KEY_").unwrap_or(&wanted);
    (1u16..=0x2ff)
        .map(evdev::KeyCode::new)
        .find(|key| key_name(*key).is_some_and(|n| n == wanted))
        .ok_or_else(|| HotkeyError::UnknownKey(name.to_owned()))
}

/// The bare name of a key, without the `KEY_` prefix.
#[must_use]
pub fn key_name(key: evdev::KeyCode) -> Option<String> {
    let debug = format!("{key:?}");
    debug.strip_prefix("KEY_").map(str::to_owned)
}

/// Watch every keyboard that can report `key`, and report its edges.
///
/// One blocking reader thread per device, because a keyboard may be added at any
/// time and a user may well have several — a laptop's built-in one and whatever
/// is plugged in. Duplicate edges from separate devices are collapsed, so
/// pressing a key that two devices both report still reads as one press.
///
/// # Errors
/// Fails if no readable device reports the key at all.
pub fn watch(key: evdev::KeyCode) -> Result<Receiver<TriggerEvent>> {
    let devices: Vec<(std::path::PathBuf, evdev::Device)> = evdev::enumerate()
        .filter(|(_, device)| device.name() != Some(SELF_DEVICE))
        .filter(|(_, device)| {
            device.supported_keys().is_some_and(|keys| keys.contains(key))
        })
        .collect();

    if devices.is_empty() {
        return Err(HotkeyError::NoDevice(
            key_name(key).unwrap_or_else(|| format!("{key:?}")),
        ));
    }

    let (tx, rx) = channel();
    let pressed = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));

    for (path, device) in devices {
        let tx = tx.clone();
        let pressed = std::sync::Arc::clone(&pressed);
        tracing::info!(device = ?device.name(), path = %path.display(), "watching for trigger");
        std::thread::Builder::new()
            .name(format!("murmur-hotkey:{}", path.display()))
            .spawn(move || read_loop(device, key, &tx, &pressed))
            .ok();
    }
    Ok(rx)
}

/// Report every key on every readable keyboard.
///
/// The diagnostic behind `murmur keys`: when a trigger "does nothing", the first
/// question is whether the key is reaching Murmur at all, and the second is what
/// that key is actually called.
///
/// # Errors
/// Fails if no readable keyboard exists at all.
pub fn watch_all() -> Result<Receiver<(evdev::KeyCode, Edge)>> {
    let devices: Vec<(std::path::PathBuf, evdev::Device)> = evdev::enumerate()
        .filter(|(_, device)| device.name() != Some(SELF_DEVICE))
        .filter(|(_, device)| device.supported_keys().is_some())
        .collect();

    if devices.is_empty() {
        return Err(HotkeyError::NoDevice("any key".into()));
    }

    let (tx, rx) = channel();
    for (path, mut device) in devices {
        let tx = tx.clone();
        std::thread::Builder::new()
            .name(format!("murmur-keys:{}", path.display()))
            .spawn(move || {
                loop {
                    let Ok(events) = device.fetch_events() else { return };
                    for event in events {
                        if let evdev::EventSummary::Key(_, code, value) = event.destructure() {
                            let edge = match value {
                                1 => Edge::Down,
                                0 => Edge::Up,
                                _ => continue,
                            };
                            if tx.send((code, edge)).is_err() {
                                return;
                            }
                        }
                    }
                }
            })
            .ok();
    }
    Ok(rx)
}

/// Names of the keyboards Murmur can read, for diagnostics.
#[must_use]
pub fn readable_keyboards(key: evdev::KeyCode) -> Vec<String> {
    evdev::enumerate()
        .filter(|(_, device)| device.name() != Some(SELF_DEVICE))
        .filter(|(_, device)| device.supported_keys().is_some_and(|keys| keys.contains(key)))
        .map(|(path, device)| {
            format!("{} ({})", device.name().unwrap_or("unnamed"), path.display())
        })
        .collect()
}

fn read_loop(
    mut device: evdev::Device,
    key: evdev::KeyCode,
    tx: &Sender<TriggerEvent>,
    pressed: &std::sync::atomic::AtomicBool,
) {
    use std::sync::atomic::Ordering;
    loop {
        let Ok(events) = device.fetch_events() else {
            tracing::warn!(device = ?device.name(), "keyboard disappeared");
            return;
        };
        for event in events {
            let evdev::EventSummary::Key(_, code, value) = event.destructure() else {
                continue;
            };
            if code != key {
                continue;
            }
            // 2 is auto-repeat: the key never went up, so it is not an edge.
            let edge = match value {
                1 => Edge::Down,
                0 => Edge::Up,
                _ => continue,
            };
            // Two keyboards reporting the same physical key must not double-fire.
            let was = pressed.swap(edge == Edge::Down, Ordering::SeqCst);
            if was == (edge == Edge::Down) {
                continue;
            }
            if tx.send(TriggerEvent { edge, at: Instant::now() }).is_err() {
                return;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_trigger_key_resolves() {
        assert_eq!(key_by_name("RIGHTCTRL").unwrap(), evdev::KeyCode::KEY_RIGHTCTRL);
    }

    #[test]
    fn key_names_are_accepted_in_the_forms_users_actually_type() {
        for spelling in ["RIGHTCTRL", "rightctrl", "KEY_RIGHTCTRL", "  RightCtrl  "] {
            assert_eq!(
                key_by_name(spelling).unwrap(),
                evdev::KeyCode::KEY_RIGHTCTRL,
                "failed for {spelling:?}"
            );
        }
    }

    #[test]
    fn an_unknown_key_names_itself_in_the_error() {
        let err = key_by_name("NOSUCHKEY").unwrap_err();
        assert!(err.to_string().contains("NOSUCHKEY"), "{err}");
    }

    #[test]
    fn names_round_trip_through_lookup() {
        for key in [
            evdev::KeyCode::KEY_RIGHTCTRL,
            evdev::KeyCode::KEY_CAPSLOCK,
            evdev::KeyCode::KEY_F13,
            evdev::KeyCode::KEY_RIGHTALT,
        ] {
            let name = key_name(key).expect("named");
            assert_eq!(key_by_name(&name).unwrap(), key, "round trip failed for {name}");
        }
    }

    #[test]
    fn candidate_trigger_keys_are_distinguishable_from_their_left_hand_twins() {
        assert_ne!(
            key_by_name("RIGHTCTRL").unwrap(),
            key_by_name("LEFTCTRL").unwrap(),
            "push-to-talk must not fire on the modifier users actually type with"
        );
    }
}
