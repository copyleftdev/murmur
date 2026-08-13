use crate::keymap;
use crate::{InjectError, Result, TextSink};
use evdev::uinput::VirtualDevice;
use evdev::{AttributeSet, InputEvent, KeyCode, KeyEvent};
use std::thread::sleep;
use std::time::Duration;

pub const UINPUT_NODE: &str = "/dev/uinput";

/// How long the compositor needs to notice a newly created input device.
///
/// libinput discovers devices through udev, asynchronously. Events emitted
/// before that completes are delivered to nobody, which is why the device is
/// created once at start-up and held open, never per injection.
const SETTLE: Duration = Duration::from_millis(600);

/// A kernel-level virtual keyboard.
///
/// Because the events enter through evdev, every compositor and toolkit treats
/// them as real hardware — no portal, no consent prompt, and no dependency on
/// Wayland protocols GNOME chooses not to implement. The cost is that we emit
/// scancodes, so we can only type what [`keymap`] can express.
#[derive(Debug)]
pub struct UinputKeyboard {
    device: VirtualDevice,
    delay: Duration,
}

impl UinputKeyboard {
    /// Create the virtual device and wait for the compositor to pick it up.
    ///
    /// # Errors
    /// Fails if `/dev/uinput` is not writable by this user.
    pub fn open(keystroke_delay_us: u64) -> Result<Self> {
        let mut keys = AttributeSet::<KeyCode>::new();
        for key in keymap::all_keys() {
            keys.insert(key);
        }

        let device = VirtualDevice::builder()
            .and_then(|b| b.name("Murmur virtual keyboard").with_keys(&keys))
            .and_then(evdev::uinput::VirtualDeviceBuilder::build)
            .map_err(InjectError::from_uinput)?;

        sleep(SETTLE);
        Ok(Self { device, delay: Duration::from_micros(keystroke_delay_us) })
    }

    /// The `/dev/input/eventN` node this virtual device appears at.
    ///
    /// Reading our own node back is how `murmur selftest` verifies the injection
    /// path without needing a window to type into.
    ///
    /// # Errors
    /// Fails if the device's sysfs entry cannot be enumerated.
    pub fn dev_node(&mut self) -> Result<Option<std::path::PathBuf>> {
        let mut nodes =
            self.device.enumerate_dev_nodes_blocking().map_err(InjectError::from_uinput)?;
        nodes.next().transpose().map_err(InjectError::from_uinput)
    }

    /// Type `text` one scancode at a time.
    ///
    /// # Errors
    /// Returns [`InjectError::Untypable`] before emitting anything if any
    /// character is outside the keymap, so a partial line is never left behind.
    pub fn type_text(&mut self, text: &str) -> Result<()> {
        if let Some(ch) = keymap::first_untypable(text) {
            return Err(InjectError::Untypable { ch });
        }
        for ch in text.chars() {
            let (key, shift) = keymap::key_for(ch).expect("checked above");
            self.tap(key, shift)?;
            if !self.delay.is_zero() {
                sleep(self.delay);
            }
        }
        Ok(())
    }

    /// Press `key` while holding `modifiers`, then release everything.
    ///
    /// # Errors
    /// Fails if the device rejects the write.
    pub fn chord(&mut self, modifiers: &[KeyCode], key: KeyCode) -> Result<()> {
        let mut events: Vec<InputEvent> =
            modifiers.iter().map(|m| *KeyEvent::new(*m, 1)).collect();
        events.push(*KeyEvent::new(key, 1));
        events.push(*KeyEvent::new(key, 0));
        events.extend(modifiers.iter().rev().map(|m| *KeyEvent::new(*m, 0)));
        self.device.emit(&events).map_err(InjectError::from_uinput)
    }

    /// Erase `count` characters to the left of the cursor.
    ///
    /// # Errors
    /// Fails if the device rejects the write.
    pub fn backspace(&mut self, count: usize) -> Result<()> {
        for _ in 0..count {
            self.tap(KeyCode::KEY_BACKSPACE, false)?;
            if !self.delay.is_zero() {
                sleep(self.delay);
            }
        }
        Ok(())
    }

    fn tap(&mut self, key: KeyCode, shift: bool) -> Result<()> {
        let events: &[InputEvent] = if shift {
            &[
                *KeyEvent::new(KeyCode::KEY_LEFTSHIFT, 1),
                *KeyEvent::new(key, 1),
                *KeyEvent::new(key, 0),
                *KeyEvent::new(KeyCode::KEY_LEFTSHIFT, 0),
            ]
        } else {
            &[*KeyEvent::new(key, 1), *KeyEvent::new(key, 0)]
        };
        self.device.emit(events).map_err(InjectError::from_uinput)
    }
}

impl TextSink for UinputKeyboard {
    fn name(&self) -> &'static str {
        "uinput"
    }

    fn inject(&mut self, text: &str) -> Result<()> {
        self.type_text(text)
    }
}
