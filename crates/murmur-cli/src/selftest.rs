use anyhow::{Context, Result, bail};
use murmur_inject::keymap;
use murmur_inject::uinput::UinputKeyboard;
use std::sync::mpsc;
use std::time::Duration;

/// Text chosen to exercise both shift states and punctuation.
const PROBE: &str = "Murmur OK! 42 (round-trip).";

const READBACK_TIMEOUT: Duration = Duration::from_secs(3);

/// Prove the injection path works, without typing on the user's screen.
///
/// The virtual keyboard emits real scancodes, so a naive test would spray text
/// into whatever window has focus. Instead we open our own device node and
/// `EVIOCGRAB` it: the kernel then delivers those events to us alone, and the
/// compositor never sees them. What comes back out is decoded and compared to
/// what went in, which verifies device creation, the keymap, shift handling and
/// event delivery in one pass.
pub fn run(keystroke_delay_us: u64) -> Result<()> {
    let mut keyboard =
        UinputKeyboard::open(keystroke_delay_us).context("creating the virtual keyboard")?;
    let node = keyboard
        .dev_node()
        .context("locating the virtual device node")?
        .context("the virtual keyboard has no /dev/input node")?;
    println!("  device    {}", node.display());

    let (ready_tx, ready_rx) = mpsc::channel();
    let (chars_tx, chars_rx) = mpsc::channel();
    let expected = PROBE.chars().count();

    let reader = std::thread::spawn(move || -> Result<()> {
        let mut device = evdev::Device::open(&node).context("opening our own device node")?;
        // Exclusive access: these keystrokes must not reach the desktop.
        device.grab().context("EVIOCGRAB on the virtual device")?;
        ready_tx.send(()).ok();

        let mut shift = false;
        let mut seen = 0usize;
        while seen < expected {
            for event in device.fetch_events()? {
                if let evdev::EventSummary::Key(_, key, value) = event.destructure() {
                    if key == evdev::KeyCode::KEY_LEFTSHIFT {
                        shift = value == 1;
                    } else if value == 1 {
                        let decoded = keymap::char_for(key, shift);
                        seen += 1;
                        if chars_tx.send(decoded).is_err() {
                            return Ok(());
                        }
                    }
                }
            }
        }
        device.ungrab().ok();
        Ok(())
    });

    ready_rx
        .recv_timeout(READBACK_TIMEOUT)
        .context("reader thread never grabbed the device")?;

    keyboard.type_text(PROBE).context("emitting the probe text")?;

    let mut decoded = String::new();
    for _ in 0..expected {
        match chars_rx.recv_timeout(READBACK_TIMEOUT) {
            Ok(Some(c)) => decoded.push(c),
            Ok(None) => decoded.push('\u{fffd}'),
            Err(_) => break,
        }
    }
    reader.join().map_err(|_| anyhow::anyhow!("reader thread panicked"))??;

    println!("  sent      {PROBE:?}");
    println!("  received  {decoded:?}");
    if decoded == PROBE {
        println!("\n  injection path verified: {expected} characters round-tripped through the kernel.");
        Ok(())
    } else {
        bail!("round-trip mismatch: the virtual keyboard did not deliver what it was given");
    }
}
