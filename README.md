# Murmur

[![Tip in tokens](https://img.shields.io/badge/tip%20in-tokens-6f42c1)](https://tokentip.to/@copyleftdev)

Voice typing that never leaves your machine. Hold a key, speak, release — the
text lands wherever your cursor already is.

Murmur is a local-first replacement for hosted dictation tools. No audio leaves
the computer, there is no account, and there is nothing to subscribe to.

## Why this is hard, and where the hard part actually is

The machine learning is the easy half. NVIDIA Parakeet transcribes faster than
you can talk on any recent GPU, and it runs offline.

The difficult half is putting the text somewhere useful. Wayland deliberately
gives an unfocused background process no way to type into another application,
and the workarounds differ per compositor:

| Path | Works on | Limits |
| --- | --- | --- |
| `/dev/uinput` virtual keyboard | everything, incl. GNOME | scancodes only, so US-layout ASCII |
| `ext-data-control` clipboard | wlroots, KDE | **not GNOME** — mutter exposes neither data-control protocol |
| RemoteDesktop portal keysyms | GNOME, KDE | full Unicode, one consent prompt |
| XTEST | X11 / XWayland | not native Wayland clients |

Measured on GNOME Shell 49 (`murmur doctor` will tell you what *your* session
supports). The clipboard-paste trick that most dictation tools rely on simply
does not work on GNOME, which is why Murmur types through the kernel by default.

## Status

Working today:

- Pure, IO-free session logic — push-to-talk, double-tap hands-free, tap
  rejection, utterance caps, transcription timeouts, continuation spacing
- Text formatting — custom dictionary (multi-word), spoken commands
  (`new line`, `new paragraph`, `scratch that`), spacing policy
- Microphone capture with rolling pre-roll, so the first syllable survives
  starting to talk before the key is fully down
- 16 kHz resampling and voice-activity trimming, verified against synthetic tones
- Global push-to-talk read from evdev, working regardless of compositor policy
- Text injection through a `/dev/uinput` virtual keyboard, verified end to end
- Latency accounting reported as percentiles of *release to text*

Not yet:

- Parakeet transcription (the engine runs against a scripted transcriber today)
- The iced overlay — the terminal surface stands in
- RemoteDesktop portal backend for full Unicode on GNOME
- Optional Nemotron polish pass

## Try it

```sh
cargo build --release
./target/release/murmur doctor      # what this session can and cannot do
./target/release/murmur selftest    # prove injection without typing on your screen
./target/release/murmur listen --mock
```

`selftest` is worth understanding: it creates the virtual keyboard, opens its own
device node, `EVIOCGRAB`s it so the kernel delivers those events to nobody else,
types a probe string, and decodes what comes back. It exercises the real
injection path without spraying text across your desktop.

`listen --mock` runs the entire loop — trigger, capture, format, inject — with a
scripted transcriber. Focus a text editor first: it really does type.

## Layout

```
murmur-core     pure logic: session FSM, formatting, latency. No IO, no clock.
murmur-audio    capture, pre-roll, resampling, VAD
murmur-asr      the Transcriber trait, and a scripted implementation
murmur-hotkey   global push-to-talk via evdev
murmur-inject   uinput virtual keyboard, clipboard, backend routing
murmur-engine   the loop that performs core's commands and reports the facts
murmur-cli      doctor, selftest, listen, devices, config
```

The core is a pure function of its event log. Every device is behind a trait
with a fixture implementation, so the whole dictation loop is asserted on
machines with no microphone, no model and no window to type into.

## Configuration

```sh
murmur config --init   # writes ~/.config/murmur/config.toml
```

The trigger defaults to Right Ctrl. That is not arbitrary: evdev cannot swallow
a single key (`EVIOCGRAB` is all-or-nothing per device), so the push-to-talk key
must be one that does nothing on its own.

## Licence

Apache-2.0
