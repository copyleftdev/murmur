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

- **NVIDIA Parakeet TDT 0.6B v3** transcription via ONNX Runtime, verified
  against a clip with a known transcript (21x realtime on CPU with int8 weights)

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

- The iced overlay — the terminal surface stands in
- RemoteDesktop portal backend for full Unicode on GNOME
- Optional Nemotron polish pass

## Models

Murmur transcribes with NVIDIA Parakeet TDT 0.6B v3, exported to ONNX. The int8
weights are 640 MB and quite fast enough on a CPU:

```sh
MD=~/.local/share/murmur/models/parakeet-tdt-0.6b-v3
B=https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main
mkdir -p "$MD" && cd "$MD"
for f in vocab.txt config.json nemo128.onnx \
         decoder_joint-model.int8.onnx encoder-model.int8.onnx; do
  curl -L -O "$B/$f"
done
```

Swap `encoder-model.int8.onnx` for `encoder-model.onnx` + `encoder-model.onnx.data`
(2.5 GB) for full precision.

### GPU

```sh
cargo build --release --features cuda
```

ONNX Runtime's prebuilt CUDA provider is linked against **CUDA 13 and cuDNN 9**.
If those are not installed, the provider fails to load and ORT falls back to the
CPU *silently* — so Murmur `dlopen`s the provider itself before use and tells
you exactly which library is missing:

```
✗ accelerator  CUDA unavailable: libcublasLt.so.13: cannot open shared object file
                 → install the matching CUDA runtime, or set asr.accelerator = "cpu"
```

Murmur never claims a device it did not actually run on.

## Try it

```sh
cargo build --release
./target/release/murmur doctor                    # what this session can and cannot do
./target/release/murmur selftest                  # prove injection, without typing on your screen
./target/release/murmur transcribe assets/jfk.wav # prove the model, offline
./target/release/murmur listen                    # the real thing
./target/release/murmur listen --mock             # the loop, without a model
```

`selftest` is worth understanding: it creates the virtual keyboard, opens its own
device node, `EVIOCGRAB`s it so the kernel delivers those events to nobody else,
types a probe string, and decodes what comes back. It exercises the real
injection path without spraying text across your desktop.

`transcribe` is the offline way to check a model and measure it — it reports the
realtime factor alongside the text, so a slow accelerator is visible immediately.

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
