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

Also working, with one important caveat:

- **NVIDIA Nemotron 3.5 cache-aware streaming ASR** (`murmur transcribe --stream`),
  which transcribes while you speak. See "Why streaming is not the default" below.

Not yet:

- Live partials in the HUD via periodic re-transcription
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

Measured on two RTX 3080 Ti, 11 seconds of speech, warm median of five runs:

| weights | device | median | realtime |
| --- | --- | --- | --- |
| int8 | CPU | 477 ms | 23x |
| int8 | CUDA | 460 ms | 24x |
| fp32 | CPU | 464 ms | 24x |
| **fp32** | **CUDA** | **36 ms** | **307x** |

**Use fp32 on a GPU** — and Murmur does this for you. int8 is a CPU
optimisation: the CUDA provider has no kernels for most quantised ops, so it
inserts hundreds of Memcpy nodes and shuttles the graph across the bus, arriving
exactly where it started.

Note that int8 is not *faster* on a CPU either; it is a quarter of the size. So
`asr.model_dir` points at a directory of models and `asr.precision = "auto"`
picks fp32 when the GPU can genuinely be used and int8 otherwise, to save 1.9 GB
of disk and memory when it cannot. `murmur doctor` shows the decision:

```
• model       parakeet-tdt-0.6b-v3 [Int8]
✓ model       parakeet-tdt-0.6b-v3-fp32 [Fp32]  ← selected
```

Set `precision` to `"fp32"` or `"int8"` to decide it yourself; an explicit
choice falls back rather than failing when those weights are not installed.

"Can the GPU genuinely be used" is three questions, not one: the build must have
a CUDA provider, the driver must report a device, *and* the userspace runtime
must be loadable. The driver is the misleading one — it ships with the kernel
module, so it is present on machines with no CUDA toolkit at all. Checking only
the driver selects 2.5 GB of fp32 weights and then runs them on the CPU.

The first call is always slower — kernel selection, autotuning and allocator
warm-up. `murmur transcribe --repeat N` reports it separately, because averaging
it in flatters or slanders the device depending only on how many times you ran it.

#### CUDA without root

ONNX Runtime's provider is linked against **CUDA 13**, which distributions lag
well behind. You do not need a system CUDA toolkit for this: only the *driver* is
privileged, and it is already installed. Everything else is ordinary userspace
shared objects, so Murmur keeps its own copy:

```sh
CD=~/.local/share/murmur/cuda
mkdir -p "$CD/wheels" "$CD/lib" && cd "$CD/wheels"
pip download --no-deps -d . nvidia-cuda-runtime nvidia-cublas nvidia-cudnn-cu13
for w in *.whl; do python3 -m zipfile -e "$w" extracted/; done
find extracted -name "*.so*" -type f -exec cp -P {} "$CD/lib/" \;
```

Undo it with `rm -rf ~/.local/share/murmur/cuda`. Nothing else on the system is
touched.

`LD_LIBRARY_PATH` cannot help here — glibc reads it once at process start — so
Murmur loads each library itself with `RTLD_GLOBAL`, which also satisfies the
sub-libraries cuDNN opens by bare name at runtime.

### Why streaming is not the default

Streaming ought to win. A batch model cannot start until the key is released, so
all of its inference lands inside the delay the user feels; a cache-aware
streaming model consumes the utterance as it happens and leaves only the last
chunk outstanding. Measured, that is exactly what happens:

| audio | batch, release to text | streaming tail |
| --- | --- | --- |
| 11 s | 35.7 ms | 14.0 ms |
| 66 s | 174.2 ms | 13.8 ms |

Batch latency scales with utterance length. The streaming tail is flat.

It still is not the default, because the streaming model decodes only on chunk
boundaries and cannot be made to emit its final partial chunk — up to 560 ms of
speech. Its API has no flush, and padding the tail does not induce one: neither
digital silence nor a noise floor yields a single token, because the decoder
emits on acoustic evidence rather than elapsed frames.

On a clip cut mid-utterance:

```
batch      "And so, my fellow Americans, ask not what"
streaming  "And so my fellow Americans, ask not"
```

Dictation ends precisely where that risk is highest — you release the key just
after your last word. A tool that usually drops it is not usable, and 20 ms of
latency is not worth a word.

So the plan is to use each for what it is good at: streaming for live partials
in the HUD, and batch for the text that actually gets typed. Since Parakeet on a
GPU transcribes 11 seconds in 36 ms, periodically re-running it over the growing
recording is affordable enough to produce partials from the *same* model, which
may remove the need for a second one entirely.

`crates/murmur-asr/tests/streaming_accuracy.rs` pins the limitation, so if a
future export fixes it the test fails and tells us to revisit this.

#### Never trusting the device

ORT does not fail when a provider cannot be registered: it logs the error and
carries on using the CPU. A build asking for the GPU will therefore report
success while running at a fraction of the speed. Murmur watches ORT's own log
and reports the device it actually got:

```
model  parakeet-tdt-0.6b-v3 (int8) on cpu (cuda registration failed)
```

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
