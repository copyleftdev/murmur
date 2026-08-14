mod models;
mod selftest;
mod settings;
mod terminal;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use murmur_asr::StreamingTranscriber as _;
use murmur_asr::{Mock, Transcriber};
use murmur_audio::Microphone;
use murmur_core::{AsrEngine, Config, Formatter, Session};
use murmur_engine::Engine;
use murmur_inject::{Injector, TextSink};
use std::time::Duration;

#[derive(Parser)]
#[command(
    name = "murmur",
    version,
    about = "Local-first voice typing that never leaves your machine"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum ModelAction {
    /// Fetch the weights into the configured model directory.
    Pull {
        /// Which weights: auto, fp32 or int8. Auto follows the hardware.
        #[arg(long, default_value = "auto")]
        precision: String,
    },
    /// Show which models are installed, and which would be used.
    List,
}

#[derive(Subcommand)]
enum Command {
    /// Hold the trigger key, speak, release. Text appears wherever your cursor is.
    Listen {
        /// Use the scripted transcriber instead of a model, to verify the loop.
        #[arg(long)]
        mock: bool,
    },
    /// Check that this machine can run Murmur, and say how to fix what it cannot.
    Doctor,
    /// Verify the injection path end to end without typing on your screen.
    Selftest,
    /// Inject text at the cursor, as a dictation would.
    Type {
        text: Vec<String>,
        /// Seconds to wait first, so you can focus the target window.
        #[arg(long, default_value_t = 3)]
        after: u64,
    },
    /// Transcribe a WAV file. The offline way to check a model and measure it.
    Transcribe {
        /// Path to a WAV file. Any sample rate; it is resampled to 16 kHz.
        path: std::path::PathBuf,
        /// Transcribe this many times, reporting warm timings separately.
        #[arg(long, default_value_t = 1)]
        repeat: usize,
        /// Override the model directory from the config.
        #[arg(long)]
        model: Option<std::path::PathBuf>,
        /// Override the accelerator: cpu, cuda or tensor-rt.
        #[arg(long)]
        accelerator: Option<String>,
        /// Use the streaming model, feeding audio in chunks as the daemon does.
        #[arg(long)]
        stream: bool,
    },
    /// Download the speech model. Run this once after installing.
    Models {
        #[command(subcommand)]
        action: ModelAction,
    },
    /// Launch the overlay, which runs dictation with a visible HUD.
    Hud,
    /// Show key presses as Murmur sees them. Use it to pick a trigger key.
    Keys,
    /// Record from the microphone and report whether anything was heard.
    Mic {
        /// Seconds to record.
        #[arg(long, default_value_t = 3)]
        seconds: u64,
        /// Transcribe what was recorded, as a dictation would.
        #[arg(long)]
        transcribe: bool,
    },
    /// List the microphones Murmur can see.
    Devices,
    /// Print the config path, and write a default config if there is none.
    Config {
        /// Write a default config file.
        #[arg(long)]
        init: bool,
    },
}

fn main() -> Result<()> {
    install_tracing();

    match Cli::parse().command {
        Command::Listen { mock } => listen(mock),
        Command::Doctor => doctor(),
        Command::Selftest => {
            println!("murmur selftest\n");
            selftest::run(Config::default().inject.keystroke_delay_us)
        }
        Command::Type { text, after } => type_text(&text.join(" "), after),
        Command::Transcribe {
            path,
            repeat,
            model,
            accelerator,
            stream,
        } => transcribe(
            &path,
            repeat,
            model.as_deref(),
            accelerator.as_deref(),
            stream,
        ),
        Command::Models { action } => models_command(&action),
        Command::Hud => hud(),
        Command::Keys => keys(),
        Command::Mic {
            seconds,
            transcribe,
        } => mic(seconds, transcribe),
        Command::Devices => devices(),
        Command::Config { init } => config(init),
    }
}

/// Set up logging, and — on CUDA builds — the layer that watches ONNX Runtime's
/// execution-provider registration so the model can report its real device.
fn install_tracing() {
    use tracing_subscriber::layer::SubscriberExt as _;
    use tracing_subscriber::util::SubscriberInitExt as _;

    let filter =
        tracing_subscriber::EnvFilter::try_from_env("MURMUR_LOG").unwrap_or_else(|_| "warn".into());
    let fmt = tracing_subscriber::fmt::layer()
        .with_target(false)
        .with_writer(std::io::stderr);

    let registry = tracing_subscriber::registry().with(filter).with(fmt);
    #[cfg(feature = "cuda")]
    let registry = registry.with(murmur_asr::cuda::ProviderWatch);
    registry.init();
}

fn listen(force_mock: bool) -> Result<()> {
    let config = settings::load()?;
    let key = murmur_hotkey::key_by_name(&config.trigger.key)?;
    let triggers = murmur_hotkey::watch(key)?;

    let microphone = Microphone::open(&config.audio).context("opening the microphone")?;
    let transcriber = transcriber(&config, force_mock)?;
    let sink = Injector::open(config.inject).context("opening the injection backend")?;

    println!("murmur \u{2014} hold {} and speak", config.trigger.key);
    println!(
        "  microphone  {} ({} Hz)",
        microphone.name(),
        microphone.sample_rate()
    );
    println!("  transcriber {}", transcriber.name());
    println!("  injection   {}", sink.name());
    println!("  partials    live text while you speak, from the same model\n");
    if config.trigger.hands_free {
        println!(
            "  double-tap {} for hands-free; press again to stop.",
            config.trigger.key
        );
    }
    println!("  Ctrl-C to quit.\n");

    let session = Session::new(config.tuning, Formatter::new(config.format));
    let mut engine = Engine::new(
        session,
        Box::new(microphone),
        transcriber,
        Box::new(sink),
        Box::new(terminal::Terminal::default()),
    );
    engine.run(&triggers)?;

    println!("\n{}", engine.stats().summary());
    Ok(())
}

fn transcriber(config: &Config, force_mock: bool) -> Result<Box<dyn Transcriber>> {
    if force_mock || config.asr.engine == AsrEngine::Mock {
        return Ok(Box::new(
            Mock::default().with_delay(Duration::from_millis(40)),
        ));
    }
    match config.asr.engine {
        AsrEngine::Parakeet => {
            let dir = settings::expand_home(&config.asr.model_dir);
            let model =
                murmur_asr::Parakeet::open(&dir, config.asr.precision, config.asr.accelerator)
                    .with_context(|| format!("loading a model from {}", dir.display()))?;
            Ok(Box::new(model))
        }
        AsrEngine::Nemotron => anyhow::bail!(
            "the streaming engine does not fit the batch Transcriber trait; \
             use `murmur transcribe --stream` while the daemon integration lands"
        ),
        AsrEngine::Whisper => {
            anyhow::bail!("the whisper engine is not wired up yet; set asr.engine = \"parakeet\"")
        }
        AsrEngine::Mock => unreachable!("handled above"),
    }
}

/// Read a WAV file as 16 kHz mono, whatever it started as.
fn read_wav(path: &std::path::Path) -> Result<Vec<f32>> {
    let mut reader =
        hound::WavReader::open(path).with_context(|| format!("opening {}", path.display()))?;
    let spec = reader.spec();
    let interleaved: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let scale = f32::from(i16::MAX);
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / scale))
                .collect::<Result<_, _>>()?
        }
    };
    let mono = murmur_audio::resample::to_mono(&interleaved, spec.channels);
    Ok(murmur_audio::resample::to_target(&mono, spec.sample_rate)?)
}

fn transcribe(
    path: &std::path::Path,
    repeat: usize,
    model_dir: Option<&std::path::Path>,
    accelerator: Option<&str>,
    stream: bool,
) -> Result<()> {
    let mut config = settings::load()?;
    if let Some(dir) = model_dir {
        config.asr.model_dir = dir.display().to_string();
    }
    if let Some(name) = accelerator {
        config.asr.accelerator = match name {
            "cpu" => murmur_core::Accelerator::Cpu,
            "cuda" => murmur_core::Accelerator::Cuda,
            "tensor-rt" | "tensorrt" => murmur_core::Accelerator::TensorRt,
            other => anyhow::bail!("unknown accelerator {other:?}"),
        };
    }
    let samples = read_wav(path)?;
    let audio = Duration::from_secs_f32(samples.len() as f32 / 16_000.0);

    if stream {
        return transcribe_streaming(&config, &samples, audio, repeat.max(1));
    }

    let mut model = transcriber(&config, false)?;
    println!("  model  {}", model.name());
    println!("  audio  {:.2}s", audio.as_secs_f32());

    let mut runs = Vec::with_capacity(repeat.max(1));
    let mut text = String::new();
    for _ in 0..repeat.max(1) {
        let transcript = model.transcribe(&samples)?;
        runs.push(transcript.elapsed);
        text = transcript.text;
    }

    // The first call pays for kernel selection, autotuning and allocator warm-up.
    // Reporting it together with the rest would flatter or slander the device
    // depending only on how many times you happened to run it.
    let first = runs[0];
    println!(
        "  first  {first:?} ({:.0}x realtime)",
        audio.as_secs_f32() / first.as_secs_f32()
    );
    if runs.len() > 1 {
        let mut warm: Vec<Duration> = runs[1..].to_vec();
        warm.sort_unstable();
        let median = warm[warm.len() / 2];
        println!(
            "  warm   {median:?} median of {} ({:.0}x realtime), best {:?}, worst {:?}",
            warm.len(),
            audio.as_secs_f32() / median.as_secs_f32(),
            warm[0],
            warm[warm.len() - 1]
        );
    }
    println!("\n{text}\n");
    Ok(())
}

/// Feed a recording through the streaming model the way the daemon will.
///
/// The number that matters is the tail: everything before it happened while the
/// user was still talking.
fn transcribe_streaming(
    config: &Config,
    samples: &[f32],
    audio: Duration,
    repeat: usize,
) -> Result<()> {
    let dir = settings::expand_home(&config.asr.model_dir);
    let mut model = murmur_asr::NemotronStream::open(
        &dir,
        config.asr.precision,
        config.asr.accelerator,
        config.asr.language.as_deref(),
    )
    .with_context(|| format!("loading a streaming model from {}", dir.display()))?;

    println!("  model  {}", model.name());
    println!("  audio  {:.2}s", audio.as_secs_f32());
    println!("  chunk  {}ms", model.chunk_samples() * 1000 / 16_000);

    let mut tails = Vec::with_capacity(repeat);
    let mut text = String::new();
    let mut total = Duration::ZERO;
    for _ in 0..repeat {
        let (transcript, tail) = murmur_asr::streaming::transcribe_all(&mut model, samples)?;
        tails.push(tail);
        total = transcript.elapsed;
        text = transcript.text;
    }

    tails.sort_unstable();
    println!(
        "  total  {total:?} of compute spread across {:.2}s of speech",
        audio.as_secs_f32()
    );
    println!(
        "  tail   {:?} median of {repeat} \u{2014} the only part the user waits for",
        tails[tails.len() / 2]
    );
    println!("\n{text}\n");
    Ok(())
}

fn doctor() -> Result<()> {
    let report = murmur_inject::probe();
    let width = report
        .iter()
        .map(|c| c.name.len())
        .max()
        .unwrap_or(0)
        .max(10);

    println!("murmur doctor\n");
    for capability in &report {
        let mark = if capability.available {
            "\u{2713}"
        } else {
            "\u{2717}"
        };
        println!("  {mark} {:width$}  {}", capability.name, capability.detail);
        if let Some(remedy) = &capability.remedy {
            println!("      {:width$}  \u{2192} {remedy}", "");
        }
    }

    let config = settings::load().unwrap_or_default();
    match murmur_hotkey::key_by_name(&config.trigger.key) {
        Ok(key) => {
            let boards = murmur_hotkey::readable_keyboards(key);
            println!(
                "  \u{2713} {:width$}  {} declared by {} device(s)",
                "trigger",
                config.trigger.key,
                boards.len()
            );
            for board in &boards {
                println!("      {:width$}    {board}", "");
            }
            if !boards.is_empty() {
                println!(
                    "      {:width$}  \u{2192} a device declaring the key is not proof it has one; \
                     run `murmur keys` and press it",
                    ""
                );
            }
        }
        Err(error) => println!("  \u{2717} {:width$}  {error}", "trigger"),
    }
    match Microphone::open(&config.audio) {
        Ok(microphone) => println!(
            "  \u{2713} {:width$}  {} ({} Hz, {} ch)",
            "microphone",
            microphone.name(),
            microphone.sample_rate(),
            microphone.channels()
        ),
        Err(error) => println!("  \u{2717} {:width$}  {error}", "microphone"),
    }

    model_report(&config, width);
    accelerator_report(width);

    let blocked = report.iter().any(|c| !c.available && c.name == "uinput");
    println!();
    if blocked {
        anyhow::bail!("cannot inject text: uinput unavailable");
    }
    println!(
        "  ready. `murmur selftest` verifies injection; `murmur listen --mock` the whole loop."
    );
    Ok(())
}

/// Report which weights are installed and which would be loaded.
///
/// The choice depends on hardware, so showing it here saves the user working out
/// why a 2.5 GB model did or did not get picked.
fn model_report(config: &Config, width: usize) {
    if config.asr.engine == AsrEngine::Mock {
        println!(
            "  \u{2713} {:width$}  scripted transcriber (no model needed)",
            "model"
        );
        return;
    }

    let root = settings::expand_home(&config.asr.model_dir);
    let variants = murmur_asr::models::discover(&root);
    if variants.is_empty() {
        println!(
            "  \u{2717} {:width$}  no model under {}",
            "model",
            root.display()
        );
        println!(
            "      {:width$}  \u{2192} see the Models section of the README",
            ""
        );
        return;
    }

    let gpu = gpu_usable();
    let chosen =
        murmur_asr::models::choose(&variants, gpu, config.asr.precision, family_for(config));
    for variant in &variants {
        let name = variant.dir.file_name().unwrap_or(variant.dir.as_os_str());
        let mark = if Some(variant) == chosen {
            "\u{2713}"
        } else {
            "\u{2022}"
        };
        let note = if Some(variant) == chosen {
            "  \u{2190} selected"
        } else {
            ""
        };
        println!(
            "  {mark} {:width$}  {} [{:?}]{note}",
            "model",
            name.to_string_lossy(),
            variant.kind
        );
        let _ = variant.family;
    }
}

/// The architecture the configured engine needs.
fn family_for(config: &Config) -> murmur_asr::models::Family {
    match config.asr.engine {
        AsrEngine::Nemotron => murmur_asr::models::Family::NemotronStreaming,
        _ => murmur_asr::models::Family::ParakeetTdt,
    }
}

/// Whether a GPU can actually be used, as the selector sees it.
fn gpu_usable() -> bool {
    #[cfg(feature = "cuda")]
    {
        murmur_asr::cuda::is_usable()
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

/// Report whether the configured accelerator can actually be used.
///
/// ONNX Runtime downgrades to CPU silently when a provider fails to load, so
/// this is checked and named rather than assumed.
fn accelerator_report(width: usize) {
    #[cfg(feature = "cuda")]
    {
        let dir = murmur_asr::cuda::bundled_dir();
        match murmur_asr::cuda::ensure_runtime() {
            0 => println!(
                "  \u{2022} {:width$}  CUDA build; no bundled runtime at {}",
                "accelerator",
                dir.display()
            ),
            n => println!(
                "  \u{2713} {:width$}  CUDA build; {n} runtime libraries from {}",
                "accelerator",
                dir.display()
            ),
        }
    }
    #[cfg(not(feature = "cuda"))]
    println!(
        "  \u{2022} {:width$}  CPU only (rebuild with --features cuda for GPU)",
        "accelerator"
    );
}

fn models_command(action: &ModelAction) -> Result<()> {
    let config = settings::load()?;
    let root = settings::expand_home(&config.asr.model_dir);

    match action {
        ModelAction::Pull { precision } => {
            let precision = match precision.as_str() {
                "auto" => murmur_core::Precision::Auto,
                "fp32" => murmur_core::Precision::Fp32,
                "int8" => murmur_core::Precision::Int8,
                other => anyhow::bail!("unknown precision {other:?}; use auto, fp32 or int8"),
            };
            models::pull(&root, precision, gpu_usable())?;
            Ok(())
        }
        ModelAction::List => {
            let variants = murmur_asr::models::discover(&root);
            if variants.is_empty() {
                println!("  no models under {}", root.display());
                println!("  \u{2192} murmur models pull");
                return Ok(());
            }
            for variant in &variants {
                println!("  {:?}  {}", variant.kind, variant.dir.display());
            }
            Ok(())
        }
    }
}

/// Hand over to the overlay binary, which owns its own event loop.
///
/// A separate process rather than a flag: iced must own the main thread, and a
/// GUI toolkit linked into `murmur doctor` would make every diagnostic depend on
/// a working GPU.
fn hud() -> Result<()> {
    use std::os::unix::process::CommandExt as _;

    let beside_us = std::env::current_exe().ok().and_then(|exe| {
        let candidate = exe.with_file_name("murmur-hud");
        candidate.exists().then_some(candidate)
    });
    let program = beside_us.unwrap_or_else(|| std::path::PathBuf::from("murmur-hud"));

    // `exec` rather than spawn: the overlay *is* the session from here on, and
    // an extra process in the middle only complicates signals and exit codes.
    let error = std::process::Command::new(&program).exec();
    Err(anyhow::Error::new(error).context(format!("launching {}", program.display())))
}

/// Print every key press, so a trigger that "does nothing" can be diagnosed.
fn keys() -> Result<()> {
    let config = settings::load()?;
    println!("murmur keys \u{2014} press keys to see their names. Ctrl-C to quit.\n");
    println!("  the configured trigger is {}\n", config.trigger.key);

    let events = murmur_hotkey::watch_all()?;
    while let Ok((code, edge)) = events.recv() {
        if edge != murmur_hotkey::Edge::Down {
            continue;
        }
        let name = murmur_hotkey::key_name(code).unwrap_or_else(|| format!("{code:?}"));
        let note = if name == config.trigger.key {
            "   \u{2190} your trigger"
        } else {
            ""
        };
        println!("  {name}{note}");
    }
    Ok(())
}

/// Record, then say plainly whether the microphone heard anything.
///
/// A trigger that appears to do nothing is usually one of two failures, and they
/// look identical from the outside: the key never arrived, or the audio was
/// silent. `murmur keys` answers the first; this answers the second.
fn mic(seconds: u64, transcribe: bool) -> Result<()> {
    let config = settings::load()?;
    let microphone = Microphone::open(&config.audio).context("opening the microphone")?;
    println!(
        "  device  {} ({} Hz, {} ch)",
        microphone.name(),
        microphone.sample_rate(),
        microphone.channels()
    );
    println!("\n  recording for {seconds}s \u{2014} say something\n");

    microphone.begin();
    let mut peak = 0.0f32;
    for _ in 0..(seconds * 20) {
        std::thread::sleep(Duration::from_millis(50));
        let level = microphone.level();
        peak = peak.max(level);
        let cells = (level.clamp(0.0, 1.0) * 40.0).round() as usize;
        eprint!(
            "\r  {}{}",
            "\u{2588}".repeat(cells),
            "\u{2591}".repeat(40 - cells)
        );
    }
    eprintln!();

    let capture = microphone.finish().context("finishing the recording")?;
    let heard = capture.samples.iter().fold(0.0f32, |m, s| m.max(s.abs()));
    println!(
        "\n  captured {:.2}s at 16 kHz, peak sample {heard:.3}, peak level {peak:.3}",
        capture.duration.as_secs_f32()
    );

    if heard < 0.005 {
        println!("\n  \u{2717} the microphone produced silence.");
        println!(
            "      \u{2192} pick a device with audio.device in the config; `murmur devices` lists them"
        );
        println!("      \u{2192} check the input is not muted, and that its level is up");
        anyhow::bail!("no audio captured");
    }
    println!("  \u{2713} audio captured");

    if transcribe {
        let mut model = transcriber(&config, false)?;
        let transcript = model.transcribe(&capture.samples)?;
        println!("\n  {:?}\n", transcript.text);
        if transcript.text.trim().is_empty() {
            println!("  \u{2717} audio was captured but transcribed to nothing.");
            println!("      \u{2192} speak closer to the microphone, or raise its input level");
        }
    }
    Ok(())
}

fn devices() -> Result<()> {
    for name in murmur_audio::list_devices()? {
        println!("  {name}");
    }
    Ok(())
}

fn config(init: bool) -> Result<()> {
    if init {
        let path = settings::write_default()?;
        println!("wrote {}", path.display());
    } else {
        let path = settings::path();
        println!(
            "{}{}",
            path.display(),
            if path.exists() {
                ""
            } else {
                "  (not created yet)"
            }
        );
    }
    Ok(())
}

fn type_text(text: &str, after: u64) -> Result<()> {
    anyhow::ensure!(!text.is_empty(), "nothing to type");
    let config = settings::load()?.inject;
    let route = if murmur_inject::routes_to_clipboard(&config, text) {
        "clipboard"
    } else {
        "keyboard"
    };

    if after > 0 {
        println!("focus the target window \u{2014} typing in {after}s via {route}");
        std::thread::sleep(Duration::from_secs(after));
    }

    let started = std::time::Instant::now();
    let mut injector = Injector::open(config).context("opening the injection backend")?;
    let ready = started.elapsed();
    injector.inject(text).context("injecting text")?;

    eprintln!(
        "injected {} chars via {route} (backend ready in {ready:?}, total {:?})",
        text.chars().count(),
        started.elapsed()
    );
    Ok(())
}
