mod selftest;
mod settings;
mod terminal;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
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
        Command::Transcribe { path, repeat, model, accelerator } =>
            transcribe(&path, repeat, model.as_deref(), accelerator.as_deref()),
        Command::Devices => devices(),
        Command::Config { init } => config(init),
    }
}

/// Set up logging, and — on CUDA builds — the layer that watches ONNX Runtime's
/// execution-provider registration so the model can report its real device.
fn install_tracing() {
    use tracing_subscriber::layer::SubscriberExt as _;
    use tracing_subscriber::util::SubscriberInitExt as _;

    let filter = tracing_subscriber::EnvFilter::try_from_env("MURMUR_LOG")
        .unwrap_or_else(|_| "warn".into());
    let fmt = tracing_subscriber::fmt::layer().with_target(false).with_writer(std::io::stderr);

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
    println!("  microphone  {} ({} Hz)", microphone.name(), microphone.sample_rate());
    println!("  transcriber {}", transcriber.name());
    println!("  injection   {}\n", sink.name());
    if config.trigger.hands_free {
        println!("  double-tap {} for hands-free; press again to stop.", config.trigger.key);
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
        return Ok(Box::new(Mock::default().with_delay(Duration::from_millis(40))));
    }
    match config.asr.engine {
        AsrEngine::Parakeet => {
            let dir = settings::expand_home(&config.asr.model_dir);
            let model =
                murmur_asr::Parakeet::open(&dir, config.asr.precision, config.asr.accelerator)
                    .with_context(|| format!("loading a model from {}", dir.display()))?;
            Ok(Box::new(model))
        }
        AsrEngine::Whisper => anyhow::bail!(
            "the whisper engine is not wired up yet; set asr.engine = \"parakeet\""
        ),
        AsrEngine::Mock => unreachable!("handled above"),
    }
}

/// Read a WAV file as 16 kHz mono, whatever it started as.
fn read_wav(path: &std::path::Path) -> Result<Vec<f32>> {
    let mut reader = hound::WavReader::open(path)
        .with_context(|| format!("opening {}", path.display()))?;
    let spec = reader.spec();
    let interleaved: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let scale = f32::from(i16::MAX);
            reader.samples::<i32>().map(|s| s.map(|v| v as f32 / scale)).collect::<Result<_, _>>()?
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
    println!("  first  {first:?} ({:.0}x realtime)", audio.as_secs_f32() / first.as_secs_f32());
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

fn doctor() -> Result<()> {
    let report = murmur_inject::probe();
    let width = report.iter().map(|c| c.name.len()).max().unwrap_or(0).max(10);

    println!("murmur doctor\n");
    for capability in &report {
        let mark = if capability.available { "\u{2713}" } else { "\u{2717}" };
        println!("  {mark} {:width$}  {}", capability.name, capability.detail);
        if let Some(remedy) = &capability.remedy {
            println!("      {:width$}  \u{2192} {remedy}", "");
        }
    }

    let config = settings::load().unwrap_or_default();
    match murmur_hotkey::key_by_name(&config.trigger.key).and_then(murmur_hotkey::watch) {
        Ok(_) => println!("  \u{2713} {:width$}  {} is readable", "trigger", config.trigger.key),
        Err(error) => {
            println!("  \u{2717} {:width$}  {error}", "trigger");
        }
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
    println!("  ready. `murmur selftest` verifies injection; `murmur listen --mock` the whole loop.");
    Ok(())
}

/// Report which weights are installed and which would be loaded.
///
/// The choice depends on hardware, so showing it here saves the user working out
/// why a 2.5 GB model did or did not get picked.
fn model_report(config: &Config, width: usize) {
    if config.asr.engine == AsrEngine::Mock {
        println!("  \u{2713} {:width$}  scripted transcriber (no model needed)", "model");
        return;
    }

    let root = settings::expand_home(&config.asr.model_dir);
    let variants = murmur_asr::models::discover(&root);
    if variants.is_empty() {
        println!("  \u{2717} {:width$}  no model under {}", "model", root.display());
        println!("      {:width$}  \u{2192} see the Models section of the README", "");
        return;
    }

    let gpu = gpu_usable();
    let chosen = murmur_asr::models::choose(&variants, gpu, config.asr.precision);
    for variant in &variants {
        let name = variant.dir.file_name().unwrap_or(variant.dir.as_os_str());
        let mark = if Some(variant) == chosen { "\u{2713}" } else { "\u{2022}" };
        let note = if Some(variant) == chosen { "  \u{2190} selected" } else { "" };
        println!(
            "  {mark} {:width$}  {} [{:?}]{note}",
            "model",
            name.to_string_lossy(),
            variant.kind
        );
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
            0 => println!("  \u{2022} {:width$}  CUDA build; no bundled runtime at {}", "accelerator", dir.display()),
            n => println!("  \u{2713} {:width$}  CUDA build; {n} runtime libraries from {}", "accelerator", dir.display()),
        }
    }
    #[cfg(not(feature = "cuda"))]
    println!("  \u{2022} {:width$}  CPU only (rebuild with --features cuda for GPU)", "accelerator");
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
        println!("{}{}", path.display(), if path.exists() { "" } else { "  (not created yet)" });
    }
    Ok(())
}

fn type_text(text: &str, after: u64) -> Result<()> {
    anyhow::ensure!(!text.is_empty(), "nothing to type");
    let config = settings::load()?.inject;
    let route =
        if murmur_inject::routes_to_clipboard(&config, text) { "clipboard" } else { "keyboard" };

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
