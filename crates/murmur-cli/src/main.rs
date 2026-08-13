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
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_env("MURMUR_LOG")
                .unwrap_or_else(|_| "warn".into()),
        )
        .with_target(false)
        .with_writer(std::io::stderr)
        .init();

    match Cli::parse().command {
        Command::Listen { mock } => listen(mock),
        Command::Doctor => doctor(),
        Command::Selftest => {
            println!("murmur selftest\n");
            selftest::run(Config::default().inject.keystroke_delay_us)
        }
        Command::Type { text, after } => type_text(&text.join(" "), after),
        Command::Devices => devices(),
        Command::Config { init } => config(init),
    }
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
    anyhow::bail!(
        "the {:?} engine is not wired up yet. Run `murmur listen --mock` to exercise \
         the full loop with a scripted transcriber.",
        config.asr.engine
    )
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

    let model = settings::expand_home(&config.asr.model_dir);
    if config.asr.engine == AsrEngine::Mock {
        println!("  \u{2713} {:width$}  scripted transcriber (no model needed)", "model");
    } else if model.exists() {
        println!("  \u{2713} {:width$}  {}", "model", model.display());
    } else {
        println!("  \u{2717} {:width$}  {} is missing", "model", model.display());
        println!("      {:width$}  \u{2192} murmur listen --mock  # exercise the loop meanwhile", "");
    }

    let blocked = report.iter().any(|c| !c.available && c.name == "uinput");
    println!();
    if blocked {
        anyhow::bail!("cannot inject text: uinput unavailable");
    }
    println!("  ready. `murmur selftest` verifies injection; `murmur listen --mock` the whole loop.");
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
