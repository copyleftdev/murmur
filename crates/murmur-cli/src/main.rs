mod selftest;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use murmur_core::Config;
use murmur_inject::{Injector, TextSink};
use std::time::Duration;

#[derive(Parser)]
#[command(name = "murmur", version, about = "Local-first voice typing that never leaves your machine")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Check that this machine can run Murmur, and say how to fix what it cannot.
    Doctor,
    /// Verify the injection path end to end without typing on your screen.
    Selftest,
    /// Inject text at the cursor, as a dictation would.
    Type {
        /// The text to inject.
        text: Vec<String>,
        /// Seconds to wait first, so you can focus the target window.
        #[arg(long, default_value_t = 3)]
        after: u64,
    },
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_env("MURMUR_LOG")
                .unwrap_or_else(|_| "warn".into()),
        )
        .with_target(false)
        .init();

    match Cli::parse().command {
        Command::Doctor => doctor(),
        Command::Selftest => {
            println!("murmur selftest\n");
            selftest::run(Config::default().inject.keystroke_delay_us)
        }
        Command::Type { text, after } => type_text(&text.join(" "), after),
    }
}

fn doctor() -> Result<()> {
    let report = murmur_inject::probe();
    let width = report.iter().map(|c| c.name.len()).max().unwrap_or(0);

    println!("murmur doctor\n");
    for capability in &report {
        let mark = if capability.available { "\u{2713}" } else { "\u{2717}" };
        println!(
            "  {mark} {:width$}  {}",
            capability.name,
            capability.detail,
            width = width
        );
        if let Some(remedy) = &capability.remedy {
            println!("      {:width$}  \u{2192} {remedy}", "", width = width);
        }
    }

    let blocking: Vec<&str> = report
        .iter()
        .filter(|c| !c.available && c.name == "uinput")
        .map(|c| c.name)
        .collect();

    println!();
    if blocking.is_empty() {
        println!("  ready. Run `murmur selftest` to verify injection end to end.");
        Ok(())
    } else {
        anyhow::bail!("cannot inject text: {} unavailable", blocking.join(", "));
    }
}

fn type_text(text: &str, after: u64) -> Result<()> {
    if text.is_empty() {
        anyhow::bail!("nothing to type");
    }
    let config = Config::default().inject;
    let route = if murmur_inject::routes_to_clipboard(&config, text) { "clipboard" } else { "keyboard" };

    if after > 0 {
        println!("focus the target window \u{2014} typing in {after}s via {route}");
        std::thread::sleep(Duration::from_secs(after));
    }

    let started = std::time::Instant::now();
    let mut injector = Injector::open(config).context("opening the injection backend")?;
    let ready = started.elapsed();
    injector.inject(text).context("injecting text")?;

    eprintln!(
        "injected {} chars via {route} (backend ready in {:?}, total {:?})",
        text.chars().count(),
        ready,
        started.elapsed()
    );
    Ok(())
}
