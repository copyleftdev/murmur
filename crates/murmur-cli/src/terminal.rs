use murmur_core::{Hud, Mode};
use murmur_engine::{Stats, Surface};
use std::io::Write;

/// A single-line status display for running Murmur in a terminal.
///
/// Stands in for the overlay until the iced HUD exists, and stays useful after:
/// it is the surface you want when running the daemon under a service manager
/// or watching latency while tuning.
pub struct Terminal {
    width: usize,
}

impl Default for Terminal {
    fn default() -> Self {
        Self { width: 0 }
    }
}

impl Terminal {
    fn line(&mut self, text: &str) {
        let mut out = std::io::stderr();
        let padding = self.width.saturating_sub(text.chars().count());
        let _ = write!(out, "\r{text}{:padding$}", "");
        let _ = out.flush();
        self.width = text.chars().count();
    }

    fn clear(&mut self) {
        if self.width > 0 {
            let mut out = std::io::stderr();
            let _ = write!(out, "\r{:width$}\r", "", width = self.width);
            let _ = out.flush();
            self.width = 0;
        }
    }
}

/// A twelve-cell meter. Fine enough to see speech, coarse enough not to flicker.
fn meter(level: f32) -> String {
    const CELLS: usize = 12;
    let filled = (level.clamp(0.0, 1.0) * CELLS as f32).round() as usize;
    format!("{}{}", "\u{2588}".repeat(filled), "\u{2591}".repeat(CELLS - filled))
}

impl Surface for Terminal {
    fn show(&mut self, hud: &Hud) {
        match hud {
            Hud::Hidden => self.clear(),
            Hud::Listening { mode } => {
                let label = match mode {
                    Mode::Hold => "listening",
                    Mode::Locked => "listening (hands-free \u{2014} press again to stop)",
                };
                self.line(&format!("\u{25cf} {label}  {}", meter(0.0)));
            }
            Hud::Partial { text } => self.line(&format!("\u{25cf} {text}")),
            Hud::Thinking => self.line("\u{25d0} transcribing\u{2026}"),
            Hud::Error { message } => {
                self.clear();
                eprintln!("  ! {message}");
            }
        }
    }

    fn level(&mut self, level: f32) {
        self.line(&format!("\u{25cf} listening  {}", meter(level)));
    }

    fn emitted(&mut self, text: &str) {
        self.clear();
        print!("{text}");
        let _ = std::io::stdout().flush();
    }

    fn completed(&mut self, stats: &Stats) {
        if let (Some(total), Some(transcribe), Some(inject)) =
            (stats.release_to_text(100.0), stats.transcribe(100.0), stats.inject(100.0))
        {
            let _ = stats;
            eprintln!(
                "  \u{2937} release \u{2192} text {total} (transcribe {transcribe}, inject {inject})"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_meter_spans_empty_to_full_without_overflowing() {
        assert_eq!(meter(0.0).chars().count(), 12);
        assert_eq!(meter(1.0).chars().count(), 12);
        assert_eq!(meter(2.0).chars().count(), 12, "a hot mic must not widen the meter");
        assert_eq!(meter(-1.0).chars().count(), 12);
        assert!(meter(0.0).starts_with('\u{2591}'));
        assert!(meter(1.0).starts_with('\u{2588}'));
    }

    #[test]
    fn the_meter_is_monotonic_in_level() {
        let filled = |l: f32| meter(l).chars().filter(|c| *c == '\u{2588}').count();
        let mut previous = 0;
        for step in 0..=10 {
            let current = filled(step as f32 / 10.0);
            assert!(current >= previous, "level {step} went backwards");
            previous = current;
        }
    }
}
