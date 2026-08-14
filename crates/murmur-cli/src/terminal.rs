use murmur_core::{Hud, Mode};
use murmur_engine::{Stats, Surface};
use std::io::Write;

/// A single-line status display for running Murmur in a terminal.
///
/// Stands in for the overlay until the iced HUD exists, and stays useful after:
/// it is the surface you want when running the daemon under a service manager
/// or watching latency while tuning.
#[derive(Default)]
pub struct Terminal {
    width: usize,
    partial: String,
}

/// Longest run of live text shown. Beyond this the *end* is kept, because that
/// is where the words are still arriving.
const PARTIAL_WIDTH: usize = 56;

impl Terminal {
    /// Repaint the capture line: meter first, then whatever text has arrived.
    ///
    /// Both live on one line on purpose. The meter is repainted every tick, so
    /// drawing them separately means the meter erases the text a moment after it
    /// appears — the feature looks broken precisely when it is working.
    fn capturing(&mut self, level: f32) {
        let meter = meter(level);
        let line = if self.partial.is_empty() {
            format!("\u{25cf} listening  {meter}")
        } else {
            format!("\u{25cf} {meter}  {}", tail(&self.partial, PARTIAL_WIDTH))
        };
        self.line(&line);
    }

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

/// The last `width` characters, so live text scrolls with the speaker.
fn tail(text: &str, width: usize) -> String {
    let count = text.chars().count();
    if count <= width {
        return text.to_owned();
    }
    let skipped = count - width + 1;
    format!("\u{2026}{}", text.chars().skip(skipped).collect::<String>())
}

/// A twelve-cell meter. Fine enough to see speech, coarse enough not to flicker.
fn meter(level: f32) -> String {
    const CELLS: usize = 12;
    let filled = (level.clamp(0.0, 1.0) * CELLS as f32).round() as usize;
    format!(
        "{}{}",
        "\u{2588}".repeat(filled),
        "\u{2591}".repeat(CELLS - filled)
    )
}

impl Surface for Terminal {
    fn show(&mut self, hud: &Hud) {
        match hud {
            Hud::Hidden => {
                self.partial.clear();
                self.clear();
            }
            Hud::Listening { mode } => {
                let label = match mode {
                    Mode::Hold => "listening",
                    Mode::Locked => "listening (hands-free \u{2014} press again to stop)",
                };
                self.partial.clear();
                self.line(&format!("\u{25cf} {label}  {}", meter(0.0)));
            }
            Hud::Partial { text } => {
                self.partial.clone_from(text);
                self.capturing(0.0);
            }
            Hud::Thinking => self.line("\u{25d0} transcribing\u{2026}"),
            Hud::Error { message } => {
                self.clear();
                eprintln!("  ! {message}");
            }
        }
    }

    fn level(&mut self, level: f32) {
        self.capturing(level);
    }

    fn emitted(&mut self, text: &str) {
        self.partial.clear();
        self.clear();
        print!("{text}");
        let _ = std::io::stdout().flush();
    }

    fn completed(&mut self, stats: &Stats) {
        if let (Some(total), Some(transcribe), Some(inject)) = (
            stats.release_to_text(100.0),
            stats.transcribe(100.0),
            stats.inject(100.0),
        ) {
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
        assert_eq!(
            meter(2.0).chars().count(),
            12,
            "a hot mic must not widen the meter"
        );
        assert_eq!(meter(-1.0).chars().count(), 12);
        assert!(meter(0.0).starts_with('\u{2591}'));
        assert!(meter(1.0).starts_with('\u{2588}'));
    }

    #[test]
    fn live_text_survives_a_meter_repaint() {
        let mut terminal = Terminal::default();
        terminal.show(&Hud::Partial {
            text: "hello there".into(),
        });
        terminal.level(0.5);
        assert_eq!(
            terminal.partial, "hello there",
            "the meter erased the live text"
        );
    }

    #[test]
    fn live_text_is_cleared_when_the_utterance_ends() {
        let mut terminal = Terminal::default();
        terminal.show(&Hud::Partial {
            text: "hello".into(),
        });
        terminal.show(&Hud::Hidden);
        assert!(terminal.partial.is_empty());

        terminal.show(&Hud::Partial {
            text: "hello".into(),
        });
        terminal.show(&Hud::Listening { mode: Mode::Hold });
        assert!(
            terminal.partial.is_empty(),
            "text from the last utterance leaked into the next"
        );
    }

    #[test]
    fn long_live_text_keeps_the_end_where_the_new_words_are() {
        let long: String = std::iter::repeat_n('a', 40)
            .chain("THE END".chars())
            .collect();
        let shown = tail(&long, 20);
        assert!(shown.ends_with("THE END"), "{shown:?}");
        assert_eq!(shown.chars().count(), 20);
        assert!(shown.starts_with('\u{2026}'));
    }

    #[test]
    fn short_live_text_is_shown_whole() {
        assert_eq!(tail("hi", 20), "hi");
        assert_eq!(tail("exactly twenty chars", 20), "exactly twenty chars");
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
