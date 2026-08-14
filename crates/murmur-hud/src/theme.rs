//! How the overlay looks.
//!
//! The whole surface is one bar with one job: be readable at a glance, from the
//! corner of the eye, over whatever the user is actually working in. So it is
//! dark, translucent, and carries exactly one accent colour at a time — the
//! colour *is* the state, and the text only elaborates.

use iced::widget::{Container, Row, Text, container, row, text};
use iced::{Background, Border, Color, Length, Theme};

pub const TEXT: Color = Color::from_rgb(0.93, 0.94, 0.96);
pub const MUTED: Color = Color::from_rgb(0.55, 0.58, 0.64);

pub const IDLE: Color = Color::from_rgb(0.38, 0.40, 0.46);
pub const LISTENING: Color = Color::from_rgb(0.36, 0.86, 0.71);
pub const THINKING: Color = Color::from_rgb(0.98, 0.76, 0.36);
pub const DONE: Color = Color::from_rgb(0.56, 0.63, 0.98);
pub const FAILED: Color = Color::from_rgb(0.98, 0.46, 0.46);

/// Bars in the level meter. Enough to read as a waveform, few enough to stay calm.
const BARS: usize = 32;
const BAR_WIDTH: f32 = 3.0;
const BAR_MIN: f32 = 3.0;
const BAR_MAX: f32 = 26.0;

/// The background: a dark, rounded, translucent slab.
pub fn pill(_theme: &Theme) -> container::Style {
    container::Style {
        background: Some(Background::Color(Color::from_rgba(0.06, 0.06, 0.08, 0.93))),
        border: Border {
            radius: 18.0.into(),
            width: 1.0,
            color: Color::from_rgba(1.0, 1.0, 1.0, 0.07),
        },
        text_color: Some(TEXT),
        ..container::Style::default()
    }
}

/// A state light. Colour carries the meaning; size keeps it quiet.
pub fn dot(color: Color) -> Container<'static, crate::Message> {
    container(text(""))
        .width(Length::Fixed(10.0))
        .height(Length::Fixed(10.0))
        .style(move |_: &Theme| container::Style {
            background: Some(Background::Color(color)),
            border: Border { radius: 5.0.into(), ..Border::default() },
            ..container::Style::default()
        })
}

/// A level meter shaped like a voice rather than a bar chart.
///
/// Bar heights follow a fixed raised-cosine profile scaled by the current level,
/// so quiet speech is a low ripple and loud speech a broad swell. A flat row of
/// equal bars reads as a progress indicator, which is the wrong idea entirely.
pub fn meter(level: f32) -> Row<'static, crate::Message> {
    let heights = bar_heights(level);

    let bars = (0..BARS).map(|i| {
        let height = heights[i];
        let profile = profile(i);
        let lit = height - BAR_MIN > (BAR_MAX - BAR_MIN) * 0.04;
        let color = if lit {
            Color { a: 0.35 + 0.65 * profile, ..LISTENING }
        } else {
            Color::from_rgba(1.0, 1.0, 1.0, 0.10)
        };

        container(text(""))
            .width(Length::Fixed(BAR_WIDTH))
            .height(Length::Fixed(height))
            .style(move |_: &Theme| container::Style {
                background: Some(Background::Color(color)),
                border: Border { radius: (BAR_WIDTH / 2.0).into(), ..Border::default() },
                ..container::Style::default()
            })
            .into()
    });

    Row::with_children(bars).spacing(2).align_y(iced::Center).height(Length::Fixed(BAR_MAX))
}

/// The fixed shape of the meter: tallest in the middle, tapering to the ends.
///
/// `sin` is clamped before the fractional power because `sin(PI)` in `f32` is
/// very slightly *negative*, and a negative base with a fractional exponent is
/// `NaN` — which drew a garbage final bar until a test caught it.
fn profile(index: usize) -> f32 {
    let position = index as f32 / (BARS - 1) as f32;
    (std::f32::consts::PI * position).sin().max(0.0).powf(0.7)
}

/// Every bar height for a given level.
///
/// Split out so the shape can be tested directly rather than through a copy of
/// the arithmetic, which would let the two drift apart precisely when it matters.
fn bar_heights(level: f32) -> [f32; BARS] {
    // Perceptual, not linear: speech spends most of its time well below unity,
    // and a linear meter therefore looks broken.
    let loudness = level.clamp(0.0, 1.0).powf(0.45);
    std::array::from_fn(|i| BAR_MIN + (BAR_MAX - BAR_MIN) * loudness * profile(i))
}

/// The close control: present, but never competing with the text.
pub fn close(_theme: &Theme, status: iced::widget::button::Status) -> iced::widget::button::Style {
    use iced::widget::button::Status;

    let (background, text_color) = match status {
        Status::Hovered | Status::Pressed => {
            (Some(Background::Color(Color::from_rgba(1.0, 1.0, 1.0, 0.12))), TEXT)
        }
        _ => (None, Color { a: 0.5, ..MUTED }),
    };

    iced::widget::button::Style {
        background,
        text_color,
        border: Border { radius: 8.0.into(), ..Border::default() },
        ..iced::widget::button::Style::default()
    }
}

/// Words the user actually said.
///
/// Owned rather than borrowed: most of what the overlay shows is assembled for
/// the frame that shows it, and threading those lifetimes through the view adds
/// nothing but noise.
pub fn speech(content: impl Into<String>) -> Text<'static> {
    text(content.into()).size(16).color(TEXT)
}

/// Everything that is not the user's words.
pub fn muted(content: impl Into<String>) -> Text<'static> {
    text(content.into()).size(14).color(MUTED)
}

pub fn failed(reason: impl Into<String>) -> Row<'static, crate::Message> {
    row![dot(FAILED), text(reason.into()).size(14).color(FAILED)]
        .spacing(12)
        .align_y(iced::Center)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn heights(level: f32) -> [f32; BARS] {
        bar_heights(level)
    }

    #[test]
    fn no_level_ever_produces_a_bar_that_is_not_a_number() {
        for step in -10..=30 {
            let level = step as f32 / 10.0;
            for (i, height) in heights(level).into_iter().enumerate() {
                assert!(height.is_finite(), "level {level} made bar {i} {height}");
            }
        }
    }

    #[test]
    fn silence_draws_a_flat_line_rather_than_nothing() {
        let bars = heights(0.0);
        assert!(bars.iter().all(|h| (*h - BAR_MIN).abs() < 1e-3), "{bars:?}");
    }

    #[test]
    fn every_bar_stays_within_its_bounds_at_any_level() {
        for step in 0..=20 {
            let level = step as f32 / 10.0 - 0.5;
            for height in heights(level) {
                assert!(
                    (BAR_MIN..=BAR_MAX).contains(&height),
                    "level {level} produced a {height}px bar"
                );
            }
        }
    }

    #[test]
    fn louder_never_draws_shorter() {
        let quiet = heights(0.2);
        let loud = heights(0.8);
        for (q, l) in quiet.iter().zip(&loud) {
            assert!(l >= q, "louder audio drew a shorter bar: {q} then {l}");
        }
    }

    #[test]
    fn the_meter_swells_in_the_middle_like_a_waveform() {
        let bars = heights(1.0);
        let middle = bars[BARS / 2];
        assert!(middle > bars[0], "the meter is flat, which reads as a progress bar");
        assert!(middle > bars[BARS - 1]);
    }

    #[test]
    fn quiet_speech_is_visible_rather_than_lost_in_the_floor() {
        // A linear meter would draw 5% of full height here; the perceptual
        // curve must lift it to something a person can actually see.
        let quiet = heights(0.05);
        let span = BAR_MAX - BAR_MIN;
        let peak = quiet.iter().copied().fold(0.0f32, f32::max) - BAR_MIN;
        assert!(peak > span * 0.15, "quiet speech drew only {peak}px of {span}px");
    }
}
