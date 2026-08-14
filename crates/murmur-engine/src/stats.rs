use murmur_core::{Latency, Millis};
use std::fmt::Write as _;
use std::time::Duration;

/// Running latency and throughput for a session.
///
/// Every sample is kept rather than averaged into a counter. Dictation latency
/// is judged by its tail — one 900 ms utterance in twenty is what a user
/// remembers, and a mean of 140 ms hides it completely.
///
/// The clock is the engine's monotonic origin, and the headline interval is
/// *trigger release to text delivered*: the gap the user actually perceives.
/// Everything before release is the user speaking, and no amount of engineering
/// makes that shorter.
#[derive(Debug, Default, Clone)]
pub struct Stats {
    release_to_text: Vec<Millis>,
    transcribe: Vec<Millis>,
    inject: Vec<Millis>,
    audio: Duration,
    inference: Duration,
    characters: usize,
}

impl Stats {
    pub fn record_latency(&mut self, latency: Latency) {
        self.release_to_text.push(latency.release_to_text);
        self.transcribe.push(latency.transcribe);
        self.inject.push(latency.inject);
    }

    pub fn record_audio(&mut self, audio: Duration, inference: Duration) {
        self.audio += audio;
        self.inference += inference;
    }

    pub fn record_emission(&mut self, characters: usize) {
        self.characters += characters;
    }

    #[must_use]
    pub fn utterances(&self) -> usize {
        self.release_to_text.len()
    }

    #[must_use]
    pub fn characters(&self) -> usize {
        self.characters
    }

    /// Aggregate audio seconds processed per second of inference.
    #[must_use]
    pub fn realtime_factor(&self) -> Option<f32> {
        let seconds = self.inference.as_secs_f32();
        (seconds > 0.0).then(|| self.audio.as_secs_f32() / seconds)
    }

    /// Nearest-rank percentile of the release-to-text interval.
    #[must_use]
    pub fn release_to_text(&self, percentile: f32) -> Option<Millis> {
        nearest_rank(&self.release_to_text, percentile)
    }

    #[must_use]
    pub fn transcribe(&self, percentile: f32) -> Option<Millis> {
        nearest_rank(&self.transcribe, percentile)
    }

    #[must_use]
    pub fn inject(&self, percentile: f32) -> Option<Millis> {
        nearest_rank(&self.inject, percentile)
    }

    /// A one-glance report, naming the interval it measures.
    #[must_use]
    pub fn summary(&self) -> String {
        if self.release_to_text.is_empty() {
            return "no utterances yet".to_owned();
        }
        let mut out = String::new();
        let _ = write!(
            out,
            "{} utterance(s), {} chars, {:.1}s audio",
            self.utterances(),
            self.characters,
            self.audio.as_secs_f32()
        );
        if let Some(rtf) = self.realtime_factor() {
            let _ = write!(out, " at {rtf:.0}x realtime");
        }
        let _ = write!(out, "\nrelease \u{2192} text  ");
        for p in [50.0, 90.0, 99.0] {
            if let Some(v) = self.release_to_text(p) {
                let _ = write!(out, " p{p:.0} {v}");
            }
        }
        if let (Some(t), Some(i)) = (self.transcribe(50.0), self.inject(50.0)) {
            let _ = write!(out, "\n  of which     transcribe {t}, inject {i} (p50)");
        }
        out
    }
}

/// Nearest-rank percentile: the smallest sample at or above the given rank.
///
/// No interpolation, so every reported figure is a latency that actually
/// happened rather than an average of two that did not.
fn nearest_rank(samples: &[Millis], percentile: f32) -> Option<Millis> {
    if samples.is_empty() {
        return None;
    }
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    let rank = (percentile / 100.0 * sorted.len() as f32).ceil().max(1.0) as usize;
    sorted.get(rank.min(sorted.len()) - 1).copied()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn latency(release_to_text: u64) -> Latency {
        Latency {
            speaking: Millis(1_000),
            transcribe: Millis(release_to_text / 2),
            inject: Millis(release_to_text / 2),
            release_to_text: Millis(release_to_text),
        }
    }

    fn stats_of(values: &[u64]) -> Stats {
        let mut stats = Stats::default();
        for v in values {
            stats.record_latency(latency(*v));
        }
        stats
    }

    #[test]
    fn an_empty_report_says_so_rather_than_showing_zeros() {
        assert_eq!(Stats::default().summary(), "no utterances yet");
        assert!(Stats::default().release_to_text(50.0).is_none());
    }

    #[test]
    fn percentiles_are_real_observed_samples_not_interpolations() {
        let stats = stats_of(&[100, 200, 300, 400]);
        for p in [1.0, 25.0, 50.0, 75.0, 99.0, 100.0] {
            let value = stats.release_to_text(p).unwrap();
            assert!(
                [100, 200, 300, 400].contains(&value.0),
                "p{p} returned {value}, which never happened"
            );
        }
    }

    #[test]
    fn nearest_rank_matches_the_worked_definition() {
        let stats = stats_of(&[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]);
        assert_eq!(stats.release_to_text(50.0), Some(Millis(50)));
        assert_eq!(stats.release_to_text(90.0), Some(Millis(90)));
        assert_eq!(stats.release_to_text(100.0), Some(Millis(100)));
    }

    #[test]
    fn a_single_sample_is_every_percentile() {
        let stats = stats_of(&[42]);
        for p in [0.0, 50.0, 99.0, 100.0] {
            assert_eq!(stats.release_to_text(p), Some(Millis(42)), "p{p}");
        }
    }

    #[test]
    fn the_tail_is_reported_rather_than_averaged_away() {
        let mut values = vec![100u64; 99];
        values.push(4_000);
        let stats = stats_of(&values);
        assert_eq!(stats.release_to_text(50.0), Some(Millis(100)));
        assert_eq!(
            stats.release_to_text(100.0),
            Some(Millis(4_000)),
            "outlier must survive"
        );
    }

    #[test]
    fn ordering_of_arrival_does_not_change_any_percentile() {
        let ascending = stats_of(&[10, 20, 30, 40, 50]);
        let descending = stats_of(&[50, 40, 30, 20, 10]);
        for p in [50.0, 90.0, 100.0] {
            assert_eq!(ascending.release_to_text(p), descending.release_to_text(p));
        }
    }

    #[test]
    fn the_realtime_factor_relates_audio_to_inference() {
        let mut stats = Stats::default();
        stats.record_audio(Duration::from_secs(10), Duration::from_millis(100));
        assert!((stats.realtime_factor().unwrap() - 100.0).abs() < 1e-3);
    }

    #[test]
    fn zero_inference_time_reports_no_factor_rather_than_infinity() {
        let mut stats = Stats::default();
        stats.record_audio(Duration::from_secs(10), Duration::ZERO);
        assert!(stats.realtime_factor().is_none());
    }

    #[test]
    fn the_summary_names_the_interval_it_measures() {
        let stats = stats_of(&[120, 130, 140]);
        let summary = stats.summary();
        assert!(summary.contains("release"), "{summary}");
        assert!(summary.contains("p50"), "{summary}");
    }
}
