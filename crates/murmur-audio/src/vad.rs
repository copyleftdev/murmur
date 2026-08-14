use earshot::Detector;
use std::ops::Range;

/// Samples per VAD frame: 16 ms at 16 kHz, which is what earshot requires.
pub const FRAME: usize = 256;

/// Frames of audio kept either side of detected speech, so trimming never
/// clips a soft consonant at the start or end of an utterance.
const PAD_FRAMES: usize = 6;

/// Voice-activity score for every whole frame in `samples`.
///
/// Samples are clamped rather than rejected: a hot microphone occasionally
/// exceeds unity, and that is not a reason to refuse to transcribe.
#[must_use]
pub fn frame_scores(samples: &[f32]) -> Vec<f32> {
    let mut detector = Detector::default_boxed();
    let mut frame = [0.0f32; FRAME];
    samples
        .chunks_exact(FRAME)
        .map(|chunk| {
            for (slot, sample) in frame.iter_mut().zip(chunk) {
                *slot = sample.clamp(-1.0, 1.0);
            }
            detector.predict_f32(&frame)
        })
        .collect()
}

/// The span of `samples` worth transcribing.
///
/// Trimming silence is a latency optimisation: Parakeet's cost scales with audio
/// length, and push-to-talk always carries dead air at the tail. When no frame
/// looks like speech the full range is returned — a transcriber that sees the
/// audio and finds nothing is a far better failure than one that never sees it.
#[must_use]
pub fn speech_range(samples: &[f32], threshold: f32) -> Range<usize> {
    if samples.is_empty() {
        return 0..0;
    }
    let scores = frame_scores(samples);
    let speech: Vec<usize> = scores
        .iter()
        .enumerate()
        .filter(|(_, score)| **score >= threshold)
        .map(|(i, _)| i)
        .collect();

    let (Some(first), Some(last)) = (speech.first(), speech.last()) else {
        return 0..samples.len();
    };

    let start = first.saturating_sub(PAD_FRAMES) * FRAME;
    let end = ((last + 1 + PAD_FRAMES) * FRAME).min(samples.len());
    start..end.max(start)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::TAU;

    fn tone(freq: f32, samples: usize) -> Vec<f32> {
        (0..samples)
            .map(|i| (TAU * freq * i as f32 / 16_000.0).sin() * 0.5)
            .collect()
    }

    #[test]
    fn empty_audio_yields_an_empty_range() {
        assert_eq!(speech_range(&[], 0.5), 0..0);
    }

    #[test]
    fn silence_is_never_trimmed_away_entirely() {
        let silence = vec![0.0f32; FRAME * 50];
        assert_eq!(speech_range(&silence, 0.5), 0..silence.len());
    }

    #[test]
    fn an_unreachable_threshold_falls_back_to_the_whole_recording() {
        let audio = tone(300.0, FRAME * 40);
        assert_eq!(speech_range(&audio, 2.0), 0..audio.len());
    }

    #[test]
    fn a_threshold_of_zero_keeps_everything() {
        let audio = tone(300.0, FRAME * 40);
        let range = speech_range(&audio, 0.0);
        assert_eq!(range.start, 0);
        assert!(
            range.end >= audio.len() - FRAME,
            "range {range:?} of {}",
            audio.len()
        );
    }

    #[test]
    fn the_range_is_always_within_bounds_and_ordered() {
        for len in [0, 1, FRAME - 1, FRAME, FRAME * 3 + 7, FRAME * 100] {
            for threshold in [0.0f32, 0.3, 0.5, 0.9] {
                let audio = tone(220.0, len);
                let range = speech_range(&audio, threshold);
                assert!(range.start <= range.end, "len {len}: {range:?}");
                assert!(range.end <= audio.len(), "len {len}: {range:?}");
            }
        }
    }

    #[test]
    fn scores_are_produced_per_whole_frame_and_bounded() {
        let audio = tone(440.0, FRAME * 10 + 5);
        let scores = frame_scores(&audio);
        assert_eq!(
            scores.len(),
            10,
            "partial trailing frame must not be scored"
        );
        assert!(scores.iter().all(|s| (0.0..=1.0).contains(s)), "{scores:?}");
    }

    #[test]
    fn clipped_input_does_not_panic_the_detector() {
        let hot: Vec<f32> = (0..FRAME * 8)
            .map(|i| if i % 2 == 0 { 4.0 } else { -4.0 })
            .collect();
        let range = speech_range(&hot, 0.5);
        assert!(range.end <= hot.len());
    }
}
