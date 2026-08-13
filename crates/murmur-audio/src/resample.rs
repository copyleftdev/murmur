use crate::AudioError;
use murmur_core::config::TARGET_SAMPLE_RATE;
use rubato::audioadapter_buffers::direct::InterleavedSlice;
use rubato::{Fft, FixedSync, Indexing, Resampler};

/// Frames per FFT chunk. Larger is marginally more efficient; smaller keeps the
/// tail handling cheap. At 48 kHz this is ~21 ms.
const CHUNK: usize = 1024;

/// Resample mono `input` from `rate` to the 16 kHz every transcriber expects.
///
/// Pure: no device, no clock, no global state — so the signal path can be tested
/// against synthetic tones rather than against a microphone.
///
/// # Errors
/// Fails only if `rate` is zero or rubato rejects the ratio.
pub fn to_target(input: &[f32], rate: u32) -> Result<Vec<f32>, AudioError> {
    if rate == TARGET_SAMPLE_RATE {
        return Ok(input.to_vec());
    }
    if rate == 0 {
        return Err(AudioError::Resample("input sample rate is zero".into()));
    }
    if input.is_empty() {
        return Ok(Vec::new());
    }

    let mut resampler =
        Fft::<f32>::new(rate as usize, TARGET_SAMPLE_RATE as usize, CHUNK, 1, FixedSync::Input)
            .map_err(|e| AudioError::Resample(e.to_string()))?;

    let expected = expected_len(input.len(), rate);
    // Rubato writes whole chunks, so the buffer must have room to overshoot.
    let mut output = vec![0.0f32; expected + 2 * CHUNK];

    let source = InterleavedSlice::new(input, 1, input.len())
        .map_err(|e| AudioError::Resample(e.to_string()))?;
    let capacity = output.len();
    let mut sink = InterleavedSlice::new_mut(&mut output, 1, capacity)
        .map_err(|e| AudioError::Resample(e.to_string()))?;

    let mut indexing = Indexing::new();
    let mut left = input.len();
    let mut next = resampler.input_frames_next();
    let mut written = 0usize;

    while left >= next {
        let (consumed, produced) = resampler
            .process_into_buffer(&source, &mut sink, Some(&indexing))
            .map_err(|e| AudioError::Resample(e.to_string()))?;
        indexing.input_offset += consumed;
        indexing.output_offset += produced;
        written += produced;
        left -= consumed;
        next = resampler.input_frames_next();
    }

    // The final, short chunk: rubato zero-pads it internally.
    indexing.partial_len = Some(left);
    let (_, produced) = resampler
        .process_into_buffer(&source, &mut sink, Some(&indexing))
        .map_err(|e| AudioError::Resample(e.to_string()))?;
    written += produced;

    output.truncate(written.min(expected));
    Ok(output)
}

fn expected_len(input_len: usize, rate: u32) -> usize {
    (input_len as u64 * u64::from(TARGET_SAMPLE_RATE) / u64::from(rate)) as usize
}

/// Average interleaved channels down to mono in place of allocating twice.
#[must_use]
pub fn to_mono(interleaved: &[f32], channels: u16) -> Vec<f32> {
    let channels = channels.max(1) as usize;
    if channels == 1 {
        return interleaved.to_vec();
    }
    interleaved
        .chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::TAU;

    fn sine(freq: f32, rate: u32, secs: f32) -> Vec<f32> {
        let n = (rate as f32 * secs) as usize;
        (0..n).map(|i| (TAU * freq * i as f32 / rate as f32).sin()).collect()
    }

    /// Frequency estimated from zero crossings — robust enough to prove the
    /// resampler preserves pitch rather than merely producing the right count.
    fn dominant_freq(samples: &[f32], rate: u32) -> f32 {
        let crossings = samples
            .windows(2)
            .filter(|w| (w[0] < 0.0) != (w[1] < 0.0))
            .count();
        crossings as f32 * rate as f32 / (2.0 * samples.len() as f32)
    }

    #[test]
    fn the_target_rate_is_returned_untouched() {
        let input = sine(440.0, TARGET_SAMPLE_RATE, 0.1);
        assert_eq!(to_target(&input, TARGET_SAMPLE_RATE).unwrap(), input);
    }

    #[test]
    fn empty_input_resamples_to_empty_output() {
        assert!(to_target(&[], 48_000).unwrap().is_empty());
    }

    #[test]
    fn a_zero_sample_rate_is_rejected_rather_than_dividing_by_zero() {
        assert!(to_target(&[0.0; 16], 0).is_err());
    }

    #[test]
    fn output_length_tracks_the_rate_ratio() {
        for rate in [44_100u32, 48_000, 96_000] {
            let input = sine(440.0, rate, 1.0);
            let out = to_target(&input, rate).unwrap();
            let expected = expected_len(input.len(), rate);
            let drift = out.len().abs_diff(expected);
            assert!(
                drift <= CHUNK,
                "{rate} Hz: got {} samples, expected ~{expected}",
                out.len()
            );
        }
    }

    #[test]
    fn a_tone_keeps_its_pitch_across_resampling() {
        for rate in [44_100u32, 48_000] {
            let input = sine(440.0, rate, 1.0);
            let out = to_target(&input, rate).unwrap();
            let freq = dominant_freq(&out[800..out.len() - 800], TARGET_SAMPLE_RATE);
            assert!(
                (freq - 440.0).abs() < 10.0,
                "{rate} Hz -> 16 kHz shifted 440 Hz to {freq} Hz"
            );
        }
    }

    #[test]
    fn content_above_the_nyquist_limit_does_not_alias_down_into_speech() {
        // 7 kHz survives; 15 kHz cannot be represented at 16 kHz and must be
        // filtered out rather than folded back over the voice band.
        let input = sine(15_000.0, 48_000, 0.5);
        let out = to_target(&input, 48_000).unwrap();
        let energy: f32 = out.iter().map(|s| s * s).sum::<f32>() / out.len() as f32;
        assert!(energy < 0.05, "aliased energy {energy} leaked into the 16 kHz band");
    }

    #[test]
    fn stereo_is_averaged_to_mono() {
        let interleaved = [1.0, -1.0, 0.5, 0.5];
        assert_eq!(to_mono(&interleaved, 2), vec![0.0, 0.5]);
    }

    #[test]
    fn mono_input_passes_through_the_downmix_unchanged() {
        let mono = [0.1, 0.2, 0.3];
        assert_eq!(to_mono(&mono, 1), mono.to_vec());
    }
}
