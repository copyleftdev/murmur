//! Microphone capture shaped for push-to-talk dictation.
//!
//! The stream is opened once and never stopped. Starting a capture only changes
//! which samples are kept, which removes device start-up — tens of milliseconds
//! on ALSA, more on Bluetooth — from the path between pressing the key and
//! speaking, and lets us keep a rolling pre-roll so the first syllable survives
//! a user who starts talking before the key is fully down.

pub mod resample;
pub mod vad;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SizedSample};
use murmur_core::config::{AudioConfig, TARGET_SAMPLE_RATE};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::Duration;

#[derive(Debug, thiserror::Error)]
pub enum AudioError {
    #[error("no input device found{}", match .0.as_str() { "" => String::new(), name => format!(" matching {name:?}") })]
    NoDevice(String),
    #[error("audio device: {0}")]
    Device(String),
    #[error("resample: {0}")]
    Resample(String),
}

type Result<T> = std::result::Result<T, AudioError>;

/// Audio ready for a transcriber: mono, 16 kHz, silence trimmed.
#[derive(Debug, Clone)]
pub struct Capture {
    pub samples: Vec<f32>,
    /// Samples the voice-activity pass removed, for latency accounting.
    pub trimmed: usize,
    pub duration: Duration,
}

impl Capture {
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}

#[derive(Debug, Default)]
struct Shared {
    preroll: VecDeque<f32>,
    preroll_capacity: usize,
    recording: Option<Vec<f32>>,
    level: f32,
}

impl Shared {
    fn push(&mut self, mono: &[f32]) {
        if let Some(recording) = &mut self.recording {
            recording.extend_from_slice(mono);
        } else {
            for sample in mono {
                if self.preroll.len() == self.preroll_capacity {
                    self.preroll.pop_front();
                }
                self.preroll.push_back(*sample);
            }
        }

        let rms = (mono.iter().map(|s| s * s).sum::<f32>() / mono.len().max(1) as f32).sqrt();
        // Fast attack, slow release: a meter that falls instantly reads as broken.
        self.level = rms.max(self.level * 0.85);
    }
}

/// An open input stream with a rolling pre-roll buffer.
///
/// Not `Send`: cpal streams are bound to the thread that created them on some
/// backends, so the daemon owns one microphone on one thread and talks to the
/// rest of the system by channel.
pub struct Microphone {
    _stream: cpal::Stream,
    shared: Arc<Mutex<Shared>>,
    rate: u32,
    channels: u16,
    name: String,
    vad: bool,
}

impl std::fmt::Debug for Microphone {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Microphone")
            .field("name", &self.name)
            .field("rate", &self.rate)
            .field("channels", &self.channels)
            .finish_non_exhaustive()
    }
}

impl Microphone {
    /// Open the configured input device and start streaming immediately.
    ///
    /// # Errors
    /// Fails if no matching device exists or the host rejects the stream.
    pub fn open(config: &AudioConfig) -> Result<Self> {
        let host = cpal::default_host();
        let device = match &config.device {
            Some(wanted) => {
                let wanted_lower = wanted.to_lowercase();
                host.input_devices()
                    .map_err(|e| AudioError::Device(e.to_string()))?
                    .find(|d| {
                        device_name(d).is_some_and(|n| n.to_lowercase().contains(&wanted_lower))
                    })
                    .ok_or_else(|| AudioError::NoDevice(wanted.clone()))?
            }
            None => host
                .default_input_device()
                .ok_or_else(|| AudioError::NoDevice(String::new()))?,
        };

        let name = device_name(&device).unwrap_or_else(|| "unknown".into());
        let supported =
            device.default_input_config().map_err(|e| AudioError::Device(e.to_string()))?;
        let rate = supported.sample_rate();
        let channels = supported.channels();
        let format = supported.sample_format();
        let stream_config: cpal::StreamConfig = supported.into();

        let preroll_capacity = (rate as u64 * u64::from(config.preroll_ms) / 1000) as usize;
        let shared = Arc::new(Mutex::new(Shared {
            preroll: VecDeque::with_capacity(preroll_capacity + 1),
            preroll_capacity,
            ..Shared::default()
        }));

        let stream = match format {
            cpal::SampleFormat::F32 => build::<f32>(&device, &stream_config, &shared, channels),
            cpal::SampleFormat::I16 => build::<i16>(&device, &stream_config, &shared, channels),
            cpal::SampleFormat::U16 => build::<u16>(&device, &stream_config, &shared, channels),
            other => Err(AudioError::Device(format!("unsupported sample format {other:?}"))),
        }?;
        stream.play().map_err(|e| AudioError::Device(e.to_string()))?;

        tracing::info!(%name, rate, channels, ?format, "microphone open");
        Ok(Self { _stream: stream, shared, rate, channels, name, vad: config.vad })
    }

    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    #[must_use]
    pub fn sample_rate(&self) -> u32 {
        self.rate
    }

    #[must_use]
    pub fn channels(&self) -> u16 {
        self.channels
    }

    /// Smoothed input level in `0.0..=1.0`, for the HUD meter.
    #[must_use]
    pub fn level(&self) -> f32 {
        self.shared.lock().map_or(0.0, |s| s.level.min(1.0))
    }

    /// Begin keeping samples, seeded with the pre-roll already in hand.
    pub fn begin(&self) {
        if let Ok(mut shared) = self.shared.lock() {
            let preroll: Vec<f32> = shared.preroll.iter().copied().collect();
            shared.recording = Some(preroll);
        }
    }

    /// Stop keeping samples and hand back 16 kHz mono audio.
    ///
    /// # Errors
    /// Fails if resampling rejects the buffer.
    pub fn finish(&self) -> Result<Capture> {
        let raw = self
            .shared
            .lock()
            .ok()
            .and_then(|mut shared| shared.recording.take())
            .unwrap_or_default();

        let resampled = resample::to_target(&raw, self.rate)?;
        let full = resampled.len();
        let samples = if self.vad {
            let range = vad::speech_range(&resampled, 0.5);
            resampled[range].to_vec()
        } else {
            resampled
        };

        let duration =
            Duration::from_secs_f32(samples.len() as f32 / TARGET_SAMPLE_RATE as f32);
        Ok(Capture { trimmed: full - samples.len(), samples, duration })
    }

    /// The audio captured so far, without ending the recording.
    ///
    /// Deliberately skips the voice-activity trim that [`finish`] applies:
    /// trimming is a latency optimisation for the final pass, and applying it to
    /// a partial would make the text jump around as the trim boundary moved.
    ///
    /// [`finish`]: Self::finish
    ///
    /// # Errors
    /// Fails if resampling rejects the buffer.
    pub fn snapshot(&self) -> Result<Option<Capture>> {
        let Some(raw) = self.shared.lock().ok().and_then(|s| s.recording.clone()) else {
            return Ok(None);
        };
        let samples = resample::to_target(&raw, self.rate)?;
        let duration = Duration::from_secs_f32(samples.len() as f32 / TARGET_SAMPLE_RATE as f32);
        Ok(Some(Capture { trimmed: 0, samples, duration }))
    }

    /// Drop whatever is being recorded without producing a capture.
    pub fn discard(&self) {
        if let Ok(mut shared) = self.shared.lock() {
            shared.recording = None;
        }
    }
}

/// cpal 0.18 replaced `Device::name` with a full description record.
fn device_name(device: &cpal::Device) -> Option<String> {
    device.description().ok().map(|d| d.name().to_owned())
}

fn build<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    shared: &Arc<Mutex<Shared>>,
    channels: u16,
) -> Result<cpal::Stream>
where
    T: SizedSample,
    f32: FromSample<T>,
{
    let sink = Arc::clone(shared);
    device
        .build_input_stream(
            config.clone(),
            move |data: &[T], _: &cpal::InputCallbackInfo| {
                let floats: Vec<f32> = data.iter().map(|s| s.to_sample::<f32>()).collect();
                let mono = resample::to_mono(&floats, channels);
                if let Ok(mut shared) = sink.lock() {
                    shared.push(&mono);
                }
            },
            |error| tracing::warn!(%error, "input stream error"),
            None,
        )
        .map_err(|e| AudioError::Device(e.to_string()))
}

/// Every input device the host can see, for `murmur devices`.
///
/// # Errors
/// Fails if the host cannot enumerate devices.
pub fn list_devices() -> Result<Vec<String>> {
    let host = cpal::default_host();
    let default = host.default_input_device().and_then(|d| device_name(&d));
    Ok(host
        .input_devices()
        .map_err(|e| AudioError::Device(e.to_string()))?
        .filter_map(|d| device_name(&d))
        .map(|name| {
            if Some(&name) == default.as_ref() { format!("{name} (default)") } else { name }
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preroll_keeps_only_the_most_recent_window() {
        let mut shared = Shared { preroll_capacity: 4, ..Shared::default() };
        shared.push(&[1.0, 2.0, 3.0]);
        shared.push(&[4.0, 5.0, 6.0]);
        let kept: Vec<f32> = shared.preroll.iter().copied().collect();
        assert_eq!(kept, vec![3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn recording_captures_everything_without_bound() {
        let mut shared = Shared { preroll_capacity: 2, recording: Some(Vec::new()), ..Shared::default() };
        for _ in 0..100 {
            shared.push(&[0.5; 16]);
        }
        assert_eq!(shared.recording.as_ref().unwrap().len(), 1_600);
        assert!(shared.preroll.is_empty(), "recording must not also fill the pre-roll");
    }

    #[test]
    fn the_level_meter_attacks_fast_and_releases_slowly() {
        let mut shared = Shared { preroll_capacity: 16, ..Shared::default() };
        shared.push(&[1.0; 8]);
        let peak = shared.level;
        assert!((peak - 1.0).abs() < 1e-6);

        shared.push(&[0.0; 8]);
        assert!(shared.level < peak, "meter must fall");
        assert!(shared.level > 0.5, "meter must not snap to zero: {}", shared.level);
    }

    #[test]
    fn a_zero_length_callback_does_not_divide_by_zero() {
        let mut shared = Shared { preroll_capacity: 4, ..Shared::default() };
        shared.push(&[]);
        assert!(shared.level.is_finite());
    }
}
