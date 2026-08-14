use std::time::Duration;

/// Audio handed to a transcriber: mono, 16 kHz.
#[derive(Debug, Clone, Default)]
pub struct Recorded {
    pub samples: Vec<f32>,
    pub trimmed: usize,
    pub duration: Duration,
}

impl Recorded {
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}

impl From<murmur_audio::Capture> for Recorded {
    fn from(capture: murmur_audio::Capture) -> Self {
        Self {
            samples: capture.samples,
            trimmed: capture.trimmed,
            duration: capture.duration,
        }
    }
}

/// Something that can be recorded from.
///
/// The engine is defined against this rather than against a microphone so the
/// whole dictation loop can be driven from fixtures. Without it, every test of
/// the loop needs a sound card, a person willing to talk into it, and a window
/// that does not mind being typed into — which in practice means no tests.
pub trait Recorder {
    fn begin(&mut self);
    /// # Errors
    /// Returns a human-readable reason the recording could not be produced.
    fn finish(&mut self) -> Result<Recorded, String>;
    fn discard(&mut self);
    fn level(&self) -> f32 {
        0.0
    }
    /// Audio captured so far, without ending the recording.
    ///
    /// `None` means partials are unavailable, which is not an error: a recorder
    /// that cannot be read mid-capture simply produces no live text.
    fn snapshot(&self) -> Option<Recorded> {
        None
    }
}

impl Recorder for murmur_audio::Microphone {
    fn begin(&mut self) {
        Self::begin(self);
    }

    fn finish(&mut self) -> Result<Recorded, String> {
        Self::finish(self)
            .map(Into::into)
            .map_err(|e| e.to_string())
    }

    fn discard(&mut self) {
        Self::discard(self);
    }

    fn level(&self) -> f32 {
        Self::level(self)
    }

    fn snapshot(&self) -> Option<Recorded> {
        Self::snapshot(self).ok().flatten().map(Into::into)
    }
}

/// A recorder that returns fixed audio, for driving the loop in tests.
#[derive(Debug, Clone, Default)]
pub struct Fixture {
    audio: Vec<f32>,
    recording: bool,
    pub begins: usize,
    pub discards: usize,
    fail: Option<String>,
}

impl Fixture {
    /// One second of a 16 kHz tone, which is enough for the mock transcriber to
    /// treat as speech rather than silence.
    #[must_use]
    pub fn speaking() -> Self {
        Self {
            audio: (0..16_000).map(|i| (i as f32 / 40.0).sin() * 0.3).collect(),
            ..Self::default()
        }
    }

    #[must_use]
    pub fn silent() -> Self {
        Self {
            audio: vec![0.0; 16_000],
            ..Self::default()
        }
    }

    #[must_use]
    pub fn failing(reason: impl Into<String>) -> Self {
        Self {
            fail: Some(reason.into()),
            ..Self::speaking()
        }
    }

    #[must_use]
    pub fn is_recording(&self) -> bool {
        self.recording
    }
}

impl Recorder for Fixture {
    fn snapshot(&self) -> Option<Recorded> {
        self.recording.then(|| Recorded {
            samples: self.audio.clone(),
            trimmed: 0,
            duration: Duration::from_secs_f32(self.audio.len() as f32 / 16_000.0),
        })
    }

    fn begin(&mut self) {
        self.begins += 1;
        self.recording = true;
    }

    fn finish(&mut self) -> Result<Recorded, String> {
        self.recording = false;
        if let Some(reason) = &self.fail {
            return Err(reason.clone());
        }
        Ok(Recorded {
            samples: self.audio.clone(),
            trimmed: 0,
            duration: Duration::from_secs_f32(self.audio.len() as f32 / 16_000.0),
        })
    }

    fn discard(&mut self) {
        self.discards += 1;
        self.recording = false;
    }
}
