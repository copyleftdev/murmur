use crate::{AsrError, Transcriber, Transcript};
use std::time::{Duration, Instant};

/// A transcriber that returns scripted text after a scripted delay.
///
/// Its purpose is not to fake speech recognition but to make the *rest* of the
/// system falsifiable: with a fixed delay we can assert the release-to-text
/// budget, and with a scripted failure we can prove the session recovers instead
/// of wedging in `Finalizing` forever.
#[derive(Debug, Clone)]
pub struct Mock {
    lines: Vec<String>,
    next: usize,
    delay: Duration,
    fail: bool,
}

impl Default for Mock {
    fn default() -> Self {
        Self::new(["This is Murmur, typing what you say."])
    }
}

impl Mock {
    pub fn new<I, S>(lines: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            lines: lines.into_iter().map(Into::into).collect(),
            next: 0,
            delay: Duration::ZERO,
            fail: false,
        }
    }

    /// Pretend inference takes this long, to exercise the latency budget.
    #[must_use]
    pub fn with_delay(mut self, delay: Duration) -> Self {
        self.delay = delay;
        self
    }

    /// Fail every call, to exercise the recovery path.
    #[must_use]
    pub fn failing() -> Self {
        Self { fail: true, ..Self::new(Vec::<String>::new()) }
    }
}

impl Transcriber for Mock {
    fn name(&self) -> String {
        "mock".to_owned()
    }

    fn transcribe(&mut self, samples: &[f32]) -> Result<Transcript, AsrError> {
        let started = Instant::now();
        if !self.delay.is_zero() {
            std::thread::sleep(self.delay);
        }
        if self.fail {
            return Err(AsrError::Inference("mock transcriber is configured to fail".into()));
        }
        // Silence transcribes to nothing, exactly as a real engine would, so the
        // "empty transcript injects nothing" path is exercised by accident too.
        if samples.iter().all(|s| s.abs() < 1e-6) {
            return Ok(Transcript::new(String::new(), started.elapsed()));
        }
        let text = if self.lines.is_empty() {
            String::new()
        } else {
            let line = self.lines[self.next % self.lines.len()].clone();
            self.next += 1;
            line
        };
        Ok(Transcript::new(text, started.elapsed()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn speech() -> Vec<f32> {
        (0..16_000).map(|i| (i as f32 / 40.0).sin() * 0.3).collect()
    }

    #[test]
    fn scripted_lines_are_returned_in_order_and_then_repeat() {
        let mut mock = Mock::new(["first", "second"]);
        assert_eq!(mock.transcribe(&speech()).unwrap().text, "first");
        assert_eq!(mock.transcribe(&speech()).unwrap().text, "second");
        assert_eq!(mock.transcribe(&speech()).unwrap().text, "first");
    }

    #[test]
    fn silence_transcribes_to_nothing() {
        let mut mock = Mock::default();
        assert_eq!(mock.transcribe(&[0.0; 16_000]).unwrap().text, "");
    }

    #[test]
    fn a_failing_transcriber_reports_rather_than_returning_empty_text() {
        assert!(Mock::failing().transcribe(&speech()).is_err());
    }

    #[test]
    fn the_scripted_delay_is_actually_spent() {
        let mut mock = Mock::default().with_delay(Duration::from_millis(30));
        let transcript = mock.transcribe(&speech()).unwrap();
        assert!(transcript.elapsed >= Duration::from_millis(30), "{:?}", transcript.elapsed);
    }

    #[test]
    fn the_realtime_factor_relates_audio_to_compute() {
        let transcript = Transcript::new("x", Duration::from_millis(100));
        assert!((transcript.realtime_factor(Duration::from_secs(1)) - 10.0).abs() < 1e-3);
    }

    #[test]
    fn an_instant_transcription_does_not_divide_by_zero() {
        let transcript = Transcript::new("x", Duration::ZERO);
        assert!(transcript.realtime_factor(Duration::from_secs(1)).is_infinite());
    }
}
