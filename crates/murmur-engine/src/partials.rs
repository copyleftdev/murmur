//! Live text while the user is still speaking, from the batch transcriber.
//!
//! Streaming models exist for this, but the one available cannot flush its final
//! chunk (see `murmur-asr/tests/streaming_accuracy.rs`), which loses the last
//! word of a dictation — the worst possible place. Batch transcription on a GPU
//! is fast enough to simply be repeated: 11 seconds of audio in 36 ms means
//! re-transcribing the whole recording every few hundred milliseconds costs a
//! small fraction of one GPU. The same model produces the partials and the final
//! text, so what the user watches appear is what they end up with.
//!
//! The one hard rule is that this must never delay the release. Inference
//! therefore happens on a worker thread, and the main loop only ever hands over
//! audio and picks up text.

use murmur_asr::Transcriber;
use murmur_core::UtteranceId;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender, TryRecvError, channel};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Shared so the final pass and the partial pass use one loaded model.
pub type SharedTranscriber = Arc<Mutex<Box<dyn Transcriber>>>;

/// Never re-transcribe more often than this, however fast the GPU is.
const MIN_INTERVAL: Duration = Duration::from_millis(300);

/// Nor less often than this, however slow it is.
const MAX_INTERVAL: Duration = Duration::from_secs(2);

/// Target duty cycle: a pass taking `d` earns a gap of `DUTY * d`.
///
/// Keeps the cost proportional rather than fixed, so a long dictation on a slow
/// machine backs off by itself instead of saturating the device it shares with
/// the final pass.
const DUTY: u32 = 5;

struct Request {
    id: UtteranceId,
    samples: Vec<f32>,
}

#[derive(Debug)]
pub struct Reply {
    pub id: UtteranceId,
    pub text: String,
    pub took: Duration,
}

/// A background transcriber that produces partial text on request.
pub struct Partials {
    requests: Sender<Request>,
    replies: Receiver<Reply>,
    busy: Arc<AtomicBool>,
    interval: Duration,
    last_request: Option<Instant>,
}

impl std::fmt::Debug for Partials {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Partials")
            .field("interval", &self.interval)
            .finish_non_exhaustive()
    }
}

impl Partials {
    /// Start a worker sharing `transcriber` with its caller.
    #[must_use]
    pub fn spawn(transcriber: SharedTranscriber) -> Self {
        let (requests, inbox) = channel::<Request>();
        let (outbox, replies) = channel::<Reply>();
        let busy = Arc::new(AtomicBool::new(false));
        let worker_busy = Arc::clone(&busy);

        std::thread::Builder::new()
            .name("murmur-partials".into())
            .spawn(move || {
                while let Ok(request) = inbox.recv() {
                    worker_busy.store(true, Ordering::SeqCst);
                    let started = Instant::now();
                    let text = transcriber
                        .lock()
                        .ok()
                        .and_then(|mut model| model.transcribe(&request.samples).ok())
                        .map(|transcript| transcript.text);
                    worker_busy.store(false, Ordering::SeqCst);

                    // A failed partial is not worth reporting: the final pass will
                    // fail too, and report properly.
                    if let Some(text) = text
                        && outbox
                            .send(Reply {
                                id: request.id,
                                text,
                                took: started.elapsed(),
                            })
                            .is_err()
                    {
                        return;
                    }
                }
            })
            .ok();

        Self {
            requests,
            replies,
            busy,
            interval: MIN_INTERVAL,
            last_request: None,
        }
    }

    /// Forget any pacing state, at the start of an utterance.
    pub fn reset(&mut self) {
        self.last_request = None;
    }

    /// Offer audio to the worker, if it is idle and enough time has passed.
    ///
    /// Dropping the request rather than queueing it is deliberate: a queued
    /// snapshot is stale by the time it is transcribed, and the next one is
    /// always better.
    pub fn offer(&mut self, id: UtteranceId, samples: Vec<f32>, now: Instant) {
        if samples.is_empty() || self.busy.load(Ordering::SeqCst) {
            return;
        }
        if self
            .last_request
            .is_some_and(|last| now.duration_since(last) < self.interval)
        {
            return;
        }
        self.last_request = Some(now);
        let _ = self.requests.send(Request { id, samples });
    }

    /// Collect whatever the worker has finished, adjusting the pace to suit.
    pub fn collect(&mut self) -> Vec<Reply> {
        let mut replies = Vec::new();
        loop {
            match self.replies.try_recv() {
                Ok(reply) => {
                    self.interval = (reply.took * DUTY).clamp(MIN_INTERVAL, MAX_INTERVAL);
                    replies.push(reply);
                }
                Err(TryRecvError::Empty | TryRecvError::Disconnected) => return replies,
            }
        }
    }

    #[must_use]
    pub fn interval(&self) -> Duration {
        self.interval
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use murmur_asr::Mock;

    fn shared() -> SharedTranscriber {
        Arc::new(Mutex::new(
            Box::new(Mock::new(["live text"])) as Box<dyn Transcriber>
        ))
    }

    fn speech() -> Vec<f32> {
        (0..16_000).map(|i| (i as f32 / 40.0).sin() * 0.3).collect()
    }

    fn wait_for_reply(partials: &mut Partials) -> Option<Reply> {
        for _ in 0..200 {
            if let Some(reply) = partials.collect().pop() {
                return Some(reply);
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        None
    }

    #[test]
    fn audio_offered_comes_back_as_text() {
        let mut partials = Partials::spawn(shared());
        partials.offer(1, speech(), Instant::now());
        let reply = wait_for_reply(&mut partials).expect("a reply");
        assert_eq!(reply.text, "live text");
        assert_eq!(reply.id, 1);
    }

    #[test]
    fn offers_inside_the_interval_are_dropped_rather_than_queued() {
        let mut partials = Partials::spawn(shared());
        let now = Instant::now();
        partials.offer(1, speech(), now);
        wait_for_reply(&mut partials).expect("first reply");

        // Immediately after: too soon, so nothing new should be produced.
        partials.offer(1, speech(), now);
        std::thread::sleep(Duration::from_millis(80));
        assert!(
            partials.collect().is_empty(),
            "a stale snapshot was queued anyway"
        );
    }

    #[test]
    fn empty_audio_is_never_sent_to_the_model() {
        let mut partials = Partials::spawn(shared());
        partials.offer(1, Vec::new(), Instant::now());
        std::thread::sleep(Duration::from_millis(50));
        assert!(partials.collect().is_empty());
    }

    #[test]
    fn the_pace_backs_off_in_proportion_to_how_long_a_pass_takes() {
        let slow: SharedTranscriber = Arc::new(Mutex::new(Box::new(
            Mock::new(["slow"]).with_delay(Duration::from_millis(120)),
        )));
        let mut partials = Partials::spawn(slow);
        assert_eq!(partials.interval(), MIN_INTERVAL);

        partials.offer(1, speech(), Instant::now());
        wait_for_reply(&mut partials).expect("a reply");

        assert!(
            partials.interval() > MIN_INTERVAL,
            "a 120ms pass should earn a longer gap, got {:?}",
            partials.interval()
        );
        assert!(partials.interval() <= MAX_INTERVAL);
    }

    #[test]
    fn the_pace_never_leaves_the_bounds_however_extreme_the_timing() {
        let mut partials = Partials::spawn(shared());
        for _ in 0..3 {
            partials.offer(1, speech(), Instant::now() + Duration::from_secs(60));
            wait_for_reply(&mut partials);
            assert!(partials.interval() >= MIN_INTERVAL);
            assert!(partials.interval() <= MAX_INTERVAL);
        }
    }

    #[test]
    fn a_failing_transcriber_produces_no_partials_and_does_not_wedge_the_worker() {
        let failing: SharedTranscriber = Arc::new(Mutex::new(Box::new(Mock::failing())));
        let mut partials = Partials::spawn(failing);
        partials.offer(1, speech(), Instant::now());
        std::thread::sleep(Duration::from_millis(100));
        assert!(partials.collect().is_empty());
    }
}
