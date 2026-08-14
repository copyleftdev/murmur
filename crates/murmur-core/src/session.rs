use crate::text::{EmitContext, Formatter};
use crate::time::Millis;
use serde::{Deserialize, Serialize};

pub type UtteranceId = u64;

/// How the current capture will be ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Mode {
    /// Push-to-talk: the capture ends when the key is released.
    Hold,
    /// Hands-free: entered by double-tapping, ended by the next press.
    Locked,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Phase {
    #[default]
    Idle,
    Capturing {
        id: UtteranceId,
        mode: Mode,
    },
    Finalizing {
        id: UtteranceId,
    },
    Injecting {
        id: UtteranceId,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    Capture,
    Transcribe,
    Inject,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Event {
    TriggerDown(Millis),
    TriggerUp(Millis),
    /// A streaming hypothesis, shown in the HUD but never injected.
    Partial {
        at: Millis,
        id: UtteranceId,
        text: String,
    },
    Final {
        at: Millis,
        id: UtteranceId,
        text: String,
    },
    Injected {
        at: Millis,
        id: UtteranceId,
    },
    Failed {
        at: Millis,
        id: UtteranceId,
        stage: Stage,
        message: String,
    },
    /// Abandon the utterance in flight without emitting anything.
    Cancel(Millis),
    Tick(Millis),
}

#[derive(Clone, Debug, PartialEq)]
pub enum Command {
    StartCapture { id: UtteranceId },
    StopCapture { id: UtteranceId },
    Discard { id: UtteranceId },
    Inject { id: UtteranceId, text: String },
    Hud(Hud),
}

#[derive(Clone, Debug, PartialEq)]
pub enum Hud {
    Hidden,
    Listening { mode: Mode },
    Partial { text: String },
    Thinking,
    Error { message: String },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct Tuning {
    /// A press shorter than this emits nothing; it only arms the double-tap.
    pub tap_max: Millis,
    pub double_tap_window: Millis,
    /// Hard cap on a single capture, so a stuck key cannot record forever.
    pub max_utterance: Millis,
    /// How long to wait for a transcript before giving up on the utterance.
    pub finalize_timeout: Millis,
    /// Utterances closer together than this are treated as one flowing dictation.
    pub continuation_window: Millis,
}

impl Default for Tuning {
    fn default() -> Self {
        Self {
            tap_max: Millis(220),
            double_tap_window: Millis(400),
            max_utterance: Millis::from_secs(120),
            finalize_timeout: Millis::from_secs(10),
            continuation_window: Millis::from_secs(8),
        }
    }
}

/// Where the time went for one utterance.
///
/// `release_to_text` is the number this product lives or dies by: the gap the
/// user actually perceives between letting go of the key and seeing their words.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Latency {
    pub speaking: Millis,
    pub transcribe: Millis,
    pub inject: Millis,
    pub release_to_text: Millis,
}

#[derive(Clone, Copy, Debug, Default)]
struct Marks {
    down: Millis,
    up: Millis,
    transcript: Millis,
}

/// The dictation state machine.
///
/// Deliberately free of IO, threads and clocks: every input is an [`Event`]
/// carrying its own timestamp, and every output is a [`Command`] for the daemon
/// to carry out. A whole session is therefore a pure function of its event log,
/// which is what makes the simulator possible.
#[derive(Clone, Debug, Default)]
pub struct Session {
    tuning: Tuning,
    formatter: Formatter,
    phase: Phase,
    next_id: UtteranceId,
    marks: Marks,
    last_tap: Option<Millis>,
    last_emit: Option<Millis>,
    last_latency: Option<Latency>,
}

impl Session {
    #[must_use]
    pub fn new(tuning: Tuning, formatter: Formatter) -> Self {
        Self {
            tuning,
            formatter,
            ..Self::default()
        }
    }

    #[must_use]
    pub fn phase(&self) -> Phase {
        self.phase
    }

    #[must_use]
    pub fn is_capturing(&self) -> bool {
        matches!(self.phase, Phase::Capturing { .. })
    }

    /// Timing of the most recently completed utterance, if any.
    #[must_use]
    pub fn last_latency(&self) -> Option<Latency> {
        self.last_latency
    }

    pub fn handle(&mut self, event: Event) -> Vec<Command> {
        match event {
            Event::TriggerDown(at) => self.on_down(at),
            Event::TriggerUp(at) => self.on_up(at),
            Event::Partial { id, text, .. } => self.on_partial(id, &text),
            Event::Final { at, id, text } => self.on_final(at, id, &text),
            Event::Injected { at, id } => self.on_injected(at, id),
            Event::Failed {
                id, stage, message, ..
            } => self.on_failed(id, stage, &message),
            Event::Cancel(_) => self.on_cancel(),
            Event::Tick(at) => self.on_tick(at),
        }
    }

    fn on_down(&mut self, at: Millis) -> Vec<Command> {
        match self.phase {
            Phase::Idle => {
                let double_tapped = self
                    .last_tap
                    .is_some_and(|tap| at.since(tap) <= self.tuning.double_tap_window);
                let mode = if double_tapped {
                    Mode::Locked
                } else {
                    Mode::Hold
                };
                self.last_tap = None;
                self.start_capture(at, mode)
            }
            // In hands-free mode the next press is the stop signal.
            Phase::Capturing {
                id,
                mode: Mode::Locked,
            } => self.stop_capture(at, id),
            _ => Vec::new(),
        }
    }

    fn on_up(&mut self, at: Millis) -> Vec<Command> {
        let Phase::Capturing {
            id,
            mode: Mode::Hold,
        } = self.phase
        else {
            return Vec::new();
        };
        if at.since(self.marks.down) <= self.tuning.tap_max {
            // Too short to be speech. Arm the double-tap instead of emitting noise.
            self.last_tap = Some(at);
            self.phase = Phase::Idle;
            return vec![Command::Discard { id }, Command::Hud(Hud::Hidden)];
        }
        self.stop_capture(at, id)
    }

    fn start_capture(&mut self, at: Millis, mode: Mode) -> Vec<Command> {
        self.next_id += 1;
        let id = self.next_id;
        self.phase = Phase::Capturing { id, mode };
        self.marks = Marks {
            down: at,
            ..Marks::default()
        };
        vec![
            Command::StartCapture { id },
            Command::Hud(Hud::Listening { mode }),
        ]
    }

    fn stop_capture(&mut self, at: Millis, id: UtteranceId) -> Vec<Command> {
        self.marks.up = at;
        self.phase = Phase::Finalizing { id };
        vec![Command::StopCapture { id }, Command::Hud(Hud::Thinking)]
    }

    /// Live text is only ever shown *while capturing*.
    ///
    /// A partial pass started before the key came up can finish after it, and
    /// showing it then repaints a stale guess over "transcribing…" — or worse,
    /// over the final text. The utterance id is not enough to catch this,
    /// because it is still the current utterance; the phase is what matters.
    fn on_partial(&mut self, id: UtteranceId, text: &str) -> Vec<Command> {
        let capturing = matches!(self.phase, Phase::Capturing { id: current, .. } if current == id);
        if capturing && !text.is_empty() {
            vec![Command::Hud(Hud::Partial {
                text: text.to_owned(),
            })]
        } else {
            Vec::new()
        }
    }

    fn on_final(&mut self, at: Millis, id: UtteranceId, text: &str) -> Vec<Command> {
        if self.phase != (Phase::Finalizing { id }) {
            return Vec::new();
        }
        self.marks.transcript = at;
        let continuation = self
            .last_emit
            .is_some_and(|emit| at.since(emit) <= self.tuning.continuation_window);
        if let Some(text) = self.formatter.format(text, EmitContext { continuation }) {
            self.phase = Phase::Injecting { id };
            vec![Command::Inject { id, text }]
        } else {
            self.phase = Phase::Idle;
            vec![Command::Hud(Hud::Hidden)]
        }
    }

    fn on_injected(&mut self, at: Millis, id: UtteranceId) -> Vec<Command> {
        if self.phase != (Phase::Injecting { id }) {
            return Vec::new();
        }
        self.phase = Phase::Idle;
        self.last_emit = Some(at);
        self.last_latency = Some(Latency {
            speaking: self.marks.up.since(self.marks.down),
            transcribe: self.marks.transcript.since(self.marks.up),
            inject: at.since(self.marks.transcript),
            release_to_text: at.since(self.marks.up),
        });
        vec![Command::Hud(Hud::Hidden)]
    }

    fn on_failed(&mut self, id: UtteranceId, stage: Stage, message: &str) -> Vec<Command> {
        if self.current_id() != Some(id) {
            return Vec::new();
        }
        self.phase = Phase::Idle;
        vec![
            Command::Discard { id },
            Command::Hud(Hud::Error {
                message: format!("{stage:?} failed: {message}"),
            }),
        ]
    }

    fn on_cancel(&mut self) -> Vec<Command> {
        match self.current_id() {
            Some(id) => {
                self.phase = Phase::Idle;
                vec![Command::Discard { id }, Command::Hud(Hud::Hidden)]
            }
            None => Vec::new(),
        }
    }

    fn on_tick(&mut self, at: Millis) -> Vec<Command> {
        match self.phase {
            Phase::Capturing { id, .. }
                if at.since(self.marks.down) >= self.tuning.max_utterance =>
            {
                self.stop_capture(at, id)
            }
            Phase::Finalizing { id } if at.since(self.marks.up) >= self.tuning.finalize_timeout => {
                self.phase = Phase::Idle;
                vec![
                    Command::Discard { id },
                    Command::Hud(Hud::Error {
                        message: "transcription timed out".into(),
                    }),
                ]
            }
            _ => Vec::new(),
        }
    }

    fn current_id(&self) -> Option<UtteranceId> {
        match self.phase {
            Phase::Idle => None,
            Phase::Capturing { id, .. } | Phase::Finalizing { id } | Phase::Injecting { id } => {
                Some(id)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn session() -> Session {
        Session::default()
    }

    /// Hold from `down` to `up`, transcribe at `+40ms`, inject at `+30ms`.
    fn dictate(s: &mut Session, down: u64, up: u64, text: &str) -> Vec<Command> {
        s.handle(Event::TriggerDown(Millis(down)));
        s.handle(Event::TriggerUp(Millis(up)));
        let Phase::Finalizing { id } = s.phase() else {
            panic!("expected Finalizing, got {:?}", s.phase());
        };
        let out = s.handle(Event::Final {
            at: Millis(up + 40),
            id,
            text: text.to_owned(),
        });
        s.handle(Event::Injected {
            at: Millis(up + 70),
            id,
        });
        out
    }

    fn injected(commands: &[Command]) -> Option<&str> {
        commands.iter().find_map(|c| match c {
            Command::Inject { text, .. } => Some(text.as_str()),
            _ => None,
        })
    }

    #[test]
    fn a_hold_produces_one_injection_and_returns_to_idle() {
        let mut s = session();
        let out = dictate(&mut s, 0, 1_500, "Hello world.");
        assert_eq!(injected(&out), Some("Hello world. "));
        assert_eq!(s.phase(), Phase::Idle);
    }

    #[test]
    fn a_tap_emits_nothing() {
        let mut s = session();
        s.handle(Event::TriggerDown(Millis(0)));
        let out = s.handle(Event::TriggerUp(Millis(100)));
        assert_eq!(
            out,
            vec![Command::Discard { id: 1 }, Command::Hud(Hud::Hidden)]
        );
        assert_eq!(s.phase(), Phase::Idle);
    }

    #[test]
    fn a_double_tap_enters_hands_free_and_the_next_press_ends_it() {
        let mut s = session();
        s.handle(Event::TriggerDown(Millis(0)));
        s.handle(Event::TriggerUp(Millis(80)));

        let out = s.handle(Event::TriggerDown(Millis(200)));
        assert!(out.contains(&Command::Hud(Hud::Listening { mode: Mode::Locked })));

        // Releasing the second tap must not end a hands-free capture.
        assert!(s.handle(Event::TriggerUp(Millis(260))).is_empty());
        assert!(s.is_capturing());

        let out = s.handle(Event::TriggerDown(Millis(9_000)));
        assert_eq!(out[0], Command::StopCapture { id: 2 });
    }

    #[test]
    fn a_slow_second_tap_is_an_ordinary_hold() {
        let mut s = session();
        s.handle(Event::TriggerDown(Millis(0)));
        s.handle(Event::TriggerUp(Millis(80)));
        let out = s.handle(Event::TriggerDown(Millis(5_000)));
        assert!(out.contains(&Command::Hud(Hud::Listening { mode: Mode::Hold })));
    }

    #[test]
    fn back_to_back_utterances_are_spaced_as_a_continuation() {
        let mut s = session();
        dictate(&mut s, 0, 1_000, "First sentence.");
        let out = dictate(&mut s, 2_000, 3_000, "Second sentence.");
        assert_eq!(injected(&out), Some(" Second sentence. "));
    }

    #[test]
    fn a_distant_utterance_is_not_a_continuation() {
        let mut s = session();
        dictate(&mut s, 0, 1_000, "First sentence.");
        let out = dictate(&mut s, 60_000, 61_000, "Much later.");
        assert_eq!(injected(&out), Some("Much later. "));
    }

    #[test]
    fn an_empty_transcript_injects_nothing() {
        let mut s = session();
        s.handle(Event::TriggerDown(Millis(0)));
        s.handle(Event::TriggerUp(Millis(1_000)));
        let out = s.handle(Event::Final {
            at: Millis(1_040),
            id: 1,
            text: String::new(),
        });
        assert_eq!(out, vec![Command::Hud(Hud::Hidden)]);
        assert_eq!(s.phase(), Phase::Idle);
    }

    #[test]
    fn live_text_is_shown_while_capturing() {
        let mut s = session();
        s.handle(Event::TriggerDown(Millis(0)));
        let out = s.handle(Event::Partial {
            at: Millis(300),
            id: 1,
            text: "hello".into(),
        });
        assert_eq!(
            out,
            vec![Command::Hud(Hud::Partial {
                text: "hello".into()
            })]
        );
    }

    #[test]
    fn live_text_arriving_after_the_key_is_released_is_discarded() {
        let mut s = session();
        s.handle(Event::TriggerDown(Millis(0)));
        s.handle(Event::TriggerUp(Millis(1_000)));

        // Same utterance, but the user has stopped talking: a partial that was
        // in flight would otherwise repaint a half-finished guess over the
        // "transcribing" state, and then over the final text.
        let out = s.handle(Event::Partial {
            at: Millis(1_010),
            id: 1,
            text: "half a gue".into(),
        });
        assert!(out.is_empty(), "a stale partial was shown: {out:?}");
    }

    #[test]
    fn live_text_arriving_during_injection_is_discarded() {
        let mut s = session();
        s.handle(Event::TriggerDown(Millis(0)));
        s.handle(Event::TriggerUp(Millis(1_000)));
        s.handle(Event::Final {
            at: Millis(1_040),
            id: 1,
            text: "done".into(),
        });

        let out = s.handle(Event::Partial {
            at: Millis(1_050),
            id: 1,
            text: "don".into(),
        });
        assert!(
            out.is_empty(),
            "a partial overwrote the final text: {out:?}"
        );
    }

    #[test]
    fn a_stale_transcript_from_a_cancelled_utterance_is_ignored() {
        let mut s = session();
        s.handle(Event::TriggerDown(Millis(0)));
        s.handle(Event::TriggerUp(Millis(1_000)));
        s.handle(Event::Cancel(Millis(1_010)));
        let out = s.handle(Event::Final {
            at: Millis(1_050),
            id: 1,
            text: "ghost".into(),
        });
        assert!(
            out.is_empty(),
            "cancelled utterance still injected: {out:?}"
        );
    }

    #[test]
    fn a_stuck_key_stops_at_the_utterance_cap() {
        let mut s = session();
        s.handle(Event::TriggerDown(Millis(0)));
        assert!(s.handle(Event::Tick(Millis(60_000))).is_empty());
        let out = s.handle(Event::Tick(Millis(120_000)));
        assert_eq!(out[0], Command::StopCapture { id: 1 });
    }

    #[test]
    fn a_hung_transcriber_releases_the_session() {
        let mut s = session();
        s.handle(Event::TriggerDown(Millis(0)));
        s.handle(Event::TriggerUp(Millis(1_000)));
        let out = s.handle(Event::Tick(Millis(11_001)));
        assert_eq!(out[0], Command::Discard { id: 1 });
        assert_eq!(s.phase(), Phase::Idle);
    }

    #[test]
    fn latency_is_attributed_to_the_right_stage() {
        let mut s = session();
        dictate(&mut s, 100, 1_600, "Timed.");
        let l = s.last_latency().expect("latency recorded");
        assert_eq!(l.speaking, Millis(1_500));
        assert_eq!(l.transcribe, Millis(40));
        assert_eq!(l.inject, Millis(30));
        assert_eq!(l.release_to_text, Millis(70));
    }

    #[test]
    fn every_utterance_id_is_unique_across_a_long_session() {
        let mut s = session();
        let mut ids = Vec::new();
        for i in 0..50 {
            let base = i * 10_000;
            s.handle(Event::TriggerDown(Millis(base)));
            s.handle(Event::TriggerUp(Millis(base + 1_000)));
            let Phase::Finalizing { id } = s.phase() else {
                panic!()
            };
            ids.push(id);
            s.handle(Event::Final {
                at: Millis(base + 1_040),
                id,
                text: "x".into(),
            });
            s.handle(Event::Injected {
                at: Millis(base + 1_070),
                id,
            });
        }
        ids.dedup();
        assert_eq!(ids.len(), 50);
    }

    #[test]
    fn events_arriving_in_the_wrong_phase_never_panic_or_emit() {
        let mut s = session();
        let noise = [
            Event::TriggerUp(Millis(1)),
            Event::Injected {
                at: Millis(2),
                id: 7,
            },
            Event::Final {
                at: Millis(3),
                id: 7,
                text: "x".into(),
            },
            Event::Partial {
                at: Millis(4),
                id: 7,
                text: "x".into(),
            },
            Event::Cancel(Millis(5)),
            Event::Tick(Millis(6)),
        ];
        for event in noise {
            assert!(
                s.handle(event.clone()).is_empty(),
                "{event:?} leaked a command"
            );
        }
        assert_eq!(s.phase(), Phase::Idle);
    }
}
