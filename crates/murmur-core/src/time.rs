use serde::{Deserialize, Serialize};
use std::fmt;
use std::ops::{Add, AddAssign, Sub};

/// A monotonic timestamp or duration in milliseconds.
///
/// The core is IO-free, so it never reads a clock: every timestamp arrives on an
/// [`Event`](crate::Event). That is what makes a whole dictation session replayable
/// from a fixture, and what lets the simulator run an hour of use in a millisecond.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct Millis(pub u64);

impl Millis {
    pub const ZERO: Self = Self(0);

    #[must_use]
    pub const fn from_secs(secs: u64) -> Self {
        Self(secs * 1000)
    }

    /// Elapsed time from `earlier` to `self`, saturating at zero.
    ///
    /// Saturation rather than panic is deliberate: clocks handed to us by the
    /// daemon come from separate threads and may arrive marginally out of order.
    #[must_use]
    pub const fn since(self, earlier: Self) -> Self {
        Self(self.0.saturating_sub(earlier.0))
    }

    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn as_secs_f32(self) -> f32 {
        self.0 as f32 / 1000.0
    }
}

impl Add for Millis {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self(self.0.saturating_add(rhs.0))
    }
}

impl AddAssign for Millis {
    fn add_assign(&mut self, rhs: Self) {
        self.0 = self.0.saturating_add(rhs.0);
    }
}

impl Sub for Millis {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        self.since(rhs)
    }
}

impl fmt::Display for Millis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0 >= 10_000 {
            write!(f, "{:.1}s", self.as_secs_f32())
        } else {
            write!(f, "{}ms", self.0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn since_saturates_on_reordered_clocks() {
        assert_eq!(Millis(5).since(Millis(9)), Millis::ZERO);
    }

    #[test]
    fn display_switches_unit_at_ten_seconds() {
        assert_eq!(Millis(450).to_string(), "450ms");
        assert_eq!(Millis(9_999).to_string(), "9999ms");
        assert_eq!(Millis(12_500).to_string(), "12.5s");
    }
}
