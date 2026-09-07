//! AXI-like stream handshake for module ports (TRANSITION §4.5).
//!
//! Mirrors `fpga-fabric/src/stream.rs` exactly: the producer sets `valid` when
//! it has data, the consumer sets `ready` when it can accept, and a transfer
//! happens on cycles where [`Stream::fire`] is true.

/// A valid/ready stream beat.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stream<T> {
    pub data: T,
    pub valid: bool,
    pub ready: bool,
}

impl<T> Stream<T> {
    /// Create a stream beat with both handshake flags low.
    pub fn new(data: T) -> Self {
        Self {
            data,
            valid: false,
            ready: false,
        }
    }

    /// A transfer occurs iff both `valid` and `ready` are asserted.
    pub fn fire(&self) -> bool {
        self.valid && self.ready
    }

    /// Producer stalled: it has data but the consumer is not ready.
    pub fn stalled(&self) -> bool {
        self.valid && !self.ready
    }

    /// Map the data payload, preserving the handshake flags.
    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> Stream<U> {
        Stream {
            data: f(self.data),
            valid: self.valid,
            ready: self.ready,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transfer_only_when_both_asserted() {
        let s = Stream {
            data: 7u32,
            valid: true,
            ready: false,
        };
        assert!(!s.fire());
        assert!(s.stalled());

        let s = Stream {
            data: 7u32,
            valid: false,
            ready: true,
        };
        assert!(!s.fire());
        assert!(!s.stalled());

        let s = Stream {
            data: 7u32,
            valid: true,
            ready: true,
        };
        assert!(s.fire());
    }

    #[test]
    fn producer_stalls_while_consumer_not_ready() {
        // Producer only advances to the next value when a beat fires —
        // modelling backpressure: it stalls while ready is deasserted.
        let mut next_value = 0u32;
        let mut consumed = Vec::new();

        for cycle in 0..5u32 {
            let consumer_ready = cycle >= 3;
            let beat = Stream {
                data: next_value,
                valid: true,
                ready: consumer_ready,
            };
            if beat.fire() {
                consumed.push(beat.data);
                next_value += 1;
            }
        }

        assert_eq!(consumed, vec![0, 1], "transfers only in cycles 3 and 4");
        assert_eq!(next_value, 2, "producer stalled for the first 3 cycles");
    }

    #[test]
    fn map_preserves_handshake_flags() {
        let s = Stream {
            data: 3u32,
            valid: true,
            ready: false,
        };
        let m = s.map(|d| d * 2);
        assert_eq!(m.data, 6);
        assert!(m.valid);
        assert!(!m.ready);
    }
}
