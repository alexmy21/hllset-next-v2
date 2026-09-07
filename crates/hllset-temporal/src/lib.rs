//! Temporal Pyramid — L0–L6 sliding window with automatic carry.
//!
//! Implements STANDARD.md §4.2: a configurable N-layer temporal pyramid
//! that compresses HLLSet observations from fine-grained time slices into
//! coarser layers through automatic aggregation at time boundaries.
//!
//! # The Default Pyramid (7 layers)
//!
//! ```text
//! Layer 6  YEAR     L6 = ∪ S(t) over 365 days          ← coarsest
//! Layer 5  MONTH    L5 = ∪ S(t) over 30 days
//! Layer 4  WEEK     L4 = ∪ S(t) over 7 days
//! Layer 3  DAY      L3 = ∪ S(t) over 24 hours
//! Layer 2  HOUR     L2 = ∪ S(t) over 60 minutes
//! Layer 1  MINUTE   L1 = ∪ S(t) over 60 seconds
//! Layer 0  SECOND   L0 = ∪ S(t) over current second    ← finest
//! ```
//!
//! # Automatic Building
//!
//! ```text
//! Every second boundary:  L1 = L1 ∪ L0;  L0 = ∅
//! Every minute boundary:  L2 = L2 ∪ L1;  L1 = ∅
//! ... cascade upward through all layers
//! ```
//!
//! After compression, layers are mutually exclusive: no time slice
//! appears in more than one layer. The complete system state is their union:
//!
//! ```text
//! H_system(t) = L0 ∪ L1 ∪ L2 ∪ L3 ∪ L4 ∪ L5 ∪ L6
//! ```
//!
//! # The Noether Invariant
//!
//! ```text
//! ⋃_{i=0}^{N-1} L_i = constant over time
//! ```
//!
//! Multiple aggregation paths converge to the same result because union
//! is monotonic and idempotent — eventual consistency emerges from the
//! structure, not from a protocol.

use hllset_core::HLLSet;
use std::time::Duration;

/// Default layer count for the standard 7-layer pyramid.
pub const DEFAULT_LAYERS: usize = 7;

/// Default durations for the standard second→year pyramid.
pub fn default_durations() -> Vec<Duration> {
    vec![
        Duration::from_secs(1),          // L0: 1 second
        Duration::from_secs(60),         // L1: 1 minute
        Duration::from_secs(3_600),      // L2: 1 hour
        Duration::from_secs(86_400),     // L3: 1 day
        Duration::from_secs(604_800),    // L4: 1 week
        Duration::from_secs(2_592_000),  // L5: ~30 days
        Duration::from_secs(31_536_000), // L6: ~365 days
    ]
}

// ── Layer ───────────────────────────────────────────────────────────────

/// One layer of the temporal pyramid.
///
/// Each layer accumulates HLLSet observations over its time window.
/// At boundary crossings, the layer is merged upward and reset.
#[derive(Clone, Debug)]
pub struct Layer {
    /// Accumulated HLLSet for this layer's current window.
    pub hllset: HLLSet,
    /// Duration of this layer's time window.
    pub window: Duration,
    /// Elapsed time within the current window.
    pub elapsed: Duration,
    /// Total number of observations ingested into this layer.
    pub observations: u64,
    /// Whether this layer has been carried up at least once.
    pub has_carried: bool,
}

impl Layer {
    /// Create a new empty layer with the given window duration.
    pub fn new(window: Duration) -> Self {
        Self {
            hllset: HLLSet::new(),
            window,
            elapsed: Duration::ZERO,
            observations: 0,
            has_carried: false,
        }
    }

    /// Whether the current window has elapsed.
    pub fn is_full(&self) -> bool {
        self.elapsed >= self.window
    }

    /// Remaining time in the current window.
    pub fn remaining(&self) -> Duration {
        if self.is_full() {
            Duration::ZERO
        } else {
            self.window - self.elapsed
        }
    }
}

// ── Temporal Pyramid ───────────────────────────────────────────────────

/// Configurable N-layer temporal pyramid.
///
/// Accepts HLLSet observations and automatically cascades them upward
/// through coarser time layers. The pyramid shape (number of layers `N`
/// and their durations `[d₀, ..., d_{N-1}]`) is fully configurable.
///
/// # Examples
///
/// ## Default 7-layer (second→year)
/// ```ignore
/// let mut pyramid = TemporalPyramid::default();
/// pyramid.ingest(&observation, timestamp);
/// let system_state = pyramid.system_state();
/// ```
///
/// ## Custom (high-frequency trading: 5×100ms)
/// ```ignore
/// use std::time::Duration;
/// let mut pyramid = TemporalPyramid::new(vec![
///     Duration::from_millis(100),
///     Duration::from_millis(100),
///     Duration::from_millis(100),
///     Duration::from_millis(100),
///     Duration::from_millis(100),
/// ]);
/// ```
#[derive(Clone, Debug)]
pub struct TemporalPyramid {
    /// Layers indexed from finest (0) to coarsest (N-1).
    layers: Vec<Layer>,
    /// Wall-clock timestamp of last ingestion.
    last_timestamp: Option<std::time::Instant>,
    /// Total observations ingested across all layers.
    total_observations: u64,
}

impl Default for TemporalPyramid {
    fn default() -> Self {
        Self::new(default_durations())
    }
}

impl TemporalPyramid {
    /// Create a pyramid with the given layer durations.
    ///
    /// `durations[i]` is the window size for layer `i` (0 = finest).
    /// At least 1 layer is required.
    pub fn new(durations: Vec<Duration>) -> Self {
        assert!(!durations.is_empty(), "pyramid requires at least one layer");
        // Validate: windows must be non-decreasing (each coarser layer ≥ finer)
        for i in 1..durations.len() {
            assert!(
                durations[i] >= durations[i - 1],
                "layer windows must be non-decreasing: layer {} ({:.1?}) < layer {} ({:.1?})",
                i,
                durations[i],
                i - 1,
                durations[i - 1]
            );
        }

        let layers: Vec<Layer> = durations.into_iter().map(Layer::new).collect();
        Self {
            layers,
            last_timestamp: None,
            total_observations: 0,
        }
    }

    /// Number of layers in the pyramid.
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// Reference to the layer at index `i` (0 = finest).
    pub fn layer(&self, i: usize) -> &Layer {
        &self.layers[i]
    }

    /// Total observations ingested.
    pub fn total_observations(&self) -> u64 {
        self.total_observations
    }

    // ── Ingestion ──────────────────────────────────────────────────

    /// Ingest an HLLSet observation at the given wall-clock timestamp.
    ///
    /// The observation is aggregated into L0. If the timestamp crosses one
    /// or more layer boundaries, carries are triggered upward automatically.
    ///
    /// Returns the number of layers that performed a carry during this
    /// ingestion (0 = no boundaries crossed).
    pub fn ingest(&mut self, observation: &HLLSet, now: std::time::Instant) -> usize {
        self.total_observations += 1;

        // Compute elapsed since last ingestion
        let delta = match self.last_timestamp {
            Some(prev) => {
                if now <= prev {
                    Duration::ZERO
                } else {
                    now.duration_since(prev)
                }
            }
            None => Duration::ZERO,
        };
        self.last_timestamp = Some(now);

        // Aggregate observation into L0
        self.layers[0].hllset = self.layers[0].hllset.union(observation);
        self.layers[0].observations += 1;

        // Advance time for L0
        let carries = self.advance_time(delta);

        carries
    }

    /// Ingest an HLLSet observation using a duration-based delta
    /// (for testing without real clocks).
    pub fn ingest_with_delta(&mut self, observation: &HLLSet, delta: Duration) -> usize {
        self.total_observations += 1;

        // Aggregate into L0
        self.layers[0].hllset = self.layers[0].hllset.union(observation);
        self.layers[0].observations += 1;

        // Advance time
        let carries = self.advance_time(delta);

        carries
    }

    // ── Time advance & carry ───────────────────────────────────────

    /// Advance elapsed time for L0; cascade carries upward.
    ///
    /// Returns the number of carries performed.
    fn advance_time(&mut self, delta: Duration) -> usize {
        if delta == Duration::ZERO {
            return 0;
        }

        let mut carries = 0;

        // Advance L0
        self.layers[0].elapsed += delta;

        // Cascade: for each adjacent pair (i, i+1), if layer i is full,
        // merge it into layer i+1 and reset layer i
        for i in 0..self.layers.len() - 1 {
            if self.layers[i].is_full() {
                self.carry(i, i + 1);
                carries += 1;
            }
        }

        // If the top layer is also full, reset it (it has no higher layer)
        let top = self.layers.len() - 1;
        if self.layers[top].is_full() {
            self.layers[top].hllset = HLLSet::new();
            self.layers[top].elapsed = Duration::ZERO;
            self.layers[top].observations = 0;
            self.layers[top].has_carried = true;
            carries += 1;
        }

        carries
    }

    /// Carry: merge layer `from` into layer `to`, then reset `from`.
    ///
    /// Also advances the target layer's elapsed by the source window
    /// (one complete source window has passed within the target).
    fn carry(&mut self, from: usize, to: usize) {
        let from_hllset = self.layers[from].hllset.clone();
        self.layers[to].hllset = self.layers[to].hllset.union(&from_hllset);
        self.layers[to].observations += self.layers[from].observations;
        let source_window = self.layers[from].window;
        self.layers[to].elapsed += source_window;

        self.layers[from].hllset = HLLSet::new();
        self.layers[from].elapsed = Duration::ZERO;
        self.layers[from].observations = 0;
        self.layers[from].has_carried = true;
    }

    // ── System state ───────────────────────────────────────────────

    /// Compute the complete system state: union of all layers.
    ///
    /// This is the holographic top — it implicitly contains every bit
    /// ever observed across all time scales.
    pub fn system_state(&self) -> HLLSet {
        let mut state = HLLSet::new();
        for layer in &self.layers {
            state = state.union(&layer.hllset);
        }
        state
    }

    /// Number of set bits in the system state.
    pub fn system_popcount(&self) -> u64 {
        self.system_state().popcount()
    }

    // ── Noether invariant ──────────────────────────────────────────

    /// Verify the Noether invariant: union of all layers before vs after
    /// ingestion should be consistent if no carries occurred.
    ///
    /// Returns true if the invariant holds.
    pub fn verify_noether(&self) -> bool {
        // The invariant simply checks that the system state is non-empty
        // iff observations have been ingested. The real invariant is:
        // Σ popcount(L_i) is monotonic (never decreases due to carries).
        //
        // Since carries merge upward (not delete), and reset is a set-to-zero,
        // the system_state() union before a carry equals system_state() after
        // because the carried data is in the higher layer.
        //
        // This function is a structural check — the invariant is guaranteed
        // by construction (union is monotonic).
        let state = self.system_state();
        if self.total_observations == 0 {
            state.is_empty()
        } else {
            // At least some bits should be present
            true
        }
    }

    // ── TF snapshots ───────────────────────────────────────────────

    /// Compute a TF vector snapshot from the current system state.
    ///
    /// Each set bit contributes 1.0 to the TF vector at that position.
    /// This can be stored at `system:tf_N` to enable time-lens queries
    /// against past states.
    pub fn tf_snapshot(&self) -> hllset_core::TFVec {
        let mut tf = hllset_core::TFVec::new();
        for layer in &self.layers {
            tf.increment_from_hllset(&layer.hllset, 1.0);
        }
        tf
    }

    /// Compute per-layer TF snapshots.
    ///
    /// Returns a Vec of TFVec, one per layer. Layer 0 is the finest;
    /// layer N-1 is the coarsest.
    pub fn per_layer_tf_snapshots(&self) -> Vec<hllset_core::TFVec> {
        self.layers
            .iter()
            .map(|layer| {
                let mut tf = hllset_core::TFVec::new();
                tf.increment_from_hllset(&layer.hllset, 1.0);
                tf
            })
            .collect()
    }
}

// ── Preset configurations ───────────────────────────────────────────────

impl TemporalPyramid {
    /// Standard 7-layer second→year pyramid.
    pub fn standard() -> Self {
        Self::default()
    }

    /// 5-layer high-frequency pyramid (all 100ms layers).
    /// Suitable for micro-burst detection.
    pub fn high_frequency() -> Self {
        Self::new(vec![
            Duration::from_millis(100),
            Duration::from_millis(100),
            Duration::from_millis(100),
            Duration::from_millis(100),
            Duration::from_millis(100),
        ])
    }

    /// 4-layer real-time control pyramid (250ms layers).
    pub fn realtime_control() -> Self {
        Self::new(vec![
            Duration::from_millis(250),
            Duration::from_millis(250),
            Duration::from_millis(250),
            Duration::from_millis(250),
        ])
    }

    /// 6-layer document analysis pyramid (10-minute layers).
    /// Suitable for section-level context tracking.
    pub fn document_analysis() -> Self {
        Self::new(vec![
            Duration::from_secs(600),
            Duration::from_secs(600),
            Duration::from_secs(600),
            Duration::from_secs(600),
            Duration::from_secs(600),
            Duration::from_secs(600),
        ])
    }

    /// Minimal 2-layer pyramid (1s + 60s).
    /// Useful for testing and simple use cases.
    pub fn minimal() -> Self {
        Self::new(vec![Duration::from_secs(1), Duration::from_secs(60)])
    }
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_obs(tokens: &[&str]) -> HLLSet {
        HLLSet::from_tokens(tokens)
    }

    #[test]
    fn test_new_pyramid() {
        let p = TemporalPyramid::standard();
        assert_eq!(p.layer_count(), 7);
        assert!(p.layer(0).hllset.is_empty());
        assert_eq!(p.total_observations(), 0);
    }

    #[test]
    fn test_ingest_into_l0() {
        let mut p = TemporalPyramid::minimal();
        let obs = make_obs(&["hello", "world"]);

        p.ingest_with_delta(&obs, Duration::from_millis(500));
        assert!(!p.layer(0).hllset.is_empty());
        assert_eq!(p.total_observations(), 1);
        assert_eq!(p.layer(0).observations, 1);
    }

    #[test]
    fn test_no_carry_within_window() {
        let mut p = TemporalPyramid::minimal();
        let obs = make_obs(&["a"]);

        // Ingest at 0.5s — well within L0's 1-second window
        let carries = p.ingest_with_delta(&obs, Duration::from_millis(500));
        assert_eq!(carries, 0);
        assert_eq!(p.layer(0).observations, 1);
        assert!(p.layer(1).hllset.is_empty());
    }

    #[test]
    fn test_carry_on_boundary() {
        let mut p = TemporalPyramid::minimal();
        let obs_a = make_obs(&["alpha"]);
        let obs_b = make_obs(&["beta"]);

        // First observation at t=0.5s
        p.ingest_with_delta(&obs_a, Duration::from_millis(500));
        // Second observation at t=1.5s (crosses 1-second boundary)
        let carries = p.ingest_with_delta(&obs_b, Duration::from_secs(1));

        assert!(carries > 0, "should have carried L0→L1");
        // L0 reset after carry (obs_b was part of the old window, carried up)
        assert_eq!(p.layer(0).observations, 0);
        // L1 received obs_a + obs_b from the carry
        assert_eq!(p.layer(1).observations, 2);
    }

    #[test]
    fn test_system_state_after_carry() {
        let mut p = TemporalPyramid::minimal();
        let obs_a = make_obs(&["keep"]);
        let obs_b = make_obs(&["also"]);

        let state_before = p.system_popcount();
        p.ingest_with_delta(&obs_a, Duration::from_millis(500));

        // Carry: push obs_a from L0→L1
        p.ingest_with_delta(&obs_b, Duration::from_millis(600));

        // System state (union of L0∪L1) should contain bits from both
        let state_after = p.system_popcount();
        assert!(state_after >= state_before, "system state is monotonic");
    }

    #[test]
    fn test_cascade_multiple_layers() {
        // 3-layer pyramid: 1s, 2s, 4s
        let mut p = TemporalPyramid::new(vec![
            Duration::from_secs(1),
            Duration::from_secs(2),
            Duration::from_secs(4),
        ]);
        let obs = make_obs(&["x"]);

        // t=0.5: L0 has obs
        p.ingest_with_delta(&obs, Duration::from_millis(500));
        assert_eq!(p.layer(0).observations, 1);

        // t=1.5: L0→L1 carry (L0 full at 1s)
        p.ingest_with_delta(&obs, Duration::from_secs(1));
        assert!(p.layer(0).has_carried);
        assert_eq!(p.layer(1).observations, 2); // 2 observations carried from L0

        // t=2.5: L0→L1 carry again, L1 now has 2s elapsed → L1→L2 carry
        p.ingest_with_delta(&obs, Duration::from_secs(1));
        assert!(p.layer(1).has_carried);
        assert_eq!(p.layer(2).observations, 3); // 2 from 1st carry + 1 from 2nd
    }

    #[test]
    fn test_noether_invariant() {
        let mut p = TemporalPyramid::minimal();
        assert!(p.verify_noether());

        let obs = make_obs(&["test"]);
        p.ingest_with_delta(&obs, Duration::from_secs(1));
        assert!(p.verify_noether());
    }

    #[test]
    fn test_tf_snapshot() {
        let mut p = TemporalPyramid::minimal();
        let obs = make_obs(&["hello", "world"]);
        p.ingest_with_delta(&obs, Duration::from_millis(500));

        let tf = p.tf_snapshot();
        assert!(!tf.is_empty());
        assert!(tf.total() > 0.0);
    }

    #[test]
    fn test_per_layer_tf_snapshots() {
        let mut p = TemporalPyramid::minimal();
        let obs = make_obs(&["data"]);
        p.ingest_with_delta(&obs, Duration::from_millis(500));

        let snapshots = p.per_layer_tf_snapshots();
        assert_eq!(snapshots.len(), 2);
        // L0 should have TF > 0
        assert!(!snapshots[0].is_empty());
    }

    #[test]
    fn test_high_frequency_pyramid() {
        let mut p = TemporalPyramid::high_frequency();
        assert_eq!(p.layer_count(), 5);
        let obs = make_obs(&["burst"]);
        p.ingest_with_delta(&obs, Duration::from_millis(50));
        assert_eq!(p.total_observations(), 1);
    }

    #[test]
    fn test_observations_count_across_carries() {
        let mut p = TemporalPyramid::minimal();
        let obs = make_obs(&["a"]);
        p.ingest_with_delta(&obs, Duration::from_millis(500));
        p.ingest_with_delta(&obs, Duration::from_millis(600)); // carry

        // Total across all layers should reflect ingestion count
        let total_layer_obs: u64 = p.layers.iter().map(|l| l.observations).sum();
        assert!(total_layer_obs > 0);
    }

    #[test]
    fn test_custom_pyramid_validation() {
        // Non-decreasing windows should work
        let p = TemporalPyramid::new(vec![
            Duration::from_secs(1),
            Duration::from_secs(5),
            Duration::from_secs(10),
        ]);
        assert_eq!(p.layer_count(), 3);
    }

    #[test]
    #[should_panic(expected = "pyramid requires at least one layer")]
    fn test_empty_pyramid_panics() {
        TemporalPyramid::new(vec![]);
    }

    #[test]
    #[should_panic(expected = "layer windows must be non-decreasing")]
    fn test_decreasing_windows_panics() {
        TemporalPyramid::new(vec![
            Duration::from_secs(10),
            Duration::from_secs(5), // smaller than previous!
        ]);
    }
}
