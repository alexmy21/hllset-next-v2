//! Monotonic term-frequency table — the TF side of the ingest module.
//!
//! Grow-only: `increment` only raises counts; `merge` is pointwise max
//! (a join in the tropical lattice), so TF is a CRDT.

use std::collections::BTreeMap;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TfTable {
    counts: BTreeMap<Vec<u8>, u64>,
}

impl TfTable {
    pub fn new() -> Self {
        Self::default()
    }

    /// One token observation.
    pub fn increment(&mut self, token: &[u8]) {
        *self.counts.entry(token.to_vec()).or_insert(0) += 1;
    }

    pub fn count(&self, token: &[u8]) -> u64 {
        self.counts.get(token).copied().unwrap_or(0)
    }

    /// CRDT join: pointwise max.
    pub fn merge(&mut self, other: &Self) {
        for (token, &count) in &other.counts {
            let slot = self.counts.entry(token.clone()).or_insert(0);
            *slot = (*slot).max(count);
        }
    }

    pub fn len(&self) -> usize {
        self.counts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.counts.is_empty()
    }
}
