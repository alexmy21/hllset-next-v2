//! Observable Mask — O(θ) threshold mask and sub-lattice degree.
//!
//! The complete collection H contains every HLLSet. The observable sample
//! O(θ) = {H | hllset-R(H) > θ} is a bitmask over H. When ranks shift,
//! the mask changes, and sub-lattice degrees update.
//!
//! FPGA-native: CMP (threshold check), POPCOUNT (degree).

use crate::hllset::{HLLSetRank, HLLSetRankIndex};
use crate::{Rank, Threshold};
use std::collections::{HashMap, HashSet};

/// The observable mask — which HLLSets are above threshold at a given moment.
#[derive(Debug, Clone)]
pub struct ObservableMask {
    /// Threshold θ.
    pub threshold: Threshold,
    /// Keys of HLLSets currently above threshold.
    pub observable: HashSet<String>,
    /// Keys of HLLSets currently below threshold.
    pub hidden: HashSet<String>,
    /// Size of the complete collection.
    pub total: usize,
}

impl ObservableMask {
    /// Apply threshold to a rank index. Returns the mask.
    pub fn apply(ranks: &HLLSetRankIndex, threshold: Threshold) -> Self {
        let mut observable = HashSet::new();
        let mut hidden = HashSet::new();
        for rank in ranks.iter() {
            if rank.value > threshold {
                observable.insert(rank.key.clone());
            } else {
                hidden.insert(rank.key.clone());
            }
        }
        let total = observable.len() + hidden.len();
        Self {
            threshold,
            observable,
            hidden,
            total,
        }
    }

    /// Is the given HLLSet currently observable?
    pub fn is_observable(&self, key: &str) -> bool {
        self.observable.contains(key)
    }

    /// How many HLLSets are observable?
    pub fn observable_count(&self) -> usize {
        self.observable.len()
    }

    /// Fraction of the collection that is observable.
    pub fn observable_fraction(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.observable.len() as f64 / self.total as f64
        }
    }

    /// Compute the difference between two masks — which HLLSets entered
    /// and which exited the observable set.
    pub fn diff(prev: &Self, curr: &Self) -> MaskDiff {
        let entered: Vec<_> = curr
            .observable
            .difference(&prev.observable)
            .cloned()
            .collect();
        let exited: Vec<_> = prev
            .observable
            .difference(&curr.observable)
            .cloned()
            .collect();
        MaskDiff { entered, exited }
    }
}

/// Result of comparing two observable masks.
#[derive(Debug, Clone)]
pub struct MaskDiff {
    /// HLLSets that entered O(θ) (were hidden, now visible).
    pub entered: Vec<String>,
    /// HLLSets that exited O(θ) (were visible, now hidden).
    pub exited: Vec<String>,
}

impl MaskDiff {
    pub fn is_stable(&self) -> bool {
        self.entered.is_empty() && self.exited.is_empty()
    }

    pub fn churn(&self) -> usize {
        self.entered.len() + self.exited.len()
    }
}

/// Sub-lattice degree tracker.
///
/// For each HLLSet, tracks its degree within the observable sub-lattice.
/// When the mask changes, degrees are updated for affected HLLSets.
#[derive(Debug, Clone, Default)]
pub struct SubLatticeDegree {
    /// key → degree within the observable sub-lattice.
    degrees: HashMap<String, usize>,
    /// Adjacency: key → set of neighbor keys. Only edges between
    /// observable HLLSets contribute to degree.
    adjacency: HashMap<String, HashSet<String>>,
}

impl SubLatticeDegree {
    pub fn new() -> Self {
        Self {
            degrees: HashMap::new(),
            adjacency: HashMap::new(),
        }
    }

    /// Add an edge between two HLLSets. If both are observable, degrees increment.
    /// Call this when a lattice operation creates a relationship.
    pub fn add_edge(&mut self, a: &str, b: &str, mask: &ObservableMask) {
        self.adjacency
            .entry(a.to_string())
            .or_default()
            .insert(b.to_string());
        self.adjacency
            .entry(b.to_string())
            .or_default()
            .insert(a.to_string());

        if mask.is_observable(a) && mask.is_observable(b) {
            *self.degrees.entry(a.to_string()).or_default() += 1;
            *self.degrees.entry(b.to_string()).or_default() += 1;
        }
    }

    /// Recompute all degrees after a mask change.
    pub fn recompute(&mut self, mask: &ObservableMask) {
        self.degrees.clear();
        for (key, neighbors) in &self.adjacency {
            if !mask.is_observable(key) {
                continue;
            }
            let degree = neighbors.iter().filter(|n| mask.is_observable(n)).count();
            if degree > 0 {
                self.degrees.insert(key.clone(), degree);
            }
        }
    }

    /// Get the observable sub-lattice degree of an HLLSet.
    pub fn degree(&self, key: &str) -> usize {
        self.degrees.get(key).copied().unwrap_or(0)
    }

    /// Number of HLLSets with non-zero observable degree.
    pub fn active_count(&self) -> usize {
        self.degrees.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hllset::{DegreeRankFn, HLLSetRank, HLLSetRankIndex};

    fn make_index(entries: &[(&str, Rank, usize, u64)]) -> HLLSetRankIndex {
        let mut idx = HLLSetRankIndex::new();
        for &(key, _value, degree, popcount) in entries {
            // We override the rank value manually for testing
            let mut rank = HLLSetRank::from_raw(key, degree, popcount, &DegreeRankFn);
            rank.value = _value;
            idx.insert(rank);
        }
        idx
    }

    #[test]
    fn test_observable_mask_basic() {
        let idx = make_index(&[("h:a", 100, 3, 50), ("h:b", 50, 2, 30), ("h:c", 10, 1, 5)]);
        let mask = ObservableMask::apply(&idx, 25);
        // a (100) and b (50) above 25, c (10) below
        assert!(mask.is_observable("h:a"));
        assert!(mask.is_observable("h:b"));
        assert!(!mask.is_observable("h:c"));
        assert_eq!(mask.observable_count(), 2);
        assert_eq!(mask.total, 3);
    }

    #[test]
    fn test_mask_diff() {
        let idx1 = make_index(&[("h:a", 100, 3, 50), ("h:b", 10, 1, 5)]);
        let idx2 = make_index(&[("h:a", 90, 3, 50), ("h:b", 30, 1, 5)]);

        let mask1 = ObservableMask::apply(&idx1, 20);
        let mask2 = ObservableMask::apply(&idx2, 20);
        // mask1: a above, b below. mask2: both above.
        let diff = ObservableMask::diff(&mask1, &mask2);
        assert_eq!(diff.entered, vec!["h:b"]);
        assert!(diff.exited.is_empty());
        assert_eq!(diff.churn(), 1);
    }

    #[test]
    fn test_sub_lattice_degree() {
        let mut degree = SubLatticeDegree::new();
        let idx = make_index(&[("h:a", 100, 3, 50), ("h:b", 50, 2, 30), ("h:c", 10, 1, 5)]);
        let mask = ObservableMask::apply(&idx, 25);

        degree.add_edge("h:a", "h:b", &mask);
        degree.add_edge("h:a", "h:c", &mask); // c is hidden, shouldn't count
        degree.add_edge("h:b", "h:c", &mask); // c is hidden

        assert_eq!(degree.degree("h:a"), 1); // only b is observable
        assert_eq!(degree.degree("h:b"), 1); // only a is observable
        assert_eq!(degree.degree("h:c"), 0); // hidden, no degree
    }

    #[test]
    fn test_mask_stability() {
        let idx = make_index(&[("h:a", 50, 3, 50)]);
        let mask = ObservableMask::apply(&idx, 25);
        let diff = ObservableMask::diff(&mask, &mask);
        assert!(diff.is_stable());
        assert_eq!(diff.churn(), 0);
    }
}
