//! Set algebra operations for HLLSet.
//!
//! Implements the lattice operations:
//! - **Join** (∪): union — bitwise OR
//! - **Meet** (∩): intersection — bitwise AND
//! - **Difference** (\): exclusion — A AND NOT B
//! - **XOR** (⊕): symmetric difference
//!
//! All operations produce new HLLSets (immutable lattice semantics).
//! In-place merge is also provided for building up sets incrementally.

use crate::core::hllset::HLLSet;

impl HLLSet {
    /// Union (join) of two HLLSets: A ∪ B.
    ///
    /// Bitwise OR of the underlying bitmaps.
    /// Corresponds to the lattice join operation.
    pub fn union(&self, other: &HLLSet) -> HLLSet {
        let mut result = self.bitmap().clone();
        result |= other.bitmap();
        HLLSet::from_bitmap(result)
    }

    /// Intersection (meet) of two HLLSets: A ∩ B.
    ///
    /// Bitwise AND of the underlying bitmaps.
    /// Corresponds to the lattice meet operation.
    pub fn intersection(&self, other: &HLLSet) -> HLLSet {
        let mut result = self.bitmap().clone();
        result &= other.bitmap();
        HLLSet::from_bitmap(result)
    }

    /// Difference of two HLLSets: A \ B.
    ///
    /// Bits that are set in A but not in B.
    pub fn difference(&self, other: &HLLSet) -> HLLSet {
        let mut result = self.bitmap().clone();
        result -= other.bitmap();
        HLLSet::from_bitmap(result)
    }

    /// Symmetric difference (XOR) of two HLLSets: A ⊕ B.
    ///
    /// Bits set in exactly one of the two sets.
    pub fn symmetric_difference(&self, other: &HLLSet) -> HLLSet {
        let mut result = self.bitmap().clone();
        result ^= other.bitmap();
        HLLSet::from_bitmap(result)
    }

    /// Jaccard similarity: |A ∩ B| / |A ∪ B|.
    ///
    /// Returns 1.0 if both sets are empty.
    pub fn jaccard_similarity(&self, other: &HLLSet) -> f64 {
        let union_card = self.union(other).cardinality();
        let inter_card = self.intersection(other).cardinality();
        if union_card == 0.0 {
            return 1.0;
        }
        inter_card / union_card
    }

    // --- In-place operations -------------------------------------------------

    /// Merge another HLLSet into this one (in-place union).
    pub fn merge(&mut self, other: &HLLSet) {
        *self.bitmap_mut() |= other.bitmap();
    }

    /// Merge multiple HLLSets into this one (in-place union).
    pub fn merge_all<I>(&mut self, others: I)
    where
        I: IntoIterator<Item = HLLSet>,
    {
        for other in others {
            self.merge(&other);
        }
    }

    /// Merge tokens into this HLLSet (in-place).
    pub fn merge_tokens<I, S>(&mut self, tokens: I)
    where
        I: IntoIterator<Item = S>,
        S: AsRef<[u8]>,
    {
        for token in tokens {
            self.add_token(token.as_ref());
        }
    }

    // --- Multi-set constructors ----------------------------------------------

    /// Create the union of multiple HLLSets.
    pub fn union_all<I>(sets: I) -> HLLSet
    where
        I: IntoIterator<Item = HLLSet>,
    {
        let mut result = HLLSet::new();
        result.merge_all(sets);
        result
    }

    /// Create the intersection of multiple HLLSets.
    pub fn intersection_all<I>(sets: I) -> HLLSet
    where
        I: IntoIterator<Item = HLLSet>,
        I::IntoIter: Clone,
    {
        let mut iter = sets.into_iter();
        let mut result = match iter.next() {
            Some(first) => first,
            None => return HLLSet::new(),
        };
        for set in iter {
            result = result.intersection(&set);
        }
        result
    }

    /// Check subset relation using bitmaps.
    ///
    /// Returns true if all bits in `self` are also set in `other`.
    /// This is an approximation — due to hash collisions, false positives
    /// are possible but false negatives are not.
    pub fn is_subset_of(&self, other: &HLLSet) -> bool {
        self.bitmap().is_subset(other.bitmap())
    }

    /// Check superset relation.
    ///
    /// Returns true if all bits in `other` are also set in `self`.
    pub fn is_superset_of(&self, other: &HLLSet) -> bool {
        other.is_subset_of(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_set(tokens: &[&str]) -> HLLSet {
        HLLSet::from_tokens(tokens.iter().map(|s| s.as_bytes()))
    }

    #[test]
    fn test_union_idempotent() {
        let a = make_set(&["hello", "world"]);
        let a2 = a.union(&a);
        assert_eq!(a.popcount(), a2.popcount());
    }

    #[test]
    fn test_union_commutative() {
        let a = make_set(&["a", "b"]);
        let b = make_set(&["b", "c"]);
        assert_eq!(a.union(&b).popcount(), b.union(&a).popcount());
    }

    #[test]
    fn test_union_card_at_least_max() {
        let a = make_set(&["a", "b"]);
        let b = make_set(&["c", "d"]);
        let union = a.union(&b);
        assert!(union.cardinality() >= a.cardinality());
        assert!(union.cardinality() >= b.cardinality());
    }

    #[test]
    fn test_intersection_subset_of_both() {
        let a = make_set(&["a", "b", "c"]);
        let b = make_set(&["b", "c", "d"]);
        let inter = a.intersection(&b);
        assert!(inter.cardinality() <= a.cardinality());
        assert!(inter.cardinality() <= b.cardinality());
    }

    #[test]
    fn test_difference_excludes_b() {
        let a = make_set(&["a", "b", "c"]);
        let b = make_set(&["b"]);
        let diff = a.difference(&b);
        // diff ∪ b should be within collision tolerance of a
        let recomposed = diff.union(&b);
        let diff_pop = a.popcount() as i64 - recomposed.popcount() as i64;
        // Due to HLL properties, recomposed may have slightly fewer bits
        assert!(diff_pop >= 0);
    }

    #[test]
    fn test_xor_symmetric() {
        let a = make_set(&["x", "y"]);
        let b = make_set(&["y", "z"]);
        assert_eq!(
            a.symmetric_difference(&b).popcount(),
            b.symmetric_difference(&a).popcount()
        );
    }

    #[test]
    fn test_jaccard_same() {
        let a = make_set(&["a", "b"]);
        assert!((a.jaccard_similarity(&a) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_jaccard_disjoint_bound() {
        let a = make_set(&["a"]);
        let b = make_set(&["b"]);
        // With HLL, disjoint sets may still have some intersection bits
        let j = a.jaccard_similarity(&b);
        assert!(j >= 0.0 && j <= 1.0, "jaccard={j}");
    }

    #[test]
    fn test_merge_equals_union() {
        let a = make_set(&["a", "b"]);
        let b = make_set(&["c"]);
        let union = a.union(&b);
        let mut merged = a.clone();
        merged.merge(&b);
        assert_eq!(union.popcount(), merged.popcount());
    }

    #[test]
    fn test_is_subset_self() {
        let a = make_set(&["x", "y"]);
        assert!(a.is_subset_of(&a));
        assert!(a.is_superset_of(&a));
    }
}
