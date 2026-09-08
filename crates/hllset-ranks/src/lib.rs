//! `hllset-ranks` — the five-level rank algebra, rewritten from the algebra.
//!
//! **TF is stored; rank is derived, never stored.** Every level is a pure
//! projection of the stored [`TfTable`] through the LUT fibers:
//!
//! ```text
//! F(TF)  →  G(bit)  →  H(register)  →  K(HLLSet)  →  L(compound)
//! ```
//!
//! - `F` token rank: the stored TF of one token.
//! - `G` bit rank: fold of `F` over the fiber `K_i` (`hllset-lut::LutIndex`).
//! - `H` register rank: fold of `G` over the register's 32 bits.
//! - `K` HLLSet rank: fold of `G` over the active atoms — the degree in the
//!   lattice.
//! - `L` compound rank: fold of `K` over a set of sketches.
//!
//! All ranks are `u64` (FPGA-native AND/OR/POPCOUNT/ADD/SUB/CMP); the fold
//! is the application's choice ([`Aggregator`]). Nothing is persisted between
//! levels — each level recomputes from TF + fibers.

use hllset_contracts::BITS_PER_REG;
use hllset_core::HLLSet;
use hllset_lut::LutIndex;
use hllset_morphisms::TfTable;

/// The universal rank type — every level produces `u64`.
pub type Rank = u64;

/// The fold used to project one level onto the next.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Aggregator {
    Sum,
    Max,
    Min,
}

impl Aggregator {
    pub fn fold(self, values: impl IntoIterator<Item = Rank>) -> Rank {
        let mut iter = values.into_iter();
        match self {
            Aggregator::Sum => iter.sum(),
            Aggregator::Max => iter.fold(0, |acc, v| acc.max(v)),
            Aggregator::Min => match iter.next() {
                Some(first) => iter.fold(first, |acc, v| acc.min(v)),
                None => 0,
            },
        }
    }
}

/// Level 1 — token rank `F(TF)`: the stored term frequency of one token.
pub fn token_rank(tf: &TfTable, token: &[u8]) -> Rank {
    tf.count(token)
}

/// Level 2 — bit rank `G(i)`: fold of token ranks over the fiber `K_i`.
pub fn bit_rank(tf: &TfTable, lut: &LutIndex, bit: u32, agg: Aggregator) -> Rank {
    agg.fold(lut.fiber(bit).iter().map(|t| token_rank(tf, t)))
}

/// Level 3 — register rank `H(reg)`: fold of `G` over the register's 32 bits.
pub fn register_rank(tf: &TfTable, lut: &LutIndex, reg: u32, agg: Aggregator) -> Rank {
    agg.fold(
        (0..BITS_PER_REG).map(|tz| bit_rank(tf, lut, reg * BITS_PER_REG + tz, agg)),
    )
}

/// Level 4 — HLLSet rank `K(H)`: fold of `G` over the active atoms.
pub fn hllset_rank(tf: &TfTable, lut: &LutIndex, hllset: &HLLSet, agg: Aggregator) -> Rank {
    agg.fold(
        hllset
            .bit_addresses()
            .iter()
            .map(|a| bit_rank(tf, lut, a.bit(), agg)),
    )
}

/// Level 5 — compound rank `L({H_j})`: fold of `K` over a set of sketches.
pub fn compound_rank(
    tf: &TfTable,
    lut: &LutIndex,
    sketches: &[HLLSet],
    agg: Aggregator,
) -> Rank {
    agg.fold(sketches.iter().map(|h| hllset_rank(tf, lut, h, agg)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use hllset_contracts::token_in_bytes;
    use hllset_morphisms::Ingest;

    fn ingest_two() -> (Ingest, HLLSet) {
        let mut ingest = Ingest::new();
        let alpha = token_in_bytes(1);
        let beta = token_in_bytes(2);
        ingest.ingest_token(&alpha);
        ingest.ingest_token(&alpha); // TF(alpha) = 2
        ingest.ingest_token(&beta); // TF(beta) = 1
        let sketch = ingest.hllset(0).clone();
        (ingest, sketch)
    }

    #[test]
    fn rank_is_derived_never_stored() {
        let (ingest, sketch) = ingest_two();
        let tf = ingest.tf();
        let lut = ingest.lut(0);

        let k1 = hllset_rank(tf, lut, &sketch, Aggregator::Sum);
        // Recomputation from the same stored TF + fibers is identical…
        let k2 = hllset_rank(tf, lut, &sketch, Aggregator::Sum);
        assert_eq!(k1, k2, "rank is a projection: same inputs, same rank");
        // …and computing ranks never mutates the stored TF.
        assert_eq!(tf.count(&token_in_bytes(1)), 2);
        assert_eq!(tf.count(&token_in_bytes(2)), 1);
    }

    #[test]
    fn levels_compose_exactly() {
        let (ingest, sketch) = ingest_two();
        let tf = ingest.tf();
        let lut = ingest.lut(0);
        let agg = Aggregator::Sum;

        // G(bit) is the fold of F over the fiber.
        for addr in sketch.bit_addresses() {
            let via_fiber: Rank = lut
                .fiber(addr.bit())
                .iter()
                .map(|t| token_rank(tf, t))
                .sum();
            assert_eq!(bit_rank(tf, lut, addr.bit(), agg), via_fiber);
        }

        // H(reg) is the fold of G over the register bits.
        let reg = sketch.bit_addresses()[0].reg();
        let via_g: Rank = (0..BITS_PER_REG)
            .map(|tz| bit_rank(tf, lut, reg * BITS_PER_REG + tz, agg))
            .sum();
        assert_eq!(register_rank(tf, lut, reg, agg), via_g);

        // K(H) is the fold of G over the active atoms.
        let via_atoms: Rank = sketch
            .bit_addresses()
            .iter()
            .map(|a| bit_rank(tf, lut, a.bit(), agg))
            .sum();
        assert_eq!(hllset_rank(tf, lut, &sketch, agg), via_atoms);

        // L is the fold of K over the sketches.
        assert_eq!(
            compound_rank(tf, lut, &[sketch.clone()], agg),
            hllset_rank(tf, lut, &sketch, agg)
        );
    }

    #[test]
    fn ranks_are_monotone_in_tf_for_sum_and_max() {
        let (ingest, sketch) = ingest_two();
        let tf = ingest.tf();
        let lut = ingest.lut(0);

        let before_sum = hllset_rank(tf, lut, &sketch, Aggregator::Sum);
        let before_max = hllset_rank(tf, lut, &sketch, Aggregator::Max);

        let mut grow = tf.clone();
        grow.increment(&token_in_bytes(1)); // TF(alpha) 2 → 3

        assert!(hllset_rank(&grow, lut, &sketch, Aggregator::Sum) >= before_sum);
        assert!(hllset_rank(&grow, lut, &sketch, Aggregator::Max) >= before_max);
    }

    #[test]
    fn empty_and_aggregator_neutrals() {
        let ingest = Ingest::new();
        let empty = hllset_core::HLLSet::new();
        assert_eq!(
            hllset_rank(ingest.tf(), ingest.lut(0), &empty, Aggregator::Sum),
            0
        );
        assert_eq!(
            hllset_rank(ingest.tf(), ingest.lut(0), &empty, Aggregator::Max),
            0
        );
        assert_eq!(
            hllset_rank(ingest.tf(), ingest.lut(0), &empty, Aggregator::Min),
            0
        );
    }
}
