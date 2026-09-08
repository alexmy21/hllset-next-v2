//! Materialization: LUT-first, TF only to resolve ambiguity.
//!
//! For each active bit of each sketch, gather candidate tokens from the
//! **corresponding pointed LUT** (every encoding); only when a bit has more
//! than one candidate does the LUT's TF decide. Normally people start with
//! TF — this module never does: TF is consulted exclusively for collided
//! bits.

use crate::tf::TfTable;
use hllset_core::HLLSet;
use hllset_lut::LutIndex;
use std::collections::{BTreeMap, BTreeSet};

/// `M(H, {L_j})` — materialize across encodings.
///
/// Takes one `(sketch, LUT)` pair per encoding; the LUT of a pair is
/// addressed by the same seed as its sketch. Candidates are collected per
/// bit across all pairs, and a bit with several candidates restores the
/// TF-maximum (ties break deterministically, to the lexicographically
/// largest token bytes).
pub fn materialize(pairs: &[(&HLLSet, &LutIndex)], tf: &TfTable) -> BTreeSet<Vec<u8>> {
    let mut by_bit: BTreeMap<u32, BTreeSet<Vec<u8>>> = BTreeMap::new();
    for (hllset, lut) in pairs {
        for addr in hllset.bit_addresses() {
            by_bit
                .entry(addr.bit())
                .or_default()
                .extend(lut.fiber(addr.bit()));
        }
    }

    let mut out = BTreeSet::new();
    for (_, candidates) in by_bit {
        match candidates.len() {
            0 => {}
            1 => {
                out.extend(candidates);
            }
            _ => {
                // Ambiguous bit: the LUT TF is the tie-break.
                if let Some(winner) = candidates.iter().max_by_key(|t| tf.count(t)) {
                    out.insert(winner.clone());
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ingest::Ingest;
    use hllset_contracts::{token_in_bytes, token_in_bytes_le, BitAddress};

    #[test]
    fn lut_is_first_tf_only_breaks_ambiguity() {
        let mut ingest = Ingest::new();
        // tid262LE and tid48300LE collide at (759, 0) under seed 0.
        let a = token_in_bytes_le(262).to_vec();
        let b = token_in_bytes_le(48_300).to_vec();
        ingest.ingest_token(&a);
        ingest.ingest_token(&a); // TF(a) = 2
        ingest.ingest_token(&b); // TF(b) = 1

        let mut sketch = hllset_core::HLLSet::new();
        sketch.add_bit(759 * 32 + 0);

        let restored = materialize(&[(&sketch, ingest.lut(0))], ingest.tf());
        assert_eq!(restored, BTreeSet::from([a.clone()]), "TF-max wins the collided bit");
    }

    #[test]
    fn unambiguous_bit_ignores_tf() {
        let mut ingest = Ingest::new();
        let rare = token_in_bytes(1);
        let frequent = token_in_bytes(2);
        ingest.ingest_token(&rare);
        for _ in 0..100 {
            ingest.ingest_token(&frequent);
        }

        // A sketch containing only `rare`'s seed-0 atom restores `rare`,
        // even though `frequent` has a much higher TF — TF is not the start.
        let mut sketch = hllset_core::HLLSet::new();
        let rare_bit = BitAddress::of_token_seeded(&rare, 0).bit();
        sketch.add_bit(rare_bit);

        let restored = materialize(&[(&sketch, ingest.lut(0))], ingest.tf());
        assert_eq!(restored, BTreeSet::from([rare]));
    }

    #[test]
    fn high_tf_token_outside_the_luts_never_appears() {
        let mut ingest = Ingest::new();
        let in_lut = token_in_bytes(5);
        let outside = token_in_bytes(6);
        ingest.ingest_token(&in_lut);
        // `outside` gets a huge TF but is removed from all LUTs.
        for _ in 0..1000 {
            ingest.ingest_token(&outside);
        }
        // Rebuild a LUT containing only `in_lut`.
        let mut lut_only_in = hllset_lut::LutIndex::default();
        lut_only_in.insert_token(in_lut.clone());

        let mut sketch = hllset_core::HLLSet::new();
        let bit = BitAddress::of_token_seeded(&in_lut, 0).bit();
        sketch.add_bit(bit);

        let restored = materialize(&[(&sketch, &lut_only_in)], ingest.tf());
        assert_eq!(restored, BTreeSet::from([in_lut]), "TF cannot conjure tokens absent from the LUTs");
    }

    #[test]
    fn candidates_come_from_all_pointed_luts() {
        let mut ingest = Ingest::new();
        let t0 = token_in_bytes(10);
        let t1 = token_in_bytes(20);
        let t2 = token_in_bytes(30);

        // One sketch + LUT pair per encoding; each token appears in its own
        // encoding's LUT only.
        let mut s0 = hllset_core::HLLSet::new();
        s0.add_bit(BitAddress::of_token_seeded(&t0, 0).bit());
        let mut l0 = hllset_lut::LutIndex::default();
        l0.insert_token_seeded(t0.clone(), 0);

        let mut s1 = hllset_core::HLLSet::new();
        s1.add_bit(BitAddress::of_token_seeded(&t1, 1).bit());
        let mut l1 = hllset_lut::LutIndex::default();
        l1.insert_token_seeded(t1.clone(), 1);

        let mut s2 = hllset_core::HLLSet::new();
        s2.add_bit(BitAddress::of_token_seeded(&t2, 2).bit());
        let mut l2 = hllset_lut::LutIndex::default();
        l2.insert_token_seeded(t2.clone(), 2);

        // Only when all three LUTs are pointed are all three restored.
        let all = materialize(&[(&s0, &l0), (&s1, &l1), (&s2, &l2)], ingest.tf());
        assert_eq!(all, BTreeSet::from([t0.clone(), t1.clone(), t2.clone()]));
        let only_l0 = materialize(&[(&s0, &l0)], ingest.tf());
        assert_eq!(only_l0, BTreeSet::from([t0]));
    }
}
