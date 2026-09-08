//! Complete ingestion: 3 hashes → HLLSet + LUT + TF, single touch, in-module.
//!
//! A token arrives once; the module computes all three seeded positions,
//! sets the atoms, registers the LUT fibers, and increments TF — without
//! handing the token to anything else.

use crate::tf::TfTable;
use hllset_contracts::BitAddress;
use hllset_core::HLLSet;
use hllset_lut::LutIndex;

/// The soldered seed set for n-seed ingestion.
///
/// For the n-gram regime the same module shape holds with channel-selected
/// seeds: each token type is ingested against the seed of its channel.
pub const SEEDS: [u64; 3] = [0, 1, 2];

/// Number of encodings (seeds) per token.
pub const N_SEEDS: usize = SEEDS.len();

/// The ingest module: one pass per token, everything in-module.
#[derive(Clone, Debug)]
pub struct Ingest {
    /// One sketch per encoding (seed).
    pub hllsets: [HLLSet; N_SEEDS],
    /// One reverse index per encoding (seed).
    pub luts: [LutIndex; N_SEEDS],
    /// The monotonic term-frequency table.
    pub tf: TfTable,
    /// Number of tokens touched (exactly once each).
    pub touched: u64,
}

impl Default for Ingest {
    fn default() -> Self {
        Self::new()
    }
}

impl Ingest {
    pub fn new() -> Self {
        Self {
            hllsets: [HLLSet::new(), HLLSet::new(), HLLSet::new()],
            luts: [
                LutIndex::default(),
                LutIndex::default(),
                LutIndex::default(),
            ],
            tf: TfTable::new(),
            touched: 0,
        }
    }

    /// One token, one touch: hash ×3, set atom ×3, LUT insert ×3, TF ×1.
    pub fn ingest_token(&mut self, token: &[u8]) {
        self.touched += 1;
        self.tf.increment(token);
        for seed_index in 0..N_SEEDS {
            let addr = BitAddress::of_token_seeded(token, SEEDS[seed_index]);
            self.hllsets[seed_index].add_bit(addr.bit());
            self.luts[seed_index].insert_token_seeded(token.to_vec(), SEEDS[seed_index]);
        }
    }

    pub fn ingest_tokens<'a, I>(&mut self, tokens: I)
    where
        I: IntoIterator<Item = &'a [u8]>,
    {
        for token in tokens {
            self.ingest_token(token);
        }
    }

    pub fn hllset(&self, seed_index: usize) -> &HLLSet {
        &self.hllsets[seed_index]
    }

    pub fn lut(&self, seed_index: usize) -> &LutIndex {
        &self.luts[seed_index]
    }

    pub fn tf(&self) -> &TfTable {
        &self.tf
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_touch_is_complete() {
        let mut ingest = Ingest::new();
        let tokens: Vec<Vec<u8>> = (0..200u32).map(hllset_contracts::token_in_bytes).collect();

        ingest.ingest_tokens(tokens.iter().map(|t| t.as_slice()));

        assert_eq!(ingest.touched, 200, "exactly one touch per token");

        for seed in 0..N_SEEDS {
            // Every token's atom is set in the sketch.
            for token in &tokens {
                let addr = BitAddress::of_token_seeded(token, SEEDS[seed]);
                assert!(
                    ingest.hllset(seed).bit_addresses().iter().any(|a| a.bit() == addr.bit()),
                    "atom {addr:?} missing from seed {seed}"
                );
                // And the token is registered in the LUT fiber.
                assert!(ingest.lut(seed).fiber(addr.bit()).contains(token));
            }
        }

        // TF counts every observation.
        assert_eq!(ingest.tf().count(&tokens[7]), 1);
        ingest.ingest_token(&tokens[7]);
        assert_eq!(ingest.tf().count(&tokens[7]), 2, "TF is monotonic");
    }
}
