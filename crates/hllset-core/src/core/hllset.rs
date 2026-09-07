//! Core HLLSet data structure.
//!
//! HLLSet stores HyperLogLog fingerprints as a compressed Roaring bitmap.
//! Each bit position (reg * 32 + tz) means register `reg` observed
//! trailing-zero count `tz`.

use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};

/// Number of precision bits (P). Determines register count M = 2^P.
/// Single source of truth: `hllset-contracts` (GOVERNANCE.md §3).
pub const P: u32 = hllset_contracts::P;

/// Number of registers (M = 2^P = 1024).
pub const M: usize = hllset_contracts::M as usize;

/// Number of bits tracked per register (trailing zeros 0..31).
pub const BITS_PER_REG: u32 = hllset_contracts::BITS_PER_REG;

/// Total bits in the bitmap tensor (M × 32 = 32768).
pub const TOTAL_BITS: u32 = hllset_contracts::TOTAL_BITS;

/// Alpha constant for standard HLL bias correction (M=1024).
pub const ALPHA_M: f64 = 0.7213 / (1.0 + 1.079 / (M as f64));

/// An HLLSet — a HyperLogLog fingerprint with set-algebra operations.
///
/// Stored as a Roaring bitmap for compression. Inflated to a dense
/// `Vec<u32>` register array for cardinality estimation.
///
/// # Lattice properties
///
/// HLLSets form a **bounded distributive lattice** under:
/// - Join (∪): bitwise OR of registers
/// - Meet (∩): bitwise AND of registers
///
/// These operations are associative, commutative, and idempotent,
/// making HLLSets ideal for content-addressed, eventually-consistent
/// distributed systems.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HLLSet {
    /// Compressed bitmap: bit at position (reg * 32 + tz) indicates
    /// that register `reg` has observed trailing-zero count `tz`.
    bitmap: RoaringBitmap,
}

impl Default for HLLSet {
    fn default() -> Self {
        Self::new()
    }
}

impl HLLSet {
    /// Create a new empty HLLSet.
    pub fn new() -> Self {
        Self {
            bitmap: RoaringBitmap::new(),
        }
    }

    /// Create an HLLSet from token byte sequences.
    ///
    /// Each token is hashed (MurmurHash3, seed 0) and the resulting
    /// bit position is set.
    pub fn from_tokens<I, S>(tokens: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<[u8]>,
    {
        let mut hllset = Self::new();
        for token in tokens {
            hllset.add_token(token.as_ref());
        }
        hllset
    }

    /// Create an HLLSet from pre-computed 64-bit hashes.
    pub fn from_hashes<I>(hashes: I) -> Self
    where
        I: IntoIterator<Item = u64>,
    {
        let mut hllset = Self::new();
        for hash in hashes {
            hllset.add_hash(hash);
        }
        hllset
    }

    /// Create an HLLSet from a dense register array.
    ///
    /// Each `u32` encodes which trailing-zero counts were observed
    /// in that register.
    pub fn from_dense(registers: &[u32]) -> Self {
        let mut hllset = Self::new();
        for (reg, &value) in registers.iter().enumerate().take(M) {
            for bit in 0..32u32 {
                if (value >> bit) & 1 == 1 {
                    let pos = (reg as u32) * BITS_PER_REG + bit;
                    hllset.bitmap.insert(pos);
                }
            }
        }
        hllset
    }

    /// Add a token (byte sequence) to this HLLSet.
    ///
    /// Hashes with MurmurHash3 (seed 0), then calls `add_hash`.
    pub fn add_token(&mut self, token: &[u8]) {
        let hash = crate::core::hashing::murmur3_hash(token);
        self.add_hash(hash);
    }

    /// Add a pre-computed 64-bit hash to this HLLSet.
    ///
    /// **Algorithm:**
    /// - Lower P bits → register index (0..1023)
    /// - Remaining bits → trailing-zero count (0..31)
    /// - Set bit `(register * 32 + trailing_zeros)` in the bitmap
    pub fn add_hash(&mut self, hash: u64) {
        let reg = (hash as u32) & ((M as u32) - 1);
        let remaining = hash >> P;
        let tz = if remaining == 0 {
            31
        } else {
            remaining.trailing_zeros().min(31)
        };
        let pos = reg * BITS_PER_REG + tz;
        self.bitmap.insert(pos);
    }

    // --- Bitmap access -------------------------------------------------------

    /// Reference to the internal RoaringBitmap.
    pub fn bitmap(&self) -> &RoaringBitmap {
        &self.bitmap
    }

    /// Mutable reference to the internal RoaringBitmap.
    pub fn bitmap_mut(&mut self) -> &mut RoaringBitmap {
        &mut self.bitmap
    }

    /// Consume and return the internal RoaringBitmap.
    pub fn into_bitmap(self) -> RoaringBitmap {
        self.bitmap
    }

    /// Create an HLLSet from a RoaringBitmap.
    pub fn from_bitmap(bitmap: RoaringBitmap) -> Self {
        Self { bitmap }
    }

    // --- Queries -------------------------------------------------------------

    /// Returns `true` if no bits are set.
    pub fn is_empty(&self) -> bool {
        self.bitmap.is_empty()
    }

    /// Number of set bits (population count).
    pub fn popcount(&self) -> u64 {
        self.bitmap.len()
    }

    /// All active (register, trailing_zeros) positions.
    pub fn active_positions(&self) -> Vec<(u32, u32)> {
        self.bitmap
            .iter()
            .map(|pos| (pos / BITS_PER_REG, pos % BITS_PER_REG))
            .collect()
    }

    /// Count of registers that have a specific bit position set.
    /// Used by the Horvitz-Thompson cardinality estimator.
    pub fn count_registers_with_bit(&self, bit_pos: u32) -> u32 {
        self.bitmap
            .iter()
            .filter(|pos| pos % BITS_PER_REG == bit_pos)
            .count() as u32
    }

    /// Bit counts per trailing-zero position (0..31).
    /// `counts[tz]` = number of registers where bit `tz` is set.
    pub fn bit_counts(&self) -> Vec<u32> {
        let mut counts = vec![0u32; BITS_PER_REG as usize];
        for pos in self.bitmap.iter() {
            counts[(pos % BITS_PER_REG) as usize] += 1;
        }
        counts
    }

    /// Number of registers with at least one bit set.
    pub fn non_zero_registers(&self) -> u32 {
        self.to_dense().iter().filter(|&&r| r != 0).count() as u32
    }

    /// Approximate serialized size in bytes.
    pub fn memory_usage(&self) -> usize {
        self.bitmap.serialized_size()
    }

    /// Check if a specific (register, bit) position is set.
    pub fn has_bit(&self, register: u32, bit: u32) -> bool {
        self.bitmap.contains(register * BITS_PER_REG + bit)
    }

    // --- Dense representation ------------------------------------------------

    /// Inflate to dense `[u32; M]` register array.
    ///
    /// Each element encodes the set of observed trailing-zero counts
    /// for that register as a bitmask.
    pub fn to_dense(&self) -> Vec<u32> {
        let mut registers = vec![0u32; M];
        for pos in self.bitmap.iter() {
            let reg = (pos / BITS_PER_REG) as usize;
            let bit = pos % BITS_PER_REG;
            if reg < M {
                registers[reg] |= 1u32 << bit;
            }
        }
        registers
    }

    /// Get the register bitmask for a specific register.
    pub fn get_register_bitmap(&self, register: usize) -> u32 {
        if register >= M {
            return 0;
        }
        self.to_dense()[register]
    }

    /// Dump (register, max_trailing_zeros) for non-empty registers.
    pub fn dump_positions(&self) -> Vec<(u32, u32)> {
        self.to_dense()
            .iter()
            .enumerate()
            .filter(|(_, &v)| v > 0)
            .map(|(reg, &value)| {
                let max_tz = 31 - value.leading_zeros();
                (reg as u32, max_tz)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_is_empty() {
        let h = HLLSet::new();
        assert!(h.is_empty());
        assert_eq!(h.popcount(), 0);
        assert_eq!(h.cardinality(), 0.0);
    }

    #[test]
    fn test_add_token_sets_bits() {
        let mut h = HLLSet::new();
        h.add_token(b"hello");
        assert!(h.popcount() > 0);
    }

    #[test]
    fn test_idempotent_insertion() {
        let mut h = HLLSet::new();
        h.add_token(b"hello");
        let after_first = h.popcount();
        h.add_token(b"hello");
        assert_eq!(h.popcount(), after_first); // same bits, no change
    }

    #[test]
    fn test_from_tokens_vs_manual() {
        let a = HLLSet::from_tokens(&["a", "b", "c"]);
        let mut b = HLLSet::new();
        b.add_token(b"a");
        b.add_token(b"b");
        b.add_token(b"c");
        assert_eq!(a.popcount(), b.popcount());
    }

    #[test]
    fn test_roundtrip_dense() {
        let mut h = HLLSet::new();
        for t in &["x", "y", "z"] {
            h.add_token(t.as_bytes());
        }
        let dense = h.to_dense();
        let h2 = HLLSet::from_dense(&dense);
        assert_eq!(h.popcount(), h2.popcount());
    }

    #[test]
    fn test_serialization_roundtrip() {
        let mut h = HLLSet::new();
        h.add_token(b"hello");
        h.add_token(b"world");
        let bytes = h.to_bytes();
        let h2 = HLLSet::from_bytes(&bytes).unwrap();
        assert_eq!(h.popcount(), h2.popcount());
    }
}
