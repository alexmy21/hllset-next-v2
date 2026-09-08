//! HLLSet hashing contract (notebook-08; `hllset-core` parity).
//!
//! One hash function system-wide: MurmurHash3 x64-128 (lower 64 bits) for
//! inscription; [`sha1_hex`] for addresses. This module pins the soldered
//! geometry and insertion rule that every HLLSet implementation must match:
//!
//! - `P = 10`, `M = 1024`, 32 bits per register;
//! - `reg = hash & (M - 1)`, `tz = trailing_zeros(hash >> P).min(31)`.

use sha1::{Digest, Sha1};
use std::io::Cursor;

/// Precision bits `P`: register count `M = 2^P`.
pub const P: u32 = 10;

/// Number of registers (`M = 2^P = 1024`).
pub const M: u32 = 1 << P;

/// Trailing-zero states tracked per register (`0..31`).
pub const BITS_PER_REG: u32 = 32;

/// Total bit positions in the dense plane (`M × 32 = 32,768`).
pub const TOTAL_BITS: u32 = M * BITS_PER_REG;

/// MurmurHash3 x64-128, lower 64 bits, seed 0.
///
/// Bit-exact with Python `mmh3.hash64(data, seed=0, signed=False)[0]` and
/// with `hllset-core::core::hashing::murmur3_hash`.
pub fn murmur3_hash(data: &[u8]) -> u64 {
    murmur3_hash_seeded(data, 0)
}

/// Seeded MurmurHash3 x64-128, lower 64 bits (seed truncated to `u32`).
pub fn murmur3_hash_seeded(data: &[u8], seed: u64) -> u64 {
    let hash = murmur3::murmur3_x64_128(&mut Cursor::new(data), seed as u32)
        .expect("in-memory Cursor cannot fail");
    hash as u64
}

/// SHA-1 hash of arbitrary bytes, hex-encoded (40 lowercase hex chars).
pub fn sha1_hex(data: &[u8]) -> String {
    let mut hasher = Sha1::new();
    hasher.update(data);
    hex::encode(hasher.finalize())
}

/// Decompose a 64-bit hash into `(reg, tz)` per the soldered insertion rule.
pub fn hash_to_position(hash: u64) -> (u32, u32) {
    let reg = (hash as u32) & (M - 1);
    let remaining = hash >> P;
    let tz = if remaining == 0 {
        31
    } else {
        remaining.trailing_zeros().min(31)
    };
    (reg, tz)
}

/// Decompose a token into its `(reg, tz)` position (seed 0).
pub fn token_to_position(token: &[u8]) -> (u32, u32) {
    hash_to_position(murmur3_hash(token))
}

/// Decompose a token into its `(reg, tz)` position using a specific seed.
///
/// Used for multi-seed catalog hashing (homogeneous consensus).
pub fn token_to_position_seeded(token: &[u8], seed: u64) -> (u32, u32) {
    hash_to_position(murmur3_hash_seeded(token, seed))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn soldered_geometry_is_pinned() {
        assert_eq!(P, 10);
        assert_eq!(M, 1024);
        assert_eq!(BITS_PER_REG, 32);
        assert_eq!(TOTAL_BITS, 32_768);
    }

    #[test]
    fn murmur3_matches_python_mmh3_known_answers() {
        // Values produced by `mmh3.hash64(data, signed=False)[0]` (notebook-08).
        assert_eq!(murmur3_hash(b"tid0"), 2_892_634_914_804_110_692);
        assert_eq!(murmur3_hash(b"hello"), 14_688_674_573_012_802_306);
        assert_eq!(murmur3_hash_seeded(b"x", 1), 4_758_147_062_373_591_632);
    }

    #[test]
    fn seeded_zero_equals_unseeded() {
        assert_eq!(murmur3_hash(b"test"), murmur3_hash_seeded(b"test", 0));
        assert_ne!(
            murmur3_hash_seeded(b"x", 0),
            murmur3_hash_seeded(b"x", 1),
            "seeds differ"
        );
    }

    #[test]
    fn sha1_hex_is_40_lowercase() {
        assert_eq!(sha1_hex(b"hello").len(), 40);
        assert!(sha1_hex(b"hello").chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn hash_to_position_matches_core_rule() {
        // Saturated convention: hash 0 → tz 31.
        assert_eq!(hash_to_position(0), (0, 31));
        // reg = hash & 1023; tz over hash >> 10.
        let hash = (1023u64) | (1u64 << 10);
        assert_eq!(hash_to_position(hash), (1023, 0));
    }

    #[test]
    fn notebook_08_collision_pair_is_pinned() {
        // 262 and 48300 collide at (759, 0) under the 4-byte LE inscription.
        assert_eq!(
            token_to_position(&262u32.to_le_bytes()),
            token_to_position(&48_300u32.to_le_bytes())
        );
        assert_eq!(token_to_position(&262u32.to_le_bytes()), (759, 0));
    }
}

/// The structural address of one bit in the HLLSet bit-vector — the hinge of
/// the `{tokens} ↔ {HLLSets}` relation that `ingest` and `materialize` both
/// pass through.
///
/// - **Structural space**: it names *where* a token inscribed itself inside
///   the sketch (`reg * 32 + tz`). The QKV interpretation ("K") is an
///   application-layer reading of this same object.
/// - **Flat `u32` storage**: `bit = reg * BITS_PER_REG + tz`, so `Ord`/`Hash`
///   are bitmap order, and serde is wire-ready as a plain integer.
/// - Invariant: `bit < TOTAL_BITS` (`reg < M` and `tz < BITS_PER_REG`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[repr(transparent)]
pub struct BitAddress {
    bit: u32,
}

impl BitAddress {
    /// Construct from the flat bit id; panics if out of range.
    pub fn new(bit: u32) -> Self {
        assert!(bit < TOTAL_BITS, "bit {bit} out of range (TOTAL_BITS={TOTAL_BITS})");
        Self { bit }
    }

    /// Construct from `(reg, tz)`; panics if out of range.
    pub fn from_reg_tz(reg: u32, tz: u32) -> Self {
        assert!(reg < M, "reg {reg} out of range (M={M})");
        assert!(tz < BITS_PER_REG, "tz {tz} out of range (BITS_PER_REG={BITS_PER_REG})");
        Self {
            bit: reg * BITS_PER_REG + tz,
        }
    }

    /// The flat bit id (`reg * 32 + tz`) — the wire representation.
    pub const fn bit(self) -> u32 {
        self.bit
    }

    /// The register index.
    pub const fn reg(self) -> u32 {
        self.bit / BITS_PER_REG
    }

    /// The trailing-zero state within the register.
    pub const fn tz(self) -> u32 {
        self.bit % BITS_PER_REG
    }

    /// The `(reg, tz)` pair (the legacy tuple convention).
    pub const fn to_pair(self) -> (u32, u32) {
        (self.reg(), self.tz())
    }

    /// Decompose a 64-bit hash into its bit address.
    pub fn from_hash(hash: u64) -> Self {
        let (reg, tz) = hash_to_position(hash);
        Self::from_reg_tz(reg, tz)
    }

    /// The bit address a token inscribes under seed 0.
    pub fn of_token(token: &[u8]) -> Self {
        Self::from_hash(murmur3_hash(token))
    }

    /// The bit address a token inscribes under a specific seed (catalog
    /// multi-seed hashing; the seed is a hash parameter, not part of the
    /// address).
    pub fn of_token_seeded(token: &[u8], seed: u64) -> Self {
        Self::from_hash(murmur3_hash_seeded(token, seed))
    }
}

impl From<(u32, u32)> for BitAddress {
    fn from((reg, tz): (u32, u32)) -> Self {
        Self::from_reg_tz(reg, tz)
    }
}

impl From<BitAddress> for u32 {
    fn from(addr: BitAddress) -> Self {
        addr.bit()
    }
}

#[cfg(test)]
mod bit_address_tests {
    use super::*;

    #[test]
    fn flat_bit_roundtrips_to_pair() {
        let addr = BitAddress::from_reg_tz(759, 0);
        assert_eq!(addr.bit(), 759 * 32);
        assert_eq!(addr.reg(), 759);
        assert_eq!(addr.tz(), 0);
        assert_eq!(addr.to_pair(), (759, 0));
        assert_eq!(BitAddress::from(addr.to_pair()), addr);
        assert_eq!(u32::from(addr), addr.bit());
    }

    #[test]
    fn order_is_bitmap_order() {
        let a = BitAddress::from_reg_tz(0, 31); // bit 31
        let b = BitAddress::from_reg_tz(1, 0); // bit 32
        assert!(a < b, "flat bit order, not reg-major");
    }

    #[test]
    fn derives_from_hash_and_token() {
        assert_eq!(
            BitAddress::from_hash(0),
            BitAddress::from_reg_tz(0, 31),
            "saturated convention"
        );
        let (reg, tz) = token_to_position(b"hello");
        assert_eq!(BitAddress::of_token(b"hello"), BitAddress::from_reg_tz(reg, tz));
    }

    #[test]
    fn notebook_08_collision_pair_is_one_address() {
        assert_eq!(
            BitAddress::of_token(&262u32.to_le_bytes()),
            BitAddress::of_token(&48_300u32.to_le_bytes())
        );
        assert_eq!(BitAddress::of_token(&262u32.to_le_bytes()).to_pair(), (759, 0));
    }

    #[test]
    #[should_panic]
    fn invalid_reg_panics() {
        BitAddress::from_reg_tz(M, 0);
    }
}
