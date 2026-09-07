//! TF Vector — bit-level term frequency accumulator.
//!
//! The TF vector is the shared frequency table that drives all rank
//! computation. It is a monotonic CRDT: TF values only increase during
//! ingestion, never decrease.
//!
//! # Wire Format (STANDARD.md §2.3)
//!
//! ```text
//! Offset  Size    Field
//! 0       4       N (entry count): uint32 LE (= 32768)
//! 4       262144  TF values: 32768 × float64 LE
//!
//! Total: 262148 bytes fixed
//! Key: system:tf (temporal)
//! ```

use crate::core::hllset::TOTAL_BITS;

/// Number of TF entries (= M × 32 = 1024 × 32).
pub const TF_ENTRIES: usize = TOTAL_BITS as usize;

/// Size of the TF vector in bytes (4 + 32768 × 8).
pub const TFVEC_BYTES: usize = 4 + TF_ENTRIES * 8;

/// A monotonic CRDT bit-level term frequency vector.
///
/// `TFVec[i]` is the accumulated frequency of bit position `i` across
/// all observed HLLSets. TF only increases during ingestion — it is
/// never decremented.
///
/// The TF vector is stored at `system:tf` via `put_tmp`/`get_tmp`.
#[derive(Clone, Debug, PartialEq)]
pub struct TFVec {
    /// TF values: 32768 × f64, indexed by bit position.
    pub values: Vec<f64>,
}

impl Default for TFVec {
    fn default() -> Self {
        Self::new()
    }
}

impl TFVec {
    /// Create a zeroed TF vector.
    pub fn new() -> Self {
        Self {
            values: vec![0.0f64; TF_ENTRIES],
        }
    }

    /// Create from existing values (must have exactly TF_ENTRIES entries).
    pub fn from_values(values: Vec<f64>) -> Option<Self> {
        if values.len() == TF_ENTRIES {
            Some(Self { values })
        } else {
            None
        }
    }

    /// Monotonically increment TF at a given bit position.
    ///
    /// Panics if `position >= TF_ENTRIES`.
    pub fn increment(&mut self, position: usize, delta: f64) {
        assert!(
            position < TF_ENTRIES,
            "TF position {} out of range (max {})",
            position,
            TF_ENTRIES - 1
        );
        self.values[position] += delta;
    }

    /// Increment TF for all bits set in an HLLSet.
    pub fn increment_from_hllset(&mut self, hllset: &crate::core::hllset::HLLSet, delta: f64) {
        for pos in hllset.bitmap().iter() {
            self.increment(pos as usize, delta);
        }
    }

    /// Get TF at a specific bit position.
    pub fn get(&self, position: usize) -> f64 {
        self.values[position]
    }

    /// Get total accumulated TF across all positions.
    pub fn total(&self) -> f64 {
        self.values.iter().sum()
    }

    /// CRDT merge: element-wise maximum with another TFVec.
    ///
    /// TF is monotonic, so the merge is the pointwise maximum.
    pub fn merge(&mut self, other: &TFVec) {
        for (a, &b) in self.values.iter_mut().zip(other.values.iter()) {
            *a = a.max(b);
        }
    }

    /// Serialize to wire format.
    ///
    /// Wire format: 4-byte LE N (always 32768) + N × 8-byte LE f64 values.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(TFVEC_BYTES);
        bytes.extend_from_slice(&(TF_ENTRIES as u32).to_le_bytes());
        for &val in &self.values {
            bytes.extend_from_slice(&val.to_le_bytes());
        }
        bytes
    }

    /// Deserialize from wire format.
    ///
    /// Returns `None` if the byte slice is malformed.
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 4 {
            return None;
        }
        let n = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
        if n != TF_ENTRIES {
            return None;
        }
        if bytes.len() < 4 + n * 8 {
            return None;
        }
        let mut values = Vec::with_capacity(n);
        for i in 0..n {
            let start = 4 + i * 8;
            let _end = start + 8;
            let val = f64::from_le_bytes([
                bytes[start],
                bytes[start + 1],
                bytes[start + 2],
                bytes[start + 3],
                bytes[start + 4],
                bytes[start + 5],
                bytes[start + 6],
                bytes[start + 7],
            ]);
            values.push(val);
        }
        Some(Self { values })
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Whether the vector is all zeros.
    pub fn is_empty(&self) -> bool {
        self.values.iter().all(|&v| v == 0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_is_zeroed() {
        let tf = TFVec::new();
        assert_eq!(tf.len(), TF_ENTRIES);
        assert!(tf.is_empty());
    }

    #[test]
    fn test_increment() {
        let mut tf = TFVec::new();
        tf.increment(0, 1.0);
        tf.increment(0, 2.0);
        assert!((tf.get(0) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_increment_out_of_range_panics() {
        let mut tf = TFVec::new();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            tf.increment(TF_ENTRIES, 1.0);
        }));
        assert!(result.is_err());
    }

    #[test]
    fn test_merge_pointwise_max() {
        let mut a = TFVec::new();
        a.increment(0, 1.0);
        a.increment(1, 2.0);

        let mut b = TFVec::new();
        b.increment(0, 3.0); // b has higher value at position 0
        b.increment(1, 1.0); // a has higher value at position 1

        a.merge(&b);
        assert!((a.get(0) - 3.0).abs() < 1e-10);
        assert!((a.get(1) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_roundtrip_bytes() {
        let mut tf = TFVec::new();
        tf.increment(0, 1.5);
        tf.increment(100, 42.0);
        tf.increment(32767, 0.125);

        let bytes = tf.to_bytes();
        assert_eq!(bytes.len(), TFVEC_BYTES);

        let tf2 = TFVec::from_bytes(&bytes).unwrap();
        assert_eq!(tf.values, tf2.values);
    }

    #[test]
    fn test_from_bytes_wrong_size_rejected() {
        let bad_bytes = vec![0u8; 10];
        assert!(TFVec::from_bytes(&bad_bytes).is_none());
    }

    #[test]
    fn test_from_values_wrong_len_rejected() {
        assert!(TFVec::from_values(vec![1.0]).is_none());
    }

    #[test]
    fn test_total() {
        let mut tf = TFVec::new();
        tf.increment(0, 1.0);
        tf.increment(1, 2.0);
        assert!((tf.total() - 3.0).abs() < 1e-10);
    }
}
