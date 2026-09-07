//! Level 3: Register Rank — H({bit-R}).
//!
//! A register spans 32 tz positions (0..31). H aggregates their bit-ranks.
//!
//! FPGA-native choices: Sum (ADD over 32 terms) or MaxPool (CMP tree).

use crate::bit::BitRank;
use crate::Rank;

/// Pluggable aggregator: {bit-R[tz]} → reg-R.
///
/// Implementations:
/// - `SumRegAggregator` — sum of all 32 bit-ranks (ADD, FPGA-native)
/// - `MaxPoolAggregator` — strongest bit dominates (CMP tree, FPGA-native)
/// - `ActiveOnlySum` — sum only over bits that are actually set in the HLLSet
pub trait RegisterRankAggregator: Send + Sync {
    /// Aggregate bit ranks within a register.
    /// `bits` may contain fewer than 32 entries if some tz slots are empty.
    fn aggregate(&self, register: u32, bits: &[BitRank]) -> Rank;

    fn name(&self) -> &'static str;
}

/// Sum of all bit-ranks in the register.
///
/// FPGA-native: 32-term integer addition chain.
#[derive(Clone, Copy, Default)]
pub struct SumRegAggregator;

impl RegisterRankAggregator for SumRegAggregator {
    fn aggregate(&self, _register: u32, bits: &[BitRank]) -> Rank {
        bits.iter().map(|b| b.value).sum()
    }
    fn name(&self) -> &'static str {
        "sum"
    }
}

/// Max-pool: the strongest bit's rank IS the register's rank.
///
/// FPGA-native: tree of 31 CMPs.
#[derive(Clone, Copy, Default)]
pub struct MaxPoolAggregator;

impl RegisterRankAggregator for MaxPoolAggregator {
    fn aggregate(&self, _register: u32, bits: &[BitRank]) -> Rank {
        bits.iter().map(|b| b.value).max().unwrap_or(0)
    }
    fn name(&self) -> &'static str {
        "max-pool"
    }
}

/// Sum only over bits that are actually set in the HLLSet's register bitmap.
/// `active_mask` is the register's u32 bitmask from `HLLSet::get_register_bitmap(r)`.
///
/// FPGA-native: AND each bit-rank with its mask bit, then ADD.
#[derive(Clone, Copy, Default)]
pub struct ActiveOnlySum;

impl ActiveOnlySum {
    /// Aggregate with a mask — only bits present in the HLLSet contribute.
    pub fn aggregate_masked(&self, _register: u32, bits: &[BitRank], active_mask: u32) -> Rank {
        bits.iter()
            .filter(|b| (active_mask >> b.tz) & 1 == 1)
            .map(|b| b.value)
            .sum()
    }
}

/// Register rank — the result of H for one register.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct RegisterRank {
    pub register: u32,
    pub value: Rank,
    /// How many of the 32 tz slots had bit-ranks.
    pub active_slots: u32,
}

impl RegisterRank {
    /// Compute register rank from bit-ranks.
    pub fn new(register: u32, bits: &[BitRank], h: &dyn RegisterRankAggregator) -> Self {
        Self {
            register,
            value: h.aggregate(register, bits),
            active_slots: bits.len() as u32,
        }
    }
}

/// Per-register TF ranking: compute rank of each register from a TFVec.
///
/// Maps the 32,768-entry TF vector down to 1,024 register-level values
/// by summing (or max-pooling) the TF values within each register.
///
/// This bridges the TFVec wire format (§2.3) with the five-level rank
/// algebra (§3.2), enabling register-level rank queries without needing
/// token-level LUT access.
pub struct TfRegisterRanker {
    aggregator: Box<dyn RegisterRankAggregator>,
}

impl Default for TfRegisterRanker {
    fn default() -> Self {
        Self {
            aggregator: Box::new(SumRegAggregator),
        }
    }
}

impl TfRegisterRanker {
    /// Create with a specific aggregator.
    pub fn new(aggregator: Box<dyn RegisterRankAggregator>) -> Self {
        Self { aggregator }
    }

    /// Compute register ranks from a TF vector.
    ///
    /// For each of the 1,024 registers, aggregates TF values across all
    /// 32 trailing-zero positions within that register. The aggregation
    /// function (sum, max-pool, etc.) is chosen at construction time.
    pub fn rank_all(&self, tf: &hllset_core::TFVec) -> Vec<RegisterRank> {
        (0..1024u32)
            .map(|reg| self.rank_register(tf, reg))
            .collect()
    }

    /// Compute rank for a single register from the TF vector.
    pub fn rank_register(&self, tf: &hllset_core::TFVec, register: u32) -> RegisterRank {
        let base = (register * 32) as usize;
        let bits: Vec<BitRank> = (0..32u32)
            .map(|tz| {
                let pos = base + tz as usize;
                BitRank {
                    register,
                    tz,
                    value: tf.get(pos) as Rank,
                    token_count: 1,
                }
            })
            .filter(|b| b.value > 0)
            .collect();

        let active_slots = bits.len() as u32;
        RegisterRank {
            register,
            value: self.aggregator.aggregate(register, &bits),
            active_slots,
        }
    }

    /// Get the top-k registers by rank.
    pub fn top_k(&self, tf: &hllset_core::TFVec, k: usize) -> Vec<RegisterRank> {
        let mut ranks = self.rank_all(tf);
        ranks.sort_by(|a, b| b.value.cmp(&a.value));
        ranks.truncate(k);
        ranks
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bit::BitRank;
    use crate::token::{IdentityRankFn, TokenRank};

    fn make_bit(reg: u32, tz: u32, value: Rank) -> BitRank {
        BitRank {
            register: reg,
            tz,
            value,
            token_count: 1,
        }
    }

    #[test]
    fn test_sum_reg() {
        let h = SumRegAggregator;
        let bits: Vec<BitRank> = (0..8).map(|tz| make_bit(0, tz, tz as u64 * 10)).collect();
        // sum = 0+10+20+...+70 = 280
        assert_eq!(h.aggregate(0, &bits), 280);
        assert_eq!(h.aggregate(0, &[]), 0);
    }

    #[test]
    fn test_max_pool() {
        let h = MaxPoolAggregator;
        let bits = vec![make_bit(0, 0, 5), make_bit(0, 1, 42), make_bit(0, 2, 17)];
        assert_eq!(h.aggregate(0, &bits), 42);
    }

    #[test]
    fn test_active_only_sum() {
        let h = ActiveOnlySum;
        let bits = vec![
            make_bit(0, 0, 100), // tz=0
            make_bit(0, 1, 200), // tz=1
            make_bit(0, 3, 400), // tz=3
        ];
        // mask: bits 0 and 3 set → 1 + 8 = 9
        assert_eq!(h.aggregate_masked(0, &bits, 0b1001), 500); // 100 + 400
                                                               // mask: only bit 1 set → 2
        assert_eq!(h.aggregate_masked(0, &bits, 0b0010), 200);
        // mask: no matching bits
        assert_eq!(h.aggregate_masked(0, &bits, 0), 0);
    }
}
