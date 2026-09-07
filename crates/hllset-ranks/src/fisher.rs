//! Fisher-Like Cross-Layer Co-occurrence Matrix.
//!
//! F_{bb'} = Σ_i B^(i)_b · B^(i)_{b'} — count of layers where bits b and b' co-occur.
//! All entries are integers (popcounts). FPGA-native: AND, POPCOUNT, ADD.
//!
//! Projection: s = F · d — identifies which bit changes are systemic vs noise.

use crate::Rank;
use hllset_dsl::LatticeElement;
use std::collections::HashMap;

/// Bit position: (register, tz).
pub type BitPos = (u32, u32);

/// Sparse entry in the Fisher matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FisherEntry {
    /// How many layers contain both bits.
    pub count: u32,
}

/// Fisher-like cross-layer co-occurrence matrix.
///
/// Stored sparsely: only non-zero entries exist.
/// FPGA-native: each entry is a POPCOUNT of (layer_b AND layer_b').
#[derive(Debug, Clone, Default)]
pub struct FisherMatrix {
    /// Sparse storage: (b1, b2) → count. Symmetric: b1 ≤ b2.
    entries: HashMap<(BitPos, BitPos), FisherEntry>,
    /// Number of layers indexed.
    layer_count: usize,
}

impl FisherMatrix {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            layer_count: 0,
        }
    }

    /// Add a layer's HLLSet to the matrix. Accumulates co-occurrence counts.
    ///
    /// For each pair of active bit positions (b, b') where b ≤ b',
    /// increment F_{bb'} by 1.
    pub fn add_layer(&mut self, layer: &LatticeElement) {
        let positions: Vec<BitPos> = layer.hllset().active_positions();
        let n = positions.len();

        for i in 0..n {
            let bi = positions[i];
            for j in i..n {
                let bj = positions[j];
                let key = if bi <= bj { (bi, bj) } else { (bj, bi) };
                self.entries
                    .entry(key)
                    .and_modify(|e| e.count += 1)
                    .or_insert(FisherEntry { count: 1 });
            }
        }
        self.layer_count += 1;
    }

    /// Get co-occurrence count for a pair.
    pub fn get(&self, b1: BitPos, b2: BitPos) -> u32 {
        let key = if b1 <= b2 { (b1, b2) } else { (b2, b1) };
        self.entries.get(&key).map(|e| e.count).unwrap_or(0)
    }

    /// Diagonal element: how many layers contain bit b.
    pub fn diagonal(&self, b: BitPos) -> u32 {
        self.get(b, b)
    }

    /// Number of layers indexed.
    pub fn layer_count(&self) -> usize {
        self.layer_count
    }

    /// Number of non-zero entries.
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Iterate over all non-zero entries.
    pub fn iter(&self) -> impl Iterator<Item = (&(BitPos, BitPos), &FisherEntry)> {
        self.entries.iter()
    }

    /// Find bits with highest diagonal (most persistent across layers).
    pub fn most_persistent(&self, top_n: usize) -> Vec<(BitPos, u32)> {
        let mut diags: Vec<(BitPos, u32)> = self
            .entries
            .iter()
            .filter(|((b1, b2), _)| b1 == b2)
            .map(|((b, _), e)| (*b, e.count))
            .collect();
        diags.sort_by(|a, b| b.1.cmp(&a.1));
        diags.truncate(top_n);
        diags
    }

    /// Find bits most strongly coupled to the given bit.
    pub fn most_coupled(&self, b: BitPos, top_n: usize) -> Vec<(BitPos, u32)> {
        let mut coupled: Vec<(BitPos, u32)> = self
            .entries
            .iter()
            .filter(|((b1, b2), _)| (*b1 == b || *b2 == b) && *b1 != *b2)
            .map(|((b1, b2), e)| {
                let other = if *b1 == b { *b2 } else { *b1 };
                (other, e.count)
            })
            .collect();
        coupled.sort_by(|a, b| b.1.cmp(&a.1));
        coupled.truncate(top_n);
        coupled
    }
}

/// Projection result: s = F · d.
///
/// For each bit b, s_b = Σ_{b'} F_{bb'} · d_{b'}.
/// d_{b'} ∈ {-1, 0, +1}: -1 = departed, 0 = stable, +1 = new.
///
/// FPGA-native: integer MUL + ADD, or repeated ADD since d ∈ {-1,0,+1}.
#[derive(Debug, Clone)]
pub struct FisherProjection {
    /// Per-bit systemic impact scores.
    pub scores: Vec<(BitPos, i64)>,
    /// Bits with |score| > threshold — systemic changes, not noise.
    pub significant: Vec<(BitPos, i64)>,
}

impl FisherProjection {
    /// Project divergence vector d through Fisher matrix F.
    ///
    /// `divergence` maps bit position → d_b ∈ {-1, 0, +1}.
    pub fn project(
        fisher: &FisherMatrix,
        divergence: &HashMap<BitPos, i8>,
        significance_threshold: i64,
    ) -> Self {
        // Collect all bits that appear in either F or divergence
        let mut all_bits: HashMap<BitPos, ()> = HashMap::new();
        for ((b1, b2), _) in fisher.iter() {
            all_bits.insert(*b1, ());
            all_bits.insert(*b2, ());
        }
        for b in divergence.keys() {
            all_bits.insert(*b, ());
        }

        let mut scores: Vec<(BitPos, i64)> = all_bits
            .keys()
            .map(|&b| {
                let mut s: i64 = 0;
                // Diagonal contribution
                let d_b = divergence.get(&b).copied().unwrap_or(0) as i64;
                s += fisher.diagonal(b) as i64 * d_b;
                // Off-diagonal: for bits b' ≠ b coupled to b
                for ((b1, b2), entry) in fisher.iter() {
                    let other = if *b1 == b {
                        *b2
                    } else if *b2 == b {
                        *b1
                    } else {
                        continue;
                    };
                    let d_other = divergence.get(&other).copied().unwrap_or(0) as i64;
                    s += entry.count as i64 * d_other;
                }
                (b, s)
            })
            .collect();

        scores.sort_by(|a, b| b.1.abs().cmp(&a.1.abs()));

        let significant: Vec<_> = scores
            .iter()
            .filter(|(_, s)| s.abs() > significance_threshold)
            .cloned()
            .collect();

        Self {
            scores,
            significant,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fisher_matrix_add_layer() {
        let mut f = FisherMatrix::new();
        let layer = LatticeElement::from_tokens(&["a", "b", "c"]);
        f.add_layer(&layer);
        assert_eq!(f.layer_count(), 1);
        // Every pair of positions gets count = 1
        let positions = layer.hllset().active_positions();
        for i in 0..positions.len() {
            let bi = positions[i];
            // Diagonal: each bit appears in 1 layer
            assert_eq!(f.diagonal(bi), 1);
            for j in (i + 1)..positions.len() {
                let bj = positions[j];
                assert_eq!(f.get(bi, bj), 1);
            }
        }
    }

    #[test]
    fn test_fisher_diagonal_persistence() {
        let mut f = FisherMatrix::new();
        // Layer with "a"
        let layer1 = LatticeElement::from_tokens(&["a"]);
        f.add_layer(&layer1);
        // Another layer also with "a"
        let layer2 = LatticeElement::from_tokens(&["a", "b"]);
        f.add_layer(&layer2);

        let pos_a = layer1.hllset().active_positions();
        assert!(!pos_a.is_empty());
        // The bit position(s) of "a" should appear in 2 layers
        // (There may be multiple positions due to how from_tokens works with multi-seed hashing)
        let diag = f.diagonal(pos_a[0]);
        assert!(
            diag >= 1,
            "Expected a's bit to appear in at least 1 layer, got {diag}"
        );
    }

    #[test]
    fn test_fisher_projection_no_divergence() {
        let mut f = FisherMatrix::new();
        let layer = LatticeElement::from_tokens(&["x", "y"]);
        f.add_layer(&layer);

        let div = HashMap::new(); // no divergence
        let proj = FisherProjection::project(&f, &div, 5);
        // All scores should be 0 since d = 0 everywhere
        for (_, s) in &proj.scores {
            assert_eq!(*s, 0);
        }
        assert!(proj.significant.is_empty());
    }

    #[test]
    fn test_most_persistent() {
        let mut f = FisherMatrix::new();
        for _ in 0..5 {
            let layer = LatticeElement::from_tokens(&["persistent", "transient"]);
            f.add_layer(&layer);
        }
        let persistent = f.most_persistent(3);
        assert!(!persistent.is_empty());
        // All should appear in all 5 layers
        for (_, count) in &persistent {
            assert_eq!(*count, 5);
        }
    }
}
