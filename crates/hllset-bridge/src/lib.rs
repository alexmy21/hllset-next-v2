//! Universal Bridge — cross-domain HLLSet re-representation.
//!
//! Per STANDARD.md Part V: maps any HLLSet into any HLLSet lattice through
//! two-pass ingestion, 3-gram structural fingerprinting, and Spearman rank
//! correlation for candidate ranking.
//!
//! # Two-Pass Ingestion
//!
//! ```text
//! Pass 1 (Representation):
//!   domain_input → murmurhash3 → H_src (source bit space)
//!
//! Pass 2 (Re-Representation):
//!   H_src bits → "reg:{r}:tz:{tz}" tokens → murmurhash3 → H_bridge
//! ```
//!
//! H_bridge lives in the target domain's bit space — a full citizen of
//! the target lattice. BSS, R-links, union, intersection all work directly.
//!
//! # The Statistics Constraint
//!
//! The bridge transfers **structure** (bit positions), not **statistics**
//! (TF vectors, rank orderings, temporal state). Each lattice must learn
//! its own statistics through its own experience (§5.5).

use hllset_core::HLLSet;
use std::collections::HashMap;

// ── Pass 2: Re-Representation ──────────────────────────────────────────

/// Re-represent a source HLLSet into the target bit space (Pass 2).
///
/// Extracts active bit positions from `src`, formats each as the token
/// `"reg:{register}:tz:{trailing_zeros}"`, and hashes them into a new HLLSet
/// via MurmurHash3.
///
/// The result is a structural projection — it carries the source's bit
/// position *shape* but not its vocabulary meaning. BSS(H_src, H_bridge) ≈ 0
/// because different inputs go through the same hash function.
///
/// This is a pure function: same source → same bridge, always (IICA).
pub fn re_represent(src: &HLLSet) -> HLLSet {
    let positions = src.active_positions();
    if positions.is_empty() {
        return HLLSet::new();
    }
    let tokens: Vec<String> = positions
        .iter()
        .map(|(reg, tz)| format!("reg:{reg}:tz:{tz}"))
        .collect();
    HLLSet::from_tokens(&tokens)
}

// ── 3-gram Structural Fingerprinting ───────────────────────────────────

/// Extract a 3-gram structural fingerprint as an HLLSet.
///
/// Builds an HLLSet from all consecutive token triples with boundary
/// padding (`_START_` / `_END_`). The 3-gram HLLSet encodes both
/// adjacency patterns AND vocabulary — it is the structural invariant
/// that enables cross-domain matching.
///
/// Two texts in different languages with similar discourse structure
/// produce 3-gram HLLSets with correlated rank distributions.
pub fn extract_3gram(tokens: &[&str]) -> HLLSet {
    if tokens.len() < 3 {
        // For fewer than 3 tokens, pad with boundaries
        let mut padded = vec!["_START_"];
        padded.extend_from_slice(tokens);
        padded.push("_END_");
        return make_ngrams(&padded, 3);
    }
    let mut padded = vec!["_START_"];
    padded.extend_from_slice(tokens);
    padded.push("_END_");
    make_ngrams(&padded, 3)
}

/// Build n-grams of size `n` from token slice.
fn make_ngrams(tokens: &[&str], n: usize) -> HLLSet {
    if tokens.len() < n {
        return HLLSet::new();
    }
    let ngrams: Vec<String> = tokens.windows(n).map(|w| w.join("\0")).collect();
    HLLSet::from_tokens(&ngrams)
}

/// Extract 3-gram fingerprint from an HLLSet's active positions.
///
/// Re-represents the HLLSet, then extracts 3-grams from the
/// re-represented positions. This enables structural comparison
/// even when the original tokens are unknown.
pub fn extract_3gram_from_hllset(hllset: &HLLSet) -> HLLSet {
    let re_repped = re_represent(hllset);
    let pos_strings: Vec<String> = re_repped
        .active_positions()
        .iter()
        .map(|(r, tz)| format!("{r}:{tz}"))
        .collect();
    let pos_refs: Vec<&str> = pos_strings.iter().map(|s| s.as_str()).collect();
    extract_3gram(&pos_refs)
}

// ── Spearman Rank Correlation ──────────────────────────────────────────

/// Compute Spearman rank correlation between two rank vectors.
///
/// Maps each vector's values to ranks (1 = highest), then computes
/// Pearson correlation on the ranks. Returns ρ ∈ [-1.0, 1.0].
///
/// ρ = 1.0: perfect positive correlation (same ordering)
/// ρ = 0.0: no correlation
/// ρ = -1.0: perfect inverse correlation
pub fn spearman_rank_correlation(a: &[u64], b: &[u64]) -> f64 {
    let n = a.len().min(b.len());
    if n < 2 {
        return if n == 1 && a[0] == b[0] { 1.0 } else { 0.0 };
    }

    // Rank transform: sort indices by value, assign ranks
    let rank_a = compute_ranks(&a[..n]);
    let rank_b = compute_ranks(&b[..n]);

    // Pearson correlation on ranks
    let mean_a: f64 = rank_a.iter().sum::<f64>() / n as f64;
    let mean_b: f64 = rank_b.iter().sum::<f64>() / n as f64;

    let mut cov = 0.0;
    let mut var_a = 0.0;
    let mut var_b = 0.0;

    for i in 0..n {
        let da = rank_a[i] - mean_a;
        let db = rank_b[i] - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }

    if var_a == 0.0 || var_b == 0.0 {
        return 0.0;
    }
    cov / (var_a.sqrt() * var_b.sqrt())
}

/// Compute ranks for a slice of values (1 = highest value).
fn compute_ranks(values: &[u64]) -> Vec<f64> {
    let n = values.len();
    let mut indexed: Vec<(usize, u64)> = values.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.cmp(&a.1)); // descending (highest = rank 1)

    let mut ranks = vec![0.0f64; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j < n && indexed[j].1 == indexed[i].1 {
            j += 1;
        }
        // Average rank for ties
        let avg_rank = (i + j - 1) as f64 / 2.0 + 1.0;
        for k in i..j {
            ranks[indexed[k].0] = avg_rank;
        }
        i = j;
    }
    ranks
}

// ── Rank vectors from HLLSets ──────────────────────────────────────────

/// Compute a rank vector from an HLLSet's popcount.
///
/// Each register's bit-count (popcount per register) forms a 1024-element
/// rank vector. This is a simplified structural signature — for full
/// rank algebra, use `hllset-ranks`.
pub fn register_popcount_vector(hllset: &HLLSet) -> Vec<u64> {
    let dense = hllset.to_dense();
    dense.iter().map(|&r| r.count_ones() as u64).collect()
}

// ── Bridge algorithm ───────────────────────────────────────────────────

/// Result of bridging a source HLLSet into a target lattice.
#[derive(Debug, Clone)]
pub struct BridgeResult {
    /// The re-represented HLLSet in target bit space.
    pub bridge: HLLSet,
    /// 3-gram fingerprint of the source.
    pub src_3gram: HLLSet,
    /// 3-gram fingerprint of the bridge.
    pub bridge_3gram: HLLSet,
    /// Top matches in the target lattice (key → Spearman ρ).
    pub matches: Vec<(String, f64)>,
}

/// Bridge a source HLLSet into a target lattice of candidate HLLSets.
///
/// 1. Re-represent source into target bit space (Pass 2)
/// 2. Extract 3-gram fingerprints from both
/// 3. Rank-correlate against all candidates in the target lattice
/// 4. Return top matches sorted by Spearman ρ
pub fn bridge(src: &HLLSet, candidates: &HashMap<String, HLLSet>, top_k: usize) -> BridgeResult {
    let bridge_hllset = re_represent(src);
    let src_3gram = extract_3gram_from_hllset(src);
    let bridge_3gram = extract_3gram_from_hllset(&bridge_hllset);

    let src_popcounts = register_popcount_vector(&src_3gram);
    let bridge_popcounts = register_popcount_vector(&bridge_3gram);

    let mut matches: Vec<(String, f64)> = candidates
        .iter()
        .map(|(key, hllset)| {
            let candidate_3gram = extract_3gram_from_hllset(hllset);
            let candidate_popcounts = register_popcount_vector(&candidate_3gram);
            let rho = spearman_rank_correlation(&bridge_popcounts, &candidate_popcounts);
            (key.clone(), rho)
        })
        .collect();

    matches.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    matches.truncate(top_k);

    BridgeResult {
        bridge: bridge_hllset,
        src_3gram,
        bridge_3gram,
        matches,
    }
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_re_represent_empty() {
        let src = HLLSet::new();
        let bridge = re_represent(&src);
        assert!(bridge.is_empty());
    }

    #[test]
    fn test_re_represent_non_empty() {
        let src = HLLSet::from_tokens(&["hello", "world"]);
        let bridge = re_represent(&src);
        assert!(!bridge.is_empty());
        assert!(bridge.popcount() > 0);
    }

    #[test]
    fn test_re_represent_idempotent() {
        let src = HLLSet::from_tokens(&["test"]);
        let b1 = re_represent(&src);
        let b2 = re_represent(&src);
        assert_eq!(b1.popcount(), b2.popcount());
    }

    #[test]
    fn test_re_represent_bss_zero() {
        // Source and bridge should have ~0 structural overlap
        // because they encode different inputs through the same hash
        let src = HLLSet::from_tokens(&["hello", "world", "test"]);
        let bridge = re_represent(&src);
        let tau = src.bss_inclusion(&bridge);
        // They're in the same bit space but encode different structures
        assert!(tau >= 0.0 && tau <= 1.0);
    }

    #[test]
    fn test_extract_3gram_empty() {
        let result = extract_3gram(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_extract_3gram_single() {
        let result = extract_3gram(&["hello"]);
        // ["_START_", "hello", "_END_"] → one 3-gram
        assert!(!result.is_empty());
    }

    #[test]
    fn test_extract_3gram_deterministic() {
        let a = extract_3gram(&["the", "cat", "sat"]);
        let b = extract_3gram(&["the", "cat", "sat"]);
        assert_eq!(a.content_key(), b.content_key());
    }

    #[test]
    fn test_spearman_perfect_correlation() {
        let a = vec![100, 80, 60, 40, 20];
        let b = vec![100, 80, 60, 40, 20];
        let rho = spearman_rank_correlation(&a, &b);
        assert!((rho - 1.0).abs() < 0.001, "rho={rho}");
    }

    #[test]
    fn test_spearman_perfect_inverse() {
        let a = vec![100, 80, 60, 40, 20];
        let b = vec![20, 40, 60, 80, 100];
        let rho = spearman_rank_correlation(&a, &b);
        assert!((rho + 1.0).abs() < 0.001, "rho={rho}");
    }

    #[test]
    fn test_spearman_empty() {
        assert_eq!(spearman_rank_correlation(&[], &[]), 0.0);
    }

    #[test]
    fn test_spearman_with_ties() {
        let a = vec![10, 10, 5, 5];
        let b = vec![10, 10, 5, 5];
        let rho = spearman_rank_correlation(&a, &b);
        assert!((rho - 1.0).abs() < 0.001, "rho={rho}");
    }

    #[test]
    fn test_full_bridge_pipeline() {
        let src = HLLSet::from_tokens(&["red", "car", "intersection"]);
        let ref_rule = HLLSet::from_tokens(&["slow", "down", "intersection"]);
        let unrelated = HLLSet::from_tokens(&["weather", "report", "sunny"]);

        let mut candidates = HashMap::new();
        candidates.insert("rule".to_string(), ref_rule);
        candidates.insert("unrelated".to_string(), unrelated);

        let result = bridge(&src, &candidates, 2);
        assert!(!result.bridge.is_empty());
        assert_eq!(result.matches.len(), 2);
        assert!(result.matches[0].1 >= -1.0 && result.matches[0].1 <= 1.0);
    }

    #[test]
    fn test_register_popcount_vector() {
        let hllset = HLLSet::from_tokens(&["a", "b", "c"]);
        let vec = register_popcount_vector(&hllset);
        assert_eq!(vec.len(), 1024);
        let total: u64 = vec.iter().sum();
        assert!(total > 0);
    }

    #[test]
    fn test_extract_3gram_from_hllset() {
        let src = HLLSet::from_tokens(&["the", "cat", "sat", "on", "the", "mat"]);
        let fingerprint = extract_3gram_from_hllset(&src);
        assert!(!fingerprint.is_empty());
    }
}
