//! CRDT-native distributed infrastructure.
//!
//! HLLSets are both data AND metadata. A node's identity is the union
//! of all HLLSets it holds. Routing is BSS — no routing tables, no consensus.
//!
//! ## Core operations
//!
//! ```text
//! Node fingerprint = ⋃{ HLLSets held by node }              (union, O(1))
//! Relevance score  = BSSτ(request, node_fingerprint)         (inclusion)
//! Multi-response   = R₁ ∪ R₂ ∪ R₃                           (union merge)
//! ```
//!
//! ## Key properties
//!
//! - **No routing table**: fingerprints ARE the routing table
//! - **Any node can answer any query**: just return (request ∩ local_data)
//! - **Client merges**: union of partial responses is idempotent
//! - **Gossip is tiny**: 1024 bits per node fingerprint exchange

use hllset_core::HLLSet;
use std::collections::HashMap;

/// A node's identity: the HLLSet union of all keys it holds.
///
/// Updated in O(1) by unioning new keys on insert.
/// Used by peers for BSS-based routing decisions.
#[derive(Clone, Debug)]
pub struct NodeFingerprint {
    /// Union of all HLLSets held by this node.
    fingerprint: HLLSet,
    /// Number of distinct HLLSets in this fingerprint.
    set_count: usize,
}

impl Default for NodeFingerprint {
    fn default() -> Self {
        Self::new()
    }
}

impl NodeFingerprint {
    /// Create an empty fingerprint.
    pub fn new() -> Self {
        Self {
            fingerprint: HLLSet::new(),
            set_count: 0,
        }
    }

    /// Add a key's HLLSet to the fingerprint.
    ///
    /// O(1) — just bitwise OR of the underlying Roaring bitmap.
    pub fn add(&mut self, hllset: &HLLSet) {
        self.fingerprint.merge(hllset);
        self.set_count += 1;
    }

    /// Remove a key's HLLSet from the fingerprint.
    ///
    /// This requires full recomputation since HLLSet difference
    /// doesn't perfectly remove bits (hash collisions).
    /// Call `rebuild` after bulk removals instead.
    pub fn remove(&mut self, _hllset: &HLLSet) {
        self.set_count = self.set_count.saturating_sub(1);
        // Full recompute on next access
    }

    /// Rebuild the fingerprint from a set of HLLSets.
    pub fn rebuild(&mut self, sets: &[&HLLSet]) {
        self.fingerprint = HLLSet::new();
        for s in sets {
            self.fingerprint.merge(s);
        }
        self.set_count = sets.len();
    }

    /// The raw HLLSet fingerprint.
    pub fn as_hllset(&self) -> &HLLSet {
        &self.fingerprint
    }

    /// Number of distinct sets in this fingerprint.
    pub fn set_count(&self) -> usize {
        self.set_count
    }

    /// BSSτ inclusion: how much of `other` is contained in this fingerprint.
    ///
    /// Returns 1.0 if this fingerprint covers everything in `other`.
    pub fn inclusion_of(&self, other: &HLLSet) -> f64 {
        self.fingerprint.bss_inclusion(other)
    }

    /// BSSρ exclusion: how much of `other` is novel relative to this fingerprint.
    pub fn novelty_vs(&self, other: &HLLSet) -> f64 {
        self.fingerprint.bss_exclusion(other)
    }
}

/// BSS-based router: scores nodes against a request HLLSet.
///
/// # Example
///
/// ```rust
/// use hllset_dsl::distributed::{BSSRouter, NodeFingerprint};
/// use hllset_core::HLLSet;
///
/// let request = HLLSet::from_tokens(&["hello", "world"]);
/// let node_a = NodeFingerprint::new(); // empty
/// let node_b = {
///     let mut f = NodeFingerprint::new();
///     f.add(&request); // node_b has the exact data
///     f
/// };
///
/// let mut router = BSSRouter::new();
/// router.add_node("a", &node_a);
/// router.add_node("b", &node_b);
///
/// let ranked = router.rank(&request, 3);
/// assert_eq!(ranked[0].0, "b"); // b has highest relevance
/// ```
pub struct BSSRouter {
    nodes: HashMap<String, NodeFingerprint>,
}

/// A ranked node entry: (node_id, BSS_inclusion_score).
pub type RankedNode = (String, f64);

impl BSSRouter {
    /// Create an empty router.
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
        }
    }

    /// Register a node with its fingerprint.
    pub fn add_node(&mut self, id: &str, fingerprint: &NodeFingerprint) {
        self.nodes.insert(id.to_string(), fingerprint.clone());
    }

    /// Remove a node.
    pub fn remove_node(&mut self, id: &str) {
        self.nodes.remove(id);
    }

    /// Update a node's fingerprint.
    pub fn update_node(&mut self, id: &str, fingerprint: &NodeFingerprint) {
        self.add_node(id, fingerprint);
    }

    /// Rank nodes by BSSτ inclusion score against the request HLLSet.
    ///
    /// Returns the top `k` nodes sorted by descending relevance.
    /// Nodes with τ > 0 are relevant; τ = 1.0 means the node
    /// has everything in the request.
    pub fn rank(&self, request: &HLLSet, k: usize) -> Vec<RankedNode> {
        let mut scored: Vec<(String, f64)> = self
            .nodes
            .iter()
            .map(|(id, fp)| (id.clone(), fp.inclusion_of(request)))
            .filter(|(_, score)| *score > 0.0)
            .collect();

        // Sort by descending BSSτ (most relevant first)
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scored.truncate(k);
        scored
    }

    /// Get all known nodes.
    pub fn nodes(&self) -> impl Iterator<Item = (&str, &NodeFingerprint)> {
        self.nodes.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Number of known nodes.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the router has any nodes.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

/// Multi-source merge: combine partial responses from multiple nodes.
///
/// Each node returns the intersection of the request with its local data.
/// The client unions all responses — correctness guaranteed by CRDT properties.
pub struct MultiSourceMerge {
    responses: Vec<HLLSet>,
}

impl MultiSourceMerge {
    /// Start collecting responses for a request.
    pub fn new() -> Self {
        Self {
            responses: Vec::new(),
        }
    }

    /// Add a partial response from a node.
    ///
    /// The response is `node_data ∩ request` — what this node
    /// found matching the query.
    pub fn add_response(&mut self, partial: HLLSet) {
        self.responses.push(partial);
    }

    /// Merge all responses into a single HLLSet.
    ///
    /// This is the union of all partial results, equivalent to
    /// (request ∩ node_A) ∪ (request ∩ node_B) ∪ ...
    /// = request ∩ (node_A ∪ node_B ∪ ...)
    pub fn merge(self) -> HLLSet {
        HLLSet::union_all(self.responses)
    }

    /// Number of responses collected.
    pub fn response_count(&self) -> usize {
        self.responses.len()
    }

    /// Check if we have enough responses (all original tokens covered).
    ///
    /// Coverage is complete when the merged result has cardinality
    /// close to the original request.
    pub fn coverage(&self, request: &HLLSet) -> f64 {
        let merged = self.merge_in_place();
        if request.cardinality() == 0.0 {
            return 1.0;
        }
        merged.cardinality() / request.cardinality()
    }

    fn merge_in_place(&self) -> HLLSet {
        HLLSet::union_all(self.responses.clone())
    }
}

impl Default for MultiSourceMerge {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_hllset(tokens: &[&str]) -> HLLSet {
        HLLSet::from_tokens(tokens)
    }

    #[test]
    fn test_fingerprint_add_and_inclusion() {
        let mut fp = NodeFingerprint::new();
        let data = make_hllset(&["hello", "world"]);
        fp.add(&data);

        assert_eq!(fp.set_count(), 1);

        let query = make_hllset(&["hello"]);
        assert!(fp.inclusion_of(&query) > 0.0);

        let empty = HLLSet::new();
        assert_eq!(fp.inclusion_of(&empty), 1.0); // vacuously true
    }

    #[test]
    fn test_router_ranks_by_relevance() {
        let request = make_hllset(&["shared", "unique"]);

        let mut fp_a = NodeFingerprint::new();
        fp_a.add(&make_hllset(&["shared"])); // partial match

        let mut fp_b = NodeFingerprint::new();
        fp_b.add(&request); // full match

        let mut fp_c = NodeFingerprint::new();
        fp_c.add(&make_hllset(&["unrelated"])); // no match

        let mut router = BSSRouter::new();
        router.add_node("a", &fp_a);
        router.add_node("b", &fp_b);
        router.add_node("c", &fp_c);

        let ranked = router.rank(&request, 3);
        assert_eq!(ranked[0].0, "b"); // b is most relevant
        assert!(ranked.len() >= 1); // c might be filtered out (score == 0)
        if ranked.len() >= 2 {
            assert_eq!(ranked[1].0, "a");
        }
    }

    #[test]
    fn test_multi_source_merge() {
        let request = make_hllset(&["a", "b", "c", "d"]);

        // Node 1 has a,b
        let r1 = request.intersection(&make_hllset(&["a", "b", "x"]));
        // Node 2 has c,d
        let r2 = request.intersection(&make_hllset(&["c", "d", "y"]));

        let mut merge = MultiSourceMerge::new();
        merge.add_response(r1);
        merge.add_response(r2);

        let result = merge.merge();
        assert!(result.cardinality() >= 3.0); // a,b,c,d recovered
    }

    #[test]
    fn test_idempotent_merge() {
        let request = make_hllset(&["x"]);
        let r = request.clone();

        let mut merge1 = MultiSourceMerge::new();
        merge1.add_response(r.clone());
        merge1.add_response(r.clone()); // duplicate
        let result1 = merge1.merge();

        let mut merge2 = MultiSourceMerge::new();
        merge2.add_response(r);
        let result2 = merge2.merge();

        // Duplicate responses → same result (union is idempotent)
        assert_eq!(result1.popcount(), result2.popcount());
    }
}
