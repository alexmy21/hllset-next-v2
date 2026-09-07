//! Materialization: reconstruct token sequences from HLLSet fingerprints.
//!
//! Materialization is the inverse of tokenization — given an HLLSet and a
//! TokenLUT (reverse index from bit positions to candidate tokens), it
//! reconstructs the most likely original token sequence.
//!
//! # Strategies
//!
//! | Strategy | Use case | How it works |
//! |----------|----------|--------------|
//! | `InLUT` | Simple lookup | Each set bit → all candidates in LUT |
//! | `NgramCrossValidate` | N-gram tokenized data | Validate bigrams by checking unigram presence |
//! | `DeBruijnReconstruct` | Ordered sequences | Build graph, find Eulerian path |
//!
//! Since HLLSets are lossy (hash collisions), materialization is probabilistic.

use hllset_core::core::hashing::token_to_position;
use hllset_core::{HLLSet, M};
use std::collections::{BTreeMap, HashMap, HashSet};

// ── TokenLUT ─────────────────────────────────────────────────────────────

/// A reverse index mapping bit positions to candidate tokens.
///
/// Each (register, trailing_zeros) pair can correspond to multiple tokens
/// due to hash collisions (different tokens may hash to the same position).
#[derive(Clone, Debug, Default)]
pub struct TokenLUT {
    /// (reg, zeros) → set of candidate tokens
    index: HashMap<(u32, u32), Vec<Vec<u8>>>,
    /// Token → (reg, zeros) — forward index for cross-validation
    forward: HashMap<Vec<u8>, (u32, u32)>,
}

impl TokenLUT {
    /// Create an empty LUT.
    pub fn new() -> Self {
        Self {
            index: HashMap::new(),
            forward: HashMap::new(),
        }
    }

    /// Register a token in the LUT — records its (reg, zeros) position.
    ///
    /// Multiple tokens can map to the same position (hash collisions).
    pub fn insert(&mut self, token: Vec<u8>) {
        let pos = token_to_position(&token);
        self.index.entry(pos).or_default().push(token.clone());
        self.forward.insert(token, pos);
    }

    /// Register multiple tokens.
    pub fn insert_all<I, B>(&mut self, tokens: I)
    where
        I: IntoIterator<Item = B>,
        B: AsRef<[u8]>,
    {
        for t in tokens {
            self.insert(t.as_ref().to_vec());
        }
    }

    /// Look up candidates for a given (reg, zeros) position.
    pub fn get(&self, reg: u32, zeros: u32) -> Option<&Vec<Vec<u8>>> {
        self.index.get(&(reg, zeros))
    }

    /// Get the position for a token.
    pub fn position_of(&self, token: &[u8]) -> Option<(u32, u32)> {
        self.forward.get(token).copied()
    }

    /// Number of unique positions in the LUT.
    pub fn len(&self) -> usize {
        self.index.len()
    }

    /// Whether the LUT is empty.
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// Batch lookup: collect all candidates that match any of the given positions.
    ///
    /// Iterates the LUT once and checks each entry against the position set —
    /// O(LUT_size) instead of O(bit_set_size × hash_cost).
    pub fn collect_candidates(&self, positions: &HashSet<(u32, u32)>) -> Vec<Vec<u8>> {
        let mut result = Vec::new();
        for (pos, tokens) in &self.index {
            if positions.contains(pos) {
                result.extend(tokens.iter().cloned());
            }
        }
        result
    }

    /// Build a LUT from a sequence of tokens.
    pub fn from_tokens<I, B>(tokens: I) -> Self
    where
        I: IntoIterator<Item = B>,
        B: AsRef<[u8]>,
    {
        let mut lut = Self::new();
        lut.insert_all(tokens);
        lut
    }
}

// ── DenseLUT: hash-free 1024×32 lookup table ──────────────────────────

/// A dense, hash-free lookup table for FPGA-native materialization.
///
/// `DenseLUT` stores tokens in a fixed-size `1024 × 32` array:
///
/// ```text
/// table[reg][zeros] → Option<Vec<Vec<u8>>>
/// ```
///
/// Where `reg` (0..1023) and `zeros` (0..31) come from MurmurHash3
/// decomposition. Lookup is O(1) direct array indexing — no HashMap hashing.
///
/// # FPGA readiness
///
/// This is the natural FPGA representation: a 1024×32 BRAM with
/// variable-width entries. The LUT can be loaded as a bitstream and
/// queried in a single cycle.
#[derive(Clone, Debug)]
pub struct DenseLUT {
    /// table[reg][zeros] = list of candidate tokens
    table: Vec<[Option<Vec<Vec<u8>>>; 32]>,
}

impl Default for DenseLUT {
    fn default() -> Self {
        Self::new()
    }
}

impl DenseLUT {
    /// Create an empty dense LUT (all cells = None).
    pub fn new() -> Self {
        Self {
            table: (0..M).map(|_| std::array::from_fn(|_| None)).collect(),
        }
    }

    /// Register a token — hashes it, gets (reg, zeros), pushes to that cell.
    pub fn insert(&mut self, token: Vec<u8>) {
        let (reg, zeros) = token_to_position(&token);
        let cell = &mut self.table[reg as usize][zeros as usize];
        cell.get_or_insert_with(Vec::new).push(token);
    }

    /// Register multiple tokens.
    pub fn insert_all<I, B>(&mut self, tokens: I)
    where
        I: IntoIterator<Item = B>,
        B: AsRef<[u8]>,
    {
        for t in tokens {
            self.insert(t.as_ref().to_vec());
        }
    }

    /// Look up candidates at (reg, zeros) — O(1) direct array access.
    pub fn get(&self, reg: u32, zeros: u32) -> Option<&Vec<Vec<u8>>> {
        self.table[reg as usize][zeros as usize].as_ref()
    }

    /// Batch lookup: collect all candidates matching any of the given positions.
    ///
    /// Iterates only the positions, each lookup is direct array index —
    /// no hashing, O(positions × 1).
    pub fn collect_candidates(&self, positions: &HashSet<(u32, u32)>) -> Vec<Vec<u8>> {
        let mut result = Vec::new();
        for &(reg, zeros) in positions {
            if let Some(tokens) = self.get(reg, zeros) {
                result.extend(tokens.iter().cloned());
            }
        }
        result
    }

    /// Total number of occupied cells.
    pub fn occupied_cells(&self) -> usize {
        self.table
            .iter()
            .flat_map(|row| row.iter())
            .filter(|cell| cell.is_some())
            .count()
    }

    /// Whether the LUT has no tokens.
    pub fn is_empty(&self) -> bool {
        self.table
            .iter()
            .flat_map(|row| row.iter())
            .all(|cell| cell.is_none())
    }

    /// Build a dense LUT from tokens.
    pub fn from_tokens<I, B>(tokens: I) -> Self
    where
        I: IntoIterator<Item = B>,
        B: AsRef<[u8]>,
    {
        let mut lut = Self::new();
        lut.insert_all(tokens);
        lut
    }
}

// ── Materialization result ────────────────────────────────────────────────

/// Result of a materialization operation.
#[derive(Clone, Debug, PartialEq)]
pub struct MaterializedResult {
    /// Reconstructed token sequences (may be multiple candidates).
    pub candidates: Vec<Vec<Vec<u8>>>,
    /// The strategy that produced this result.
    pub strategy: String,
    /// Confidence — fraction of HLLSet bits that were resolved.
    pub confidence: f64,
}

impl MaterializedResult {
    /// Flatten all candidate sequences into a single deduplicated set of tokens.
    pub fn flat_tokens(&self) -> Vec<Vec<u8>> {
        let mut seen = HashSet::new();
        let mut result = Vec::new();
        for seq in &self.candidates {
            for token in seq {
                if seen.insert(token.clone()) {
                    result.push(token.clone());
                }
            }
        }
        result
    }

    /// Convert flat tokens to strings (where valid UTF-8).
    pub fn flat_strings(&self) -> Vec<String> {
        self.flat_tokens()
            .into_iter()
            .map(|t| String::from_utf8_lossy(&t).to_string())
            .collect()
    }

    /// Total number of candidate sequences.
    pub fn sequence_count(&self) -> usize {
        self.candidates.len()
    }
}

// ── Materialization strategies ────────────────────────────────────────────

/// Materialize using simple LUT lookup.
///
/// For each bit set in the HLLSet, find all tokens in the LUT that map to
/// that (reg, zeros). Results are returned as individual candidate sequences.
pub fn materialize_inlut(hllset: &HLLSet, lut: &TokenLUT) -> MaterializedResult {
    let positions = hllset.active_positions();
    let total_bits = positions.len() as f64;
    if total_bits == 0.0 {
        return MaterializedResult {
            candidates: vec![],
            strategy: "InLUT".to_string(),
            confidence: 1.0,
        };
    }

    let bit_set: HashSet<(u32, u32)> = positions.into_iter().collect();

    // Batch lookup: single pass through LUT checking against bit_set
    let candidates = lut.collect_candidates(&bit_set);
    let resolved = candidates.len() as u64;

    MaterializedResult {
        candidates: vec![candidates],
        strategy: "InLUT".to_string(),
        confidence: resolved as f64 / total_bits,
    }
}

/// Materialize using n-gram cross-validation.
///
/// For an HLLSet created with bigrams (or higher n-grams), this strategy:
/// 1. Finds all n-gram candidates via LUT lookup
/// 2. Validates each n-gram by checking its constituent tokens against the LUT
/// 3. Returns only validated n-grams
///
/// An n-gram "a\0b" is valid iff both "a" and "b" are in the LUT and their
/// positions correspond to bits set in the HLLSet.
pub fn materialize_ngram_cross_validate(hllset: &HLLSet, lut: &TokenLUT) -> MaterializedResult {
    let positions = hllset.active_positions();
    if positions.is_empty() {
        return MaterializedResult {
            candidates: vec![],
            strategy: "NgramCrossValidate".to_string(),
            confidence: 1.0,
        };
    }

    let total_bits = positions.len() as f64;
    let bit_set: HashSet<(u32, u32)> = positions.into_iter().collect();
    let mut resolved = 0u64;
    let mut validated_tokens: Vec<Vec<u8>> = Vec::new();

    for (reg, zeros) in &bit_set {
        if let Some(tokens) = lut.get(*reg, *zeros) {
            resolved += 1;
            for token in tokens {
                // Check if this is an n-gram (contains NUL separator)
                if token.contains(&0u8) {
                    // Split on NUL and validate each constituent
                    let parts: Vec<&[u8]> = token.split(|b| *b == 0u8).collect();
                    let all_valid = parts.iter().all(|part| {
                        if part.is_empty() {
                            return false;
                        }
                        lut.position_of(part)
                            .map(|p| bit_set.contains(&p))
                            .unwrap_or(false)
                    });

                    if all_valid {
                        validated_tokens.push(token.clone());
                    }
                } else {
                    // Unigram — always valid if in LUT
                    validated_tokens.push(token.clone());
                }
            }
        }
    }

    MaterializedResult {
        candidates: vec![validated_tokens],
        strategy: "NgramCrossValidate".to_string(),
        confidence: resolved as f64 / total_bits,
    }
}

/// De Bruijn graph reconstruction from n-gram tokens.
///
/// Given validated n-grams (typically bigrams) with boundary markers,
/// this builds a De Bruijn graph and finds the most likely path:
///
/// 1. Each (n-1)-gram prefix/suffix becomes a node
/// 2. Each n-gram becomes an edge from prefix to suffix
/// 3. Find an Eulerian path through the graph
///
/// If START/END markers are present, the path is constrained to begin
/// at START and end at END.
pub fn materialize_debruijn(
    hllset: &HLLSet,
    lut: &TokenLUT,
    start_marker: &[u8],
    end_marker: &[u8],
) -> MaterializedResult {
    let positions = hllset.active_positions();
    if positions.is_empty() {
        return MaterializedResult {
            candidates: vec![],
            strategy: "DeBruijnReconstruct".to_string(),
            confidence: 1.0,
        };
    }

    let total_bits = positions.len() as f64;
    let bit_set: HashSet<(u32, u32)> = positions.into_iter().collect();
    let mut resolved = 0u64;

    // Step 1: Collect validated n-grams
    let mut edges: Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> = Vec::new();
    // (prefix, suffix, full_ngram)

    for (reg, zeros) in &bit_set {
        if let Some(tokens) = lut.get(*reg, *zeros) {
            resolved += 1;
            for token in tokens {
                if let Some((prefix, suffix)) = split_at_nul(token) {
                    edges.push((prefix, suffix.clone(), token.clone()));
                }
            }
        }
    }

    if edges.is_empty() {
        return MaterializedResult {
            candidates: vec![],
            strategy: "DeBruijnReconstruct".to_string(),
            confidence: resolved as f64 / total_bits,
        };
    }

    // Step 2: Build adjacency list
    let mut adj: BTreeMap<Vec<u8>, Vec<(Vec<u8>, Vec<u8>)>> = BTreeMap::new();
    for (prefix, suffix, full) in &edges {
        adj.entry(prefix.clone())
            .or_default()
            .push((suffix.clone(), full.clone()));
    }

    // Step 3: Find path — greedy DFS from start marker
    let path = find_path(&adj, start_marker, end_marker);

    // Step 4: Reconstruct token sequence from path
    let mut tokens: Vec<Vec<u8>> = Vec::new();
    if let Some(ref p) = path {
        for node in p {
            tokens.push(node.clone());
        }
    }

    let candidates = if tokens.is_empty() {
        // Fallback: just return all unique tokens found
        let mut unique: Vec<Vec<u8>> = Vec::new();
        let mut seen = HashSet::new();
        for (_, _, full) in &edges {
            if seen.insert(full.clone()) {
                unique.push(full.clone());
            }
        }
        vec![unique]
    } else {
        vec![tokens]
    };

    MaterializedResult {
        candidates,
        strategy: "DeBruijnReconstruct".to_string(),
        confidence: resolved as f64 / total_bits,
    }
}

// ── Helpers ────────────────────────────────────────────────────────────────

/// Split a byte sequence at the first NUL separator.
fn split_at_nul(data: &[u8]) -> Option<(Vec<u8>, Vec<u8>)> {
    data.iter()
        .position(|&b| b == 0u8)
        .map(|pos| (data[..pos].to_vec(), data[pos + 1..].to_vec()))
}

/// Find a greedy path through the De Bruijn graph from start to end.
fn find_path(
    adj: &BTreeMap<Vec<u8>, Vec<(Vec<u8>, Vec<u8>)>>,
    start: &[u8],
    end: &[u8],
) -> Option<Vec<Vec<u8>>> {
    let mut path: Vec<Vec<u8>> = Vec::new();
    let mut current = start.to_vec();
    let mut visited = HashSet::new();

    path.push(current.clone());

    for _ in 0..1000 {
        // safety limit
        if current == end {
            return Some(path);
        }

        if let Some(nexts) = adj.get(&current) {
            // Find first unvisited next node
            let mut found = false;
            for (next, _full) in nexts {
                let edge_key = (current.clone(), next.clone());
                if !visited.contains(&edge_key) {
                    visited.insert(edge_key);
                    path.push(next.clone());
                    current = next.clone();
                    found = true;
                    break;
                }
            }
            if !found {
                // Dead end — return what we have
                return Some(path);
            }
        } else {
            return Some(path);
        }
    }

    Some(path)
}

// ── CatalogLUT: multi-seed lookup table for homogeneous data ───────────

/// Seeds used for multi-seed cross-validation (G1 convention).
pub const CATALOG_SEEDS: [u64; 3] = [0, 1, 2];

/// Minimum seeds required for consensus validation (≥ 2 of 3).
pub const MIN_CONSENSUS_SEEDS: usize = 2;

/// A reverse index for catalog (homogeneous/enumerable) data.
///
/// Unlike `TokenLUT` which uses a single hash seed, `CatalogLUT` hashes
/// each value with multiple seeds (default: 0, 1, 2). During materialization,
/// the homogeneous consensus requires that a value appears at ≥ 2 of its
/// 3 hashed positions.
#[derive(Clone, Debug, Default)]
pub struct CatalogLUT {
    /// Value → list of (reg, zeros) positions, one per seed
    forward: HashMap<Vec<u8>, Vec<(u32, u32)>>,
    /// (reg, zeros) → set of candidate values
    reverse: HashMap<(u32, u32), Vec<Vec<u8>>>,
    /// Seeds used (default: [0, 1, 2])
    seeds: Vec<u64>,
}

impl CatalogLUT {
    /// Create an empty catalog LUT with default seeds [0, 1, 2].
    pub fn new() -> Self {
        Self {
            forward: HashMap::new(),
            reverse: HashMap::new(),
            seeds: CATALOG_SEEDS.to_vec(),
        }
    }

    /// Use custom seeds (must have at least 2 for consensus).
    pub fn with_seeds(mut self, seeds: &[u64]) -> Self {
        assert!(seeds.len() >= 2, "need at least 2 seeds for consensus");
        self.seeds = seeds.to_vec();
        self
    }

    /// Register a catalog value — hashes it with all seeds.
    pub fn insert(&mut self, value: Vec<u8>) {
        let positions: Vec<(u32, u32)> = self
            .seeds
            .iter()
            .map(|&seed| hllset_core::core::hashing::token_to_position_seeded(&value, seed))
            .collect();

        for &pos in &positions {
            self.reverse.entry(pos).or_default().push(value.clone());
        }

        self.forward.insert(value, positions);
    }

    /// Register multiple values.
    pub fn insert_all<I, B>(&mut self, values: I)
    where
        I: IntoIterator<Item = B>,
        B: AsRef<[u8]>,
    {
        for v in values {
            self.insert(v.as_ref().to_vec());
        }
    }

    /// Look up candidate values at a given (reg, zeros) position.
    pub fn get(&self, reg: u32, zeros: u32) -> Option<&Vec<Vec<u8>>> {
        self.reverse.get(&(reg, zeros))
    }

    /// Get all positions for a value.
    pub fn positions_of(&self, value: &[u8]) -> Option<&Vec<(u32, u32)>> {
        self.forward.get(value)
    }

    /// Number of catalog values registered.
    pub fn len(&self) -> usize {
        self.forward.len()
    }

    /// Whether the LUT is empty.
    pub fn is_empty(&self) -> bool {
        self.forward.is_empty()
    }

    /// Batch lookup: collect all candidate values matching any of the given positions.
    ///
    /// Single pass through the reverse index — O(LUT_size) instead of per-position hashing.
    pub fn collect_candidates(&self, positions: &HashSet<(u32, u32)>) -> Vec<Vec<u8>> {
        let mut result = Vec::new();
        for (pos, values) in &self.reverse {
            if positions.contains(pos) {
                result.extend(values.iter().cloned());
            }
        }
        result
    }

    /// Create from an iterator of values.
    pub fn from_values<I, B>(values: I) -> Self
    where
        I: IntoIterator<Item = B>,
        B: AsRef<[u8]>,
    {
        let mut lut = Self::new();
        lut.insert_all(values);
        lut
    }
}

// ── Homogeneous Consensus materialization ──────────────────────────────

/// Materialize using homogeneous (multi-seed) consensus.
///
/// For each value in the catalog LUT:
/// 1. Check how many of its seeded positions have bits set in the HLLSet
/// 2. If ≥ `min_seeds` positions are set, accept the value as "present"
/// 3. Returns all consensus-validated values
///
/// This provides collision resistance: a false positive from one seed
/// is unlikely to coincide with false positives from other seeds.
pub fn materialize_homogeneous_consensus(hllset: &HLLSet, lut: &CatalogLUT) -> MaterializedResult {
    let positions = hllset.active_positions();
    if positions.is_empty() || lut.is_empty() {
        return MaterializedResult {
            candidates: vec![],
            strategy: "HomogeneousConsensus".to_string(),
            confidence: if positions.is_empty() { 1.0 } else { 0.0 },
        };
    }

    let bit_set: HashSet<(u32, u32)> = positions.into_iter().collect();
    let total_bits = bit_set.len() as f64;

    let min_seeds = std::cmp::max(1, lut.seeds.len() - 1); // ≥ 2 of 3 by default

    // Batch lookup: single pass through LUT to collect all candidates
    let all_candidates = lut.collect_candidates(&bit_set);
    let candidate_values: HashSet<Vec<u8>> = all_candidates.into_iter().collect();

    // Validate each candidate by seed consensus
    let mut validated = Vec::new();
    let mut resolved = 0u64;

    for value in &candidate_values {
        if let Some(positions) = lut.positions_of(value) {
            let seeds_matched = positions.iter().filter(|p| bit_set.contains(p)).count();
            if seeds_matched >= min_seeds {
                resolved += seeds_matched as u64;
                validated.push(value.clone());
            }
        }
    }

    let mut result = Vec::new();
    if !validated.is_empty() {
        result.push(validated);
    }

    MaterializedResult {
        candidates: result,
        strategy: "HomogeneousConsensus".to_string(),
        confidence: if total_bits > 0.0 {
            (resolved as f64 / total_bits).min(1.0)
        } else {
            1.0
        },
    }
}

// ── Higher-level API ──────────────────────────────────────────────────────

/// A materializer that holds a LUT and can apply different strategies.
#[derive(Clone, Debug)]
pub struct Materializer {
    lut: TokenLUT,
    start_marker: Vec<u8>,
    end_marker: Vec<u8>,
}

impl Materializer {
    /// Create a new materializer with an empty LUT.
    pub fn new() -> Self {
        Self {
            lut: TokenLUT::new(),
            start_marker: b"<S>".to_vec(),
            end_marker: b"</S>".to_vec(),
        }
    }

    /// Set the boundary markers for De Bruijn reconstruction.
    pub fn with_boundaries(mut self, start: &[u8], end: &[u8]) -> Self {
        self.start_marker = start.to_vec();
        self.end_marker = end.to_vec();
        self
    }

    /// Register tokens in the LUT.
    pub fn insert<I, B>(&mut self, tokens: I)
    where
        I: IntoIterator<Item = B>,
        B: AsRef<[u8]>,
    {
        self.lut.insert_all(tokens);
    }

    /// Access the underlying LUT.
    pub fn lut(&self) -> &TokenLUT {
        &self.lut
    }

    /// Mutable access to the LUT.
    pub fn lut_mut(&mut self) -> &mut TokenLUT {
        &mut self.lut
    }

    /// Materialize using InLUT strategy.
    pub fn inlut(&self, hllset: &HLLSet) -> MaterializedResult {
        materialize_inlut(hllset, &self.lut)
    }

    /// Materialize using n-gram cross-validation.
    pub fn ngram_cross_validate(&self, hllset: &HLLSet) -> MaterializedResult {
        materialize_ngram_cross_validate(hllset, &self.lut)
    }

    /// Materialize using De Bruijn reconstruction.
    pub fn debruijn(&self, hllset: &HLLSet) -> MaterializedResult {
        materialize_debruijn(hllset, &self.lut, &self.start_marker, &self.end_marker)
    }

    /// Auto-select strategy based on bit count and LUT properties.
    pub fn auto(&self, hllset: &HLLSet) -> MaterializedResult {
        // If LUT has n-grams (tokens containing NUL), try De Bruijn
        let has_ngrams = self
            .lut
            .index
            .values()
            .any(|tokens| tokens.iter().any(|t| t.contains(&0u8)));

        if has_ngrams && hllset.popcount() > 2 {
            self.debruijn(hllset)
        } else {
            self.inlut(hllset)
        }
    }
}

impl Default for Materializer {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::Tokenizer;

    /// Helper: create HLLSet from text tokens and populate LUT.
    fn setup(text: &str) -> (HLLSet, TokenLUT) {
        let tok = Tokenizer::new().lowercase();
        let hllset = tok.apply(text.as_bytes()).into_hllset();
        let tokens = tok.tokenize(text.as_bytes());
        let lut = TokenLUT::from_tokens(&tokens);
        (hllset, lut)
    }

    #[test]
    fn test_lut_insert_and_get() {
        let mut lut = TokenLUT::new();
        lut.insert(b"hello".to_vec());
        let pos = token_to_position(b"hello");
        let candidates = lut.get(pos.0, pos.1).unwrap();
        assert!(candidates.contains(&b"hello".to_vec()));
    }

    #[test]
    fn test_lut_from_tokens() {
        let tokens = vec![b"hello", b"world"];
        let lut = TokenLUT::from_tokens(tokens.iter());
        assert!(!lut.is_empty());
    }

    #[test]
    fn test_materialize_inlut_basic() {
        let (hllset, lut) = setup("hello world");
        let result = materialize_inlut(&hllset, &lut);
        assert!(result.confidence > 0.0);
        let flat = result.flat_strings();
        assert!(flat.contains(&"hello".to_string()));
        assert!(flat.contains(&"world".to_string()));
    }

    #[test]
    fn test_materialize_inlut_empty() {
        let hllset = HLLSet::new();
        let lut = TokenLUT::new();
        let result = materialize_inlut(&hllset, &lut);
        assert_eq!(result.candidates, vec![] as Vec<Vec<Vec<u8>>>);
        assert_eq!(result.confidence, 1.0);
    }

    #[test]
    fn test_ngram_cross_validate() {
        // Create n-gram HLLSet
        let tok = Tokenizer::new().lowercase().ngrams(1, 2);
        let hllset = tok.apply(b"the cat sat").into_hllset();

        // Build LUT with both unigrams and bigrams
        let tokens = tok.tokenize(b"the cat sat");
        let lut = TokenLUT::from_tokens(&tokens);

        let result = materialize_ngram_cross_validate(&hllset, &lut);
        assert!(result.confidence > 0.0);
    }

    #[test]
    fn test_debruijn_reconstruction() {
        let tok = Tokenizer::new()
            .lowercase()
            .pad(b"<S>", b"</S>")
            .ngrams(2, 2);
        let hllset = tok.apply(b"the cat sat").into_hllset();
        let tokens = tok.tokenize(b"the cat sat");
        let lut = TokenLUT::from_tokens(&tokens);

        let result = materialize_debruijn(&hllset, &lut, b"<S>", b"</S>");
        assert!(result.confidence > 0.0);
        // De Bruijn should reconstruct some path
        assert!(!result.candidates.is_empty());
    }

    #[test]
    fn test_materializer_auto() {
        let tok = Tokenizer::new().lowercase().ngrams(1, 2);
        let hllset = tok.apply(b"hello world").into_hllset();
        let tokens = tok.tokenize(b"hello world");

        let mut m = Materializer::new();
        m.insert(&tokens);
        let result = m.auto(&hllset);
        assert!(result.confidence > 0.0);
    }

    #[test]
    fn test_split_at_nul() {
        let (a, b) = split_at_nul(b"hello\0world").unwrap();
        assert_eq!(a, b"hello");
        assert_eq!(b, b"world");
    }

    #[test]
    fn test_split_at_nul_no_separator() {
        assert!(split_at_nul(b"hello").is_none());
    }

    #[test]
    fn test_materialize_roundtrip() {
        // Tokenize → extract HLLSet → populate LUT → materialize → verify
        let tok = Tokenizer::new().lowercase();
        let hllset = tok.apply(b"alpha beta gamma").into_hllset();
        let tokens = tok.tokenize(b"alpha beta gamma");
        let lut = TokenLUT::from_tokens(&tokens);

        let result = materialize_inlut(&hllset, &lut);
        let flat = result.flat_strings();

        assert!(flat.contains(&"alpha".to_string()));
        assert!(flat.contains(&"beta".to_string()));
        assert!(flat.contains(&"gamma".to_string()));
    }

    #[test]
    fn test_full_roundtrip_with_ngrams() {
        // Full roundtrip: tokenize → HLLSet → materialize → verify
        let input = "the quick brown fox jumps over the lazy dog";
        let tok = Tokenizer::new().lowercase().ngrams(1, 2);
        let hllset = tok.apply(input.as_bytes()).into_hllset();
        let all_tokens = tok.tokenize(input.as_bytes());
        let lut = TokenLUT::from_tokens(&all_tokens);

        let result = materialize_inlut(&hllset, &lut);
        let flat = result.flat_strings();

        // Should recover all original words (possibly with some collisions)
        for word in input.split_whitespace() {
            assert!(
                flat.contains(&word.to_lowercase()),
                "missing word: {}",
                word
            );
        }
    }

    #[test]
    fn test_materializer_with_boundaries() {
        let tok = Tokenizer::new()
            .lowercase()
            .pad(b"<S>", b"</S>")
            .ngrams(2, 2);
        let hllset = tok.apply(b"the end").into_hllset();
        let tokens = tok.tokenize(b"the end");

        let mut m = Materializer::new().with_boundaries(b"<S>", b"</S>");
        m.insert(&tokens);
        let result = m.debruijn(&hllset);
        assert!(result.confidence > 0.0);
    }

    // ── CatalogLUT & Homogeneous Consensus tests ────────────────────

    #[test]
    fn test_catalog_lut_insert_and_get() {
        let mut lut = CatalogLUT::new();
        lut.insert(b"alice@example.com".to_vec());
        let pos = hllset_core::core::hashing::token_to_position_seeded(b"alice@example.com", 0);
        let candidates = lut.get(pos.0, pos.1).unwrap();
        assert!(candidates.contains(&b"alice@example.com".to_vec()));
    }

    #[test]
    fn test_catalog_lut_multi_seed_positions() {
        let mut lut = CatalogLUT::new();
        lut.insert(b"test".to_vec());
        let positions = lut.positions_of(b"test").unwrap();
        // Should have 3 positions (one per seed)
        assert_eq!(positions.len(), 3);
        // Different seeds should give different positions (with high probability)
        assert!(positions[0] != positions[1] || positions[1] != positions[2]);
    }

    #[test]
    fn test_homogeneous_consensus_full_match() {
        // Create HLLSet with all 3 seeds for each value
        let values: Vec<&[u8]> = vec![b"alice", b"bob", b"carol"];
        let mut hllset = HLLSet::new();
        for &seed in &[0u64, 1, 2] {
            for &v in &values {
                let hash = hllset_core::core::hashing::murmur3_hash_seeded(v, seed);
                hllset.add_hash(hash);
            }
        }
        let lut = CatalogLUT::from_values(values.iter());
        let result = materialize_homogeneous_consensus(&hllset, &lut);
        assert_eq!(result.strategy, "HomogeneousConsensus");
        assert!(result.confidence > 0.0);
        let flat = result.flat_strings();
        for &v in &values {
            assert!(flat.contains(&String::from_utf8_lossy(v).to_string()));
        }
    }

    #[test]
    fn test_consensus_rejects_weak_matches() {
        // Only seed 0 bits set → no value should pass consensus (need ≥ 2)
        let values = vec![b"alice"];
        let mut hllset = HLLSet::new();
        let hash = hllset_core::core::hashing::murmur3_hash_seeded(b"alice", 0);
        hllset.add_hash(hash);
        // Seeds 1 and 2 NOT set → should not pass 2-of-3 consensus
        let lut = CatalogLUT::from_values(values.iter());
        let result = materialize_homogeneous_consensus(&hllset, &lut);
        assert!(result.candidates.is_empty() || result.candidates[0].is_empty());
    }

    #[test]
    fn test_consensus_accepts_two_of_three() {
        // Seeds 0 and 1 set → should pass 2-of-3 consensus
        let mut hllset = HLLSet::new();
        for &seed in &[0u64, 1] {
            let hash = hllset_core::core::hashing::murmur3_hash_seeded(b"alice", seed);
            hllset.add_hash(hash);
        }
        let lut = CatalogLUT::from_values(&[b"alice"]);
        let result = materialize_homogeneous_consensus(&hllset, &lut);
        assert!(!result.candidates.is_empty());
        assert!(result.flat_strings().contains(&"alice".to_string()));
    }

    #[test]
    fn test_catalog_roundtrip() {
        let emails = vec![b"a@x.com", b"b@x.com", b"c@x.com", b"d@x.com", b"e@x.com"];
        // Build HLLSet with all 3 seeds
        let mut hllset = HLLSet::new();
        for &seed in &[0u64, 1, 2] {
            for &e in &emails {
                let hash = hllset_core::core::hashing::murmur3_hash_seeded(e, seed);
                hllset.add_hash(hash);
            }
        }
        let lut = CatalogLUT::from_values(emails.iter());
        let result = materialize_homogeneous_consensus(&hllset, &lut);
        let flat = result.flat_strings();
        for &e in &emails {
            assert!(
                flat.contains(&String::from_utf8_lossy(e).to_string()),
                "missing: {}",
                String::from_utf8_lossy(e)
            );
        }
    }
}

// ── MaterializeEngine trait bridge ─────────────────────────────────

impl hllset_materialize::MaterializeEngine for Materializer {
    fn materialize(
        &self,
        hllset: &HLLSet,
        positions: &[(u16, u8)],
    ) -> Result<Vec<Vec<u8>>, hllset_materialize::MaterializeError> {
        // Use the existing inlut materializer which handles position extraction internally
        let result = materialize_inlut(hllset, &self.lut);
        let tokens: Vec<Vec<u8>> = result
            .flat_strings()
            .into_iter()
            .map(|s| s.into_bytes())
            .collect();
        Ok(tokens)
    }
    fn name(&self) -> &str {
        "inmemory"
    }
    fn lut_count(&self) -> usize {
        self.lut.len()
    }
}
