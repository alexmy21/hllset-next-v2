//! Sparse adjacency matrix over HLLSet bit positions.
//!
//! The conceptual matrix is `2^32768 × 2^32768` — one row/column per
//! possible HLLSet bit-vector — but it is populated only at the ≤ 32768
//! bit positions that tokens actually hash to. In practice the effective
//! size is even smaller:
//!
//! - the dense view is **never allocated**;
//! - the trailing-zero decomposition makes high `tz` bits geometrically
//!   rare, so most activity sits in the low-`tz` columns;
//! - the number of distinct rows/columns is bounded by the number of
//!   distinct tokens in the conversation, not by 32768.
//!
//! Cell `(row_bit, col_bit)` counts how many times a 2-gram `p\0s` was
//! observed with `row_bit = pos(p)` and `col_bit = pos(s)`, i.e. the
//! accumulated number of times `s` followed `p`.
//!
//! # Collision groups
//!
//! One bit position can correspond to **several** candidate tokens (hash
//! collisions, or the LUT being consulted without full disambiguation).
//! The matrix therefore keeps a token-level index in parallel, and a
//! [`TokenIndex`] can unfold each bit cell into one row/column per
//! candidate token — duplicating the cell value across the collision
//! group.

use std::collections::{HashMap, HashSet};

use crate::ngrams;

/// Sparse follow-frequency matrix.
///
/// Keeps both a bit-level index (for HLLSet-native scoring) and a
/// token-level index (for human-readable prompt summaries and collision
/// unfolding).
#[derive(Clone, Debug, Default)]
pub struct SparseAdjacencyMatrix {
    /// `(row_bit, col_bit) -> follow count`.
    counts: HashMap<(u32, u32), u64>,
    /// `(prefix_token, suffix_token) -> follow count`.
    token_counts: HashMap<(Vec<u8>, Vec<u8>), u64>,
}

/// Reverse index: bit position → candidate tokens (a collision group).
///
/// Built from a vocabulary by hashing each token with the same MurmurHash3
/// decomposition used by the HLLSet layers.
#[derive(Clone, Debug, Default)]
pub struct TokenIndex {
    groups: HashMap<u32, Vec<Vec<u8>>>,
}

impl TokenIndex {
    /// Create an empty index.
    pub fn new() -> Self {
        Self::default()
    }

    /// Build an index from a vocabulary of tokens.
    pub fn from_vocabulary<I, S>(tokens: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<[u8]>,
    {
        let mut idx = Self::new();
        for t in tokens {
            idx.insert(t.as_ref());
        }
        idx
    }

    /// Register a token under its bit position (deduplicated).
    pub fn insert(&mut self, token: &[u8]) {
        let bit = ngrams::token_bit_position(token);
        let group = self.groups.entry(bit).or_default();
        if !group.iter().any(|t| t.as_slice() == token) {
            group.push(token.to_vec());
        }
    }

    /// Candidate tokens for a bit position (empty if unknown).
    pub fn get(&self, bit: u32) -> &[Vec<u8>] {
        self.groups.get(&bit).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Number of occupied bit positions (collision groups).
    pub fn len(&self) -> usize {
        self.groups.len()
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.groups.is_empty()
    }

    /// Collision groups sorted by bit position: `(bit, group_size)`.
    pub fn group_sizes(&self) -> Vec<(u32, usize)> {
        let mut v: Vec<(u32, usize)> = self.groups.iter().map(|(b, g)| (*b, g.len())).collect();
        v.sort_by(|a, b| a.0.cmp(&b.0));
        v
    }

    /// Largest collision group size.
    pub fn max_group_size(&self) -> usize {
        self.groups.values().map(|g| g.len()).max().unwrap_or(0)
    }
}

/// One cell of the unfolded (token-resolved) adjacency matrix.
///
/// The same `count` is duplicated across every token pair in the row and
/// column collision groups — "duplicating rows and columns".
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnfoldedCell {
    /// Row bit position.
    pub row_bit: u32,
    /// Column bit position.
    pub col_bit: u32,
    /// Candidate tokens for the row bit (collision group).
    pub row_tokens: Vec<Vec<u8>>,
    /// Candidate tokens for the column bit (collision group).
    pub col_tokens: Vec<Vec<u8>>,
    /// Follow count shared by this cell.
    pub count: u64,
}

impl SparseAdjacencyMatrix {
    /// Create an empty matrix.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record one follow occurrence `prefix -> suffix`.
    pub fn increment_edge(&mut self, prefix: &[u8], suffix: &[u8]) {
        let row = ngrams::token_bit_position(prefix);
        let col = ngrams::token_bit_position(suffix);
        *self.counts.entry((row, col)).or_insert(0) += 1;
        *self
            .token_counts
            .entry((prefix.to_vec(), suffix.to_vec()))
            .or_insert(0) += 1;
    }

    /// Record every valid 2-gram in a layer-2 token list.
    ///
    /// Tokens that do not split into exactly two parts are ignored.
    pub fn increment_2grams(&mut self, two_grams: &[Vec<u8>]) {
        for tg in two_grams {
            let parts = ngrams::split_ngram(tg);
            if parts.len() == 2 {
                self.increment_edge(&parts[0], &parts[1]);
            }
        }
    }

    /// Follow count at bit cell `(row, col)`.
    pub fn get(&self, row: u32, col: u32) -> u64 {
        self.counts.get(&(row, col)).copied().unwrap_or(0)
    }

    /// Follow count at token cell `(prefix, suffix)`.
    pub fn get_token(&self, prefix: &[u8], suffix: &[u8]) -> u64 {
        self.token_counts
            .get(&(prefix.to_vec(), suffix.to_vec()))
            .copied()
            .unwrap_or(0)
    }

    /// Ranked successors of a row bit: `(col_bit, count)`, descending by count.
    pub fn successors(&self, row: u32) -> Vec<(u32, u64)> {
        let mut v: Vec<(u32, u64)> = self
            .counts
            .iter()
            .filter(|((r, _), _)| *r == row)
            .map(|((_, c), n)| (*c, *n))
            .collect();
        v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        v
    }

    /// Top-k successors of a row bit (cell-value path-switching rule).
    pub fn continuations(&self, row: u32, k: usize) -> Vec<(u32, u64)> {
        let mut v = self.successors(row);
        v.truncate(k);
        v
    }

    /// Greedy walk through the matrix starting at `start_bit`.
    ///
    /// At each node the highest-count unused outgoing edge is taken; when
    /// the top edge is exhausted the walk **switches** to the next-best
    /// cell. This is the cell-value path-switching generation rule.
    pub fn greedy_walk(&self, start_bit: u32, max_len: usize) -> Vec<u32> {
        if max_len == 0 {
            return Vec::new();
        }
        let mut path = vec![start_bit];
        let mut used: HashSet<(u32, u32)> = HashSet::new();
        let mut current = start_bit;
        while path.len() < max_len {
            let next = self
                .successors(current)
                .into_iter()
                .find(|(col, _)| used.insert((current, *col)));
            match next {
                Some((col, _)) => {
                    path.push(col);
                    current = col;
                }
                None => break,
            }
        }
        path
    }

    /// Number of non-zero bit cells.
    pub fn nonzero_cells(&self) -> usize {
        self.counts.len()
    }

    /// All row and column bits currently used by the matrix.
    pub fn index_bits(&self) -> HashSet<u32> {
        let mut bits = HashSet::new();
        for (row, col) in self.counts.keys() {
            bits.insert(*row);
            bits.insert(*col);
        }
        bits
    }

    /// Whether the matrix's row/column space is a subset of `vocab_bits`.
    ///
    /// Used to verify the invariant that the matrix is built over the
    /// vocabulary space (the union of the exchanges' 1-HLLSets), not over
    /// the full union HLLSet.
    pub fn index_is_subset_of(&self, vocab_bits: &HashSet<u32>) -> bool {
        self.counts
            .keys()
            .all(|(row, col)| vocab_bits.contains(row) && vocab_bits.contains(col))
    }

    /// Number of distinct row bits.
    pub fn distinct_row_bits(&self) -> usize {
        self.counts
            .keys()
            .map(|(r, _)| *r)
            .collect::<HashSet<_>>()
            .len()
    }

    /// Number of distinct column bits.
    pub fn distinct_col_bits(&self) -> usize {
        self.counts
            .keys()
            .map(|(_, c)| *c)
            .collect::<HashSet<_>>()
            .len()
    }

    /// Effective sparse shape: `(distinct_rows, distinct_cols, cells)`.
    ///
    /// This is the actual working size of the matrix — bounded by the
    /// number of distinct tokens in the conversation, not by 2^32768.
    pub fn shape(&self) -> (usize, usize, usize) {
        (
            self.distinct_row_bits(),
            self.distinct_col_bits(),
            self.nonzero_cells(),
        )
    }

    /// Total accumulated follow occurrences.
    pub fn total_follows(&self) -> u64 {
        self.counts.values().sum()
    }

    /// Top-k human-readable follow links `(prefix, suffix, count)`.
    pub fn top_links(&self, k: usize) -> Vec<(Vec<u8>, Vec<u8>, u64)> {
        let mut v: Vec<(Vec<u8>, Vec<u8>, u64)> = self
            .token_counts
            .iter()
            .map(|((p, s), n)| (p.clone(), s.clone(), *n))
            .collect();
        v.sort_by(|a, b| b.2.cmp(&a.2).then(a.0.cmp(&b.0)));
        v.truncate(k);
        v
    }

    /// Unfold every bit cell into token-resolved cells.
    ///
    /// For each bit cell `(row, col, count)` the row and column collision
    /// groups are looked up in `index`; the cell is **duplicated** once per
    /// `(row_token, col_token)` pair, sharing the same `count`. Bits absent
    /// from the index are represented by a `bit:<n>` placeholder.
    pub fn unfolded_cells(&self, index: &TokenIndex) -> Vec<UnfoldedCell> {
        let mut out = Vec::new();
        let mut cells: Vec<((u32, u32), u64)> = self.counts.iter().map(|(k, n)| (*k, *n)).collect();
        cells.sort_by(|a, b| a.0.cmp(&b.0));
        for ((row, col), count) in cells {
            let row_tokens = resolve_group(index, row);
            let col_tokens = resolve_group(index, col);
            for rt in &row_tokens {
                for ct in &col_tokens {
                    out.push(UnfoldedCell {
                        row_bit: row,
                        col_bit: col,
                        row_tokens: vec![rt.clone()],
                        col_tokens: vec![ct.clone()],
                        count,
                    });
                }
            }
        }
        out
    }

    /// Whether no edges have been recorded.
    pub fn is_empty(&self) -> bool {
        self.counts.is_empty()
    }
}

/// Resolve a bit position to a collision group, or a placeholder token.
fn resolve_group(index: &TokenIndex, bit: u32) -> Vec<Vec<u8>> {
    let group = index.get(bit);
    if group.is_empty() {
        vec![format!("bit:{bit}").into_bytes()]
    } else {
        group.to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_increment_and_get() {
        let mut m = SparseAdjacencyMatrix::new();
        m.increment_edge(b"a", b"b");
        m.increment_edge(b"a", b"b");
        m.increment_edge(b"b", b"c");

        let row_a = ngrams::token_bit_position(b"a");
        let col_b = ngrams::token_bit_position(b"b");
        let col_c = ngrams::token_bit_position(b"c");

        assert_eq!(m.get(row_a, col_b), 2);
        assert_eq!(m.get(ngrams::token_bit_position(b"b"), col_c), 1);
        assert_eq!(m.get(row_a, col_c), 0);
        assert_eq!(m.get_token(b"a", b"b"), 2);
        assert_eq!(m.nonzero_cells(), 2);
        assert_eq!(m.total_follows(), 3);
    }

    #[test]
    fn test_successors_sorted() {
        let mut m = SparseAdjacencyMatrix::new();
        m.increment_edge(b"a", b"x");
        m.increment_edge(b"a", b"y");
        m.increment_edge(b"a", b"y");

        let row_a = ngrams::token_bit_position(b"a");
        let succ = m.successors(row_a);
        assert_eq!(succ.len(), 2);
        // y has count 2 → first
        assert_eq!(succ[0].0, ngrams::token_bit_position(b"y"));
        assert_eq!(succ[0].1, 2);
        assert_eq!(succ[1].0, ngrams::token_bit_position(b"x"));
        assert_eq!(succ[1].1, 1);
    }

    #[test]
    fn test_increment_2grams_ignores_malformed() {
        let mut m = SparseAdjacencyMatrix::new();
        m.increment_2grams(&[b"a\0b".to_vec(), b"not-a-2gram".to_vec()]);
        assert_eq!(m.total_follows(), 1);
    }

    #[test]
    fn test_top_links() {
        let mut m = SparseAdjacencyMatrix::new();
        m.increment_edge(b"the", b"cat");
        m.increment_edge(b"the", b"cat");
        m.increment_edge(b"cat", b"sat");

        let top = m.top_links(1);
        assert_eq!(top.len(), 1);
        assert_eq!(top[0].0, b"the".to_vec());
        assert_eq!(top[0].1, b"cat".to_vec());
        assert_eq!(top[0].2, 2);
    }

    #[test]
    fn test_shape_effective_size() {
        let mut m = SparseAdjacencyMatrix::new();
        m.increment_edge(b"a", b"b");
        m.increment_edge(b"a", b"c");
        m.increment_edge(b"b", b"c");
        let (rows, cols, cells) = m.shape();
        assert_eq!(rows, 2); // a, b
        assert_eq!(cols, 2); // b, c
        assert_eq!(cells, 3);
    }

    #[test]
    fn test_index_is_subset_of() {
        let mut m = SparseAdjacencyMatrix::new();
        m.increment_edge(b"a", b"b");
        m.increment_edge(b"b", b"c");

        let mut vocab: HashSet<u32> = HashSet::new();
        vocab.insert(ngrams::token_bit_position(b"a"));
        vocab.insert(ngrams::token_bit_position(b"b"));
        vocab.insert(ngrams::token_bit_position(b"c"));
        assert!(m.index_is_subset_of(&vocab));

        let mut smaller: HashSet<u32> = HashSet::new();
        smaller.insert(ngrams::token_bit_position(b"a"));
        assert!(!m.index_is_subset_of(&smaller));
        assert_eq!(m.index_bits().len(), 3);
    }

    #[test]
    fn test_token_index_collision_group() {
        let mut idx = TokenIndex::new();
        idx.insert(b"alpha");
        idx.insert(b"alpha"); // dedup
        let bit = ngrams::token_bit_position(b"alpha");
        assert_eq!(idx.get(bit), &[b"alpha".to_vec()]);
        assert_eq!(idx.len(), 1);
        assert_eq!(idx.max_group_size(), 1);
    }

    #[test]
    fn test_unfolded_cells_duplicate_rows_and_cols() {
        let mut m = SparseAdjacencyMatrix::new();
        m.increment_edge(b"a", b"b");

        let bit_a = ngrams::token_bit_position(b"a");
        let bit_b = ngrams::token_bit_position(b"b");

        // Synthetic index: bit_a resolves to a two-token collision group.
        let mut groups = std::collections::HashMap::new();
        groups.insert(bit_a, vec![b"a1".to_vec(), b"a2".to_vec()]);
        groups.insert(bit_b, vec![b"b".to_vec()]);
        let idx = TokenIndex { groups };

        let unfolded = m.unfolded_cells(&idx);
        // one bit cell × (2 row tokens × 1 col token) = 2 unfolded cells
        assert_eq!(unfolded.len(), 2);
        assert_eq!(unfolded[0].row_tokens, vec![b"a1".to_vec()]);
        assert_eq!(unfolded[0].col_tokens, vec![b"b".to_vec()]);
        assert_eq!(unfolded[0].count, 1);
        assert_eq!(unfolded[1].row_tokens, vec![b"a2".to_vec()]);
    }

    #[test]
    fn test_greedy_walk_switches_cells() {
        let mut m = SparseAdjacencyMatrix::new();
        // a → b ×2, a → c ×1 ; b → d ×1 ; c → e ×1
        m.increment_edge(b"a", b"b");
        m.increment_edge(b"a", b"b");
        m.increment_edge(b"a", b"c");
        m.increment_edge(b"b", b"d");
        m.increment_edge(b"c", b"e");

        let walk = m.greedy_walk(ngrams::token_bit_position(b"a"), 4);
        let b = ngrams::token_bit_position(b"b");
        let c = ngrams::token_bit_position(b"c");
        let d = ngrams::token_bit_position(b"d");
        let e = ngrams::token_bit_position(b"e");
        // start a → b (count 2) → d (only edge) → then d has no outgoing,
        // so stop. Path = [a, b, d].
        assert_eq!(walk.len(), 3);
        assert_eq!(walk[0], ngrams::token_bit_position(b"a"));
        assert_eq!(walk[1], b);
        assert_eq!(walk[2], d);
        let _ = (c, e);
    }
}
