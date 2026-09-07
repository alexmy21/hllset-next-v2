//! Conversation context — the top union plus the exchange path collection
//! and the sparse follow-frequency matrix.
//!
//! # Problem 1: Context building
//!
//! A conversation is the union of its exchange HLLSets (`top`), the list of
//! exchanges that produced it (path evidence), and the sparse adjacency
//! matrix accumulated from the padded 2-gram layer of every exchange.
//!
//! ```text
//! top.hll   = ∪ exch-HLLSet                    (the context HLLSet)
//! top.hll_n = ∪ exch-HLLSet layer n            (per-layer unions)
//! matrix    = follow frequencies from 2-grams  (multiplicity lives here)
//! ```
//!
//! **The matrix's row/column space is `top.hll_1`** — the union of the
//! exchanges' 1-HLLSets — plus the two constant boundary bits
//! `pos(_START_)`/`pos(_END_)`. The 2-HLLSets and 3-HLLSets are structural
//! equivalents: they contribute the edges and the order that point into the
//! 1-HLLSet vocabulary, never their own bit positions as matrix indices.
//!
//! Because union is monotonic and idempotent, building is incremental and
//! never rewrites history: adding an exchange only ORs bits in and
//! increments matrix counters.

use std::collections::HashSet;

use hllset_core::HLLSet;

use crate::debruijn::{DeBruijnRestorer, RestoredConversation};
use crate::matrix::{SparseAdjacencyMatrix, TokenIndex};
use crate::ngrams;
use crate::prompt::{build_prompt_with, PromptOptions};
use crate::Exchange;

/// Per-layer unions of a conversation context.
#[derive(Clone, Debug, Default)]
pub struct ContextTop {
    /// Union of 1-gram HLLSets (vocabulary).
    pub hll_1: HLLSet,
    /// Union of padded 2-gram HLLSets (follow edges).
    pub hll_2: HLLSet,
    /// Union of padded 3-gram HLLSets (De Bruijn edges).
    pub hll_3: HLLSet,
    /// `hll_1 ∪ hll_2 ∪ hll_3` — the full context HLLSet.
    pub hll: HLLSet,
}

impl ContextTop {
    /// Number of set bits in the full context HLLSet.
    pub fn popcount(&self) -> u64 {
        self.hll.popcount()
    }

    /// Horvitz-Thompson cardinality estimate of the full context.
    pub fn cardinality(&self) -> f64 {
        self.hll.cardinality()
    }

    /// Whether the context is empty.
    pub fn is_empty(&self) -> bool {
        self.hll.is_empty()
    }

    /// Trailing-zero histogram of the full context HLLSet.
    ///
    /// `hist[tz]` = number of registers in which bit `tz` is set. Because
    /// `tz` follows a geometric distribution, the low entries dominate and
    /// the high entries are (very) sparse — one reason the dense
    /// `1024 × 32` view is never worth materializing.
    pub fn tz_histogram(&self) -> Vec<u32> {
        self.hll.bit_counts()
    }

    /// Number of non-empty trailing-zero columns.
    pub fn active_tz_columns(&self) -> usize {
        self.tz_histogram().iter().filter(|&&c| c > 0).count()
    }
}

/// One shared vocabulary bit between two exchanges and its per-exchange
/// materializations.
///
/// This is the "coarse and beautiful" property of multi-gram HLLSets: the
/// same `<reg, zeros>` position in the 1-HLLSet layer can materialize to
/// *different* tokens in different exchanges. The 2-/3-HLLSet layers then
/// disambiguate through the structural anchors.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SharedBitOrigin {
    /// The shared 1-HLLSet bit position.
    pub bit: u32,
    /// Tokens that exchange `i` materializes at this bit.
    pub from_a: Vec<Vec<u8>>,
    /// Tokens that exchange `j` materializes at this bit.
    pub from_b: Vec<Vec<u8>>,
    /// Whether both exchanges materialize the same token set at this bit.
    pub same_token: bool,
}

/// A conversation context: exchange collection + top unions + adjacency matrix.
#[derive(Clone, Debug, Default)]
pub struct ConversationContext {
    /// Every user/assistant turn in order.
    pub exchanges: Vec<Exchange>,
    /// Unioned layers of the whole conversation.
    pub top: ContextTop,
    /// Sparse follow-frequency matrix (from padded 2-grams).
    pub matrix: SparseAdjacencyMatrix,
}

impl ConversationContext {
    /// Create an empty conversation context.
    pub fn new() -> Self {
        Self::default()
    }

    /// Build a context from a list of exchanges.
    pub fn build(exchanges: Vec<Exchange>) -> Self {
        let mut ctx = Self::new();
        for ex in exchanges {
            ctx.add_exchange(ex);
        }
        ctx
    }

    /// Incrementally add one exchange (monotonic, idempotent per bit).
    pub fn add_exchange(&mut self, exchange: Exchange) {
        let two_grams = exchange.ngrams(2);
        self.matrix.increment_2grams(&two_grams);

        self.top.hll_1.merge(&exchange.hll_1);
        self.top.hll_2.merge(&exchange.hll_2);
        self.top.hll_3.merge(&exchange.hll_3);
        self.top.hll.merge(&exchange.hll);

        self.exchanges.push(exchange);
    }

    /// Number of exchanges in the context.
    pub fn len(&self) -> usize {
        self.exchanges.len()
    }

    /// Whether the context has no exchanges.
    pub fn is_empty(&self) -> bool {
        self.exchanges.is_empty()
    }

    /// BSS inclusion of exchange `i` within the context top: how much of the
    /// exchange's content is already covered by the conversation.
    pub fn bss_inclusion(&self, exchange_index: usize) -> Option<f64> {
        self.exchanges
            .get(exchange_index)
            .map(|ex| self.top.hll.bss_inclusion(&ex.hll))
    }

    /// Jaccard overlap between two exchanges (their shared vocabulary weight).
    pub fn exchange_overlap(&self, i: usize, j: usize) -> Option<f64> {
        match (self.exchanges.get(i), self.exchanges.get(j)) {
            (Some(a), Some(b)) => Some(a.hll.jaccard_similarity(&b.hll)),
            _ => None,
        }
    }

    /// Vocabulary index: bit position → candidate 1-gram tokens.
    ///
    /// Includes the boundary markers so matrix rows/columns for
    /// `_START_`/`_END_` also resolve. This is the collision-group source
    /// used to unfold bit cells into token cells (observation 3).
    pub fn vocabulary_index(&self) -> TokenIndex {
        let mut idx = TokenIndex::new();
        idx.insert(ngrams::START_MARKER);
        idx.insert(ngrams::END_MARKER);
        for ex in &self.exchanges {
            for token in ex.ngrams(1) {
                idx.insert(&token);
            }
        }
        idx
    }

    /// Extract the matrix cells that belong to exchange `i`, in 2-gram order.
    ///
    /// Observation 4: the matrix is built over the union; the exchange
    /// HLLSets let us project it back onto the individual paths.
    pub fn exchange_path_cells(&self, exchange_index: usize) -> Vec<(u32, u32, u64)> {
        let mut out = Vec::new();
        if let Some(ex) = self.exchanges.get(exchange_index) {
            for tg in ex.ngrams(2) {
                let parts = ngrams::split_ngram(&tg);
                if parts.len() == 2 {
                    let row = ngrams::token_bit_position(&parts[0]);
                    let col = ngrams::token_bit_position(&parts[1]);
                    out.push((row, col, self.matrix.get(row, col)));
                }
            }
        }
        out
    }

    /// Ranked next-token suggestions after the last token of exchange `i`.
    ///
    /// Cell-value path-switching rule (observation 4): take the last 1-gram
    /// of the exchange, follow its highest-count matrix successors, and
    /// resolve each successor bit through the vocabulary collision group.
    pub fn suggest_continuations(&self, exchange_index: usize, k: usize) -> Vec<(Vec<u8>, u64)> {
        let mut out = Vec::new();
        let Some(ex) = self.exchanges.get(exchange_index) else {
            return out;
        };
        let Some(last) = ex.ngrams(1).last().cloned() else {
            return out;
        };
        let row = ngrams::token_bit_position(&last);
        let index = self.vocabulary_index();
        for (col, count) in self.matrix.continuations(row, k * 4) {
            let group = index.get(col);
            if group.is_empty() {
                out.push((format!("bit:{col}").into_bytes(), count));
            } else {
                for token in group {
                    out.push((token.clone(), count));
                }
            }
        }
        out.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        out.truncate(k);
        out
    }

    /// De Bruijn-restore the exchange paths from the top union + LUT only
    /// (drops `self.exchanges`; the restorer's LUT is built from the
    /// exchanges' n-grams, simulating a persisted vocabulary LUT).
    pub fn restore(&self) -> RestoredConversation {
        let restorer = DeBruijnRestorer::from_context(self);
        restorer.restore(self)
    }

    /// For each exchange, the 1-gram tokens that hash to `bit`.
    ///
    /// The same bit position can be materialized by different tokens in
    /// different exchanges; this returns the per-exchange origins.
    pub fn bit_origins(&self, bit: u32) -> Vec<(usize, Vec<Vec<u8>>)> {
        self.exchanges
            .iter()
            .enumerate()
            .filter_map(|(i, ex)| {
                let tokens = tokens_at_bit(ex, bit);
                if tokens.is_empty() {
                    None
                } else {
                    Some((i, tokens))
                }
            })
            .collect()
    }

    /// Shared vocabulary bits of exchanges `i` and `j` with their origins.
    ///
    /// Computes `hll_1(i) ∩ hll_1(j)` and, for each shared bit, lists the
    /// tokens that each exchange materializes there. Entries where
    /// `same_token == false` are the collision cases: the same position,
    /// different tokens — which the 2-/3-HLLSet structural layers resolve.
    pub fn shared_bit_origins(&self, i: usize, j: usize) -> Vec<SharedBitOrigin> {
        let (Some(a), Some(b)) = (self.exchanges.get(i), self.exchanges.get(j)) else {
            return Vec::new();
        };
        let shared = a.hll_1.intersection(&b.hll_1);
        let mut out: Vec<SharedBitOrigin> = shared
            .bitmap()
            .iter()
            .map(|bit| {
                let from_a = tokens_at_bit(a, bit);
                let from_b = tokens_at_bit(b, bit);
                SharedBitOrigin {
                    bit,
                    same_token: from_a == from_b,
                    from_a,
                    from_b,
                }
            })
            .collect();
        out.sort_by_key(|o| o.bit);
        out
    }

    /// Verify the matrix invariant: its row/column space is a subset of the
    /// union of the exchanges' 1-HLLSets (`top.hll_1`) plus the two constant
    /// structural boundary bits.
    ///
    /// The matrix is built over the vocabulary space — the 2-HLLSets and
    /// 3-HLLSets are structural equivalents that contribute edges/order but
    /// never their own bit positions as rows or columns.
    pub fn matrix_index_is_vocabulary(&self) -> bool {
        let mut vocab: HashSet<u32> = self.top.hll_1.bitmap().iter().collect();
        vocab.insert(ngrams::token_bit_position(ngrams::START_MARKER));
        vocab.insert(ngrams::token_bit_position(ngrams::END_MARKER));
        self.matrix.index_is_subset_of(&vocab)
    }

    /// Materialize the live context into an LLM text prompt.
    pub fn to_prompt(&self) -> String {
        build_prompt_with(self, &PromptOptions::default())
    }

    /// Materialize the live context into an LLM text prompt with options.
    pub fn to_prompt_with(&self, options: &PromptOptions) -> String {
        build_prompt_with(self, options)
    }
}

/// Sorted, deduplicated 1-gram tokens of `exchange` that hash to `bit`.
fn tokens_at_bit(exchange: &Exchange, bit: u32) -> Vec<Vec<u8>> {
    let mut tokens: Vec<Vec<u8>> = exchange
        .ngrams(1)
        .into_iter()
        .filter(|t| ngrams::token_bit_position(t) == bit)
        .collect();
    tokens.sort();
    tokens.dedup();
    tokens
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Role;

    fn ex(role: Role, text: &str) -> Exchange {
        Exchange::from_text(role, text)
    }

    #[test]
    fn test_build_union_equals_exchanges() {
        let exchanges = vec![
            ex(Role::User, "the cat sat"),
            ex(Role::Assistant, "the dog ran"),
        ];
        let ctx = ConversationContext::build(exchanges);

        let mut manual = HLLSet::new();
        for e in &ctx.exchanges {
            manual.merge(&e.hll);
        }
        assert_eq!(ctx.top.hll.popcount(), manual.popcount());
        assert_eq!(
            ctx.top.hll_2.popcount(),
            manual_union_layer(&ctx.exchanges, 2).popcount()
        );
        assert_eq!(ctx.len(), 2);
        assert!(!ctx.is_empty());
    }

    fn manual_union_layer(exchanges: &[Exchange], n: usize) -> HLLSet {
        let mut out = HLLSet::new();
        for e in exchanges {
            let layer = match n {
                1 => &e.hll_1,
                2 => &e.hll_2,
                _ => &e.hll_3,
            };
            out.merge(layer);
        }
        out
    }

    #[test]
    fn test_matrix_counts_shared_follows() {
        let exchanges = vec![
            ex(Role::User, "the cat sat"),
            ex(Role::Assistant, "the cat purred"),
        ];
        let ctx = ConversationContext::build(exchanges);

        let row = crate::ngrams::token_bit_position(b"the");
        let col = crate::ngrams::token_bit_position(b"cat");
        // "the → cat" occurs once per exchange
        assert_eq!(ctx.matrix.get(row, col), 2);
        assert!(ctx.matrix.total_follows() >= 2);
    }

    #[test]
    fn test_incremental_equals_batch() {
        let a = ex(Role::User, "alpha beta gamma");
        let b = ex(Role::Assistant, "alpha beta delta");

        let batch = ConversationContext::build(vec![a.clone(), b.clone()]);
        let mut inc = ConversationContext::new();
        inc.add_exchange(a);
        inc.add_exchange(b);

        assert_eq!(batch.top.hll.popcount(), inc.top.hll.popcount());
        assert_eq!(batch.matrix.total_follows(), inc.matrix.total_follows());
    }

    #[test]
    fn test_bss_inclusion() {
        let ctx = ConversationContext::build(vec![
            ex(Role::User, "the cat sat"),
            ex(Role::Assistant, "the cat sat"),
        ]);
        let tau = ctx.bss_inclusion(1).unwrap();
        assert!((tau - 1.0).abs() < 0.01, "tau={tau}");
    }

    #[test]
    fn test_tz_histogram_and_active_columns() {
        let ctx = ConversationContext::build(vec![
            ex(Role::User, "the cat sat"),
            ex(Role::Assistant, "the dog ran"),
        ]);
        let hist = ctx.top.tz_histogram();
        assert_eq!(hist.len(), 32);
        assert!(hist.iter().sum::<u32>() > 0);
        assert!(ctx.top.active_tz_columns() > 0);
        // high-tz columns are usually empty for tiny vocabularies
        assert!(hist
            .iter()
            .rev()
            .take(8)
            .all(|&c| c <= ctx.top.popcount() as u32));
    }

    #[test]
    fn test_vocabulary_index_resolves_bits_to_tokens() {
        let ctx = ConversationContext::build(vec![ex(Role::User, "the cat sat")]);
        let idx = ctx.vocabulary_index();
        let bit = crate::ngrams::token_bit_position(b"cat");
        assert!(idx.get(bit).contains(&b"cat".to_vec()));
        // boundary markers are in the index too
        let start_bit = crate::ngrams::token_bit_position(crate::ngrams::START_MARKER);
        assert!(idx
            .get(start_bit)
            .contains(&crate::ngrams::START_MARKER.to_vec()));
    }

    #[test]
    fn test_exchange_path_cells_follow_exchange() {
        let ctx = ConversationContext::build(vec![
            ex(Role::User, "the cat sat"),
            ex(Role::Assistant, "the dog ran"),
        ]);
        let cells = ctx.exchange_path_cells(0);
        // 2-grams of exchange 0: _START_\0the, the\0cat, cat\0sat, sat\0_END_
        assert_eq!(cells.len(), 4);
        // all cells come from the union matrix with non-zero counts
        for (r, c, n) in &cells {
            assert_eq!(ctx.matrix.get(*r, *c), *n);
            assert!(*n >= 1);
        }
    }

    #[test]
    fn test_suggest_continuations_ranked() {
        let ctx = ConversationContext::build(vec![
            ex(Role::User, "the cat sat"),
            ex(Role::Assistant, "the cat purred"),
            ex(Role::User, "the cat"),
        ]);
        // last exchange ends with "cat"; top successors are its follow tokens.
        let preds = ctx.suggest_continuations(2, 5);
        assert!(!preds.is_empty());
        assert!(preds[0].1 >= 1);
    }

    #[test]
    fn test_matrix_index_is_vocabulary() {
        let ctx = ConversationContext::build(vec![
            ex(Role::User, "the cat sat"),
            ex(Role::Assistant, "the dog ran"),
        ]);
        assert!(ctx.matrix_index_is_vocabulary());
    }

    #[test]
    fn test_shared_bit_origins_same_token() {
        let ctx = ConversationContext::build(vec![
            ex(Role::User, "the cat sat"),
            ex(Role::Assistant, "the cat purred"),
        ]);
        let origins = ctx.shared_bit_origins(0, 1);
        assert!(!origins.is_empty());
        // the bit for "cat" is shared and materializes the same token
        let cat_bit = crate::ngrams::token_bit_position(b"cat");
        let cat = origins.iter().find(|o| o.bit == cat_bit).unwrap();
        assert_eq!(cat.from_a, vec![b"cat".to_vec()]);
        assert!(cat.same_token);
    }

    #[test]
    fn test_shared_bit_origins_collision_different_tokens() {
        // Deterministically find two distinct tokens whose MurmurHash3
        // decomposes to the same (reg, tz) — the same bit position.
        let Some((a, b)) = find_collision_pair(3000) else {
            return; // astronomically unlikely with 3000 candidates
        };
        let ctx = ConversationContext::build(vec![
            Exchange::from_tokens(Role::User, vec![a.clone()]),
            Exchange::from_tokens(Role::Assistant, vec![b.clone()]),
        ]);
        let origins = ctx.shared_bit_origins(0, 1);
        assert!(
            origins.iter().any(|o| !o.same_token),
            "expected a shared bit with different tokens; origins={origins:?}"
        );
    }

    /// Find two distinct tokens that hash to the same HLLSet bit position.
    fn find_collision_pair(n: u32) -> Option<(Vec<u8>, Vec<u8>)> {
        let mut seen: std::collections::HashMap<u32, Vec<u8>> = std::collections::HashMap::new();
        for i in 0..n {
            let token = format!("w{i}").into_bytes();
            let bit = crate::ngrams::token_bit_position(&token);
            if let Some(prev) = seen.get(&bit) {
                return Some((prev.clone(), token));
            }
            seen.insert(bit, token);
        }
        None
    }

    #[test]
    fn test_empty_context() {
        let ctx = ConversationContext::new();
        assert!(ctx.is_empty());
        assert!(ctx.top.is_empty());
        assert!(ctx.matrix.is_empty());
        assert!(ctx.to_prompt().contains("assistant:"));
    }
}
