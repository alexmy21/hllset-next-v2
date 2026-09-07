//! De Bruijn restoration of exchange paths from the context union.
//!
//! # Problem 2a: restore order from the 3-gram layer
//!
//! The top union is multiplicity-free, but its padded 3-gram bits encode a
//! De Bruijn graph:
//!
//! ```text
//! node  = 2-gram   (a\0b)
//! edge  = 3-gram   (a\0b\0c) : node (a\0b) -> node (b\0c)
//! START = 2-gram whose first token is _START_
//! END   = 2-gram whose last  token is _END_
//! ```
//!
//! Every exchange corresponds to one START→END path. Multiple exchanges
//! produce multiple START/END anchors and therefore multiple paths.
//!
//! The restorer uses an `hllset_materialize::InMemoryEngine` as the reverse
//! LUT (bit position → candidate tokens), then cross-validates 3-grams
//! against the 1-gram and 2-gram unions to suppress hash-collision noise,
//! and finally enumerates START→END paths.

use std::collections::{HashMap, HashSet};

use hllset_materialize::{InMemoryEngine, MaterializeEngine};

use crate::ngrams;
use crate::ConversationContext;

/// One restored exchange path (tokens, pads stripped).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RestoredPath {
    /// The reconstructed token sequence.
    pub tokens: Vec<Vec<u8>>,
}

impl RestoredPath {
    /// The path as a space-joined string.
    pub fn as_text(&self) -> String {
        self.tokens
            .iter()
            .map(|t| String::from_utf8_lossy(t).to_string())
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// Result of restoring a conversation from union + LUT.
#[derive(Clone, Debug, Default)]
pub struct RestoredConversation {
    /// Restored exchange paths (canonical order).
    pub paths: Vec<RestoredPath>,
    /// Fraction of 3-gram bits in the union explained by the restored paths.
    pub confidence: f64,
}

impl RestoredConversation {
    /// Number of restored paths.
    pub fn len(&self) -> usize {
        self.paths.len()
    }

    /// Whether no paths were restored.
    pub fn is_empty(&self) -> bool {
        self.paths.is_empty()
    }

    /// Flatten all restored paths into one deduplicated token list.
    pub fn flat_tokens(&self) -> Vec<Vec<u8>> {
        let mut seen = HashSet::new();
        let mut out = Vec::new();
        for path in &self.paths {
            for t in &path.tokens {
                if seen.insert(t.clone()) {
                    out.push(t.clone());
                }
            }
        }
        out
    }

    /// Materialize the restored paths into an LLM prompt (alternating roles
    /// starting with `first_role`, latest user query appended).
    pub fn to_prompt(&self, latest_query: &str) -> String {
        crate::prompt::build_prompt_from_restored(self, latest_query, crate::Role::User)
    }
}

/// De Bruijn restorer backed by an `InMemoryEngine` reverse LUT.
pub struct DeBruijnRestorer {
    engine: InMemoryEngine,
}

impl DeBruijnRestorer {
    /// Build a restorer from a vocabulary of n-gram tokens.
    pub fn from_vocabulary(tokens: &[&[u8]]) -> Self {
        let mut engine = InMemoryEngine::new("debruijn-lut");
        engine.build(tokens);
        Self { engine }
    }

    /// Build a restorer whose LUT contains every n-gram of every exchange in
    /// the context. This simulates a persisted vocabulary LUT: after
    /// restoring, the exchange list itself is no longer consulted.
    pub fn from_context(ctx: &ConversationContext) -> Self {
        let mut tokens: Vec<Vec<u8>> = Vec::new();
        for ex in &ctx.exchanges {
            for n in 1..=3 {
                tokens.extend(ex.ngrams(n));
            }
        }
        let refs: Vec<&[u8]> = tokens.iter().map(|t| t.as_slice()).collect();
        Self::from_vocabulary(&refs)
    }

    /// Restore exchange paths from a context's top unions.
    pub fn restore(&self, ctx: &ConversationContext) -> RestoredConversation {
        self.restore_with_limits(ctx, 64, 512)
    }

    /// Restore with explicit path/depth limits.
    pub fn restore_with_limits(
        &self,
        ctx: &ConversationContext,
        max_paths: usize,
        max_depth: usize,
    ) -> RestoredConversation {
        let top = &ctx.top;
        let total_bits = top.hll_3.popcount().max(1) as f64;

        // 1. Candidate tokens per layer, straight from the LUT.
        let pos1 = InMemoryEngine::extract_positions(&top.hll_1);
        let pos2 = InMemoryEngine::extract_positions(&top.hll_2);
        let pos3 = InMemoryEngine::extract_positions(&top.hll_3);

        let cand1 = self
            .engine
            .materialize(&top.hll_1, &pos1)
            .unwrap_or_default();
        let cand2 = self
            .engine
            .materialize(&top.hll_2, &pos2)
            .unwrap_or_default();
        let cand3 = self
            .engine
            .materialize(&top.hll_3, &pos3)
            .unwrap_or_default();

        // Keep only tokens with the right shape for each layer.
        let bits1: HashSet<u32> = top.hll_1.bitmap().iter().collect();
        let bits2: HashSet<u32> = top.hll_2.bitmap().iter().collect();

        let grams1: HashSet<Vec<u8>> = cand1
            .into_iter()
            .filter(|t| ngrams::split_ngram(t).len() == 1)
            .collect();
        let grams2: HashSet<Vec<u8>> = cand2
            .into_iter()
            .filter(|t| ngrams::split_ngram(t).len() == 2)
            .collect();
        let grams3: HashSet<Vec<u8>> = cand3
            .into_iter()
            .filter(|t| ngrams::split_ngram(t).len() == 3)
            .collect();

        // 2. Cross-validate 3-grams and build the De Bruijn graph.
        //    node -> [(next_node, triple)]
        let mut adj: HashMap<Vec<u8>, Vec<(Vec<u8>, Vec<u8>)>> = HashMap::new();

        for triple in &grams3 {
            let parts = ngrams::split_ngram(triple);
            if parts.len() != 3 {
                continue;
            }
            let (a, b, c) = (&parts[0], &parts[1], &parts[2]);

            // 1-gram cross-validation (boundaries are structural, always ok).
            if !valid_1gram(a, &grams1, &bits1)
                || !valid_1gram(b, &grams1, &bits1)
                || !valid_1gram(c, &grams1, &bits1)
            {
                continue;
            }

            // 2-gram cross-validation.
            let ab = ngrams::join_tokens(&[a.clone(), b.clone()]);
            let bc = ngrams::join_tokens(&[b.clone(), c.clone()]);
            if !valid_2gram(&ab, &grams2, &bits2) || !valid_2gram(&bc, &grams2, &bits2) {
                continue;
            }

            adj.entry(ab.clone())
                .or_default()
                .push((bc.clone(), triple.clone()));
        }

        // 3. Enumerate START→END paths.
        let mut starts: Vec<Vec<u8>> = adj
            .keys()
            .filter(|node| {
                let parts = ngrams::split_ngram(node);
                parts
                    .first()
                    .map(|p| p.as_slice() == ngrams::START_MARKER)
                    .unwrap_or(false)
            })
            .cloned()
            .collect();
        starts.sort();

        let mut results = Vec::new();
        for start in &starts {
            let mut used: HashSet<(Vec<u8>, Vec<u8>)> = HashSet::new();
            let mut path = vec![start.clone()];
            dfs_paths(
                &adj,
                start,
                &mut used,
                &mut path,
                max_paths,
                max_depth,
                &mut results,
            );
        }

        // Deduplicate and canonicalize.
        results.sort();
        results.dedup();
        results.truncate(max_paths);

        // 4. Confidence = fraction of 3-gram union bits covered by the
        //    validated edges that were actually used in a returned path.
        let used_triples: HashSet<&Vec<u8>> = {
            let mut s = HashSet::new();
            for path in &results {
                for w in path.windows(2) {
                    if let Some(nexts) = adj.get(&w[0]) {
                        for (next, triple) in nexts {
                            if *next == w[1] {
                                s.insert(triple);
                                break;
                            }
                        }
                    }
                }
            }
            s
        };
        let confidence = if results.is_empty() {
            0.0
        } else {
            (used_triples.len() as f64 / total_bits).min(1.0)
        };

        RestoredConversation {
            paths: results
                .into_iter()
                .map(|path| RestoredPath {
                    tokens: path_to_tokens(&path),
                })
                .collect(),
            confidence,
        }
    }
}

/// 1-gram cross-validation: boundary markers always pass; otherwise the
/// token must be a known 1-gram whose bit is set in the 1-gram union.
fn valid_1gram(token: &[u8], grams1: &HashSet<Vec<u8>>, bits1: &HashSet<u32>) -> bool {
    if ngrams::is_boundary(token) {
        return true;
    }
    if !grams1.contains(token) {
        return false;
    }
    bits1.contains(&ngrams::token_bit_position(token))
}

/// 2-gram cross-validation: the 2-gram must be a known 2-gram whose bit is
/// set in the 2-gram union.
fn valid_2gram(two_gram: &[u8], grams2: &HashSet<Vec<u8>>, bits2: &HashSet<u32>) -> bool {
    if !grams2.contains(two_gram) {
        return false;
    }
    bits2.contains(&ngrams::token_bit_position(two_gram))
}

/// Depth-first enumeration of all START→END paths.
fn dfs_paths(
    adj: &HashMap<Vec<u8>, Vec<(Vec<u8>, Vec<u8>)>>,
    node: &[u8],
    used: &mut HashSet<(Vec<u8>, Vec<u8>)>,
    path: &mut Vec<Vec<u8>>,
    max_paths: usize,
    max_depth: usize,
    results: &mut Vec<Vec<Vec<u8>>>,
) {
    if results.len() >= max_paths || path.len() > max_depth {
        return;
    }

    // Record when we reach an END node.
    if is_end_node(node) {
        results.push(path.clone());
        // END nodes have no outgoing edges in well-formed data; return to
        // avoid extending through them.
        return;
    }

    let mut nexts: Vec<&(Vec<u8>, Vec<u8>)> = adj
        .get(node)
        .map(|v| v.iter().collect())
        .unwrap_or_default();
    nexts.sort_by(|a, b| a.0.cmp(&b.0));

    for (next, _triple) in nexts {
        let edge_key = (node.to_vec(), next.clone());
        if used.contains(&edge_key) {
            continue;
        }
        used.insert(edge_key.clone());
        path.push(next.clone());
        dfs_paths(adj, next, used, path, max_paths, max_depth, results);
        path.pop();
        used.remove(&edge_key);
    }
}

/// Whether a 2-gram node is an END node (last token is `_END_`).
fn is_end_node(node: &[u8]) -> bool {
    ngrams::split_ngram(node)
        .last()
        .map(|p| p.as_slice() == ngrams::END_MARKER)
        .unwrap_or(false)
}

/// Collapse a path of 2-gram nodes back into the original token sequence.
///
/// Emits the last token of each node; the first token of the first node and
/// the boundary markers are stripped.
fn path_to_tokens(path: &[Vec<u8>]) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    for (i, node) in path.iter().enumerate() {
        let parts = ngrams::split_ngram(node);
        if parts.is_empty() {
            continue;
        }
        if i == 0 && !ngrams::is_boundary(&parts[0]) {
            out.push(parts[0].clone());
        }
        if let Some(last) = parts.last() {
            if !ngrams::is_boundary(last) {
                out.push(last.clone());
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Exchange, Role};

    fn ctx_of(turns: &[(&str, &str)]) -> ConversationContext {
        let exchanges = turns
            .iter()
            .map(|(role, text)| {
                let role = if *role == "user" {
                    Role::User
                } else {
                    Role::Assistant
                };
                Exchange::from_text(role, text)
            })
            .collect();
        ConversationContext::build(exchanges)
    }

    fn tokens(paths: &RestoredConversation) -> Vec<Vec<String>> {
        paths
            .paths
            .iter()
            .map(|p| {
                p.tokens
                    .iter()
                    .map(|t| String::from_utf8_lossy(t).to_string())
                    .collect()
            })
            .collect()
    }

    #[test]
    fn test_single_exchange_roundtrip() {
        let ctx = ctx_of(&[("user", "the cat sat")]);
        let restored = ctx.restore();
        assert_eq!(restored.paths.len(), 1, "paths={restored:?}");
        assert_eq!(tokens(&restored), vec![vec!["the", "cat", "sat"]]);
        assert!(restored.confidence > 0.0);
    }

    #[test]
    fn test_two_exchanges_roundtrip() {
        let ctx = ctx_of(&[("user", "the cat sat"), ("assistant", "the dog ran")]);
        let restored = ctx.restore();
        let got = tokens(&restored);
        assert!(
            got.contains(&vec![
                "the".to_string(),
                "cat".to_string(),
                "sat".to_string()
            ]),
            "got={got:?}"
        );
        assert!(
            got.contains(&vec![
                "the".to_string(),
                "dog".to_string(),
                "ran".to_string()
            ]),
            "got={got:?}"
        );
    }

    #[test]
    fn test_shared_prefix_branching() {
        let ctx = ctx_of(&[("user", "a b c"), ("assistant", "a b d")]);
        let restored = ctx.restore();
        let got = tokens(&restored);
        assert!(
            got.contains(&vec!["a".to_string(), "b".to_string(), "c".to_string()]),
            "got={got:?}"
        );
        assert!(
            got.contains(&vec!["a".to_string(), "b".to_string(), "d".to_string()]),
            "got={got:?}"
        );
    }

    #[test]
    fn test_single_token_exchange() {
        let ctx = ctx_of(&[("user", "ok")]);
        let restored = ctx.restore();
        assert_eq!(tokens(&restored), vec![vec!["ok"]]);
    }

    #[test]
    fn test_two_token_exchange() {
        let ctx = ctx_of(&[("user", "hello world")]);
        let restored = ctx.restore();
        assert_eq!(tokens(&restored), vec![vec!["hello", "world"]]);
    }

    #[test]
    fn test_empty_context_restores_nothing() {
        let ctx = ConversationContext::new();
        let restored = ctx.restore();
        assert!(restored.is_empty());
        assert_eq!(restored.confidence, 0.0);
    }

    #[test]
    fn test_restored_prompt() {
        let ctx = ctx_of(&[
            ("user", "how does a cat sit"),
            ("assistant", "a cat sits on the mat"),
        ]);
        let restored = ctx.restore();
        let prompt = restored.to_prompt("what about a dog");
        assert!(
            prompt.contains("user: how does a cat sit"),
            "prompt={prompt}"
        );
        assert!(prompt.contains("assistant: a cat sits on the mat"));
        assert!(prompt.contains("user: what about a dog"));
        assert!(prompt.ends_with("assistant:"));
    }
}
