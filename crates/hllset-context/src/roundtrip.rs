//! Round-trip composition law.
//!
//! For any composition
//!
//! ```text
//! hllA ──materialize──▶ {tokens} ──ingest──▶ hllB
//! ```
//!
//! the law states:
//!
//! - **1-hllA = 1-hllB** — exact. The 1-HLLSet is order-independent, and
//!   materialization only ever emits tokens whose bits are set in the
//!   source 1-HLLSet.
//! - **2-hllA = 2-hllB and 3-hllA = 3-hllB** — with high probability.
//!   Deviations are caused exclusively by hash-collision ambiguity during
//!   materialization (e.g. a noisy/global LUT supplying a false candidate
//!   that survives cross-validation).
//!
//! With a **perfect LUT** (built from the same n-grams that produced `hllA`)
//! the structural layers round-trip exactly as well: every enumerated
//! De Bruijn path is composed of true edges, so re-ingesting all paths
//! re-emits only original n-grams, and every original exchange is among the
//! enumerated paths.
//!
//! The re-ingest step unions all restored paths, so even hybrid paths
//! (chains stitched at shared 2-gram nodes) and repeated tokens preserve
//! every layer exactly.

use hllset_core::HLLSet;

use crate::{ConversationContext, DeBruijnRestorer, Exchange, RestoredConversation, Role};

/// Result of one materialize → re-ingest round trip.
#[derive(Clone, Debug, PartialEq)]
pub struct RoundTripReport {
    /// The restored token sequences that were re-ingested.
    pub paths: Vec<Vec<Vec<u8>>>,
    /// `1-hllA == 1-hllB` (bit-exact).
    pub hll_1_exact: bool,
    /// `2-hllA == 2-hllB` (bit-exact).
    pub hll_2_exact: bool,
    /// `3-hllA == 3-hllB` (bit-exact).
    pub hll_3_exact: bool,
    /// Full union `hllA == hllB` (bit-exact).
    pub hll_exact: bool,
    /// Jaccard similarity of the re-ingested 1-HLLSet vs the original.
    pub hll_1_jaccard: f64,
    /// Jaccard similarity of the re-ingested 2-HLLSet vs the original.
    pub hll_2_jaccard: f64,
    /// Jaccard similarity of the re-ingested 3-HLLSet vs the original.
    pub hll_3_jaccard: f64,
    /// Jaccard similarity of the full re-ingested HLLSet vs the original.
    pub hll_jaccard: f64,
}

impl RoundTripReport {
    /// Whether all three layers (and the full union) round-trip exactly.
    pub fn all_exact(&self) -> bool {
        self.hll_1_exact && self.hll_2_exact && self.hll_3_exact && self.hll_exact
    }
}

/// Round-trip a single exchange with a perfect LUT (its own n-grams).
pub fn roundtrip_exchange(exchange: &Exchange) -> RoundTripReport {
    let ctx = ConversationContext::build(vec![exchange.clone()]);
    let restorer = DeBruijnRestorer::from_context(&ctx);
    reingest_and_compare(&ctx, &restorer.restore(&ctx))
}

/// Round-trip a single exchange with a noisy LUT: the exchange's own
/// n-grams plus `extra_tokens` (simulating a global vocabulary LUT).
///
/// This is where the law's "high probability" clause is observable: false
/// candidates only break a layer if they survive shape filtering *and*
/// 1-gram/2-gram cross-validation, which is rare.
pub fn roundtrip_exchange_with_extra_tokens(
    exchange: &Exchange,
    extra_tokens: &[&[u8]],
) -> RoundTripReport {
    let ctx = ConversationContext::build(vec![exchange.clone()]);

    let mut lut: Vec<Vec<u8>> = Vec::new();
    for n in 1..=3 {
        lut.extend(exchange.ngrams(n));
    }
    lut.extend(extra_tokens.iter().map(|t| t.to_vec()));
    let refs: Vec<&[u8]> = lut.iter().map(|t| t.as_slice()).collect();
    let restorer = DeBruijnRestorer::from_vocabulary(&refs);

    reingest_and_compare(&ctx, &restorer.restore(&ctx))
}

/// Materialize a context, re-ingest every restored path, and compare the
/// per-layer unions.
fn reingest_and_compare(
    original: &ConversationContext,
    restored: &RestoredConversation,
) -> RoundTripReport {
    let mut reborn = ConversationContext::new();
    let mut role = original
        .exchanges
        .first()
        .map(|e| e.role)
        .unwrap_or(Role::User);
    for path in &restored.paths {
        reborn.add_exchange(Exchange::from_tokens(role, path.tokens.clone()));
        role = role.toggle();
    }

    let (hll_1_exact, hll_1_jaccard) = layer_report(&original.top.hll_1, &reborn.top.hll_1);
    let (hll_2_exact, hll_2_jaccard) = layer_report(&original.top.hll_2, &reborn.top.hll_2);
    let (hll_3_exact, hll_3_jaccard) = layer_report(&original.top.hll_3, &reborn.top.hll_3);
    let (hll_exact, hll_jaccard) = layer_report(&original.top.hll, &reborn.top.hll);

    RoundTripReport {
        paths: restored.paths.iter().map(|p| p.tokens.clone()).collect(),
        hll_1_exact,
        hll_2_exact,
        hll_3_exact,
        hll_exact,
        hll_1_jaccard,
        hll_2_jaccard,
        hll_3_jaccard,
        hll_jaccard,
    }
}

/// Bit-exact equality and Jaccard similarity of two layers.
fn layer_report(a: &HLLSet, b: &HLLSet) -> (bool, f64) {
    (a.bitmap() == b.bitmap(), a.jaccard_similarity(b))
}

impl Exchange {
    /// `hllA → materialize → {tokens} → ingest → hllB`, reported per layer.
    pub fn roundtrip(&self) -> RoundTripReport {
        roundtrip_exchange(self)
    }
}

impl ConversationContext {
    /// Round-trip the whole context: restore every path, re-ingest, compare
    /// the per-layer unions.
    pub fn roundtrip(&self) -> RoundTripReport {
        let restorer = DeBruijnRestorer::from_context(self);
        reingest_and_compare(self, &restorer.restore(self))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ex(text: &str) -> Exchange {
        Exchange::from_text(Role::User, text)
    }

    #[test]
    fn test_single_exchange_roundtrip_exact() {
        for text in ["the cat sat", "hello world", "ok"] {
            let report = ex(text).roundtrip();
            assert!(report.all_exact(), "text={text}, report={report:?}");
            assert!((report.hll_1_jaccard - 1.0).abs() < 1e-9);
            assert!((report.hll_2_jaccard - 1.0).abs() < 1e-9);
            assert!((report.hll_3_jaccard - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn test_roundtrip_repeated_tokens() {
        // Repeated tokens create self-loops in the De Bruijn graph; the
        // union of all enumerated paths must still re-emit exactly the
        // original layers.
        for tokens in [
            vec!["a", "a", "a"],
            vec!["the", "the", "cat"],
            vec!["x", "y", "x", "y"],
        ] {
            let exchange = Exchange::from_tokens(
                Role::User,
                tokens.iter().map(|t| t.as_bytes().to_vec()).collect(),
            );
            let report = exchange.roundtrip();
            assert!(report.all_exact(), "tokens={tokens:?}, report={report:?}");
        }
    }

    #[test]
    fn test_context_roundtrip_exact_non_overlapping() {
        let ctx = ConversationContext::build(vec![
            ex("the cat sat"),
            Exchange::from_text(Role::Assistant, "the dog ran"),
        ]);
        let report = ctx.roundtrip();
        assert!(report.all_exact(), "report={report:?}");
    }

    #[test]
    fn test_context_roundtrip_exact_overlapping_hybrid_paths() {
        // Two chains share the middle 2-gram node "b\0c", so the restore
        // enumerates hybrid paths as well. Every enumerated path is made of
        // true edges, so re-ingesting them all still preserves every layer.
        let ctx = ConversationContext::build(vec![
            Exchange::from_tokens(
                Role::User,
                vec![b"a".to_vec(), b"b".to_vec(), b"c".to_vec(), b"d".to_vec()],
            ),
            Exchange::from_tokens(
                Role::Assistant,
                vec![b"e".to_vec(), b"b".to_vec(), b"c".to_vec(), b"f".to_vec()],
            ),
        ]);
        let report = ctx.roundtrip();
        assert!(report.paths.len() >= 2);
        assert!(report.all_exact(), "report={report:?}");
    }

    #[test]
    fn test_roundtrip_empty_exchange() {
        let report = Exchange::from_text(Role::User, "").roundtrip();
        assert!(report.all_exact());
    }

    #[test]
    fn test_roundtrip_with_noisy_lut_high_probability() {
        // A global LUT with thousands of distractor tokens. False candidates
        // must survive shape filtering AND 1-gram/2-gram cross-validation,
        // so the structural layers round-trip exactly with high probability
        // (deterministically true for these distractor tokens).
        let exchange = ex("the cat sat");
        let mut distractors: Vec<Vec<u8>> = Vec::new();
        for i in 0..2000u32 {
            let t = format!("dx{i}\0dy{i}\0dz{i}").into_bytes();
            distractors.push(t);
        }
        let refs: Vec<&[u8]> = distractors.iter().map(|t| t.as_slice()).collect();
        let report = roundtrip_exchange_with_extra_tokens(&exchange, &refs);
        assert!(
            report.hll_1_exact,
            "1-HLLSet must stay exact; report={report:?}"
        );
        assert!(
            report.hll_2_exact && report.hll_3_exact,
            "high-probability structural exactness; report={report:?}"
        );
    }

    #[test]
    fn test_composition_law_over_random_exchanges() {
        let mut rng = XorShift::new(0x9E37_79B9_7F4A_7C15);
        for i in 0..80u32 {
            let len = 2 + (rng.next() % 6) as usize;
            let tokens: Vec<Vec<u8>> = (0..len).map(|j| format!("t{i}_{j}").into_bytes()).collect();
            let report = Exchange::from_tokens(Role::User, tokens).roundtrip();
            assert!(
                report.hll_1_exact,
                "1-HLLSet must round-trip exactly (i={i})"
            );
            assert!(
                report.hll_2_exact,
                "2-HLLSet must round-trip exactly with a perfect LUT (i={i})"
            );
            assert!(
                report.hll_3_exact,
                "3-HLLSet must round-trip exactly with a perfect LUT (i={i})"
            );
        }
    }

    /// Tiny deterministic xorshift PRNG for the random-exchange law test.
    struct XorShift(u64);

    impl XorShift {
        fn new(seed: u64) -> Self {
            Self(seed.max(1))
        }

        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
    }
}
