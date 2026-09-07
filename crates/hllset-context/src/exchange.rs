//! Exchange encoding — one user/LLM turn as three n-gram HLLSets.
//!
//! ```text
//! tokens      = [t1, ..., tn]
//! hll_1       = { hash(ti) }                      (unpadded vocabulary)
//! hll_2       = { hash of padded 2-grams }        (follow edges)
//! hll_3       = { hash of padded 3-grams }        (De Bruijn edges)
//! hll         = hll_1 ∪ hll_2 ∪ hll_3
//! ```

use hllset_core::HLLSet;

use crate::ngrams;

/// Speaker of an exchange.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Role {
    /// Human user turn.
    User,
    /// LLM assistant turn.
    Assistant,
}

impl Role {
    /// Role label used in materialized prompts.
    pub fn label(&self) -> &'static str {
        match self {
            Role::User => "user",
            Role::Assistant => "assistant",
        }
    }

    /// Toggle to the opposite role.
    pub fn toggle(&self) -> Role {
        match self {
            Role::User => Role::Assistant,
            Role::Assistant => Role::User,
        }
    }
}

/// One exchange (a user prompt or an assistant reply) encoded as HLLSets.
#[derive(Clone, Debug)]
pub struct Exchange {
    /// Who said it.
    pub role: Role,
    /// Original text (kept for high-fidelity prompt materialization).
    pub text: String,
    /// Tokenized text (unpadded).
    pub tokens: Vec<Vec<u8>>,
    /// 1-gram HLLSet (unpadded vocabulary bits).
    pub hll_1: HLLSet,
    /// 2-gram HLLSet (padded follow edges).
    pub hll_2: HLLSet,
    /// 3-gram HLLSet (padded De Bruijn edges).
    pub hll_3: HLLSet,
    /// `hll_1 ∪ hll_2 ∪ hll_3` — the complete exchange fingerprint.
    pub hll: HLLSet,
}

impl Exchange {
    /// Build an exchange from token bytes, joining them with spaces as text.
    pub fn from_tokens(role: Role, tokens: Vec<Vec<u8>>) -> Self {
        let text = tokens
            .iter()
            .map(|t| String::from_utf8_lossy(t).to_string())
            .collect::<Vec<_>>()
            .join(" ");
        Self::from_tokens_with_text(role, tokens, text)
    }

    /// Build an exchange from token bytes and an explicit text form.
    pub fn from_tokens_with_text(role: Role, tokens: Vec<Vec<u8>>, text: String) -> Self {
        let hll_1 = HLLSet::from_tokens(&grams_for(&tokens, 1));
        let hll_2 = HLLSet::from_tokens(&grams_for(&tokens, 2));
        let hll_3 = HLLSet::from_tokens(&grams_for(&tokens, 3));
        let hll = hll_1.union(&hll_2).union(&hll_3);
        Self {
            role,
            text,
            tokens,
            hll_1,
            hll_2,
            hll_3,
            hll,
        }
    }

    /// Build an exchange from raw text (ASCII whitespace tokenization, lowercased).
    pub fn from_text(role: Role, text: &str) -> Self {
        let tokens = tokenize_text(text);
        Self::from_tokens_with_text(role, tokens, text.to_string())
    }

    /// The n-grams of this exchange for layer `n` (1, 2, or 3).
    ///
    /// Layers 2 and 3 are boundary-padded; layer 1 is not.
    pub fn ngrams(&self, n: usize) -> Vec<Vec<u8>> {
        grams_for(&self.tokens, n)
    }

    /// Number of tokens in this exchange.
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Whether this exchange has no tokens.
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Structural anchors for a vocabulary bit in this exchange.
    ///
    /// The 2-HLLSet is the structural equivalent of the 1-HLLSet: for a
    /// bit set in `hll_1`, the padded 2-grams tell us which tokens precede
    /// and follow the token(s) that hashed to that bit. This is what
    /// disambiguates a bit when different exchanges materialize different
    /// tokens at the same `<reg, zeros>` position.
    pub fn bit_anchors(&self, bit: u32) -> Option<BitAnchors> {
        let tokens: Vec<Vec<u8>> = self
            .ngrams(1)
            .into_iter()
            .filter(|t| ngrams::token_bit_position(t) == bit)
            .collect();
        if tokens.is_empty() {
            return None;
        }

        let mut left = Vec::new();
        let mut right = Vec::new();
        for tg in self.ngrams(2) {
            let parts = ngrams::split_ngram(&tg);
            if parts.len() != 2 {
                continue;
            }
            if ngrams::token_bit_position(&parts[1]) == bit && !ngrams::is_boundary(&parts[0]) {
                left.push(parts[0].clone());
            }
            if ngrams::token_bit_position(&parts[0]) == bit && !ngrams::is_boundary(&parts[1]) {
                right.push(parts[1].clone());
            }
        }

        Some(BitAnchors {
            bit,
            tokens,
            left,
            right,
        })
    }
}

/// Structural anchors of one vocabulary bit inside one exchange.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BitAnchors {
    /// The 1-HLLSet bit position.
    pub bit: u32,
    /// 1-gram tokens in this exchange that hash to `bit`.
    pub tokens: Vec<Vec<u8>>,
    /// Tokens immediately preceding `tokens` (from the padded 2-grams).
    pub left: Vec<Vec<u8>>,
    /// Tokens immediately following `tokens` (from the padded 2-grams).
    pub right: Vec<Vec<u8>>,
}

/// Tokenize text into lowercase word tokens.
///
/// The crate intentionally ships with a simple, dependency-free tokenizer;
/// swap in any tokenizer (BPE, char-level for CJK, ...) by building the
/// exchange from `Vec<Vec<u8>>` directly.
pub fn tokenize_text(text: &str) -> Vec<Vec<u8>> {
    text.split_whitespace()
        .map(|w| w.to_ascii_lowercase().into_bytes())
        .collect()
}

/// Generate n-grams for an exchange: layer 1 unpadded, layers 2/3 padded.
fn grams_for(tokens: &[Vec<u8>], n: usize) -> Vec<Vec<u8>> {
    if tokens.is_empty() {
        return Vec::new();
    }
    ngrams::generate_ngrams(tokens, n, n > 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_text_layers() {
        let ex = Exchange::from_text(Role::User, "The cat sat");
        assert_eq!(
            ex.tokens,
            vec![b"the".to_vec(), b"cat".to_vec(), b"sat".to_vec()]
        );
        assert!(!ex.hll_1.is_empty());
        assert!(!ex.hll_2.is_empty());
        assert!(!ex.hll_3.is_empty());
        // combined hll is the union of the three layers
        let manual = ex.hll_1.union(&ex.hll_2).union(&ex.hll_3);
        assert_eq!(ex.hll.popcount(), manual.popcount());
    }

    #[test]
    fn test_empty_exchange_is_empty() {
        let ex = Exchange::from_text(Role::Assistant, "");
        assert!(ex.is_empty());
        assert!(ex.hll.is_empty());
        assert!(ex.hll_1.is_empty());
        assert!(ex.hll_2.is_empty());
        assert!(ex.hll_3.is_empty());
    }

    #[test]
    fn test_single_token_exchange() {
        let ex = Exchange::from_tokens(Role::User, vec![b"ok".to_vec()]);
        // 2-grams: _START_\0ok, ok\0_END_ ; 3-gram: _START_\0ok\0_END_
        assert_eq!(ex.ngrams(2).len(), 2);
        assert_eq!(ex.ngrams(3).len(), 1);
        assert_eq!(ex.ngrams(1), vec![b"ok".to_vec()]);
    }

    #[test]
    fn test_bit_anchors_structure() {
        let ex = Exchange::from_text(Role::User, "the cat sat");
        let bit = ngrams::token_bit_position(b"cat");
        let anchors = ex.bit_anchors(bit).unwrap();
        assert_eq!(anchors.tokens, vec![b"cat".to_vec()]);
        assert_eq!(anchors.left, vec![b"the".to_vec()]);
        assert_eq!(anchors.right, vec![b"sat".to_vec()]);
    }

    #[test]
    fn test_deterministic_content_keys() {
        let a = Exchange::from_text(Role::User, "hello world");
        let b = Exchange::from_text(Role::User, "hello world");
        assert_eq!(a.hll.content_key(), b.hll.content_key());
        assert_eq!(a.hll_2.content_key(), b.hll_2.content_key());
    }

    #[test]
    fn test_3gram_matches_bridge() {
        // Our padded 3-gram layer must be bit-identical to
        // hllset_bridge::extract_3gram for UTF-8 tokens.
        let tokens = vec!["the", "cat", "sat", "on", "the", "mat"];
        let ex = Exchange::from_tokens(
            Role::User,
            tokens.iter().map(|t| t.as_bytes().to_vec()).collect(),
        );
        let bridge = hllset_bridge::extract_3gram(&tokens);
        assert_eq!(ex.hll_3.content_key(), bridge.content_key());
        assert_eq!(ex.hll_3.popcount(), bridge.popcount());
    }
}
