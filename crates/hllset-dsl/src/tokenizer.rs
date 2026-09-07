//! Composable tokenization pipeline: pattern → normalize → n-grams → inscribe.
//!
//! A `Tokenizer` turns raw byte sequences into token streams and,
//! optionally, directly into `LatticeElement` fingerprints.
//!
//! # Pipeline
//!
//! ```text
//! Bytes → [Pattern Match] → Tokens → [Normalize] → [N-grams] → [Boundary Pad] → LatticeElement
//! ```
//!
//! # Example
//!
//! ```rust
//! use hllset_dsl::pattern::Pattern;
//! use hllset_dsl::tokenizer::Tokenizer;
//!
//! let tok = Tokenizer::new()
//!     .word_pattern()
//!     .lowercase()
//!     .ngrams(1, 2);
//!
//! let tokens = tok.tokenize(b"The Cat Sat");
//! // tokens: ["the", "the", "the", "cat", "the cat", "cat", "cat", "cat sat", "sat", "sat"]
//! ```

use crate::lattice::LatticeElement;
use crate::pattern::Pattern;

/// A normalizer transforms an individual token byte sequence.
///
/// Can be extended with user-defined normalizers via the Lua surface.
#[derive(Clone, Debug)]
pub enum Normalizer {
    /// Convert ASCII letters to lowercase.
    Lowercase,
    /// Remove leading and trailing ASCII whitespace.
    Trim,
    /// Keep only bytes that are in the allowed set.
    KeepOnly(Vec<u8>),
}

impl Normalizer {
    /// Apply the normalizer to a token. Returns `None` if the token
    /// should be discarded entirely.
    pub fn apply(&self, token: &[u8]) -> Option<Vec<u8>> {
        match self {
            Normalizer::Lowercase => Some(token.iter().map(|b| b.to_ascii_lowercase()).collect()),
            Normalizer::Trim => {
                let start = token
                    .iter()
                    .position(|b| !b.is_ascii_whitespace())
                    .unwrap_or(token.len());
                let end = token
                    .iter()
                    .rposition(|b| !b.is_ascii_whitespace())
                    .map(|i| i + 1)
                    .unwrap_or(start);
                if start < end {
                    Some(token[start..end].to_vec())
                } else {
                    None
                }
            }
            Normalizer::KeepOnly(allowed) => {
                let set: std::collections::HashSet<u8> = allowed.iter().copied().collect();
                let result: Vec<u8> = token.iter().copied().filter(|b| set.contains(b)).collect();
                if result.is_empty() {
                    None
                } else {
                    Some(result)
                }
            }
        }
    }
}

/// A composable tokenization pipeline.
///
/// # Configuration
///
/// | Field | Default | Purpose |
/// |-------|---------|---------|
/// | `pattern` | Word pattern | How to extract tokens |
/// | `normalizers` | `[]` | Post-extraction transforms |
/// | `ngram_min` | 1 | Minimum n-gram size |
/// | `ngram_max` | 1 | Maximum n-gram size |
/// | `boundary_start` | `None` | Start-of-sequence marker |
/// | `boundary_end` | `None` | End-of-sequence marker |
#[derive(Clone, Debug)]
pub struct Tokenizer {
    pattern: Pattern,
    normalizers: Vec<Normalizer>,
    ngram_min: usize,
    ngram_max: usize,
    boundary_start: Option<Vec<u8>>,
    boundary_end: Option<Vec<u8>>,
}

impl Default for Tokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Tokenizer {
    /// Create a new tokenizer with sensible defaults (word pattern, unigrams).
    pub fn new() -> Self {
        // Default pattern: one or more ASCII letters or digits
        let word_pattern =
            Pattern::span(b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789");

        Self {
            pattern: word_pattern,
            normalizers: Vec::new(),
            ngram_min: 1,
            ngram_max: 1,
            boundary_start: None,
            boundary_end: None,
        }
    }

    /// Set the pattern for extracting tokens from input bytes.
    pub fn pattern(mut self, p: Pattern) -> Self {
        self.pattern = p;
        self
    }

    /// Use the default word pattern (ASCII letters + digits).
    pub fn word_pattern(mut self) -> Self {
        self.pattern =
            Pattern::span(b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789");
        self
    }

    /// Add the lowercase normalizer.
    pub fn lowercase(mut self) -> Self {
        self.normalizers.push(Normalizer::Lowercase);
        self
    }

    /// Add the trim normalizer.
    pub fn trim(mut self) -> Self {
        self.normalizers.push(Normalizer::Trim);
        self
    }

    /// Keep only bytes in the allowed set.
    pub fn keep_only(mut self, allowed: &[u8]) -> Self {
        self.normalizers
            .push(Normalizer::KeepOnly(allowed.to_vec()));
        self
    }

    /// Add a custom normalizer.
    pub fn normalizer(mut self, n: Normalizer) -> Self {
        self.normalizers.push(n);
        self
    }

    /// Generate n-grams in the range `[min, max]` (inclusive).
    ///
    /// For example, `.ngrams(1, 3)` produces unigrams, bigrams, and trigrams.
    pub fn ngrams(mut self, min: usize, max: usize) -> Self {
        assert!(min >= 1, "n-gram minimum must be >= 1");
        assert!(max >= min, "n-gram max must be >= min");
        self.ngram_min = min;
        self.ngram_max = max;
        self
    }

    /// Add boundary tokens to the sequence before n-gram generation.
    ///
    /// START token is prepended; END token is appended. This enables
    /// n-grams that include start/end boundaries for context.
    pub fn pad(mut self, start: &[u8], end: &[u8]) -> Self {
        self.boundary_start = Some(start.to_vec());
        self.boundary_end = Some(end.to_vec());
        self
    }

    // ── Token extraction ────────────────────────────────────────────────

    /// Extract raw tokens from input bytes using the configured pattern.
    pub fn extract(&self, input: &[u8]) -> Vec<Vec<u8>> {
        let matches = self.pattern.find_all(input, 0);
        matches
            .into_iter()
            .map(|m| input[m.range].to_vec())
            .collect()
    }

    /// Tokenize input bytes through the full pipeline.
    ///
    /// Steps:
    /// 1. Extract tokens using the pattern
    /// 2. Apply normalizers (may discard tokens)
    /// 3. Apply boundary padding
    /// 4. Generate n-grams
    pub fn tokenize(&self, input: &[u8]) -> Vec<Vec<u8>> {
        // Step 1+2: extract and normalize
        let mut tokens: Vec<Vec<u8>> = self
            .extract(input)
            .into_iter()
            .filter_map(|t| self.apply_normalizers(&t))
            .collect();

        if tokens.is_empty() {
            return Vec::new();
        }

        // Step 3: boundary padding
        if let Some(ref start) = self.boundary_start {
            tokens.insert(0, start.clone());
        }
        if let Some(ref end) = self.boundary_end {
            tokens.push(end.clone());
        }

        // Step 4: n-grams
        self.generate_ngrams(&tokens)
    }

    /// Apply the full pipeline and inscribe into a `LatticeElement`.
    ///
    /// This is the main entry point: bytes in, LatticeElement out.
    pub fn apply(&self, input: &[u8]) -> LatticeElement {
        let tokens = self.tokenize(input);
        LatticeElement::from_tokens(&tokens)
    }

    // ── Internal helpers ────────────────────────────────────────────────

    fn apply_normalizers(&self, token: &[u8]) -> Option<Vec<u8>> {
        let mut t = token.to_vec();
        for norm in &self.normalizers {
            t = norm.apply(&t)?;
        }
        Some(t)
    }

    fn generate_ngrams(&self, tokens: &[Vec<u8>]) -> Vec<Vec<u8>> {
        if tokens.is_empty() {
            return Vec::new();
        }

        let mut ngrams = Vec::new();
        for n in self.ngram_min..=self.ngram_max {
            for window in tokens.windows(n) {
                let ngram: Vec<u8> = join_tokens(window);
                ngrams.push(ngram);
            }
        }
        ngrams
    }
}

/// Join multiple tokens with a NUL separator. This is the standard
/// HLLSet n-token joining convention: uses 0x00 as separator.
///
/// For single-token n-grams (n=1), no separator is inserted.
fn join_tokens(tokens: &[Vec<u8>]) -> Vec<u8> {
    if tokens.len() == 1 {
        return tokens[0].clone();
    }
    let total_len: usize = tokens.iter().map(|t| t.len()).sum::<usize>() + tokens.len() - 1;
    let mut result = Vec::with_capacity(total_len);
    for (i, token) in tokens.iter().enumerate() {
        if i > 0 {
            result.push(0u8); // NUL separator
        }
        result.extend_from_slice(token);
    }
    result
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_words() {
        let tok = Tokenizer::new();
        let tokens = tok.extract(b"hello world lua");
        let strs: Vec<&str> = tokens
            .iter()
            .map(|t| std::str::from_utf8(t).unwrap())
            .collect();
        assert_eq!(strs, vec!["hello", "world", "lua"]);
    }

    #[test]
    fn test_lowercase() {
        let tok = Tokenizer::new().lowercase();
        let tokens = tok.tokenize(b"Hello WORLD Lua");
        let strs: Vec<&str> = tokens
            .iter()
            .map(|t| std::str::from_utf8(t).unwrap())
            .collect();
        assert_eq!(strs, vec!["hello", "world", "lua"]);
    }

    #[test]
    fn test_bigrams() {
        let tok = Tokenizer::new().lowercase().ngrams(2, 2);
        let tokens = tok.tokenize(b"the cat sat");
        let strs: Vec<&str> = tokens
            .iter()
            .map(|t| std::str::from_utf8(t).unwrap())
            .collect();
        // \0 = NUL separator between words in bigrams
        assert_eq!(strs, vec!["the\0cat", "cat\0sat"]);
    }

    #[test]
    fn test_bigrams_and_unigrams() {
        let tok = Tokenizer::new().lowercase().ngrams(1, 2);
        let tokens = tok.tokenize(b"hello world");
        let strs: Vec<&str> = tokens
            .iter()
            .map(|t| std::str::from_utf8(t).unwrap())
            .collect();
        assert_eq!(strs, vec!["hello", "world", "hello\0world"]);
    }

    #[test]
    fn test_boundary_padding() {
        let tok = Tokenizer::new()
            .lowercase()
            .pad(b"<S>", b"</S>")
            .ngrams(2, 2);
        let tokens = tok.tokenize(b"hello world");
        let strs: Vec<&str> = tokens
            .iter()
            .map(|t| std::str::from_utf8(t).unwrap())
            .collect();
        assert_eq!(strs, vec!["<S>\0hello", "hello\0world", "world\0</S>"]);
    }

    #[test]
    fn test_apply_returns_lattice_element() {
        let tok = Tokenizer::new().lowercase();
        let elem = tok.apply(b"Hello World");
        assert!(elem.key().starts_with("h:"));
        assert!(elem.cardinality() > 0.0);
    }

    #[test]
    fn test_apply_deterministic() {
        let tok = Tokenizer::new().lowercase().ngrams(1, 1);
        let elem1 = tok.apply(b"The Cat Sat On The Mat");
        let elem2 = tok.apply(b"The Cat Sat On The Mat");
        assert_eq!(elem1.key(), elem2.key());
        assert_eq!(elem1.popcount(), elem2.popcount());
    }

    #[test]
    fn test_apply_order_independent_for_same_tokens() {
        let tok = Tokenizer::new().lowercase().ngrams(1, 1);
        let elem1 = tok.apply(b"cat dog");
        let elem2 = tok.apply(b"dog cat");
        // Content keys should be the same because tokens are sorted before hashing
        assert_eq!(elem1.key(), elem2.key());
    }

    #[test]
    fn test_empty_input() {
        let tok = Tokenizer::new();
        let tokens = tok.tokenize(b"");
        assert!(tokens.is_empty());
        let elem = tok.apply(b"");
        assert!(elem.is_empty());
    }

    #[test]
    fn test_custom_pattern() {
        // Match only digits
        let digit = Pattern::span(b"0123456789");
        let tok = Tokenizer::new().pattern(digit);
        let tokens = tok.extract(b"x=42, y=137");
        let strs: Vec<&str> = tokens
            .iter()
            .map(|t| std::str::from_utf8(t).unwrap())
            .collect();
        assert_eq!(strs, vec!["42", "137"]);
    }

    #[test]
    fn test_custom_pattern_with_alt() {
        // Match either words or numbers (via alt)
        let alpha = Pattern::span(b"abcdefghijklmnopqrstuvwxyz");
        let digit = Pattern::span(b"0123456789");
        let tok = Tokenizer::new().pattern(alpha.alt(digit)).lowercase();
        let tokens = tok.tokenize(b"hello 42 world");
        let strs: Vec<&str> = tokens
            .iter()
            .map(|t| std::str::from_utf8(t).unwrap())
            .collect();
        assert_eq!(strs, vec!["hello", "42", "world"]);
    }

    #[test]
    fn test_trim_normalizer() {
        let tok = Tokenizer::new().trim();
        // Create a pattern that captures whitespace too
        let any_char = Pattern::any(b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ");
        let tok = Tokenizer::new().pattern(any_char).trim();
        let tokens = tok.tokenize(b"  hello  world  ");
        let strs: Vec<&str> = tokens
            .iter()
            .map(|t| std::str::from_utf8(t).unwrap())
            .collect();
        // Trim should remove leading/trailing spaces on each token
        assert!(strs.iter().all(|s| !s.contains(' ')));
    }

    #[test]
    fn test_trigrams() {
        let tok = Tokenizer::new().lowercase().ngrams(3, 3);
        let tokens = tok.tokenize(b"the cat sat on the mat");
        let strs: Vec<&str> = tokens
            .iter()
            .map(|t| std::str::from_utf8(t).unwrap())
            .collect();
        assert!(strs.len() > 0);
        assert_eq!(strs[0], "the\0cat\0sat");
    }

    #[test]
    fn test_ngram_range_preserves_cardinality_monotonicity() {
        let tok_small = Tokenizer::new().lowercase().ngrams(1, 1);
        let tok_large = Tokenizer::new().lowercase().ngrams(1, 3);

        let elem_small = tok_small.apply(b"the cat sat");
        let elem_large = tok_large.apply(b"the cat sat");

        // N-grams 1..3 should have >= cardinality than just unigrams
        assert!(elem_large.cardinality() >= elem_small.cardinality());
    }
}
