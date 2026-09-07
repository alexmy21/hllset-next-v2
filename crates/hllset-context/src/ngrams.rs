//! N-gram generation, boundary padding, and bit-position helpers.
//!
//! Conventions are bit-identical to `hllset-bridge`'s `extract_3gram`:
//! padding markers `_START_` / `_END_` and NUL (`0x00`) as the n-gram
//! token separator. 1-grams are generated *without* padding (pads are
//! structural, not vocabulary); 2-grams and 3-grams are generated *with*
//! padding so that START/END path anchors survive into the HLLSet.

use hllset_core::hashing::token_to_position;
use hllset_core::BITS_PER_REG;

/// Start-of-sequence padding marker (matches `hllset-bridge`).
pub const START_MARKER: &[u8] = b"_START_";

/// End-of-sequence padding marker (matches `hllset-bridge`).
pub const END_MARKER: &[u8] = b"_END_";

/// N-gram token separator (matches `hllset-bridge` / `hllset-dsl`).
pub const SEP: u8 = 0x00;

/// Whether a token is a structural boundary marker.
pub fn is_boundary(token: &[u8]) -> bool {
    token == START_MARKER || token == END_MARKER
}

/// Pad a token sequence with START/END markers.
pub fn pad_tokens(tokens: &[Vec<u8>]) -> Vec<Vec<u8>> {
    let mut padded = Vec::with_capacity(tokens.len() + 2);
    padded.push(START_MARKER.to_vec());
    padded.extend(tokens.iter().cloned());
    padded.push(END_MARKER.to_vec());
    padded
}

/// Join tokens into a single n-gram byte sequence with NUL separators.
///
/// A single-token window is returned unchanged (no separator).
pub fn join_tokens(tokens: &[Vec<u8>]) -> Vec<u8> {
    match tokens.len() {
        0 => Vec::new(),
        1 => tokens[0].clone(),
        _ => {
            let total: usize = tokens.iter().map(|t| t.len()).sum::<usize>() + tokens.len() - 1;
            let mut out = Vec::with_capacity(total);
            for (i, t) in tokens.iter().enumerate() {
                if i > 0 {
                    out.push(SEP);
                }
                out.extend_from_slice(t);
            }
            out
        }
    }
}

/// Split an n-gram byte sequence on NUL into its constituent tokens.
pub fn split_ngram(ngram: &[u8]) -> Vec<Vec<u8>> {
    if ngram.is_empty() {
        return Vec::new();
    }
    ngram.split(|&b| b == SEP).map(|s| s.to_vec()).collect()
}

/// Generate all n-grams of size `n` from `tokens`.
///
/// If `pad` is `true`, the sequence is padded with `_START_`/`_END_` before
/// the sliding window. Returns an empty vector if the (padded) sequence is
/// shorter than `n` or if `n == 0`.
pub fn generate_ngrams(tokens: &[Vec<u8>], n: usize, pad: bool) -> Vec<Vec<u8>> {
    if n == 0 || tokens.is_empty() {
        return Vec::new();
    }
    let seq: Vec<Vec<u8>> = if pad {
        pad_tokens(tokens)
    } else {
        tokens.to_vec()
    };
    if seq.len() < n {
        return Vec::new();
    }
    seq.windows(n).map(join_tokens).collect()
}

/// Bit position `reg * 32 + tz` (0..32767) of a token under MurmurHash3 seed 0.
///
/// This is the flattened index used to address the sparse adjacency matrix.
pub fn token_bit_position(token: &[u8]) -> u32 {
    let (reg, tz) = token_to_position(token);
    reg * BITS_PER_REG + tz
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tok(s: &str) -> Vec<u8> {
        s.as_bytes().to_vec()
    }

    #[test]
    fn test_pad_tokens() {
        let tokens = vec![tok("a"), tok("b")];
        let padded = pad_tokens(&tokens);
        assert_eq!(padded.len(), 4);
        assert_eq!(padded[0], START_MARKER);
        assert_eq!(padded[3], END_MARKER);
    }

    #[test]
    fn test_join_split_roundtrip() {
        let tokens = vec![tok("a"), tok("b"), tok("c")];
        let joined = join_tokens(&tokens);
        assert_eq!(joined, b"a\0b\0c");
        assert_eq!(split_ngram(&joined), tokens);
    }

    #[test]
    fn test_generate_unigrams_no_pad() {
        let tokens = vec![tok("a"), tok("b")];
        let grams = generate_ngrams(&tokens, 1, false);
        assert_eq!(grams, vec![tok("a"), tok("b")]);
    }

    #[test]
    fn test_generate_bigrams_padded() {
        let tokens = vec![tok("a"), tok("b")];
        let grams = generate_ngrams(&tokens, 2, true);
        assert_eq!(
            grams,
            vec![
                b"_START_\0a".to_vec(),
                b"a\0b".to_vec(),
                b"b\0_END_".to_vec()
            ]
        );
    }

    #[test]
    fn test_generate_trigrams_single_token() {
        let tokens = vec![tok("a")];
        let grams = generate_ngrams(&tokens, 3, true);
        assert_eq!(grams, vec![b"_START_\0a\0_END_".to_vec()]);
    }

    #[test]
    fn test_generate_empty() {
        assert!(generate_ngrams(&[], 1, false).is_empty());
        assert!(generate_ngrams(&[], 2, true).is_empty());
        assert!(generate_ngrams(&[], 3, true).is_empty());
    }

    #[test]
    fn test_token_bit_position_range() {
        for word in ["hello", "world", "_START_", "_END_"] {
            let pos = token_bit_position(word.as_bytes());
            assert!(pos < hllset_core::core::hllset::TOTAL_BITS);
        }
    }
}
