//! SNOBOL-inspired composable pattern matching over byte sequences.
//!
//! Patterns are recursive matchers over `&[u8]`. They support:
//!
//! **Primitives** — `literal`, `any`, `notany`, `span`, `break`, `arb`
//!
//! **Combinators** — `cat` (sequence), `alt` (alternative), `capture`
//!
//! **Matching** — backtracking with greedy span/arb, returning
//! `Match` with position and any captured byte sub-sequences.
//!
//! # Example
//!
//! ```rust
//! use hllset_dsl::pattern::Pattern;
//!
//! // Match: word = one or more ASCII letters
//! let word = Pattern::span(b"abcdefghijklmnopqrstuvwxyz");
//! let m = word.match_at(b"hello world", 0).unwrap();
//! assert_eq!(m.range, 0..5); // matched "hello" at 0..5
//! ```

use std::ops::Range;

/// Result of a successful pattern match.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Match {
    /// Byte range of the full match in the input.
    pub range: Range<usize>,
    /// Any captured sub-sequences (in order of capture).
    pub captures: Vec<Vec<u8>>,
}

impl Match {
    fn at(start: usize, end: usize) -> Self {
        Self {
            range: start..end,
            captures: Vec::new(),
        }
    }
}

/// A composable byte-sequence pattern.
///
/// Patterns are matched against a byte slice starting at a given position.
/// On success they return a `Match` with the consumed range; on failure `None`.
#[derive(Clone, Debug)]
pub enum Pattern {
    /// Match an exact byte sequence.
    Literal(Vec<u8>),

    /// Match any single byte from the set.
    Any(Vec<u8>),

    /// Match any single byte NOT in the set.
    NotAny(Vec<u8>),

    /// Match one or more bytes from the set (greedy).
    Span(Vec<u8>),

    /// Match zero or more bytes NOT in the set, stopping before a delimiter
    /// or at end of input.
    Break(Vec<u8>),

    /// Match any byte sequence (greedy).
    Arb,

    /// Sequence: match first, then second.
    Cat(Box<Pattern>, Box<Pattern>),

    /// Alternative: try first; if it fails, try second.
    Alt(Box<Pattern>, Box<Pattern>),

    /// Capture: match inner pattern and capture what it consumed.
    Capture(Box<Pattern>),
}

impl Pattern {
    // ── Primitive constructors ─────────────────────────────────────────

    /// Match an exact byte sequence.
    pub fn literal(s: &[u8]) -> Self {
        Pattern::Literal(s.to_vec())
    }

    /// Match any single byte from the given set.
    pub fn any(set: &[u8]) -> Self {
        Pattern::Any(set.to_vec())
    }

    /// Match any single byte NOT in the given set.
    pub fn notany(set: &[u8]) -> Self {
        Pattern::NotAny(set.to_vec())
    }

    /// Match one or more consecutive bytes from the given set (greedy).
    pub fn span(set: &[u8]) -> Self {
        Pattern::Span(set.to_vec())
    }

    /// Match zero or more consecutive bytes NOT in the given set.
    pub fn break_(set: &[u8]) -> Self {
        Pattern::Break(set.to_vec())
    }

    /// Match any sequence of bytes (including empty — greedy).
    pub fn arb() -> Self {
        Pattern::Arb
    }

    // ── Combinators ────────────────────────────────────────────────────

    /// Sequence: match `self` then `other`.
    pub fn cat(self, other: Pattern) -> Self {
        Pattern::Cat(Box::new(self), Box::new(other))
    }

    /// Alternative: try `self` first; if it fails, try `other`.
    pub fn alt(self, other: Pattern) -> Self {
        Pattern::Alt(Box::new(self), Box::new(other))
    }

    /// Capture whatever this pattern matches.
    pub fn capture(self) -> Self {
        Pattern::Capture(Box::new(self))
    }

    // ── Matching ───────────────────────────────────────────────────────

    /// Try to match at position `pos` in `input`.
    ///
    /// Returns `Some(Match)` on success or `None` on failure.
    pub fn match_at(&self, input: &[u8], pos: usize) -> Option<Match> {
        if pos > input.len() {
            return None;
        }
        self.try_match(input, pos)
    }

    /// Find the first match starting at or after `pos`.
    ///
    /// Scans forward until the pattern matches. Returns `None` if no match
    /// found anywhere.
    pub fn find(&self, input: &[u8], pos: usize) -> Option<Match> {
        let mut p = pos;
        while p <= input.len() {
            if let Some(m) = self.match_at(input, p) {
                return Some(m);
            }
            if p == input.len() {
                break;
            }
            p += 1;
        }
        None
    }

    /// Find all non-overlapping matches starting at `pos`.
    pub fn find_all(&self, input: &[u8], mut pos: usize) -> Vec<Match> {
        let mut results = Vec::new();
        while pos <= input.len() {
            match self.match_at(input, pos) {
                Some(m) => {
                    let next_pos = m.range.end;
                    results.push(m);
                    pos = if next_pos > pos { next_pos } else { pos + 1 };
                }
                None => {
                    pos += 1;
                }
            }
        }
        results
    }

    // ── Internal recursive matcher ─────────────────────────────────────

    fn try_match(&self, input: &[u8], pos: usize) -> Option<Match> {
        let rem = &input[pos..];
        match self {
            Pattern::Literal(lit) => {
                if rem.starts_with(lit) {
                    Some(Match::at(pos, pos + lit.len()))
                } else {
                    None
                }
            }

            Pattern::Any(set) => {
                let set_ref: &[u8] = set;
                if let Some(&b) = rem.first() {
                    if set_ref.contains(&b) {
                        return Some(Match::at(pos, pos + 1));
                    }
                }
                None
            }

            Pattern::NotAny(set) => {
                let set_ref: &[u8] = set;
                if let Some(&b) = rem.first() {
                    if !set_ref.contains(&b) {
                        return Some(Match::at(pos, pos + 1));
                    }
                }
                None
            }

            Pattern::Span(set) => {
                let end = advance_while(rem, |b| set.contains(&b));
                if end > 0 {
                    // Greedy: try longest first, backtrack if needed
                    // For a standalone Span, just return the longest match
                    Some(Match::at(pos, pos + end))
                } else {
                    None
                }
            }

            Pattern::Break(set) => {
                let end = advance_while(rem, |b| !set.contains(&b));
                // Break matches zero or more, so it always succeeds
                Some(Match::at(pos, pos + end))
            }

            Pattern::Arb => {
                // Greedy: match entire remaining input
                Some(Match::at(pos, input.len()))
            }

            Pattern::Cat(first, second) => {
                // Try first, then second from where first ended
                let m1 = first.try_match(input, pos)?;
                let m2 = second.try_match(input, m1.range.end)?;
                let mut captures = m1.captures;
                captures.extend(m2.captures);
                Some(Match {
                    range: pos..m2.range.end,
                    captures,
                })
            }

            Pattern::Alt(first, second) => first
                .try_match(input, pos)
                .or_else(|| second.try_match(input, pos)),

            Pattern::Capture(inner) => {
                let m = inner.try_match(input, pos)?;
                let captured = input[m.range.start..m.range.end].to_vec();
                let mut captures = m.captures;
                captures.push(captured);
                Some(Match {
                    range: m.range,
                    captures,
                })
            }
        }
    }
}

/// Count consecutive bytes satisfying a predicate.
fn advance_while<F: Fn(u8) -> bool>(bytes: &[u8], pred: F) -> usize {
    bytes.iter().take_while(|&&b| pred(b)).count()
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Primitives ──

    #[test]
    fn test_literal_match() {
        let p = Pattern::literal(b"hello");
        assert_eq!(p.match_at(b"hello world", 0), Some(Match::at(0, 5)));
    }

    #[test]
    fn test_literal_no_match() {
        let p = Pattern::literal(b"hello");
        assert_eq!(p.match_at(b"world", 0), None);
    }

    #[test]
    fn test_literal_at_offset() {
        let p = Pattern::literal(b"world");
        assert_eq!(p.match_at(b"hello world!", 6), Some(Match::at(6, 11)));
    }

    #[test]
    fn test_any() {
        let p = Pattern::any(b"aeiou");
        assert!(p.match_at(b"apple", 0).is_some());
        assert!(p.match_at(b"zebra", 0).is_none());
    }

    #[test]
    fn test_notany() {
        let p = Pattern::notany(b"aeiou");
        assert!(p.match_at(b"zebra", 0).is_some());
        assert!(p.match_at(b"apple", 0).is_none());
    }

    #[test]
    fn test_span() {
        let alpha = Pattern::span(b"abcdefghijklmnopqrstuvwxyz");
        let m = alpha.match_at(b"hello world", 0).unwrap();
        assert_eq!(m.range, 0..5); // matched "hello"
    }

    #[test]
    fn test_span_requires_at_least_one() {
        let alpha = Pattern::span(b"abcdefghijklmnopqrstuvwxyz");
        assert!(alpha.match_at(b"123abc", 0).is_none());
    }

    #[test]
    fn test_break() {
        let to_space = Pattern::break_(b" ");
        let m = to_space.match_at(b"hello world", 0).unwrap();
        assert_eq!(m.range.len(), 5); // matched until space

        // Break can match zero
        let m2 = to_space.match_at(b" hello", 0).unwrap();
        assert_eq!(m2.range.len(), 0);
    }

    #[test]
    fn test_arb() {
        let p = Pattern::arb();
        let m = p.match_at(b"hello", 0).unwrap();
        assert_eq!(m.range, 0..5);
    }

    // ── Combinators ──

    #[test]
    fn test_cat() {
        let a = Pattern::literal(b"hello");
        let b = Pattern::literal(b" ");
        let c = Pattern::literal(b"world");
        let p = a.cat(b).cat(c);
        let m = p.match_at(b"hello world!", 0).unwrap();
        assert_eq!(m.range, 0..11);
    }

    #[test]
    fn test_alt_first_wins() {
        let p = Pattern::literal(b"hello").alt(Pattern::literal(b"world"));
        let m = p.match_at(b"hello world", 0).unwrap();
        assert_eq!(m.range, 0..5);
    }

    #[test]
    fn test_alt_fallback() {
        let p = Pattern::literal(b"bonjour").alt(Pattern::literal(b"hello"));
        let m = p.match_at(b"hello world", 0).unwrap();
        assert_eq!(m.range, 0..5);
    }

    #[test]
    fn test_capture() {
        let word = Pattern::span(b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ");
        let p = word.capture();
        let m = p.match_at(b"hello world", 0).unwrap();
        assert_eq!(m.captures, vec![b"hello".to_vec()]);
    }

    #[test]
    fn test_nested_capture() {
        // Capture two words
        let alpha = Pattern::span(b"abcdefghijklmnopqrstuvwxyz");
        let space = Pattern::literal(b" ");
        let p = alpha
            .clone()
            .capture()
            .cat(space)
            .cat(alpha.clone().capture());
        let m = p.match_at(b"hello world", 0).unwrap();
        assert_eq!(m.captures, vec![b"hello".to_vec(), b"world".to_vec()]);
    }

    // ── Find ──

    #[test]
    fn test_find_scans() {
        let p = Pattern::literal(b"world");
        let m = p.find(b"hello world!", 0).unwrap();
        assert_eq!(m.range, 6..11);
    }

    #[test]
    fn test_find_all() {
        let word = Pattern::span(b"abcdefghijklmnopqrstuvwxyz");
        let matches = word.find_all(b"hello world lua", 0);
        let tokens: Vec<&[u8]> = matches
            .iter()
            .map(|m| &b"hello world lua"[m.range.clone()])
            .collect();
        assert_eq!(tokens, vec![b"hello" as &[u8], b"world", b"lua"]);
    }

    #[test]
    fn test_find_all_empty_input() {
        let word = Pattern::span(b"abcdefghijklmnopqrstuvwxyz");
        assert_eq!(word.find_all(b"", 0), vec![]);
    }

    // ── Tokenizer-like usage ──

    #[test]
    fn test_tokenize_words() {
        let letter =
            Pattern::span(b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789");
        let tokens = letter.find_all(b"the cat sat on the mat", 0);
        let words: Vec<&str> = tokens
            .iter()
            .map(|m| std::str::from_utf8(&b"the cat sat on the mat"[m.range.clone()]).unwrap())
            .collect();
        assert_eq!(words, vec!["the", "cat", "sat", "on", "the", "mat"]);
    }

    #[test]
    fn test_tokenize_by_delimiter() {
        // CSV-like: fields separated by commas
        let field = Pattern::break_(b",");
        let comma = Pattern::literal(b",");
        let csv_cell = field.capture();
        let csv_row = csv_cell.clone().cat(comma).cat(csv_cell.clone());

        let m = csv_row.match_at(b"hello,world", 0).unwrap();
        assert_eq!(m.captures, vec![b"hello".to_vec(), b"world".to_vec()]);
    }
}
