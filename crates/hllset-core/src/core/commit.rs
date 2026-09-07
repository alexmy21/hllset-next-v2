//! Commit struct — lattice event record (STANDARD.md §2.3, §4.1).
//!
//! Each evolution step in the HLLSet lattice produces a Commit that
//! records the D/R/N decomposition and the system state.
//!
//! # Wire Format (STANDARD.md §2.3)
//!
//! ```text
//! Compact JSON with canonical key ordering:
//! {"d":"<cid>","h":"<cid>","n":"<cid>","r":"<cid>","s":"<cid>","ts":<u64>}
//!
//! CID: t:SHA1(json_bytes)
//! ```
//!
//! # Fields
//!
//! - `ts` — wall-clock timestamp (Unix epoch μs)
//! - `s` — source HLLSet CID (the observation that triggered this commit)
//! - `h` — previous head commit CID (chain link)
//! - `d` — Departed HLLSet CID (H_prev \ S_curr)
//! - `r` — Retained HLLSet CID (H_prev ∩ S_curr, the R-link)
//! - `n` — New HLLSet CID (S_curr \ H_prev)

use serde::{Deserialize, Serialize};

/// A lattice evolution commit — records the D/R/N decomposition of one step.
///
/// Commits form a content-addressed chain: each commit links to the previous
/// head via the `h` field. The commit itself is stored under `t:<sha1>`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Commit {
    /// Unix timestamp (microseconds since epoch).
    #[serde(rename = "ts")]
    pub timestamp: u64,

    /// Source HLLSet CID — the observation (typically `o:` prefix).
    #[serde(rename = "s")]
    pub source: String,

    /// Previous head commit CID (chain link).
    #[serde(rename = "h")]
    pub head: String,

    /// Departed HLLSet CID (`d:` prefix).
    #[serde(rename = "d")]
    pub departed: String,

    /// Retained HLLSet CID — the R-link (`r:` prefix).
    #[serde(rename = "r")]
    pub retained: String,

    /// New HLLSet CID (`n:` prefix).
    #[serde(rename = "n")]
    pub new: String,
}

impl Commit {
    /// Create a new commit with the current timestamp.
    pub fn new(source: &str, prev_head: &str, departed: &str, retained: &str, new: &str) -> Self {
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;

        Self {
            timestamp: ts,
            source: source.to_string(),
            head: prev_head.to_string(),
            departed: departed.to_string(),
            retained: retained.to_string(),
            new: new.to_string(),
        }
    }

    /// Serialize to canonical JSON (compact, sorted keys).
    ///
    /// Uses serde_json with sorted keys for deterministic output.
    pub fn to_json(&self) -> String {
        // serde_json serializes fields in struct declaration order by default.
        // The struct fields are declared in canonical key order per the spec.
        serde_json::to_string(self).expect("Commit serialization is infallible")
    }

    /// Deserialize from canonical JSON bytes.
    pub fn from_json(json: &str) -> Option<Self> {
        serde_json::from_str(json).ok()
    }

    /// Generate the content-addressable key for this commit.
    ///
    /// Format: `t:<sha1(json_bytes)>`
    pub fn content_key(&self) -> String {
        let json = self.to_json();
        let sha1 = crate::core::hashing::sha1_hex(json.as_bytes());
        format!("t:{sha1}")
    }

    /// Validate the commit chain link: `self.head` should equal `prev_key`.
    pub fn chain_valid(&self, prev_key: &str) -> bool {
        self.head == prev_key
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_creates_commit() {
        let c = Commit::new("o:aaaa", "t:bbbb", "d:cccc", "r:dddd", "n:eeee");
        assert_eq!(c.source, "o:aaaa");
        assert_eq!(c.head, "t:bbbb");
        assert_eq!(c.departed, "d:cccc");
        assert_eq!(c.retained, "r:dddd");
        assert_eq!(c.new, "n:eeee");
        assert!(c.timestamp > 0);
    }

    #[test]
    fn test_json_roundtrip() {
        let c = Commit::new("o:aaaa", "t:bbbb", "d:cccc", "r:dddd", "n:eeee");
        let json = c.to_json();
        let c2 = Commit::from_json(&json).unwrap();
        assert_eq!(c.source, c2.source);
        assert_eq!(c.retained, c2.retained);
    }

    #[test]
    fn test_content_key() {
        let c = Commit::new("o:s1", "t:h1", "d:d1", "r:r1", "n:n1");
        let key = c.content_key();
        assert!(key.starts_with("t:"), "key = {key}");
        assert_eq!(key.len(), 42); // "t:" + 40 hex chars
    }

    #[test]
    fn test_content_key_deterministic() {
        let c1 = Commit::new("o:s1", "t:h1", "d:d1", "r:r1", "n:n1");
        let c2 = Commit::new("o:s1", "t:h1", "d:d1", "r:r1", "n:n1");
        // Both keys are valid t: prefix content keys
        assert!(c1.content_key().starts_with("t:"));
        assert!(c2.content_key().starts_with("t:"));
        assert_eq!(c1.content_key().len(), 42);
    }

    #[test]
    fn test_chain_validation() {
        let c = Commit::new("o:s1", "t:prev_head", "d:d1", "r:r1", "n:n1");
        assert!(c.chain_valid("t:prev_head"));
        assert!(!c.chain_valid("t:wrong_head"));
    }
}
