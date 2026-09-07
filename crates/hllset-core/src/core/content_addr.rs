//! Content-addressable key generation for HLLSet.
//!
//! Every HLLSet has a deterministic content key derived from its data.
//! This module provides key generation for all namespace prefixes per
//! STANDARD.md §2.2.
//!
//! ## Namespace Prefixes
//!
//! | Prefix | Name        | Meaning                                    |
//! |--------|-------------|--------------------------------------------|
//! | `o:`   | Original    | From tokenizer, immutable                  |
//! | `h:`   | HLLSet      | Any operation result                       |
//! | `r:`   | Retained    | R-link (intersection)                      |
//! | `d:`   | Departed    | Difference                                 |
//! | `n:`   | New         | Difference                                 |
//! | `t:`   | Commit      | Lattice evolution record                   |
//! | `v:`   | View        | Ephemeral, not persisted                   |
//! | `l:`   | LLM context | Human annotation                           |
//! | `c:`   | Catalog     | Homogeneous/enumerable data                |
//! | `u:`   | User        | UUID, user-assigned temporal               |
//! | `system:` | System   | Named global (tf, head, globals)           |

use crate::core::hashing::sha1_hex;

/// Valid namespace prefixes for content-addressed identifiers.
pub const VALID_PREFIXES: &[&str] = &["o", "h", "r", "d", "n", "t", "v", "l", "c", "u"];

/// System prefix for temporal/named keys.
pub const SYSTEM_PREFIX: &str = "system:";

/// Check whether a CID prefix is valid.
pub fn is_valid_prefix(prefix: &str) -> bool {
    VALID_PREFIXES.contains(&prefix)
}

/// Build a content-addressed ID: `{prefix}:{sha1_hex}`.
///
/// The prefix must be one of the valid namespace prefixes.
/// Returns `None` if the prefix is invalid.
pub fn make_cid(prefix: &str, sha1_hex: &str) -> Option<String> {
    if !is_valid_prefix(prefix) {
        return None;
    }
    if sha1_hex.len() != 40 || !sha1_hex.chars().all(|c| c.is_ascii_hexdigit()) {
        return None;
    }
    Some(format!("{prefix}:{sha1_hex}"))
}

/// Parse a CID string into (prefix, sha1_hex).
///
/// Returns `None` if the string doesn't match `{prefix}:{40-hex}` format.
pub fn parse_cid(cid: &str) -> Option<(&str, &str)> {
    let colon_pos = cid.find(':')?;
    let prefix = &cid[..colon_pos];
    let sha1 = &cid[colon_pos + 1..];
    if !is_valid_prefix(prefix) || sha1.len() != 40 {
        return None;
    }
    if !sha1.chars().all(|c| c.is_ascii_hexdigit()) {
        return None;
    }
    Some((prefix, sha1))
}

// ── Per-prefix generators ──────────────────────────────────────────────

/// Generate a content key with a specific prefix.
///
/// Generic helper: sorts and deduplicates tokens, hashes, returns `{prefix}:{sha1}`.
fn content_key_with_prefix<'a, I, B>(tokens: I, prefix: &str) -> String
where
    I: IntoIterator<Item = &'a B>,
    B: AsRef<[u8]> + 'a,
{
    let mut sorted: Vec<Vec<u8>> = tokens.into_iter().map(|t| t.as_ref().to_vec()).collect();
    sorted.sort();
    sorted.dedup();

    let mut canonical = Vec::new();
    for (i, token) in sorted.iter().enumerate() {
        if i > 0 {
            canonical.push(0u8);
        }
        canonical.extend_from_slice(token);
    }
    format!("{prefix}:{}", sha1_hex(&canonical))
}

/// Generate a heterogeneous content key: `h:<sha1>`.
pub fn content_key_from_tokens<'a, I, B>(tokens: I) -> String
where
    I: IntoIterator<Item = &'a B>,
    B: AsRef<[u8]> + 'a,
{
    content_key_with_prefix(tokens, "h")
}

/// Generate an original content key: `o:<sha1>`.
///
/// For HLLSets produced directly from a tokenizer — immutable source data.
pub fn original_key_from_tokens<'a, I, B>(tokens: I) -> String
where
    I: IntoIterator<Item = &'a B>,
    B: AsRef<[u8]> + 'a,
{
    content_key_with_prefix(tokens, "o")
}

/// Generate a retained (R-link) content key: `r:<sha1>`.
pub fn retained_key_from_tokens<'a, I, B>(tokens: I) -> String
where
    I: IntoIterator<Item = &'a B>,
    B: AsRef<[u8]> + 'a,
{
    content_key_with_prefix(tokens, "r")
}

/// Generate a departed content key: `d:<sha1>`.
pub fn departed_key_from_tokens<'a, I, B>(tokens: I) -> String
where
    I: IntoIterator<Item = &'a B>,
    B: AsRef<[u8]> + 'a,
{
    content_key_with_prefix(tokens, "d")
}

/// Generate a new content key: `n:<sha1>`.
pub fn new_key_from_tokens<'a, I, B>(tokens: I) -> String
where
    I: IntoIterator<Item = &'a B>,
    B: AsRef<[u8]> + 'a,
{
    content_key_with_prefix(tokens, "n")
}

/// Generate a view content key: `v:<sha1>`.
///
/// Views are ephemeral — they are not persisted by default.
pub fn view_key_from_tokens<'a, I, B>(tokens: I) -> String
where
    I: IntoIterator<Item = &'a B>,
    B: AsRef<[u8]> + 'a,
{
    content_key_with_prefix(tokens, "v")
}

/// Generate an LLM context key: `l:<sha1>`.
///
/// For human-written or auto-generated annotations bridging prompts to code.
pub fn llm_context_key_from_tokens<'a, I, B>(tokens: I) -> String
where
    I: IntoIterator<Item = &'a B>,
    B: AsRef<[u8]> + 'a,
{
    content_key_with_prefix(tokens, "l")
}

/// Generate a commit key from raw bytes: `t:<sha1>`.
pub fn commit_key_from_bytes(data: &[u8]) -> String {
    format!("t:{}", sha1_hex(data))
}

/// Generate a homogeneous (catalog) content key: `c:<sha1>`.
pub fn content_key_from_catalog<I, B>(catalog_values: I) -> String
where
    I: IntoIterator<Item = B>,
    B: AsRef<[u8]>,
{
    let mut sorted: Vec<Vec<u8>> = catalog_values
        .into_iter()
        .map(|v| v.as_ref().to_vec())
        .collect();
    sorted.sort();
    sorted.dedup();

    let mut canonical = Vec::new();
    for (i, val) in sorted.iter().enumerate() {
        if i > 0 {
            canonical.push(0u8);
        }
        canonical.extend_from_slice(val);
    }

    format!("c:{}", sha1_hex(&canonical))
}

/// Generate a user-assigned temporal key: `u:<uuid>`.
///
/// UUID is 32 hex chars (no dashes, per canonical format).
pub fn user_key_from_uuid(uuid: &str) -> Option<String> {
    let uuid = uuid.replace('-', "");
    if uuid.len() != 32 || !uuid.chars().all(|c| c.is_ascii_hexdigit()) {
        return None;
    }
    Some(format!("u:{uuid}"))
}

/// Generate a system key: `system:{name}`.
pub fn system_key(name: &str) -> String {
    format!("{SYSTEM_PREFIX}{name}")
}

/// Generate an ontological catalog key from structural position.
///
/// Returns a raw SHA1 hex string (no prefix — the consumer adds `c:`).
pub fn ontological_key(parent_sha1: &str, seq: u64) -> String {
    let mut data = parent_sha1.as_bytes().to_vec();
    data.extend_from_slice(&seq.to_le_bytes());
    sha1_hex(&data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_content_key_from_tokens_deterministic() {
        let k1 = content_key_from_tokens(&[b"hello", b"world"]);
        let k2 = content_key_from_tokens(&[b"world", b"hello"]); // different order
        assert_eq!(k1, k2, "key must be order-independent");
    }

    #[test]
    fn test_content_key_from_tokens_prefix() {
        let key = content_key_from_tokens(&[b"test"]);
        assert!(key.starts_with("h:"));
        assert_eq!(key.len(), 42); // "h:" + 40 hex
    }

    #[test]
    fn test_content_key_from_tokens_dedup() {
        let k1 = content_key_from_tokens(&[b"a", b"a", b"b"]);
        let k2 = content_key_from_tokens(&[b"a", b"b"]);
        assert_eq!(k1, k2, "duplicates should not affect key");
    }

    #[test]
    fn test_content_key_different_for_different_data() {
        let k1 = content_key_from_tokens(&[b"x"]);
        let k2 = content_key_from_tokens(&[b"y"]);
        assert_ne!(k1, k2);
    }

    #[test]
    fn test_catalog_key_prefix() {
        let key = content_key_from_catalog(&[b"alice@example.com"]);
        assert!(key.starts_with("c:"));
        assert_eq!(key.len(), 42);
    }

    #[test]
    fn test_ontological_key_deterministic() {
        let k1 = ontological_key("a3f82c1d", 42);
        let k2 = ontological_key("a3f82c1d", 42);
        assert_eq!(k1, k2);
    }

    #[test]
    fn test_ontological_key_differs_by_seq() {
        let k1 = ontological_key("a3f82c1d", 1);
        let k2 = ontological_key("a3f82c1d", 2);
        assert_ne!(k1, k2);
    }

    // ── Namespace prefix tests ─────────────────────────────────────

    #[test]
    fn test_valid_prefixes() {
        assert!(is_valid_prefix("h"));
        assert!(is_valid_prefix("o"));
        assert!(is_valid_prefix("r"));
        assert!(is_valid_prefix("d"));
        assert!(is_valid_prefix("n"));
        assert!(is_valid_prefix("t"));
        assert!(is_valid_prefix("v"));
        assert!(is_valid_prefix("l"));
        assert!(is_valid_prefix("c"));
        assert!(is_valid_prefix("u"));
        assert!(!is_valid_prefix("x"));
        assert!(!is_valid_prefix(""));
    }

    #[test]
    fn test_make_cid() {
        let cid = make_cid("h", "a3f82c1d00000000000000000000000000000000").unwrap();
        assert_eq!(cid, "h:a3f82c1d00000000000000000000000000000000");
    }

    #[test]
    fn test_make_cid_invalid_prefix() {
        assert!(make_cid("x", "a3f82c1d00000000000000000000000000000000").is_none());
    }

    #[test]
    fn test_make_cid_invalid_sha1() {
        assert!(make_cid("h", "too_short").is_none());
    }

    #[test]
    fn test_parse_cid() {
        let (prefix, sha1) = parse_cid("h:a3f82c1d00000000000000000000000000000000").unwrap();
        assert_eq!(prefix, "h");
        assert_eq!(sha1, "a3f82c1d00000000000000000000000000000000");
    }

    #[test]
    fn test_parse_cid_invalid() {
        assert!(parse_cid("x:abc").is_none());
        assert!(parse_cid("h:short").is_none());
        assert!(parse_cid("no_colon").is_none());
    }

    #[test]
    fn test_original_key() {
        let key = original_key_from_tokens(&[b"source"]);
        assert!(key.starts_with("o:"));
        assert_eq!(key.len(), 42);
    }

    #[test]
    fn test_retained_key() {
        let key = retained_key_from_tokens(&[b"shared"]);
        assert!(key.starts_with("r:"));
    }

    #[test]
    fn test_departed_key() {
        let key = departed_key_from_tokens(&[b"old"]);
        assert!(key.starts_with("d:"));
    }

    #[test]
    fn test_new_key() {
        let key = new_key_from_tokens(&[b"fresh"]);
        assert!(key.starts_with("n:"));
    }

    #[test]
    fn test_view_key() {
        let key = view_key_from_tokens(&[b"ephemeral"]);
        assert!(key.starts_with("v:"));
    }

    #[test]
    fn test_llm_context_key() {
        let key = llm_context_key_from_tokens(&[b"doc comment"]);
        assert!(key.starts_with("l:"));
    }

    #[test]
    fn test_user_key() {
        let key = user_key_from_uuid("550e8400e29b41d4a716446655440000").unwrap();
        assert_eq!(key, "u:550e8400e29b41d4a716446655440000");
    }

    #[test]
    fn test_user_key_rejects_invalid() {
        assert!(user_key_from_uuid("too_short").is_none());
    }

    #[test]
    fn test_system_key() {
        let key = system_key("tf");
        assert_eq!(key, "system:tf");
    }

    #[test]
    fn test_commit_key() {
        let key = commit_key_from_bytes(b"{\"ts\":123}");
        assert!(key.starts_with("t:"));
        assert_eq!(key.len(), 42);
    }

    #[test]
    fn test_all_prefix_keys_are_deterministic() {
        let tokens = &[b"x", b"y"];
        let k1 = content_key_from_tokens(tokens);
        let k2 = content_key_from_tokens(tokens);
        assert_eq!(k1, k2);
        let o1 = original_key_from_tokens(tokens);
        let o2 = original_key_from_tokens(tokens);
        assert_eq!(o1, o2);
    }
}
