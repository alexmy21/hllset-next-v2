//! `hllset-cid` — embedded content-addressed identifiers.
//!
//! The gen2 default substitute for the external `ipfrs-core` dependency: a
//! [`Cid`] is the SHA-1 of the block bytes, displayed with the HLLSet `h:`
//! prefix. Std-only, deterministic, content-addressed — the same identity law
//! as the rest of the algebra. A full ipfrs/IPFS CID backend can be plugged
//! in later behind this type without touching storage.

use hllset_contracts::sha1_hex;

/// A content-addressed identifier: SHA-1 over the data.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Cid {
    /// 40-hex lowercase SHA-1 of the data.
    hex: String,
}

impl Cid {
    /// Compute the CID of a block of bytes.
    pub fn compute(data: &[u8]) -> Self {
        Self {
            hex: sha1_hex(data),
        }
    }

    /// The bare 40-hex digest.
    pub fn as_str(&self) -> &str {
        &self.hex
    }

    /// The HLLSet content key form: `h:<sha1>`.
    pub fn prefixed(&self) -> String {
        format!("h:{}", self.hex)
    }
}

impl std::fmt::Display for Cid {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("h:")?;
        f.write_str(&self.hex)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_is_identity() {
        assert_eq!(
            Cid::compute(b"hello").to_string(),
            format!("h:{}", sha1_hex(b"hello"))
        );
    }

    #[test]
    fn deterministic_and_distinct() {
        assert_eq!(Cid::compute(b"a"), Cid::compute(b"a"));
        assert_ne!(Cid::compute(b"a"), Cid::compute(b"b"));
        assert_ne!(Cid::compute(b"a").as_str().len(), 0);
    }

    #[test]
    fn matches_contracts_sha1() {
        let cid = Cid::compute(b"hello world");
        assert_eq!(cid.as_str(), sha1_hex(b"hello world"));
        assert_eq!(cid.prefixed(), format!("h:{}", sha1_hex(b"hello world")));
    }
}
