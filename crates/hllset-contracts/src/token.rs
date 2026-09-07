//! Token identity contract (TRANSITION §3.1; nanoLM pre-flight Q1).
//!
//! The id EWM hashes (`tid{n}`) is exactly the integer that indexes the LLM's
//! `wte[n]` / `lm_head[:,n]`. Divergence is a correctness bug, never a
//! configuration option.

/// An LLM encoding id (`llm.token`), matching nanoLM's `TokenId`.
pub type TokenId = u32;

/// Inscribe a token id for EWM as opaque bytes: `tid{n}`.
///
/// This is the bit-exact byte encoding pinned by nanoLM
/// (`IMPLEMENTATION_PLAN.md` §1: LLM id `n` → `"tid{n}"`).
pub fn token_in_bytes(id: TokenId) -> Vec<u8> {
    format!("tid{id}").into_bytes()
}

/// Inscribe a token id as 4-byte little-endian bytes.
///
/// This is the bit-exact byte encoding used by notebook-08
/// (`token_bytes(tid) = tid.to_bytes(4, "little")`), kept alongside
/// [`token_in_bytes`] because the bridge refactors **both** golden projects:
/// notebook-08 (LE) and nanoLM (`tid{n}`).
pub fn token_in_bytes_le(id: TokenId) -> [u8; 4] {
    id.to_le_bytes()
}

/// Parse the nanoLM inscription `tid{n}` back to the id.
pub fn parse_token_id(bytes: &[u8]) -> Option<TokenId> {
    let s = std::str::from_utf8(bytes).ok()?;
    s.strip_prefix("tid")?.parse().ok()
}

/// Parse the notebook-08 inscription (4-byte LE) back to the id.
pub fn parse_token_id_le(bytes: &[u8]) -> Option<TokenId> {
    let arr: [u8; 4] = bytes.try_into().ok()?;
    Some(TokenId::from_le_bytes(arr))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_in_bytes_is_tid_n() {
        assert_eq!(token_in_bytes(0), b"tid0");
        assert_eq!(token_in_bytes(7), b"tid7");
        assert_eq!(token_in_bytes(50256), b"tid50256");
        assert_eq!(
            token_in_bytes(u32::MAX),
            format!("tid{}", u32::MAX).into_bytes()
        );
    }

    #[test]
    fn token_in_bytes_le_is_4_byte_little_endian() {
        assert_eq!(token_in_bytes_le(44), [44, 0, 0, 0]);
        assert_eq!(token_in_bytes_le(0x0102_0304), [4, 3, 2, 1]);
        assert_eq!(token_in_bytes_le(u32::MAX), [255, 255, 255, 255]);
    }

    #[test]
    fn parsers_are_the_inverse_of_the_inscriptions() {
        assert_eq!(parse_token_id(b"tid0"), Some(0));
        assert_eq!(parse_token_id(b"tid50256"), Some(50256));
        assert_eq!(parse_token_id(b"token"), None);
        assert_eq!(parse_token_id_le(&token_in_bytes_le(42)), Some(42));
        assert_eq!(parse_token_id_le(b"tid50256"), None, "not 4 bytes");
    }

    #[test]
    fn token_id_is_u32() {
        // The wire id width is part of the soldered contract.
        assert_eq!(TokenId::BITS, 32);
    }
}
