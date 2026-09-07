//! HLPP Storage trait for HLLSet data.
//!
//! Implements the HLPP (HLLSet Lattice Persistence Protocol) algebraic
//! specification as a Rust trait. See STANDARD.md §2.5 for the canonical
//! specification.

/// Storage errors.
#[derive(Debug, thiserror::Error)]
pub enum StorageError {
    #[error("backend error: {0}")]
    Backend(String),
    #[error("not found: {0}")]
    NotFound(String),
    #[error("serialization: {0}")]
    Serialization(String),
    #[error("CAS mismatch: expected {expected:?}, got {actual:?}")]
    CasMismatch { expected: Vec<u8>, actual: Vec<u8> },
}

/// Convenience result type.
pub type Result<T> = std::result::Result<T, StorageError>;

/// Content-addressed + temporal storage backend (HLPP).
///
/// Implements the HLPP algebraic specification (§2.5 of STANDARD.md).
/// Keys follow HLLSet conventions: `h:<sha1>` for heterogeneous data,
/// `c:<sha1>` for homogeneous/catalog data.
///
/// # CA (Content-Addressed) Operations
///
/// The canonical names are `put`/`get`/`has` (from the HLPP spec). The
/// legacy names `store`/`load`/`exists` are provided as default methods
/// that delegate to the canonical names. New code should use the canonical
/// names; old code continues to work via the default delegations.
///
/// # Temporal Operations
///
/// `put_tmp`/`get_tmp`/`cas_tmp` provide mutable, named-key storage for
/// system state (TF vectors, head pointer, globals). These are NOT
/// content-addressed — keys are user-assigned names like `system:tf`.
pub trait Storage {
    // ── CA Operations (canonical HLPP names) ───────────────────────────

    /// PUT: store raw bytes under a content-addressed key.
    /// Idempotent — multiple PUTs with the same key have no effect.
    fn put(&self, key: &str, data: &[u8]) -> Result<()>;

    /// GET: load raw bytes by key. Returns `None` if not found.
    fn get(&self, key: &str) -> Result<Option<Vec<u8>>>;

    /// HAS: check whether a key exists in storage.
    fn has(&self, key: &str) -> Result<bool>;

    /// Delete a key and its data.
    fn delete(&self, key: &str) -> Result<bool>;

    /// LIST: list keys matching a prefix (e.g., "h:" or "c:").
    fn list(&self, prefix: &str) -> Result<Vec<String>>;

    /// PIN: prevent garbage collection. Idempotent.
    /// Default: no-op (backends that don't support GC don't need pins).
    fn pin(&self, _key: &str) -> Result<()> {
        Ok(())
    }

    /// UNPIN: allow garbage collection. Idempotent.
    fn unpin(&self, _key: &str) -> Result<()> {
        Ok(())
    }

    /// GC: garbage collect — remove all unpinned keys. Returns removed keys.
    /// Default: returns empty (backends without GC support).
    fn gc(&self) -> Result<Vec<String>> {
        Ok(Vec::new())
    }

    // ── Temporal Operations ────────────────────────────────────────────

    /// PUT_TMP: store bytes under a temporal (user-assigned) key.
    /// Not content-addressed — key is a human-readable name like `system:tf`.
    /// Default: no-op (backends without temporal support).
    fn put_tmp(&self, _key: &str, _data: &[u8]) -> Result<()> {
        Ok(())
    }

    /// GET_TMP: load bytes from a temporal key. Returns `None` if not found.
    /// Default: returns `None`.
    fn get_tmp(&self, _key: &str) -> Result<Option<Vec<u8>>> {
        Ok(None)
    }

    /// CAS_TMP: compare-and-swap on a temporal key.
    /// If current value equals `old`, atomically replace with `new`.
    /// Returns `true` if the swap succeeded, `false` if the current value
    /// doesn't match `old`.
    /// Default: returns `Err` (backends without CAS support).
    fn cas_tmp(&self, _key: &str, _old: &[u8], _new: &[u8]) -> Result<bool> {
        Err(StorageError::Backend(
            "CAS_TMP not supported by this backend".into(),
        ))
    }

    // ── Legacy aliases (delegate to canonical names) ──────────────────

    /// Legacy alias for `put`. New code should use `put`.
    fn store(&self, key: &str, data: &[u8]) -> Result<()> {
        self.put(key, data)
    }

    /// Legacy alias for `get`. New code should use `get`.
    fn load(&self, key: &str) -> Result<Option<Vec<u8>>> {
        self.get(key)
    }

    /// Legacy alias for `has`. New code should use `has`.
    fn exists(&self, key: &str) -> Result<bool> {
        self.has(key)
    }
}
