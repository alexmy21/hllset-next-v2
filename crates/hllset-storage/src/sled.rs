//! Sled-backed storage — the embedded default backend (gen2).
//!
//! Content-addressing is provided by `hllset-cid` (embedded SHA-1 CIDs):
//! no external daemon, no network calls, no `ipfrs-core` dependency, pure
//! Rust on top of `sled`.

use crate::storage::{Result, Storage, StorageError};
use hllset_cid::Cid;
use sled::Db;
use std::path::PathBuf;

/// Sled-backed content-addressed storage — the default embedded backend.
#[derive(Clone)]
pub struct SledStorage {
    db: Db,
}

impl SledStorage {
    /// Open (or create) a sled database at `path`.
    pub fn open(path: impl Into<PathBuf>) -> Result<Self> {
        let db = sled::open(path.into())
            .map_err(|e| StorageError::Backend(format!("sled open: {e}")))?;
        Ok(Self { db })
    }

    /// Open a temporary database (for testing).
    pub fn open_temp() -> Result<Self> {
        let db = sled::Config::new()
            .temporary(true)
            .open()
            .map_err(|e| StorageError::Backend(format!("sled temp open: {e}")))?;
        Ok(Self { db })
    }

    /// Compute the embedded CID for the given data.
    pub fn compute_cid(data: &[u8]) -> Cid {
        Cid::compute(data)
    }
}

impl Storage for SledStorage {
    fn put(&self, key: &str, data: &[u8]) -> Result<()> {
        self.db
            .insert(key.as_bytes(), data)
            .map_err(|e| StorageError::Backend(format!("sled insert: {e}")))?;
        Ok(())
    }

    fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        self.db
            .get(key.as_bytes())
            .map_err(|e| StorageError::Backend(format!("sled get: {e}")))
            .map(|opt| opt.map(|ivec| ivec.to_vec()))
    }

    fn has(&self, key: &str) -> Result<bool> {
        self.db
            .contains_key(key.as_bytes())
            .map_err(|e| StorageError::Backend(format!("sled contains_key: {e}")))
    }

    fn delete(&self, key: &str) -> Result<bool> {
        self.db
            .remove(key.as_bytes())
            .map_err(|e| StorageError::Backend(format!("sled remove: {e}")))
            .map(|opt| opt.is_some())
    }

    fn list(&self, prefix: &str) -> Result<Vec<String>> {
        let prefix_bytes = prefix.as_bytes();
        let mut keys = Vec::new();
        for item in self.db.scan_prefix(prefix_bytes) {
            let (k, _) = item.map_err(|e| StorageError::Backend(format!("sled scan: {e}")))?;
            if let Ok(s) = String::from_utf8(k.to_vec()) {
                keys.push(s);
            }
        }
        Ok(keys)
    }
}

/// Compatibility alias for the former ipfrs-native backend name.
pub type IpfrsNativeStorage = SledStorage;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::MemoryStorage;

    #[test]
    fn test_sled_store_and_load() {
        let s = SledStorage::open_temp().unwrap();
        let key = "h:test_native";
        let data = b"hello sled-native";

        s.store(key, data).unwrap();
        assert!(s.exists(key).unwrap());

        let loaded = s.load(key).unwrap().unwrap();
        assert_eq!(loaded, data);
    }

    #[test]
    fn test_sled_list() {
        let s = SledStorage::open_temp().unwrap();
        s.store("h:alpha", b"a").unwrap();
        s.store("h:beta", b"b").unwrap();
        s.store("c:gamma", b"c").unwrap();

        let h_keys = s.list("h:").unwrap();
        assert_eq!(h_keys.len(), 2);
        assert!(h_keys.contains(&"h:alpha".to_string()));
        assert!(h_keys.contains(&"h:beta".to_string()));

        let c_keys = s.list("c:").unwrap();
        assert_eq!(c_keys.len(), 1);
        assert_eq!(c_keys[0], "c:gamma");
    }

    #[test]
    fn test_sled_delete() {
        let s = SledStorage::open_temp().unwrap();
        s.store("h:del", b"x").unwrap();
        assert!(s.exists("h:del").unwrap());
        assert!(s.delete("h:del").unwrap());
        assert!(!s.exists("h:del").unwrap());
        // Deleting again returns false
        assert!(!s.delete("h:del").unwrap());
    }

    #[test]
    fn test_compute_cid_is_embedded_and_content_addressed() {
        let cid = SledStorage::compute_cid(b"hello world");
        let s = cid.to_string();
        assert!(s.starts_with("h:"));
        assert_eq!(s.len(), 42, "h: + 40 hex SHA-1");
        assert_eq!(cid, Cid::compute(b"hello world"), "deterministic");
    }

    /// MemoryStorage still passes its original tests unchanged.
    #[test]
    fn test_memory_storage_still_works() {
        let s = MemoryStorage::new();
        s.store("h:mem", b"data").unwrap();
        assert_eq!(s.load("h:mem").unwrap().unwrap(), b"data");
    }
}
