//! In-memory storage backend for development and testing.
//!
//! Implements the full HLPP Storage trait including temporal operations
//! (put_tmp, get_tmp, cas_tmp).

use crate::storage::{Result, Storage, StorageError};
use std::collections::{BTreeMap, HashSet};

/// Content-addressed in-memory storage with temporal support.
///
/// All data lives in a `BTreeMap<String, Vec<u8>>`. Keys follow
/// HLLSet naming conventions (`h:`, `c:`). Temporal keys
/// (`system:`, `u:`) are stored separately.
#[derive(Clone, Debug, Default)]
pub struct MemoryStorage {
    data: std::rc::Rc<std::cell::RefCell<BTreeMap<String, Vec<u8>>>>,
    pinned: std::rc::Rc<std::cell::RefCell<HashSet<String>>>,
    temporal: std::rc::Rc<std::cell::RefCell<BTreeMap<String, Vec<u8>>>>,
}

impl MemoryStorage {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Storage for MemoryStorage {
    // ── CA operations (canonical names) ───────────────────────────────

    fn put(&self, key: &str, data: &[u8]) -> Result<()> {
        self.data
            .borrow_mut()
            .insert(key.to_string(), data.to_vec());
        Ok(())
    }

    fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        Ok(self.data.borrow().get(key).cloned())
    }

    fn has(&self, key: &str) -> Result<bool> {
        Ok(self.data.borrow().contains_key(key))
    }

    fn delete(&self, key: &str) -> Result<bool> {
        Ok(self.data.borrow_mut().remove(key).is_some())
    }

    fn list(&self, prefix: &str) -> Result<Vec<String>> {
        let keys: Vec<String> = self
            .data
            .borrow()
            .keys()
            .filter(|k| k.starts_with(prefix))
            .cloned()
            .collect();
        Ok(keys)
    }

    fn pin(&self, key: &str) -> Result<()> {
        self.pinned.borrow_mut().insert(key.to_string());
        Ok(())
    }

    fn unpin(&self, key: &str) -> Result<()> {
        self.pinned.borrow_mut().remove(key);
        Ok(())
    }

    fn gc(&self) -> Result<Vec<String>> {
        let pinned = self.pinned.borrow();
        let mut data = self.data.borrow_mut();
        let to_remove: Vec<String> = data
            .keys()
            .filter(|k| !pinned.contains(*k))
            .cloned()
            .collect();
        for k in &to_remove {
            data.remove(k);
        }
        Ok(to_remove)
    }

    // ── Temporal operations ───────────────────────────────────────────

    fn put_tmp(&self, key: &str, data: &[u8]) -> Result<()> {
        self.temporal
            .borrow_mut()
            .insert(key.to_string(), data.to_vec());
        Ok(())
    }

    fn get_tmp(&self, key: &str) -> Result<Option<Vec<u8>>> {
        Ok(self.temporal.borrow().get(key).cloned())
    }

    fn cas_tmp(&self, key: &str, old: &[u8], new: &[u8]) -> Result<bool> {
        let mut temporal = self.temporal.borrow_mut();
        match temporal.get(key) {
            Some(current) if current == old => {
                temporal.insert(key.to_string(), new.to_vec());
                Ok(true)
            }
            Some(current) => Err(StorageError::CasMismatch {
                expected: old.to_vec(),
                actual: current.clone(),
            }),
            None => Err(StorageError::NotFound(key.to_string())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_store_and_load() {
        let s = MemoryStorage::new();
        s.store("h:abc", b"hello").unwrap();
        assert!(s.exists("h:abc").unwrap());
        assert_eq!(s.load("h:abc").unwrap(), Some(b"hello".to_vec()));
    }

    #[test]
    fn test_missing_returns_none() {
        let s = MemoryStorage::new();
        assert_eq!(s.load("h:nonexistent").unwrap(), None);
        assert!(!s.exists("h:nonexistent").unwrap());
    }

    #[test]
    fn test_delete() {
        let s = MemoryStorage::new();
        s.store("h:x", b"test").unwrap();
        assert!(s.delete("h:x").unwrap());
        assert!(!s.exists("h:x").unwrap());
    }

    #[test]
    fn test_list_by_prefix() {
        let s = MemoryStorage::new();
        s.store("h:a", b"1").unwrap();
        s.store("h:b", b"2").unwrap();
        s.store("c:x", b"3").unwrap();

        let h_keys = s.list("h:").unwrap();
        assert_eq!(h_keys.len(), 2);
        let c_keys = s.list("c:").unwrap();
        assert_eq!(c_keys.len(), 1);
    }

    // ── HLPP canonical name tests ─────────────────────────────────────

    #[test]
    fn test_put_get_has() {
        let s = MemoryStorage::new();
        s.put("h:canon", b"data").unwrap();
        assert!(s.has("h:canon").unwrap());
        assert_eq!(s.get("h:canon").unwrap(), Some(b"data".to_vec()));
    }

    // ── Temporal operation tests ──────────────────────────────────────

    #[test]
    fn test_put_tmp_and_get_tmp() {
        let s = MemoryStorage::new();
        s.put_tmp("system:test", b"temporal data").unwrap();
        let val = s.get_tmp("system:test").unwrap();
        assert_eq!(val, Some(b"temporal data".to_vec()));
    }

    #[test]
    fn test_get_tmp_missing() {
        let s = MemoryStorage::new();
        assert_eq!(s.get_tmp("system:nonexistent").unwrap(), None);
    }

    #[test]
    fn test_cas_tmp_success() {
        let s = MemoryStorage::new();
        s.put_tmp("system:head", b"old_cid").unwrap();
        let ok = s.cas_tmp("system:head", b"old_cid", b"new_cid").unwrap();
        assert!(ok);
        assert_eq!(s.get_tmp("system:head").unwrap(), Some(b"new_cid".to_vec()));
    }

    #[test]
    fn test_cas_tmp_mismatch() {
        let s = MemoryStorage::new();
        s.put_tmp("system:head", b"actual_value").unwrap();
        let result = s.cas_tmp("system:head", b"expected_value", b"new_value");
        assert!(result.is_err());
    }

    #[test]
    fn test_cas_tmp_missing_key() {
        let s = MemoryStorage::new();
        let result = s.cas_tmp("system:unknown", b"old", b"new");
        assert!(result.is_err());
    }
}
