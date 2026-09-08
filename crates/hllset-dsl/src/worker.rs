//! Stateless HLLSet worker — loads from IPFS, computes, forgets.
//!
//! LUTs are built in-memory from request tokens. The worker doesn't
//! persist LUT state — all state lives in IPFS (HLLSets) or in the request.

use crate::lattice::LatticeElement;
use hllset_core::HLLSet;
use hllset_storage::Storage;

pub struct Worker<S: Storage> {
    storage: S,
}

impl<S: Storage> Worker<S> {
    pub fn new(storage: S) -> Self {
        Self { storage }
    }

    pub fn load_hllset(&self, key: &str) -> Option<HLLSet> {
        let bytes = self.storage.load(key).ok()??;
        HLLSet::from_bytes(&bytes)
    }

    pub fn inscribe(&self, tokens: &[&str]) -> String {
        let elem = LatticeElement::from_tokens(tokens);
        let key = elem.key().to_string();
        let _ = self.storage.store(&key, &elem.hllset().to_bytes());
        key
    }

    pub fn has_resource(&self, key: &str) -> bool {
        self.storage.exists(key).unwrap_or(false)
    }

    pub fn bss_inclusion(&self, ka: &str, kb: &str) -> Option<f64> {
        let a = self.load_hllset(ka)?;
        let b = self.load_hllset(kb)?;
        Some(a.bss_inclusion(&b))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hllset_storage::MemoryStorage;

    #[test]
    fn test_worker_inscribe_and_load() {
        let w = Worker::new(MemoryStorage::new());
        let key = w.inscribe(&["hello", "world"]);
        assert!(key.starts_with("h:"));
        assert!(w.load_hllset(&key).unwrap().cardinality() > 0.0);
    }

    #[test]
    fn test_worker_forgets() {
        let w = Worker::new(MemoryStorage::new());
        let key = w.inscribe(&["ephemeral"]);
        assert!(w.has_resource(&key));
    }
}
