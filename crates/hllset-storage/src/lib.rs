//! Content-addressed storage for HLLSets.
//!
//! Provides a sync `Storage` trait with:
//! - `MemoryStorage` — in-memory HashMap (dev/testing)
//! - `SledStorage` — the embedded default backend: sled + `hllset-cid`
//!   (embedded SHA-1 CIDs; no external daemon, no `ipfrs-core` dependency)
//!
//! # Example
//!
//! ```rust
//! use hllset_storage::{MemoryStorage, Storage};
//!
//! let store = MemoryStorage::new();
//! store.store("h:abc123", b"hello").unwrap();
//! let data = store.load("h:abc123").unwrap();
//! assert_eq!(data, Some(b"hello".to_vec()));
//! ```

pub mod cache;
pub mod memory;
pub mod sled;
pub mod storage;

pub use cache::CacheStorage;
pub use memory::MemoryStorage;
pub use sled::{IpfrsNativeStorage, SledStorage};
pub use storage::{Result, Storage, StorageError};
