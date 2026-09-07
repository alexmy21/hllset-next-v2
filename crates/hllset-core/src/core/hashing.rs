//! Hashing utilities for HLLSet — re-exported from the shared leaf crate
//! `hllset-contracts` (GOVERNANCE.md §3, §5). The soldered geometry
//! (`P=10, M=1024, 32 bits/reg`), MurmurHash3 x64-128, SHA-1 addresses, and
//! position decomposition live in exactly one place.

pub use hllset_contracts::hashing::*;
