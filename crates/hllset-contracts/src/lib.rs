//! `hllset-contracts` — the leaf of the cross-project dependency graph
//! (GOVERNANCE.md §3, §5).
//!
//! This crate is the **single source of truth** for the soldered invariants
//! and the wire vocabulary shared by `hllset-next` (reference),
//! `ewm-fpga-bridge` (SPI/modules/DSL), `hllset-fpga-simulator` (backends),
//! and all application projects. Nothing above this crate may re-declare a
//! soldered invariant.
//!
//! - [`hashing`] — `P=10, M=1024, 32 bits/reg`, MurmurHash3 x64-128, SHA-1
//!   addresses, position decomposition;
//! - [`token`] — `TokenId = u32` and the two golden inscriptions
//!   (`tid{n}` and 4-byte LE);
//! - [`module`] — the soldered "bag of modules" (`ModuleKind`, ports);
//! - [`stream`] — the AXI-like `Stream<T> { data, valid, ready }` handshake;
//! - [`PROTOCOL_VERSION`] / [`CONTRACT_VERSION`] — the version stamps.
//!
//! std-only by default; optional `serde` for the wire vocabulary.

#![forbid(unsafe_code)]

pub mod hashing;
pub mod module;
pub mod stream;
pub mod token;

/// Wire protocol version. Bump on any wire-format change.
pub const PROTOCOL_VERSION: u32 = 1;

/// Contract (vocabulary) version. Bump on any soldered-invariant change:
/// additive changes bump the patch component, breaking changes the minor.
pub const CONTRACT_VERSION: u32 = 2;

pub use hashing::{
    hash_to_position, murmur3_hash, murmur3_hash_seeded, sha1_hex, token_to_position,
    token_to_position_seeded, BitAddress, BITS_PER_REG, M, P, TOTAL_BITS,
};
pub use module::{
    input_port_in_range, module_ports, output_port_in_range, ModuleKind, ModulePorts, NodeId,
    PortId, MODULE_KINDS,
};
pub use stream::Stream;
pub use token::{parse_token_id, parse_token_id_le, token_in_bytes, token_in_bytes_le, TokenId};
