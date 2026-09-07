//! HLLSet Algebra DSL — Lua-embedded domain-specific language.
//!
//! This crate provides:
//!
//! - **`LatticeElement`** — the core DSL type wrapping an HLLSet with its
//!   content-addressable key. Supports union (∪), intersection (∩),
//!   difference (\), BSS morphisms.
//!
//! - **`DslRuntime`** — a Lua VM with the global `hllset` table, operator
//!   overloading (`+`, `*`, `-`, `#`), and all LatticeElement methods.
//!
//! # Quick example
//!
//! ```rust
//! use hllset_dsl::DslRuntime;
//!
//! let mut rt = DslRuntime::new().unwrap();
//! let cardinality: f64 = rt.eval(r#"
//!     local a = hllset.inscribe({"hello", "world", "lua"})
//!     local b = hllset.inscribe({"lua", "programming"})
//!     local c = a + b    -- union
//!     return #c          -- cardinality
//! "#).unwrap();
//! assert!(cardinality > 0.0);
//! ```
//!
//! # Lua API reference
//!
//! | Expression | Operation | Returns |
//! |-----------|-----------|---------|
//! | `hllset.inscribe({...})` | Create element from tokens | `LatticeElement` |
//! | `hllset.empty()` | Create empty element (⊥) | `LatticeElement` |
//! | `#a` | Cardinality | `number` |
//! | `a + b` | Union (∪) | `LatticeElement` |
//! | `a * b` | Intersection (∩) | `LatticeElement` |
//! | `a - b` | Difference (\) | `LatticeElement` |
//! | `a:key()` | Content key | `string` |
//! | `a:card()` | Cardinality | `number` |
//! | `a:popcount()` | Bits set | `integer` |
//! | `a:is_empty()` | Emptiness check | `boolean` |
//! | `a:bss_inclusion(b)` | BSSτ | `number` |
//! | `a:bss_exclusion(b)` | BSSρ | `number` |
//! | `a:morph_to(b, τ, ρ)` | BSS morphism | `table` |
//! | `a:jaccard(b)` | Jaccard similarity | `number` |
//! | `a:is_subset_of(b)` | Subset check | `boolean` |
//! | `a:is_superset_of(b)` | Superset check | `boolean` |
//! | `a:to_bytes()` | Serialize to binary | `string` |
//! | `tostring(a)` | Human-readable | `string` |

pub mod distributed;
pub mod lattice;
pub mod materialize;
pub mod pattern;
pub mod runtime;
pub mod tokenizer;
pub mod worker;

pub use distributed::{BSSRouter, MultiSourceMerge, NodeFingerprint};
pub use lattice::LatticeElement;
pub use materialize::{
    materialize_debruijn, materialize_homogeneous_consensus, materialize_inlut,
    materialize_ngram_cross_validate, CatalogLUT, DenseLUT, MaterializedResult, Materializer,
    TokenLUT,
};

// Re-export the pluggable materialization engine trait
pub use hllset_materialize::{
    DuckDBEngine, FPGASimEngine, InMemoryEngine, MaterializeEngine, MaterializeError,
    MaterializeRegistry,
};
pub use runtime::DslRuntime;
pub use tokenizer::{Normalizer, Tokenizer};
pub use worker::Worker;

// Re-export core types for convenient single-crate usage in notebooks
pub use hllset_core::{
    core::bss::BSSResult,
    core::hashing::{
        hash_to_position, murmur3_hash, murmur3_hash_seeded, token_to_position,
        token_to_position_seeded,
    },
    HLLSet, BITS_PER_REG, M, P,
};
