//! Pluggable materialization engines — abstract backends for HLLSet→Token bridging.
//!
//! The Core produces HLLSets. Materialization bridges them back to tokens.
//! Different engines implement the same trait: in-memory LUTs, chunked SQLite,
//! FPGA (simulated or physical). The registry maps LUT keys to engine instances.
//!
//! # Sync with hllset-dsl
//!
//! This crate is the **standard materialization interface**.
//! `hllset-dsl::materialize` provides the existing TokenLUT/DenseLUT/CatalogLUT
//! implementations. `InMemoryEngine` wraps them as a `MaterializeEngine`.
//!
//! `hllset-dsl` does NOT depend on this crate (to avoid circular deps).
//! Instead, `hllset-dsl` re-exports `MaterializeEngine` from this crate
//! and its `Materializer` struct implements the trait.

use hllset_core::HLLSet;
use hllset_storage::MemoryStorage;
use std::collections::HashMap;

// ── MaterializeEngine Trait ─────────────────────────────────────────

/// Abstract materialization backend.
///
/// Takes an HLLSet + (reg, zeros) positions → candidate tokens.
/// Different engines have different performance characteristics:
///   InMemoryEngine — fastest, for small LUTs
///   DuckDBEngine   — chunked, for large LUTs on IPFS
///   FPGASimEngine  — cycle-model, matches hardware bit-exact
///   FPGABoardEngine — physical FPGA (future)
pub trait MaterializeEngine {
    fn materialize(
        &self,
        hllset: &HLLSet,
        positions: &[(u16, u8)],
    ) -> Result<Vec<Vec<u8>>, MaterializeError>;
    fn name(&self) -> &str;
    fn lut_count(&self) -> usize;
    fn is_hardware(&self) -> bool {
        false
    }
}

#[derive(Debug, thiserror::Error)]
pub enum MaterializeError {
    #[error("LUT not loaded: {0}")]
    LutNotLoaded(String),
    #[error("Query error: {0}")]
    Query(String),
    #[error("IO: {0}")]
    IO(#[from] std::io::Error),
}

// ── In-Memory Engine ────────────────────────────────────────────────

/// Reference engine: HashMap-based (reg,zeros)→tokens lookup.
/// All other engines must produce bit-exact results matching this one.
pub struct InMemoryEngine {
    name: String,
    lut: HashMap<(u16, u8), Vec<Vec<u8>>>,
}

impl InMemoryEngine {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.into(),
            lut: HashMap::new(),
        }
    }

    pub fn build(&mut self, tokens: &[&[u8]]) {
        for &t in tokens {
            let (r, z) = hllset_core::hashing::token_to_position(t);
            self.lut
                .entry((r as u16, z as u8))
                .or_default()
                .push(t.to_vec());
        }
    }

    pub fn token_count(&self) -> usize {
        self.lut.values().map(|v| v.len()).sum()
    }

    /// Extract (reg, zeros) positions from an HLLSet by iterating non-zero registers.
    /// This mirrors the internal logic of `materialize_inlut`.
    pub fn extract_positions(hllset: &HLLSet) -> Vec<(u16, u8)> {
        hllset
            .active_positions()
            .into_iter()
            .map(|(r, z)| (r as u16, z as u8))
            .collect()
    }
}

impl MaterializeEngine for InMemoryEngine {
    fn materialize(
        &self,
        _hllset: &HLLSet,
        positions: &[(u16, u8)],
    ) -> Result<Vec<Vec<u8>>, MaterializeError> {
        let mut res = Vec::new();
        for &p in positions {
            if let Some(ts) = self.lut.get(&p) {
                res.extend(ts.iter().cloned());
            }
        }
        Ok(res)
    }
    fn name(&self) -> &str {
        &self.name
    }
    fn lut_count(&self) -> usize {
        self.token_count()
    }
}

// ── DuckDB/SQLite Chunked Engine ────────────────────────────────────

/// Chunked LUT engine — register-range partitioned, IPFS-stored.
/// Maps to the 4-chunk Algebraic Chunk Space architecture.
pub struct DuckDBEngine {
    name: String,
    storage: MemoryStorage,
    cache_dir: std::path::PathBuf,
}

impl DuckDBEngine {
    pub fn new(name: &str, storage: MemoryStorage, cache_dir: std::path::PathBuf) -> Self {
        Self {
            name: name.into(),
            storage,
            cache_dir,
        }
    }

    pub fn open(&self) -> Result<hllset_duckdb::ChunkMaterializer, Box<dyn std::error::Error>> {
        hllset_duckdb::ChunkMaterializer::open_with(
            self.storage.clone(),
            &self.name,
            self.cache_dir.clone(),
        )
    }
}

impl MaterializeEngine for DuckDBEngine {
    fn materialize(
        &self,
        _hllset: &HLLSet,
        positions: &[(u16, u8)],
    ) -> Result<Vec<Vec<u8>>, MaterializeError> {
        let mat = self
            .open()
            .map_err(|e| MaterializeError::Query(e.to_string()))?;
        mat.query(positions)
            .map_err(|e| MaterializeError::Query(e.to_string()))
    }
    fn name(&self) -> &str {
        &self.name
    }
    fn lut_count(&self) -> usize {
        4
    } // 4 chunks
}

// ── FPGA Simulation Engine ──────────────────────────────────────────

/// Cycle-accurate FPGA simulation. Same results as InMemoryEngine,
/// with configurable pipeline delay to model hardware timing.
pub struct FPGASimEngine {
    name: String,
    inner: InMemoryEngine,
    pipeline_cycles: u32,
}

impl FPGASimEngine {
    pub fn new(name: &str, pipeline_cycles: u32) -> Self {
        Self {
            name: name.into(),
            inner: InMemoryEngine::new(name),
            pipeline_cycles,
        }
    }
    pub fn build(&mut self, tokens: &[&[u8]]) {
        self.inner.build(tokens);
    }
    pub fn simulated_cycles(&self, num_positions: usize) -> u32 {
        self.pipeline_cycles + (num_positions as u32 / 256)
    }
}

impl MaterializeEngine for FPGASimEngine {
    fn materialize(
        &self,
        hllset: &HLLSet,
        positions: &[(u16, u8)],
    ) -> Result<Vec<Vec<u8>>, MaterializeError> {
        let _cycles = self.simulated_cycles(positions.len());
        self.inner.materialize(hllset, positions)
    }
    fn name(&self) -> &str {
        &self.name
    }
    fn lut_count(&self) -> usize {
        self.inner.token_count()
    }
    fn is_hardware(&self) -> bool {
        true
    }
}

// ── Engine Registry ─────────────────────────────────────────────────

/// Registry mapping LUT keys to materialization engine instances.
pub struct MaterializeRegistry {
    engines: HashMap<String, Box<dyn MaterializeEngine>>,
}

impl MaterializeRegistry {
    pub fn new() -> Self {
        Self {
            engines: HashMap::new(),
        }
    }

    pub fn register(&mut self, lut_key: &str, engine: Box<dyn MaterializeEngine>) {
        self.engines.insert(lut_key.to_string(), engine);
    }

    pub fn get(&self, lut_key: &str) -> Option<&dyn MaterializeEngine> {
        self.engines.get(lut_key).map(|e| e.as_ref())
    }

    /// Get the best engine for a LUT key (prefers hardware).
    pub fn best_for(&self, lut_key: &str) -> Option<&dyn MaterializeEngine> {
        // Try exact match first, then hardware, then any
        if let Some(e) = self.engines.get(lut_key) {
            return Some(e.as_ref());
        }
        self.engines
            .values()
            .find(|e| e.is_hardware())
            .map(|e| e.as_ref())
            .or_else(|| self.engines.values().next().map(|e| e.as_ref()))
    }

    pub fn list(&self) -> Vec<String> {
        self.engines.keys().cloned().collect()
    }
    pub fn hardware_engines(&self) -> Vec<&dyn MaterializeEngine> {
        self.engines
            .values()
            .filter(|e| e.is_hardware())
            .map(|e| e.as_ref())
            .collect()
    }
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_in_memory_roundtrip() {
        let mut e = InMemoryEngine::new("t");
        e.build(&[b"hello", b"world"]);
        let (r, z) = hllset_core::hashing::token_to_position(b"hello");
        let h = HLLSet::from_tokens(&["hello"]);
        let r = e.materialize(&h, &[(r as u16, z as u8)]).unwrap();
        assert!(r.iter().any(|t| t == b"hello"));
    }

    #[test]
    fn test_registry() {
        let mut e1 = InMemoryEngine::new("a");
        e1.build(&[b"one"]);
        let e2 = InMemoryEngine::new("b");
        let mut reg = MaterializeRegistry::new();
        reg.register("lut:a", Box::new(e1));
        reg.register("lut:b", Box::new(e2));
        assert!(reg.get("lut:a").is_some());
        assert!(reg.get("lut:c").is_none());
    }

    #[test]
    fn test_fpga_matches_memory() {
        let mut ref_e = InMemoryEngine::new("ref");
        ref_e.build(&[b"t1", b"t2"]);
        let mut fpga = FPGASimEngine::new("fpga", 3);
        fpga.build(&[b"t1", b"t2"]);
        let (r, z) = hllset_core::hashing::token_to_position(b"t1");
        let h = HLLSet::from_tokens(&["t1"]);
        let a = ref_e.materialize(&h, &[(r as u16, z as u8)]).unwrap();
        let b = fpga.materialize(&h, &[(r as u16, z as u8)]).unwrap();
        assert_eq!(a, b, "FPGA sim must match reference");
    }

    #[test]
    fn test_extract_positions() {
        let h = HLLSet::from_tokens(&["x", "y", "z"]);
        let pos = InMemoryEngine::extract_positions(&h);
        assert!(!pos.is_empty());
        // Every position should be a valid HLLSet register
        for (r, _) in &pos {
            assert!(*r < 1024);
        }
    }
}
