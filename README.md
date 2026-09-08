# hllset-next-v2 — the HLLSet algebra line (gen2)

The foundation of the `fractal_manifold_gen2` collection. This project is the
reference implementation of the HLLSet algebra recorded in
[`ewm-cortex-fpga-v2/docs/ALGEBRAIC_FOUNDATION.md`](../../ewm-cortex-fpga-v2/docs/ALGEBRAIC_FOUNDATION.md)
and consolidated, with the working rules and implementation details, in
[`_DOCS/dev/HLLSET_DEVELOPER_GUIDE.md`](_DOCS/dev/HLLSET_DEVELOPER_GUIDE.md):
two spaces, one hinge (`BitAddress`), one LUT
lattice, two morphisms with contracts, the Noether context, and derived ranks —
all Rust, `cargo` only, no Go/Python/ROS 2/Redis.

## Architecture

```text
hllset-next-v2/
├── Cargo.toml                 # Workspace manifest (12 crates)
├── crates/
│   ├── hllset-contracts/      # Soldered invariants: hashing, BitAddress, token encodings
│   ├── hllset-core/           # HLLSet bitmap, IICA lattice ops, serialization, cardinality
│   ├── hllset-lut/            # The LUT lattice: LutNode (nodes of 2^T), fibers K_i, AtomTree
│   ├── hllset-morphisms/      # The two morphisms: complete single-touch ingest,
│   │                          #   LUT-first materialization (TF only for ambiguity)
│   ├── hllset-context/        # The Noether context: S(t) generators, D/R/N, tropical FollowMatrix
│   ├── hllset-ranks/          # Five-level rank algebra: TF stored, rank derived (F→G→H→K→L)
│   ├── hllset-cid/            # Embedded content-addressed identifiers (SHA-1 CIDs)
│   ├── hllset-dsl/            # Lua VM bindings, tokenizer, LatticeElement, storage API
│   ├── hllset-forth/          # Forth frontend: source → AST → Lua
│   ├── hllset-bridge/         # 3-gram fingerprinting bridge (n-gram ingest helpers)
│   ├── hllset-storage/        # Storage trait + MemoryStorage + SledStorage (embedded CID)
│   └── hllset-cli/            # CLI: Lua -e, Forth --forth, REPL
├── _DOCS/
│   ├── dev/                   # Design and governance documents
│   └── notebooks/             # 6 gen2 notebooks (Rust kernel, executed green)
└── README.md
```

Dropped from the legacy line (kept in the frozen `fractal_manifold/` tree):
`hllset-mesh` (no coordination needed), `hllset-materialize` (the morphism lives
in `hllset-morphisms`), `hllset-storage-redis`, `hllset-duckdb`,
`hllset-temporal`.

## Quick Start

```bash
# Build
cargo build

# Lua evaluation
cargo run -- -e 'return #hllset.tokenize("hello world")'

# Forth DSL
cargo run -- --forth '"neural" "network" 2 INSCRIBE KEY'

# Interactive REPL
cargo run -- --repl
```

## Test Suite

```bash
cargo test --workspace
# 245 tests, 0 failures (12 crates)
```

## Notebooks

Six notebooks, aligned with the gen2 architecture, all executed green in the
Rust (evcxr) Jupyter kernel. The legacy 14-notebook set is discharged with the
frozen line.

| # | Notebook | Description |
| --- | ---------- | ------------- |
| 01 | `hllset_algebra` | The hinge, LUT lattice, ingest/materialize, Noether context, ranks, finest symmetry |
| 02 | `tokenizer_materialization` | DSL Tokenizer + the one materialization morphism (LUT-first, TF for ambiguity) |
| 03 | `hllset_core` | IICA laws, content keys (o:/h:), sub-lattice collapse, serialization, atoms, cardinality |
| 04 | `lua_dsl_and_forth` | Lua DSL (inscribe, operators, store/load) + Forth parse→lower→run |
| 05 | `noether_context_tree` | S(t) generators, D/R/N invariants, tropical follow matrix, atom tree = same element |
| 06 | `embedded_storage_cid` | Embedded CID + SledStorage: the default IPFS substitute |

## Key Features (gen2)

- **Contracts first.** `hllset-contracts` holds the soldered invariants —
  `PROTOCOL_VERSION`, `CONTRACT_VERSION`, Murmur3/sha1, `BitAddress` (the
  structural hinge), and both token encodings (`tid{n}`, 4-byte LE).
- **IICA core.** `hllset-core`: bitmap over `B` = 1024×32, union/intersection/
  difference, popcount vs Horvitz–Thompson cardinality, serialization,
  content keys (`h:`, `o:`, …).
- **The LUT lattice.** `hllset-lut`: every named LUT is a labeled node of
  `2^T`; `K_i` (the fiber of bit address `i`) is first-class; `AtomTree` is
  the sparse Merkle tree over `B` and `HLLSet(LUT) = HLLSet(MerkleTree)`.
- **The two morphisms.** `hllset-morphisms`: ingest is complete and
  single-touch (3 seeded hashes → atoms + LUT fibers + TF, in-module);
  materialize is LUT-first with TF consulted only for collided bits.
- **The Noether context.** `hllset-context`: `S(t)` as a lattice element with
  declared generators, `H(t) = (S(t), H(t-1), D, R, N)` with invariants, and
  the tropical follow matrix (max/+, grow-only, projection to `V(H)`).
- **Derived ranks.** `hllset-ranks`: five levels `F(TF) → G(bit) →
  H(register) → K(HLLSet) → L(compound)`, all `u64`; TF stored, rank never
  stored.
- **DSL & tooling.** Lua runtime, Forth frontend, REPL — pure algebra,
  no materialization baked in.
- **Embedded storage.** `hllset-storage`: `MemoryStorage` for dev/testing and
  `SledStorage` as the embedded default — sled + `hllset-cid` (SHA-1 CIDs).
  No external daemon, no `ipfrs-core` path, nothing outside the collection.

## Collection roadmap

1. **hllset-next-v2** (this project) — the algebra foundation. ✅ Refactor
   complete.
2. **hllset-fpga-simulator-v2** — consume the updated foundation.
3. **ewm-fpga-bridge-v2** — consume the updated foundation.
4. **ewm-cortex-fpga-v2** — structural/persistence layer on top.

## Key Changes from the Legacy Line

| Aspect | Legacy | gen2 |
| -------- | ---------------------- | ------------------- |
| Materialization | 4 strategies + engine trait + registry | one morphism: LUT-first, TF for ambiguity |
| LUT | ad-hoc tables (TokenLUT/CatalogLUT/…) | labeled nodes of `2^T`, fibers `K_i` |
| Ranks | traits + Fisher + masks + derivatives | five pure projections over TF + fibers |
| Context | conversation crate (prompts, De Bruijn) | Noether context + tropical follow matrix |
| Messaging | mesh bus (tokio) | none — no coordination needed |
| Storage | Memory / Redis / DuckDB / ipfrs-core path | Memory + `SledStorage` with embedded `hllset-cid` |
| Notebooks | 14 legacy | 5 gen2, executed green |
| Tests | 291 (13 crates) | 245 (12 crates) |
