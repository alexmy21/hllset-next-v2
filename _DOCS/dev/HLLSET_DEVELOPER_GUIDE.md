# hllset-next-v2 — Developer Guide

The consolidated working model of `hllset-next-v2`: the algebra, the
operational contracts, the implementation map, and the working rules we
discovered while refactoring the foundation of `fractal_manifold_gen2`.

Audience: anyone joining the development. Read this before the code; the code
is disposable, this document is the contract.

---

## 1. The model in one paragraph

There are **two spaces** and **one hinge**:

```text
Token space T (bytes)          HLLSet space 2^B (bitmaps)
     │                                ▲
     │  ingest (complete, single-touch)│
     ▼                                │
LUT lattice 2^T ── materialize M(H,L) ─┘
```

- `B` = flat bit addresses (`reg * 32 + tz`), `|B| = 1024 × 32 = 32768`.
- `BitAddress` (in `hllset-contracts`) is the hinge: the one object both
  morphisms pass through.
- Ingest maps tokens to atoms; materialize maps atoms back to candidate
  tokens through a LUT node. They are the only two crossings between spaces.

## 2. The two lattices

### 2.1 The HLLSet lattice `2^B`

- Elements are bitmaps; ops are `∪`, `∩`, `\` (and later `△`).
- **IICA**: Idempotent, Immutable, Content-Addressed. Keys are SHA-1 of the
  content, with the prefix as the type (`h:`, `o:`, `r:`, `d:`, `n:`, `v:`, …).
- Every sketch decomposes **uniquely** into join-irreducibles — its active
  atoms `{i}`. The atoms are the canonical basis.

### 2.2 The LUT lattice `2^T`

- Elements are token sets; a **named LUT** is a labeled node (`LutNode`):
  `tokenLUT₁/₂/₃` (1-/2-/3-gram), `catalogLUT`, `gateLUT`, … Each node is a
  physically isolated collection in storage terms.
- `K_i` = the fiber of bit address `i`: `{ t ∈ T : pos(t) = i }`. The fibers
  partition `T`, and each `K_i` is itself a node of `2^T`.
- Materialization is a join in `2^T`:

```text
M(H, L) = ⋃ { L ∩ K_i : i ∈ A(H) }
```

  and `M(H, ·)` is a join-homomorphism: `M(H, L₁ ∪ L₂) = M(H, L₁) ∪ M(H, L₂)`.

### 2.3 Working rule — totality is a blessing and a curse

Every composition in either lattice is a *valid* element. There are no type
errors, so mistakes are **semantic drift**: you always get a proper set or a
proper HLLSet, but it may not be the one you meant. The LUT lattice is the
dangerous one — `2^T` is astronomically larger than `2^B`.

Consequences:

- Do not build new ad-hoc set operations by hand; derive them from the
  lattice ops and name them by what they *are*.
- The correspondence rule (below) is what keeps the two lattices locked
  together and prevents anonymous objects.

## 3. The two morphisms (operational contracts)

### 3.1 Ingest — complete, single touch, in-module

For each token, in one pass and without leaving the processing module:

1. compute the **3 seeded hashes** (n-seed encodings; the n-gram regime is
   the same shape with channel-selected seeds),
2. set the atom in the corresponding HLLSet,
3. insert the token into the corresponding LUT fiber,
4. increment TF.

Implementation: `hllset-morphisms::Ingest` (`SEEDS = [0, 1, 2]`).
Nothing is handed to another module; the token is touched once.

### 3.2 Materialize — LUT-first, TF only for ambiguity

For each active bit, collect candidate tokens from **all pointed LUTs**
(every encoding); only when a bit has more than one candidate does the LUT's
TF break the tie. TF is never the starting point.

Implementation: `hllset-morphisms::materialize(pairs, tf)` — one
`(sketch, LUT)` pair per encoding. Default restoration = the join of all
known LUT nodes (the full image).

### 3.3 Persistence rule — LUT sets are run-time, pr-HLLSets are persistent

Raw LUT sets may be operated on in run-time, but **if we want to persist a
LUT for future use, we convert it into an HLLSet** — the short version of
ingest: no tokenization, just collect the related n-grams/n-seeds and set
their atoms.

The generated sketches are **projections** (`pr-HLLSet`) of the original
token collections:

```text
pr-HLLSet(L) = ingest(L) = ⋁ { {pos(t)} : t ∈ L }
{ tokens }    = materialize(pr-HLLSet(L), L)
```

This is the correspondence rule in operational form (§5).

## 4. K_i, the constant atoms, and where time lives

- `K_i` is a **projection** of tokens sharing the same `<reg, zeros>` value
  in a LUT record — nothing more.
- Its HLLSet is the atom `{i}`: explicit, single-bit, no disambiguation.
- The atoms are **constant**: `{i}` is always the same bitmap; it never ages.
  Time does not exist for atoms.
- Time enters only through the **coefficient vector** — which atoms are
  present in `S(t)`. The Noether derivation is the difference of two
  coefficient vectors: `D/R/N` at bit level. This is exactly why the atom
  tree diff equals the bit-level D/R/N.

Operational reading: persist the basis once (implicitly, it is fixed); all
evolution is a sequence of state vectors over that basis.

## 5. The correspondence rule (mandatory)

```text
C = the defined token collections: { L ⊆ T : L has an extraction rule }
H = the HLLSet lattice

Rule A (forward):   every L ∈ C has its HLLSet  H_L = ⋁_{t∈L} {pos(t)} = ingest(L)
Rule B (backward):  every H ∈ H carries a restoration rule — a declared
                    token-collection type (LUT node) L such that V(H) = M(H, L),
                    or a declared family of LUT nodes for the full image
```

Consequences:

- **Anonymous sketches are illegal.** Every sketch travels with its
  restoration pointer; the store persists them together.
- Named LUT nodes carry their own typed sketches; `K_i` inscribes to `{i}`.
- Prefixes are the type system: `o:` original, `v:` views, `r/d/n:` Noether
  derivations, `h:` the only untyped prefix (its restoration record names
  the LUT).

## 6. Procedure scope (immutability at the boundary)

Mutability is confined to the scope of a procedure. Inside it, an HLLSet may
be built up (`add_bit`, `add_token`, `merge`). Outside it, HLLSets are
immutable values delivered with their SHA-1. Processing takes HLLSets and
returns **new** HLLSets with new keys; inputs are never mutated.

## 7. Context and time

- `S(t)` is a lattice element with **declared generators**
  (`Context::from_generators`; insert/remove are persistent).
- `H(t) = (S(t), H(t-1), D, R, N)` with invariants
  `D∪R = H(t-1)`, `R∪N = S(t)`, `D∩N = ∅` (`hllset-context::evolve`).
- The follow matrix is **tropical** (`⊕ = max`, `⊗ = +`): grow-only, merge
  is a join, restriction to `V(H)` is a lattice projection.
- The context tree is the **sparse Merkle tree over `B`**: leaves are the
  active atoms, `leaf_i = ({i}, K_i)` — the restoration rule of the `K_i`
  HLLSet, and the finest symmetry holds:

```text
HLLSet(LUT) = HLLSet(MerkleTree)     (ingest(L) = ⋁ atoms of L)
```

## 8. Ranks — TF stored, rank derived

Five levels, all `u64`, FPGA-native ops:

```text
F(TF) → G(bit) → H(register) → K(HLLSet) → L(compound)
```

TF is stored (monotonic, CRDT max-merge). Rank is a pure projection and is
**never stored**. The fold (`Sum`/`Max`/`Min`) is the application's choice.

## 9. Storage — embedded by default

- `hllset-cid`: embedded SHA-1 CIDs (`h:<sha1>`), std-only.
- `SledStorage` (`hllset-storage`): sled + `hllset-cid` — the default
  embedded backend. No external daemon, no network, no `ipfrs-core` path.
- A distributed/ipfrs backend can be added later as another implementation
  of the `Storage` trait; nothing above the trait changes.

## 10. Naming and files

| Concept | Crate |
| --- | --- |
| Soldered invariants, `BitAddress`, hashing, encodings | `hllset-contracts` |
| HLLSet bitmap, IICA ops, serialization, cardinality, content keys | `hllset-core` |
| LUT lattice: `LutNode`, `LutIndex`, fibers, `AtomTree` | `hllset-lut` |
| Ingest + materialize + `TfTable` | `hllset-morphisms` |
| Noether context + tropical follow matrix | `hllset-context` |
| Five-level ranks | `hllset-ranks` |
| Embedded CIDs | `hllset-cid` |
| Lua DSL + tokenizer + `LatticeElement` | `hllset-dsl` |
| Forth frontend | `hllset-forth` |
| 3-gram bridge | `hllset-bridge` |
| Storage trait, `MemoryStorage`, `SledStorage` | `hllset-storage` |
| CLI shell | `hllset-cli` |

Notebooks: `_DOCS/notebooks/01..06` — the executable form of this guide.

## 11. Future work (marked, not started)

1. **Boolean ring on HLLSets.** `(2^B, △, ∩)` is a Boolean ring / GF(2)
   vector space. Choose **base HLLSets** (`b-HLLSets`); every sketch is a
   linear combination (XOR) of base elements. The atoms are the canonical
   basis; other bases (e.g. generator-grouped) become compression and
   provenance tools. Base elements can serve as nodes of a **Merkle tree /
   Merkle forest** (XOR-Merkle style). Not built now — recorded here so the
   door stays open.
2. **Distributed CID backend** behind `hllset_cid::Cid` / `Storage` (ipfrs
   or similar), restoring IPFS-grade addressing without changing callers.
3. **Consensus and De Bruijn restoration**, re-expressed from the algebra
   when the application line needs them (they were discharged with the
   legacy line).
4. **Temporal rank derivatives** (ΔR etc.) if a use case demands them —
   currently the five levels are pure projections and that suffices.

## 12. Collection roadmap

```text
hllset-next-v2          ✅ foundation (this project)
hllset-fpga-simulator-v2   → consume the foundation
ewm-fpga-bridge-v2         → consume the foundation
ewm-cortex-fpga-v2         → structural/persistence layer on top
```

The frozen `fractal_manifold/` line keeps the legacy implementations and
notebooks; gen2 never reaches back into it.
