//! `hllset-lut` — the LUT lattice (ALGEBRAIC_FOUNDATION §1.2, §1.5).
//!
//! Written from the algebra, not from legacy code. The contract:
//!
//! - `T` = the token universe (byte strings). The **general LUT is the
//!   subset lattice `2^T`**: every named LUT is a labeled node [`LutNode`].
//! - `K_i` = the fiber of bit address `i` — the tokens with `pos(t) = i`.
//!   The fibers partition `T`, and each `K_i` is itself a node of `2^T`.
//! - **Materialization is a join in `2^T`**:
//!   `M(H, L) = ⋃ { L ∩ K_i : i ∈ A(H) }`, and `M(H, ·)` is a
//!   join-homomorphism: `M(H, L₁ ∪ L₂) = M(H, L₁) ∪ M(H, L₂)`.
//! - [`AtomTree`] is the sparse Merkle tree over `B` — the sketch-space
//!   presentation of a LUT node. The **finest symmetry** holds:
//!   `HLLSet(LUT) = HLLSet(MerkleTree)`, i.e. ingesting `L` equals joining
//!   the atoms `{ {i} : K_i ∩ L ≠ ∅ }`.

use hllset_contracts::{sha1_hex, BitAddress};
use hllset_core::HLLSet;
use std::collections::{BTreeMap, BTreeSet};

/// A token of the universe `T`: opaque bytes.
pub type Token = Vec<u8>;

/// A named LUT: a labeled subset of `T` — one node of the lattice `2^T`.
///
/// Named nodes are physically isolated collections (`tokenLUT₁`, `tokenLUT₂`,
/// `catalogLUT`, `gateLUT`, …); the node's label is its extraction rule.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LutNode {
    pub label: String,
    pub tokens: BTreeSet<Token>,
}

impl LutNode {
    pub fn new(label: impl Into<String>, tokens: impl IntoIterator<Item = Token>) -> Self {
        Self {
            label: label.into(),
            tokens: tokens.into_iter().collect(),
        }
    }

    /// The fiber node `K_i` of this node: the tokens that hash to bit `i`.
    ///
    /// `K_i` is itself a node of the LUT lattice; its HLLSet is the atom
    /// `{i}`.
    pub fn fiber(label: impl Into<String>, i: u32, tokens: impl IntoIterator<Item = Token>) -> Self {
        Self::new(
            label,
            tokens
                .into_iter()
                .filter(|t| BitAddress::of_token(t).bit() == i),
        )
    }

    /// Lattice join in `2^T`: the union of two nodes.
    pub fn union(&self, other: &Self) -> Self {
        Self {
            label: format!("{} ∪ {}", self.label, other.label),
            tokens: self.tokens.union(&other.tokens).cloned().collect(),
        }
    }

    /// The node's own HLLSet: `ingest(L) = ⋁_{t∈L} {pos(t)}`.
    pub fn to_hllset(&self) -> HLLSet {
        HLLSet::from_tokens(self.tokens.iter())
    }
}

/// The fiber decomposition `{ L ∩ K_i }` — the reverse index of one LUT node.
///
/// This is the algebraic form of the InLUT table: bit address → candidate
/// tokens.
#[derive(Clone, Debug, Default)]
pub struct LutIndex {
    by_bit: BTreeMap<u32, BTreeSet<Token>>,
}

impl LutIndex {
    /// Build the reverse index of one LUT node.
    pub fn build(lut: &LutNode) -> Self {
        let mut by_bit: BTreeMap<u32, BTreeSet<Token>> = BTreeMap::new();
        for token in &lut.tokens {
            by_bit
                .entry(BitAddress::of_token(token).bit())
                .or_default()
                .insert(token.clone());
        }
        Self { by_bit }
    }

    /// `M(H, L) = ⋃ { L ∩ K_i : i ∈ A(H) }` — materialization as a join in
    /// `2^T`. Slice and Gate are both this function at different nodes.
    pub fn materialize(&self, hllset: &HLLSet) -> BTreeSet<Token> {
        let mut out = BTreeSet::new();
        for addr in hllset.bit_addresses() {
            if let Some(candidates) = self.by_bit.get(&addr.bit()) {
                out.extend(candidates.iter().cloned());
            }
        }
        out
    }

    /// Incremental insertion for single-touch ingestion: register a token in
    /// the fiber of its bit address (seed 0).
    pub fn insert_token(&mut self, token: Token) {
        self.insert_token_seeded(token, 0);
    }

    /// Incremental insertion under a specific seed: register a token in the
    /// fiber of its seeded bit address. Per-encoding LUTs are addressed by
    /// the same seed as their sketches.
    pub fn insert_token_seeded(&mut self, token: Token, seed: u64) {
        let bit = BitAddress::of_token_seeded(&token, seed).bit();
        self.by_bit.entry(bit).or_default().insert(token);
    }

    /// The fiber `K_i` of this index (empty if no candidate hashes to `i`).
    pub fn fiber(&self, i: u32) -> BTreeSet<Token> {
        self.by_bit.get(&i).cloned().unwrap_or_default()
    }
}

/// The sparse Merkle tree over `B` — the sketch-space presentation of a LUT
/// node (or of any HLLSet).
///
/// Leaves are the active atoms `{i}`; internal nodes are joins; the root is
/// the content address of the atom set. By the finest symmetry, the tree of
/// `L` and `ingest(L)` are the same lattice element: the tree is the
/// restoration rule of the `K_i` HLLSets, `leaf_i = ({i}, K_i ∩ L)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AtomTree {
    /// Active bit addresses, in bitmap order.
    pub atoms: Vec<u32>,
    /// Merkle root over the atoms.
    root: String,
}

impl AtomTree {
    /// The atom tree of an HLLSet: leaves = its active atoms.
    pub fn from_hllset(hllset: &HLLSet) -> Self {
        let atoms: Vec<u32> = hllset.bit_addresses().iter().map(|a| a.bit()).collect();
        Self::build(atoms)
    }

    /// The atom tree of a LUT node: leaves = `{ i : K_i ∩ L ≠ ∅ }`.
    pub fn from_lut(lut: &LutNode) -> Self {
        let atoms: BTreeSet<u32> = lut
            .tokens
            .iter()
            .map(|t| BitAddress::of_token(t).bit())
            .collect();
        Self::build(atoms.into_iter().collect())
    }

    fn build(mut atoms: Vec<u32>) -> Self {
        atoms.sort_unstable();
        atoms.dedup();
        let root = merkle_root(&atoms);
        Self { atoms, root }
    }

    pub fn root(&self) -> &str {
        &self.root
    }
}

/// Merkle root over sorted atom bits: leaf = `sha1(0x00 ‖ bit_le)`,
/// internal = `sha1(0x01 ‖ left ‖ right)`, odd level duplicates the last.
fn merkle_root(atoms: &[u32]) -> String {
    if atoms.is_empty() {
        return sha1_hex(b"empty-atom-tree");
    }
    let mut level: Vec<String> = atoms
        .iter()
        .map(|bit| {
            let mut data = vec![0x00u8];
            data.extend_from_slice(&bit.to_le_bytes());
            sha1_hex(&data)
        })
        .collect();
    while level.len() > 1 {
        let mut next = Vec::new();
        for pair in level.chunks(2) {
            let right = pair.get(1).cloned().unwrap_or_else(|| pair[0].clone());
            let mut data = vec![0x01u8];
            data.extend_from_slice(pair[0].as_bytes());
            data.extend_from_slice(right.as_bytes());
            next.push(sha1_hex(&data));
        }
        level = next;
    }
    level[0].clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(n: u32) -> Token {
        hllset_contracts::token_in_bytes(n)
    }

    fn assert_same_atoms(a: &HLLSet, tree: &AtomTree) {
        let atoms: Vec<u32> = a.bit_addresses().iter().map(|x| x.bit()).collect();
        assert_eq!(atoms, tree.atoms, "sketch and tree present the same atoms");
        assert_eq!(merkle_root(&atoms), tree.root(), "and the same root");
    }

    #[test]
    fn fibers_partition_the_lut_node() {
        let lut = LutNode::new("demo", (0..40u32).map(t));
        let idx = LutIndex::build(&lut);

        // Every token lands in exactly its fiber; fibers are disjoint.
        let mut seen = BTreeSet::new();
        for token in &lut.tokens {
            let i = BitAddress::of_token(token).bit();
            assert!(idx.fiber(i).contains(token));
            seen.insert(i);
        }
        assert_eq!(seen.len(), lut.tokens.iter().map(|x| BitAddress::of_token(x).bit()).collect::<BTreeSet<_>>().len());
    }

    #[test]
    fn materialization_is_a_join_over_fibers() {
        let lut = LutNode::new("demo", (0..64u32).map(t));
        let idx = LutIndex::build(&lut);
        let h = HLLSet::from_tokens([t(3), t(7), t(200)]); // 200 not in LUT

        let joined: BTreeSet<Token> = h
            .bit_addresses()
            .iter()
            .flat_map(|addr| idx.fiber(addr.bit()))
            .collect();
        assert_eq!(idx.materialize(&h), joined);
        assert_eq!(idx.materialize(&h).len(), 2, "only tid3 and tid7 restore");
    }

    #[test]
    fn materialization_is_a_join_homomorphism() {
        let l1 = LutNode::new("L1", (0..32u32).map(t));
        let l2 = LutNode::new("L2", (16..48u32).map(t));
        let all = l1.union(&l2);
        let h = HLLSet::from_tokens([t(0), t(20), t(40), t(999)]);

        let m_all = LutIndex::build(&all).materialize(&h);
        let m1 = LutIndex::build(&l1).materialize(&h);
        let m2 = LutIndex::build(&l2).materialize(&h);
        let union: BTreeSet<Token> = m1.union(&m2).cloned().collect();

        assert_eq!(m_all, union, "M(H, L1 ∪ L2) = M(H, L1) ∪ M(H, L2)");
    }

    #[test]
    fn slice_and_gate_are_the_same_morphism() {
        let lut = LutNode::new("tokenLUT_1", (0..64u32).map(t));
        let vocab = LutNode::new("gateLUT", (0..16u32).map(t));
        let h = HLLSet::from_tokens([t(5), t(50)]);

        // Same function M(H, L), two different LUT nodes.
        let slice = LutIndex::build(&lut).materialize(&h);
        let gate = LutIndex::build(&vocab).materialize(&h);
        assert_eq!(slice.len(), 2, "slice restores all LUT candidates");
        assert_eq!(gate.len(), 1, "gate restores only the vocabulary node");
        assert!(gate.is_subset(&slice));
    }

    #[test]
    fn default_restoration_is_the_full_image() {
        let l1 = LutNode::new("tokenLUT_1", (0..32u32).map(t));
        let l2 = LutNode::new("tokenLUT_2", (32..64u32).map(t));
        let all_known = l1.union(&l2);
        let h = HLLSet::from_tokens([t(1), t(40)]);

        let full = LutIndex::build(&all_known).materialize(&h);
        let per_node: BTreeSet<Token> = [&l1, &l2]
            .iter()
            .flat_map(|l| LutIndex::build(l).materialize(&h))
            .collect();
        assert_eq!(full, per_node, "full image = materialize against the join of known nodes");
    }

    #[test]
    fn k_i_is_a_node_and_its_hllset_is_the_atom() {
        let tokens: Vec<Token> = (0..256u32).map(t).collect();
        let i = BitAddress::of_token(&t(44)).bit();
        let k_i = LutNode::fiber("K_i", i, tokens.iter().cloned());

        assert!(!k_i.tokens.is_empty());
        assert!(k_i.tokens.iter().all(|x| BitAddress::of_token(x).bit() == i));

        // ingest(K_i) = the atom {i}.
        let atom = k_i.to_hllset();
        assert_eq!(atom.popcount(), 1);
        assert_eq!(atom.bit_addresses()[0].bit(), i);
    }

    #[test]
    fn the_finest_symmetry_holds() {
        let lut = LutNode::new("tokenLUT_all", (0..300u32).map(t));

        // HLLSet(LUT): ingest the token collection.
        let from_tokens = lut.to_hllset();
        // HLLSet(MerkleTree): the atom tree of the same node.
        let tree = AtomTree::from_lut(&lut);

        assert_same_atoms(&from_tokens, &tree);
        assert_eq!(from_tokens.popcount(), tree.atoms.len() as u64);

        // And the tree of an arbitrary sketch equals its atom presentation.
        let h = HLLSet::from_tokens([t(10), t(100), t(1000)]);
        assert_same_atoms(&h, &AtomTree::from_hllset(&h));
    }
}
