//! `hllset-context` — the Noether context algebra, rewritten from the
//! algebra, not from the legacy conversation crate.
//!
//! `H(t) = (S(t), H(t-1), D, R, N)`, expressed structurally:
//!
//! - [`Context`]: `S(t)` as a lattice element with **declared generators**
//!   (`S(t) = ⋁ generators`) — the coarsest lawful leaf set. Operations are
//!   persistent: they return new contexts, inputs are never mutated.
//! - [`evolve`]: the D/R/N derivation between two states, with the Noether
//!   invariants `D∪R = H(t-1)`, `R∪N = S(t)`, `D∩N = ∅`.
//! - [`FollowMatrix`]: the tropical matrix over tokens (`⊕ = max`, `⊗ = +`)
//!   — grow-only, `merge` is a join, and restriction to the active
//!   vocabulary `V(H)` is a lattice projection.
//!
//! Restoration is *not* reimplemented here: it goes through
//! `hllset-morphisms` (LUT-first materialization, TF only for ambiguity).
//! This crate holds the structural context those morphisms operate on.

use hllset_core::HLLSet;
use std::collections::{BTreeMap, BTreeSet};

/// Bitmap equality (HLLSet has no `PartialEq`): both differences are empty.
fn same_set(a: &HLLSet, b: &HLLSet) -> bool {
    a.difference(b).popcount() == 0 && b.difference(a).popcount() == 0
}

/// A context state: `S(t)` with its declared generator leaves.
#[derive(Clone, Debug)]
pub struct Context {
    state: HLLSet,
    generators: Vec<HLLSet>,
}

impl Context {
    /// Build `S(t) = ⋁ generators` — the join of the declared leaves.
    pub fn from_generators(generators: Vec<HLLSet>) -> Self {
        let state = HLLSet::union_all(generators.iter().cloned());
        Self { state, generators }
    }

    pub fn state(&self) -> &HLLSet {
        &self.state
    }

    pub fn generators(&self) -> &[HLLSet] {
        &self.generators
    }

    /// Persistent insert of a generator: a new context, inputs untouched.
    pub fn with_generator(&self, generator: HLLSet) -> Self {
        let mut generators = self.generators.clone();
        generators.push(generator);
        Self::from_generators(generators)
    }

    /// Persistent remove of a generator (by bitmap equality): a new context.
    pub fn without_generator(&self, generator: &HLLSet) -> Self {
        let generators = self
            .generators
            .iter()
            .filter(|g| !same_set(g, generator))
            .cloned()
            .collect();
        Self::from_generators(generators)
    }
}

/// The Noether derivation between two states.
#[derive(Clone, Debug)]
pub struct Noether {
    pub departed: HLLSet,
    pub retained: HLLSet,
    pub novel: HLLSet,
}

/// `H(t) = (S(t), H(t-1), D, R, N)`:
/// `D = H(t-1) \ S(t)`, `R = H(t-1) ∩ S(t)`, `N = S(t) \ H(t-1)`.
pub fn evolve(previous: &HLLSet, current: &HLLSet) -> Noether {
    Noether {
        departed: previous.difference(current),
        retained: previous.intersection(current),
        novel: current.difference(previous),
    }
}

/// The three Noether invariants.
pub fn invariants_hold(n: &Noether, previous: &HLLSet, current: &HLLSet) -> bool {
    same_set(&n.departed.union(&n.retained), previous)
        && same_set(&n.retained.union(&n.novel), current)
        && n.departed.intersection(&n.novel).popcount() == 0
}

/// The tropical follow matrix over tokens: `⊕ = max`, `⊗ = +`.
///
/// Grow-only (`observe` never lowers a cell), `merge` is the join, and
/// [`FollowMatrix::restrict`] is the lattice projection onto `V(H)`.
#[derive(Clone, Debug, Default)]
pub struct FollowMatrix {
    cells: BTreeMap<(Vec<u8>, Vec<u8>), u64>,
}

impl FollowMatrix {
    /// Observe one transition with a weight: `cell = max(cell, weight)`.
    pub fn observe(&mut self, from: &[u8], to: &[u8], weight: u64) {
        let slot = self.cells.entry((from.to_vec(), to.to_vec())).or_insert(0);
        *slot = (*slot).max(weight);
    }

    pub fn get(&self, from: &[u8], to: &[u8]) -> u64 {
        self.cells
            .get(&(from.to_vec(), to.to_vec()))
            .copied()
            .unwrap_or(0)
    }

    /// CRDT join: pointwise max over the cells.
    pub fn merge(&mut self, other: &Self) {
        for ((a, b), &w) in &other.cells {
            let slot = self.cells.entry((a.clone(), b.clone())).or_insert(0);
            *slot = (*slot).max(w);
        }
    }

    /// Lattice projection: keep only transitions between tokens in `tokens`.
    pub fn restrict(&self, tokens: &BTreeSet<Vec<u8>>) -> Self {
        let cells = self
            .cells
            .iter()
            .filter(|((a, b), _)| tokens.contains(a) && tokens.contains(b))
            .map(|(k, &w)| (k.clone(), w))
            .collect();
        Self { cells }
    }

    pub fn len(&self) -> usize {
        self.cells.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hllset_contracts::token_in_bytes;
    use hllset_core::HLLSet;

    fn hll(ids: &[u32]) -> HLLSet {
        HLLSet::from_tokens(ids.iter().map(|&n| token_in_bytes(n)))
    }

    #[test]
    fn generators_collapse_to_the_state_and_are_reversible() {
        let a = hll(&[0, 1, 2]);
        let b = hll(&[2, 3]);
        let ctx = Context::from_generators(vec![a.clone(), b.clone()]);

        assert!(same_set(
            ctx.state(),
            &HLLSet::union_all(vec![a.clone(), b.clone()])
        ));

        let c = hll(&[9]);
        let with_c = ctx.with_generator(c.clone());
        assert_eq!(with_c.generators().len(), 3);
        assert!(same_set(with_c.state(), &ctx.state().union(&c)));

        assert_eq!(
            with_c.without_generator(&c).generators().len(),
            2,
            "insert∘remove = id at the generator level"
        );
    }

    #[test]
    fn noether_invariants_hold_and_are_exact() {
        // H(t-1) = {a, b}, S(t) = {a, c}.
        let a = hll(&[0, 1, 2]);
        let b = hll(&[3, 4]);
        let c = hll(&[2, 5, 6]);
        let previous = HLLSet::union_all(vec![a.clone(), b.clone()]);
        let current = HLLSet::union_all(vec![a.clone(), c.clone()]);

        let n = evolve(&previous, &current);
        assert!(invariants_hold(&n, &previous, &current));
        assert_eq!(n.departed.popcount(), 2, "bits of b only");
        assert_eq!(n.novel.popcount(), 2, "bits of c only");
        assert_eq!(n.retained.popcount(), 3, "bits of a");
    }

    #[test]
    fn follow_matrix_is_tropical() {
        let mut m = FollowMatrix::default();
        m.observe(b"a", b"b", 2);
        m.observe(b"a", b"b", 1); // max keeps 2
        m.observe(b"b", b"c", 3);
        assert_eq!(m.get(b"a", b"b"), 2, "observe is max");
        assert_eq!(m.get(b"b", b"c"), 3);
        assert_eq!(m.get(b"c", b"a"), 0);

        // Merge is the CRDT join: commutative, idempotent.
        let mut n = FollowMatrix::default();
        n.observe(b"a", b"b", 5);
        n.observe(b"x", b"y", 1);
        let mut mn = m.clone();
        mn.merge(&n);
        assert_eq!(mn.get(b"a", b"b"), 5, "join takes the max");
        assert_eq!(mn.get(b"b", b"c"), 3, "join keeps other side");
        assert_eq!(mn.get(b"x", b"y"), 1);

        let mut nm = n.clone();
        nm.merge(&m);
        assert_eq!(mn.len(), nm.len(), "commutative");
        assert_eq!(mn.get(b"a", b"b"), nm.get(b"a", b"b"));

        // Restrict is a projection onto the active vocabulary.
        let active: BTreeSet<Vec<u8>> = [b"a".to_vec(), b"b".to_vec()].into_iter().collect();
        let restricted = mn.restrict(&active);
        assert_eq!(restricted.get(b"a", b"b"), 5);
        assert_eq!(restricted.get(b"b", b"c"), 0, "c dropped");
        assert_eq!(restricted.get(b"x", b"y"), 0, "x,y dropped");
    }

    #[test]
    fn context_restores_through_the_morphisms() {
        // Build a context from a generator; restoration goes through
        // hllset-morphisms (LUT-first) — not reimplemented here.
        use hllset_morphisms::{materialize, Ingest};

        let mut ingest = Ingest::new();
        // Pick ids whose seed-0 atoms are pairwise distinct.
        let mut chosen: Vec<u32> = Vec::new();
        let mut bits = BTreeSet::new();
        for id in 0..1000u32 {
            let tok = token_in_bytes(id);
            let bit = hllset_contracts::BitAddress::of_token_seeded(&tok, 0).bit();
            if bits.insert(bit) {
                chosen.push(id);
                ingest.ingest_token(&tok);
                if chosen.len() == 20 {
                    break;
                }
            }
        }
        assert_eq!(chosen.len(), 20);

        let ctx = Context::from_generators(vec![ingest.hllset(0).clone()]);
        let restored = materialize(&[(ctx.state(), ingest.lut(0))], ingest.tf());

        for id in &chosen {
            assert!(
                restored.contains(&token_in_bytes(*id)),
                "tid{id} restored from the context through the LUT"
            );
        }
    }
}
