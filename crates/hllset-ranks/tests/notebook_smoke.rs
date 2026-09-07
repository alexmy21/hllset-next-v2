use hllset_dsl::LatticeElement;
use hllset_ranks::compound::*;
use hllset_ranks::derivatives::*;
use hllset_ranks::fisher::*;
use hllset_ranks::hllset::*;
use hllset_ranks::mask::*;
use std::collections::HashMap;

#[test]
fn test_level4_notebook_cell() {
    let deg_k = DegreeRankFn;
    let alpha = LatticeElement::from_tokens(&["alpha", "beta", "gamma"]);
    let beta = LatticeElement::from_tokens(&["beta", "gamma", "delta"]);
    let gamma = LatticeElement::from_tokens(&["gamma", "delta", "epsilon"]);
    let mut idx = HLLSetRankIndex::new();
    for (elem, deg) in [(&alpha, 2usize), (&beta, 2), (&gamma, 2)] {
        let r = HLLSetRank::new(elem, deg, &deg_k);
        idx.insert(r);
    }
    assert_eq!(idx.len(), 3);
    for rank in idx.iter() {
        assert!(!rank.key.is_empty());
    }
}

#[test]
fn test_full_pipeline() {
    let doc_a = LatticeElement::from_tokens(&["machine", "learning", "neural", "network"]);
    let doc_b = LatticeElement::from_tokens(&["deep", "learning", "gradient", "network"]);
    let rank_a = HLLSetRank::new(&doc_a, 4, &DegreeRankFn);
    let rank_b = HLLSetRank::new(&doc_b, 4, &DegreeRankFn);
    let union_rank = CompoundRank::union(rank_a.value, rank_b.value);
    let inter_rank = CompoundRank::intersection(rank_a.value, rank_b.value);
    assert!(union_rank.value >= rank_a.value);
    assert!(inter_rank.value <= rank_a.value);
}

#[test]
fn test_mask_notebook() {
    let mut idx = HLLSetRankIndex::new();
    for (key, value, degree) in [
        ("h:a", 100u64, 3usize),
        ("h:b", 80, 4),
        ("h:c", 60, 2),
        ("h:d", 40, 1),
        ("h:e", 20, 1),
    ] {
        let mut r = HLLSetRank::from_raw(key, degree, value * 10, &DegreeRankFn);
        r.value = value;
        idx.insert(r);
    }
    let mask50 = ObservableMask::apply(&idx, 50);
    assert_eq!(mask50.total, 5);
    assert!(mask50.observable_count() > 0);
    assert!(mask50.observable_count() < 5); // some must be hidden
}
