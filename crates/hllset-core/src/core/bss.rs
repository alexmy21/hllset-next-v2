//! Bell State Similarity (BSS) morphisms for HLLSet.
//!
//! BSS is the directed similarity measure from the HLLSet theoretical
//! foundation. It provides two fundamental measures:
//!
//! - **BSSτ** (inclusion): |A ∩ B| / |B| — the fraction of B's content
//!   that is also in A. Measures how much B is "contained in" A.
//!
//! - **BSSρ** (exclusion): |A \ B| / |B| — the fraction of B's capacity
//!   that is novel relative to A. Measures how much of A is NOT in B.
//!
//! ## Morphisms
//!
//! A **morphism** A → B exists when B is sufficiently included in A
//! (high τ) and A has limited novelty relative to B (low ρ):
//!
//! ```text
//! A → B  iff  BSSτ(A, B) ≥ τ_min  AND  BSSρ(A, B) ≤ ρ_max
//! ```
//!
//! Morphisms compose naturally:
//!
//! ```text
//! If A → B (τ₁, ρ₁) and B → C (τ₂, ρ₂)
//! Then A → C (min(τ₁, τ₂), max(ρ₁, ρ₂))
//! ```

use crate::core::hllset::HLLSet;

/// Result of a BSS morphism check.
#[derive(Clone, Debug, PartialEq)]
pub struct BSSResult {
    /// BSSτ: inclusion measure |A ∩ B| / |B|.
    pub inclusion: f64,
    /// BSSρ: exclusion measure |A \ B| / |B|.
    pub exclusion: f64,
    /// Whether the morphism A → B holds under the given thresholds.
    pub morphism_holds: bool,
}

impl HLLSet {
    /// BSSτ — Bell State Similarity inclusion: |A ∩ B| / |B|.
    ///
    /// Answers: "How much of B's content is also in A?"
    ///
    /// Returns 1.0 if B is empty (vacuously all of nothing is in A).
    pub fn bss_inclusion(&self, other: &HLLSet) -> f64 {
        let b_card = other.cardinality();
        if b_card == 0.0 {
            return 1.0;
        }
        let inter_card = self.intersection(other).cardinality();
        (inter_card / b_card).min(1.0)
    }

    /// BSSρ — Bell State Similarity exclusion: |A \ B| / |B|.
    ///
    /// Answers: "How much novel content does A have that B does not,
    /// relative to B's size?"
    ///
    /// Returns 0.0 if B is empty.
    pub fn bss_exclusion(&self, other: &HLLSet) -> f64 {
        let b_card = other.cardinality();
        if b_card == 0.0 {
            return 0.0;
        }
        let diff_card = self.difference(other).cardinality();
        (diff_card / b_card).min(1.0)
    }

    /// Check if a morphism A → B exists under the given thresholds.
    ///
    /// A morphism holds when:
    /// - `bss_inclusion(self, other) >= tau_min` (enough of B is in A)
    /// - `bss_exclusion(self, other) <= rho_max` (A has limited novelty)
    ///
    /// # Arguments
    /// - `tau_min`: minimum inclusion threshold (0.0..1.0)
    /// - `rho_max`: maximum exclusion threshold (0.0..1.0)
    pub fn morph_to(&self, other: &HLLSet, tau_min: f64, rho_max: f64) -> BSSResult {
        let inclusion = self.bss_inclusion(other);
        let exclusion = self.bss_exclusion(other);
        let morphism_holds = inclusion >= tau_min && exclusion <= rho_max;

        BSSResult {
            inclusion,
            exclusion,
            morphism_holds,
        }
    }

    /// Check if this HLLSet is approximately equal to another.
    ///
    /// Uses symmetric BSS: both mutual inclusions must meet the threshold.
    /// This is more appropriate for HLLSets than exact bitmap equality
    /// since hash collisions prevent truly identical bitmaps.
    pub fn approx_eq(&self, other: &HLLSet, tau: f64) -> bool {
        self.morph_to(other, tau, 0.0).morphism_holds
            && other.morph_to(self, tau, 0.0).morphism_holds
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set(tokens: &[&str]) -> HLLSet {
        HLLSet::from_tokens(tokens.iter().map(|s| s.as_bytes()))
    }

    #[test]
    fn test_bss_inclusion_self_is_one() {
        let a = set(&["a", "b", "c"]);
        let tau = a.bss_inclusion(&a);
        assert!((tau - 1.0).abs() < 0.01, "tau = {tau}");
    }

    #[test]
    fn test_bss_exclusion_self_is_zero() {
        let a = set(&["a", "b", "c"]);
        let rho = a.bss_exclusion(&a);
        assert!(rho < 0.01, "rho = {rho}");
    }

    #[test]
    fn test_bss_inclusion_subset() {
        let a = set(&["a", "b", "c", "d", "e"]);
        let b = set(&["a", "b"]);
        // All of B's content is in A → τ should be close to 1.0
        let tau = a.bss_inclusion(&b);
        assert!(tau > 0.8, "tau = {tau}");
    }

    #[test]
    fn test_bss_inclusion_bounds() {
        let a = set(&["a", "b"]);
        let b = set(&["c", "d"]);
        let tau = a.bss_inclusion(&b);
        assert!((0.0..=1.0).contains(&tau), "tau out of bounds: {tau}");
    }

    #[test]
    fn test_morphism_self_holds() {
        let a = set(&["x", "y", "z"]);
        let result = a.morph_to(&a, 0.8, 0.2);
        assert!(result.morphism_holds);
        assert!((result.inclusion - 1.0).abs() < 0.01);
        assert!(result.exclusion < 0.01);
    }

    #[test]
    fn test_morphism_subset_holds() {
        // A is a large set; B is mostly overlapping with A.
        // Many tokens shared → high τ, similar sizes → low ρ.
        let shared: Vec<String> = (0..50).map(|i| format!("shared_{i}")).collect();
        let extra_a: Vec<String> = (0..3).map(|i| format!("only_a_{i}")).collect();
        let extra_b: Vec<String> = (0..2).map(|i| format!("only_b_{i}")).collect();

        let mut tokens_a: Vec<&[u8]> = shared.iter().map(|s| s.as_bytes()).collect();
        tokens_a.extend(extra_a.iter().map(|s| s.as_bytes()));
        let mut tokens_b: Vec<&[u8]> = shared.iter().map(|s| s.as_bytes()).collect();
        tokens_b.extend(extra_b.iter().map(|s| s.as_bytes()));

        let a = HLLSet::from_tokens(&tokens_a);
        let b = HLLSet::from_tokens(&tokens_b);

        let result = a.morph_to(&b, 0.8, 0.1);
        assert!(
            result.morphism_holds,
            "subset morphism should hold; τ={}, ρ={}",
            result.inclusion, result.exclusion
        );
    }

    #[test]
    fn test_morphism_disjoint_fails() {
        let a = set(&["a", "b"]);
        let b = set(&["c", "d"]);
        let result = a.morph_to(&b, 0.8, 0.2);
        assert!(
            !result.morphism_holds,
            "disjoint should not morph; τ={}, ρ={}",
            result.inclusion, result.exclusion
        );
    }

    #[test]
    fn test_approx_eq_same() {
        let a = set(&["x", "y", "z"]);
        assert!(a.approx_eq(&a, 0.9));
    }

    #[test]
    fn test_approx_eq_different() {
        let a = set(&["a", "b"]);
        let b = set(&["c", "d"]);
        assert!(!a.approx_eq(&b, 0.9));
    }

    #[test]
    fn test_bss_empty_set() {
        let a = set(&["a"]);
        let empty = HLLSet::new();
        // Inclusion: all of nothing is in A → 1.0
        assert!((a.bss_inclusion(&empty) - 1.0).abs() < 0.01);
        // Exclusion: nothing novel over empty → 0.0
        assert!(a.bss_exclusion(&empty) < 0.01);
    }

    #[test]
    fn test_bss_into_empty() {
        let a = set(&["a"]);
        let empty = HLLSet::new();
        // How much of A is in empty → 0.0
        assert!(empty.bss_inclusion(&a) < 0.01);
    }
}
