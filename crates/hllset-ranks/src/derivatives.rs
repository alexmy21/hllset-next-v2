//! Rank Derivatives and Noether Steering.
//!
//! Ranks are static; derivatives measure motion. All integer — FPGA-native.
//!
//! - ΔR(t) = R(t) - R(t-1) — rank velocity, decomposed via D/R/N
//! - Δ²R(t) = ΔR(t) - ΔR(t-1) — rank acceleration
//! - Noether steering: |ΣR(N) - ΣR(D)| → 0 (rank-weighted conservation)

use crate::Rank;
use hllset_dsl::LatticeElement;

/// Decomposition of rank change across D/R/N bitmasks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RankFlux {
    /// Sum of rank changes for retained bits: Σ(R_b(t) - R_b(t-1)) for b ∈ H(t) ∩ H(t-1)
    pub rank_drift: Rank,
    /// Sum of ranks for newly added bits: Σ R_b(t) for b ∈ N(t)
    pub rank_influx: Rank,
    /// Sum of ranks for departed bits: Σ R_b(t-1) for b ∈ D(t-1)
    pub rank_outflux: Rank,
    /// Net flux: influx - outflux
    pub net_flux: i64,
}

/// First and second discrete derivatives of the rank signal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RankDerivatives {
    /// First derivative (velocity) at time t.
    pub velocity: i64,
    /// Second derivative (acceleration) at time t.
    /// Requires at least 3 time steps: t, t-1, t-2.
    pub acceleration: Option<i64>,
    /// Time step index.
    pub t: usize,
}

/// Noether steering signal: rank-weighted conservation check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NoetherSteering {
    /// |card(N) - card(D)| — raw bit-count divergence.
    pub bit_divergence: u64,
    /// |Σ R(N) - Σ R(D)| — rank-weighted divergence.
    pub rank_divergence: u64,
    /// Is the system in structural equilibrium? (bit_divergence below threshold)
    pub structural_equilibrium: bool,
    /// Is the system in rank equilibrium? (rank_divergence below threshold)
    pub rank_equilibrium: bool,
}

impl NoetherSteering {
    /// Compute Noether steering from D/R/N decomposition of two HLLSets.
    ///
    /// `prev` = H(t-1), `curr` = H(t).
    /// `rank_of_bit` is a function mapping bit position to its current rank.
    pub fn compute(
        prev: &LatticeElement,
        curr: &LatticeElement,
        rank_of_bit: &dyn Fn(u32, u32) -> Rank,
        bit_threshold: u64,
        rank_threshold: u64,
    ) -> Self {
        let prev_hll = prev.hllset();
        let curr_hll = curr.hllset();

        // Collect positions
        let prev_positions: std::collections::HashSet<(u32, u32)> =
            prev_hll.active_positions().into_iter().collect();
        let curr_positions: std::collections::HashSet<(u32, u32)> =
            curr_hll.active_positions().into_iter().collect();

        // N(t) = curr - prev (newly added)
        let n_count = curr_positions.difference(&prev_positions).count() as u64;
        let n_rank: Rank = curr_positions
            .difference(&prev_positions)
            .map(|&(reg, tz)| rank_of_bit(reg, tz))
            .sum();

        // D(t-1) = prev - curr (departed)
        let d_count = prev_positions.difference(&curr_positions).count() as u64;
        let d_rank: Rank = prev_positions
            .difference(&curr_positions)
            .map(|&(reg, tz)| rank_of_bit(reg, tz))
            .sum();

        let bit_divergence = if n_count > d_count {
            n_count - d_count
        } else {
            d_count - n_count
        };

        let rank_divergence = if n_rank > d_rank {
            n_rank - d_rank
        } else {
            d_rank - n_rank
        };

        Self {
            bit_divergence,
            rank_divergence,
            structural_equilibrium: bit_divergence <= bit_threshold,
            rank_equilibrium: rank_divergence <= rank_threshold,
        }
    }

    /// Quick check using popcount-based ranks (each bit's rank = 1).
    /// This reduces to the original bit-count Noether steering.
    pub fn compute_popcount(prev: &LatticeElement, curr: &LatticeElement, threshold: u64) -> Self {
        Self::compute(prev, curr, &|_, _| 1, threshold, threshold)
    }
}

impl RankFlux {
    /// Compute rank flux from two HLLSets and a per-bit rank function.
    pub fn compute(
        prev: &LatticeElement,
        curr: &LatticeElement,
        rank_of_bit: &dyn Fn(u32, u32) -> Rank,
    ) -> Self {
        let prev_hll = prev.hllset();
        let curr_hll = curr.hllset();

        let prev_positions: std::collections::HashSet<(u32, u32)> =
            prev_hll.active_positions().into_iter().collect();
        let curr_positions: std::collections::HashSet<(u32, u32)> =
            curr_hll.active_positions().into_iter().collect();

        // Retained: in both
        let retained: Vec<_> = prev_positions.intersection(&curr_positions).collect();
        let rank_drift: i64 = retained
            .iter()
            .map(|&&(reg, tz)| rank_of_bit(reg, tz) as i64 - rank_of_bit(reg, tz) as i64)
            .sum();
        // For drift we need prev vs curr rank difference. With the same rank function,
        // drift is zero unless ranks changed between samples. Simplified: drift = 0.

        let rank_influx: Rank = curr_positions
            .difference(&prev_positions)
            .map(|&(reg, tz)| rank_of_bit(reg, tz))
            .sum();

        let rank_outflux: Rank = prev_positions
            .difference(&curr_positions)
            .map(|&(reg, tz)| rank_of_bit(reg, tz))
            .sum();

        let net_flux = rank_influx as i64 - rank_outflux as i64;

        Self {
            rank_drift: 0, // simplified: ranks static between samples
            rank_influx,
            rank_outflux,
            net_flux,
        }
    }
}

impl RankDerivatives {
    /// Compute derivatives from a sliding window of net flux values.
    ///
    /// `flux_history` is a vec of net_flux values at times [t-N, ..., t].
    pub fn from_flux_history(flux_history: &[i64]) -> Vec<Self> {
        let mut result = Vec::with_capacity(flux_history.len());
        for (i, &flux) in flux_history.iter().enumerate() {
            let prev_flux = if i > 0 { flux_history[i - 1] } else { flux };
            let velocity = flux - prev_flux;
            let acceleration = if i > 1 {
                let prev_velocity = prev_flux - flux_history[i - 2];
                Some(velocity - prev_velocity)
            } else {
                None
            };
            result.push(Self {
                velocity,
                acceleration,
                t: i,
            });
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hllset_dsl::LatticeElement;

    #[test]
    fn test_noether_steering_identical() {
        let a = LatticeElement::from_tokens(&["hello", "world"]);
        let ns = NoetherSteering::compute_popcount(&a, &a, 5);
        assert_eq!(ns.bit_divergence, 0);
        assert!(ns.structural_equilibrium);
    }

    #[test]
    fn test_noether_steering_different() {
        let a = LatticeElement::from_tokens(&["hello", "world", "foo", "bar"]);
        let b = LatticeElement::from_tokens(&["hello", "world", "baz"]);
        let ns = NoetherSteering::compute_popcount(&a, &b, 5);
        // a and b share "hello" and "world", differ on foo/bar vs baz
        assert!(ns.bit_divergence > 0);
    }

    #[test]
    fn test_derivatives_from_flux() {
        let fluxes = vec![10, 12, 11, 15, 14];
        let derivs = RankDerivatives::from_flux_history(&fluxes);
        assert_eq!(derivs.len(), 5);
        // t=0: velocity = 10-10 = 0
        assert_eq!(derivs[0].velocity, 0);
        assert_eq!(derivs[0].acceleration, None);
        // t=1: velocity = 12-10 = 2, acceleration = None
        assert_eq!(derivs[1].velocity, 2);
        assert_eq!(derivs[1].acceleration, None);
        // t=2: velocity = 11-12 = -1, acceleration = -1-2 = -3
        assert_eq!(derivs[2].velocity, -1);
        assert_eq!(derivs[2].acceleration, Some(-3));
        // t=3: velocity = 15-11 = 4, acceleration = 4-(-1) = 5
        assert_eq!(derivs[3].velocity, 4);
        assert_eq!(derivs[3].acceleration, Some(5));
        // t=4: velocity = 14-15 = -1, acceleration = -1-4 = -5
        assert_eq!(derivs[4].velocity, -1);
        assert_eq!(derivs[4].acceleration, Some(-5));
    }

    #[test]
    fn test_rank_flux() {
        let a = LatticeElement::from_tokens(&["a", "b", "c"]);
        let b = LatticeElement::from_tokens(&["b", "c", "d"]);
        // Each bit has rank = 1 (popcount-based)
        let flux = RankFlux::compute(&a, &b, &|_, _| 1);
        // N: d (1 bit), D: a (1 bit)
        assert_eq!(flux.rank_influx, 1);
        assert_eq!(flux.rank_outflux, 1);
        assert_eq!(flux.net_flux, 0);
    }
}
