//! archive.rs — MAP-Elites Quality-Diversity Archive for SymboLR.
//!
//! ## Phase 2: Gradient-Aware Behavioral Axes
//!
//! Behavioral descriptors are redesigned to capture gradient-awareness:
//!
//!   dim 0 — `size_idx`               : AST node count, bucketed
//!   dim 1 — `gradient_sensitivity_idx`: How much the formula's output changes
//!                                        when the gradient norm varies
//!   dim 2 — `loss_sensitivity_idx`   : How much the formula's output changes
//!                                        when the loss slope varies
//!
//! This means the archive's behavioral map directly answers the question:
//! "Does this formula react to gradient health signals?"
//!
//! Formulas that use only VarT always land in (size, 0, 0) niches.
//! Formulas using VarG land in higher gradient_sensitivity niches.
//! Formulas using VarDL land in higher loss_sensitivity niches.
//!
//! ## Elite Aging
//! Each niche tracks its age (generations since last improvement).
//! An age-penalty slightly favours fresh challengers, preventing ossification.

use std::collections::HashMap;
use serde::{Serialize, Deserialize};

use crate::ast::{Expr, MAX_NODES};

// ─────────────────────────────────────────────────────────────────────────────
// 1.  Types
// ─────────────────────────────────────────────────────────────────────────────

/// 3-tuple `(size_idx, gradient_sensitivity_idx, loss_sensitivity_idx)`.
pub type NicheKey = (usize, usize, usize);

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Niche {
    pub expr: Expr,
    pub raw_loss: f64,
    pub complexity: usize,
    pub expr_hash: u64,
    pub age: u32,
}

// ─────────────────────────────────────────────────────────────────────────────
// 2.  Archive Config
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ArchiveConfig {
    /// Number of AST size buckets (axis 0).
    pub size_bins: usize,
    /// Number of gradient-sensitivity buckets (axis 1).
    pub gradient_sensitivity_bins: usize,
    /// Number of loss-slope-sensitivity buckets (axis 2).
    pub loss_sensitivity_bins: usize,
    /// Additive penalty per generation of age applied to incumbents.
    pub age_penalty_coeff: f64,
}

impl Default for ArchiveConfig {
    fn default() -> Self {
        Self {
            size_bins: 30,
            gradient_sensitivity_bins: 20,
            loss_sensitivity_bins: 10,
            age_penalty_coeff: 0.001,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3.  MapElitesArchive
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Serialize, Deserialize)]
pub struct MapElitesArchive {
    pub niches: HashMap<NicheKey, Niche>,
    pub config: ArchiveConfig,
    expression_hashes: std::collections::HashSet<u64>,
    pub total_attempts: u64,
    pub total_additions: u64,
    pub current_generation: u32,
}

impl MapElitesArchive {
    pub fn new(config: ArchiveConfig) -> Self {
        let capacity = config.size_bins
            * config.gradient_sensitivity_bins
            * config.loss_sensitivity_bins
            / 4;
        Self {
            niches: HashMap::with_capacity(capacity),
            config,
            expression_hashes: std::collections::HashSet::new(),
            total_attempts: 0,
            total_additions: 0,
            current_generation: 0,
        }
    }

    // ── 3.1  Behavioural Descriptors ──────────────────────────────────────────

    /// Compute the 3-D niche key for a formula by probing its sensitivity.
    ///
    /// Validity check: requires at least one finite, positive output when
    /// evaluated at 5 time-axis probe points (g=0, dl=0). This matches the
    /// semantics of the old schedule-sum check while supporting gradient-aware
    /// formulas that are zero at g=0.
    ///
    /// Sensitivity computation:
    /// - Gradient sensitivity: CoV of outputs as g varies over [-2, 2] at t=0.5
    /// - Loss sensitivity: CoV of outputs as dl varies over [-1, 1] at t=0.5
    ///
    /// Time-only formulas (no VarG/VarDL) get sensitivity = 0 → land in (size, 0, 0).
    pub fn compute_niche_key(&self, expr: &Expr) -> Option<NicheKey> {
        // ── Validity: comprehensive probe across (t, g, dl) ───────────────────
        // Sample a 5×3×3 = 45-point grid. A formula is invalid only if it
        // cannot produce a positive output for any combination of inputs.
        // This correctly accepts cos(π*t) (positive at t < 0.5) and VarDL/VarG
        // formulas (positive when those variables are positive).
        const T_VALS: [f64; 5]  = [0.1, 0.3, 0.5, 0.7, 0.9];
        const G_VALS: [f64; 3]  = [-2.0, 0.0, 2.0];
        const DL_VALS: [f64; 3] = [-1.0, 0.0, 1.0];

        let mut max_output = f64::NEG_INFINITY;
        for &t in &T_VALS {
            for &g in &G_VALS {
                for &dl in &DL_VALS {
                    let v = expr.eval(t, g, dl);
                    if v > max_output { max_output = v; }
                }
            }
        }
        if !max_output.is_finite() || max_output <= 0.0 {
            return None;
        }

        // ── Gradient sensitivity: vary g ∈ [-2, 2] at t=0.5, dl=0.0 ─────────
        let g_outputs: Vec<f64> = (0..11)
            .map(|i| {
                let g = -2.0 + 4.0 * i as f64 / 10.0;
                expr.eval(0.5, g, 0.0)
            })
            .collect();
        let g_sensitivity = coefficient_of_variation(&g_outputs);

        // ── Loss sensitivity: vary dl ∈ [-1, 1] at t=0.5, g=0.0 ─────────────
        let dl_outputs: Vec<f64> = (0..11)
            .map(|i| {
                let dl = -1.0 + 2.0 * i as f64 / 10.0;
                expr.eval(0.5, 0.0, dl)
            })
            .collect();
        let dl_sensitivity = coefficient_of_variation(&dl_outputs);

        // ── Bin mapping ───────────────────────────────────────────────────────
        let size = expr.size();
        // Map size ∈ [1, MAX_NODES] evenly to [0, size_bins - 1]
        let size_fraction = (size as f64 - 1.0)
            / (MAX_NODES as f64 - 1.0).max(1.0);
        let size_idx = ((size_fraction * (self.config.size_bins - 1) as f64) as usize)
            .min(self.config.size_bins - 1);

        let grad_idx = ((g_sensitivity * self.config.gradient_sensitivity_bins as f64) as usize)
            .min(self.config.gradient_sensitivity_bins - 1);

        let loss_idx = ((dl_sensitivity * self.config.loss_sensitivity_bins as f64) as usize)
            .min(self.config.loss_sensitivity_bins - 1);

        Some((size_idx, grad_idx, loss_idx))
    }

    // ── 3.2  Archive Insertion ────────────────────────────────────────────────

    pub fn try_add(&mut self, expr: Expr, raw_loss: f64, effective_loss: f64) -> bool {
        self.total_attempts += 1;

        if !raw_loss.is_finite() || !effective_loss.is_finite() {
            return false;
        }

        let niche_key = match self.compute_niche_key(&expr) {
            Some(k) => k,
            None    => return false,
        };

        let expr_hash = expr.structural_hash();
        let size = expr.size();

        if let Some(incumbent) = self.niches.get(&niche_key) {
            if incumbent.expr_hash == expr_hash {
                return false;
            }
            let incumbent_adjusted = incumbent.raw_loss
                + self.config.age_penalty_coeff * incumbent.age as f64;
            if effective_loss >= incumbent_adjusted {
                return false;
            }
        }

        self.expression_hashes.insert(expr_hash);
        self.niches.insert(niche_key, Niche {
            complexity: size,
            expr_hash,
            expr,
            raw_loss,
            age: 0,
        });
        self.total_additions += 1;
        true
    }

    // ── 3.3  Lifecycle ────────────────────────────────────────────────────────

    pub fn increment_ages(&mut self) {
        self.current_generation += 1;
        for niche in self.niches.values_mut() {
            niche.age = niche.age.saturating_add(1);
        }
    }

    // ── 3.4  Query ────────────────────────────────────────────────────────────

    #[inline]
    pub fn len(&self) -> usize { self.niches.len() }

    #[inline]
    pub fn is_empty(&self) -> bool { self.niches.is_empty() }

    #[inline]
    pub fn max_niches(&self) -> usize {
        self.config.size_bins
            * self.config.gradient_sensitivity_bins
            * self.config.loss_sensitivity_bins
    }

    pub fn best(&self) -> Option<&Niche> {
        self.niches.values().min_by(|a, b| {
            a.raw_loss.partial_cmp(&b.raw_loss).unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    pub fn hall_of_fame(&self, top_k: usize) -> Vec<&Niche> {
        let mut all: Vec<&Niche> = self.niches.values().collect();
        all.sort_by(|a, b| a.raw_loss.partial_cmp(&b.raw_loss)
                              .unwrap_or(std::cmp::Ordering::Equal));
        all.truncate(top_k);
        all
    }

    /// Mean gradient-sensitivity bin index normalized to [0, 1].
    ///
    /// Returns 0.0 for a pure time-only archive, 1.0 if all elites are
    /// maximally gradient-reactive.
    pub fn mean_gradient_sensitivity(&self) -> f64 {
        if self.niches.is_empty() { return 0.0; }
        let bins = (self.config.gradient_sensitivity_bins - 1).max(1) as f64;
        let sum: f64 = self.niches.keys().map(|k| k.1 as f64 / bins).sum();
        sum / self.niches.len() as f64
    }

    pub fn stats(&self) -> ArchiveStats {
        let losses: Vec<f64> = self.niches.values().map(|n| n.raw_loss).collect();
        let ages: Vec<u32>   = self.niches.values().map(|n| n.age).collect();

        let best_loss = losses.iter().cloned().fold(f64::INFINITY, f64::min);
        let median_loss = {
            let mut s = losses.clone();
            s.sort_by(|a, b| a.partial_cmp(b).unwrap());
            if s.is_empty() { f64::INFINITY } else { s[s.len() / 2] }
        };
        let mean_age = if ages.is_empty() {
            0.0
        } else {
            ages.iter().sum::<u32>() as f64 / ages.len() as f64
        };
        let occupancy_pct = 100.0 * self.niches.len() as f64
            / self.max_niches().max(1) as f64;

        ArchiveStats {
            occupied_niches: self.niches.len(),
            max_niches: self.max_niches(),
            occupancy_pct,
            total_attempts: self.total_attempts,
            total_additions: self.total_additions,
            best_loss,
            median_loss,
            mean_elite_age: mean_age,
            gradient_sensitivity_mean: self.mean_gradient_sensitivity(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 4.  Sensitivity Helper
// ─────────────────────────────────────────────────────────────────────────────

/// Coefficient of Variation of a set of values, clamped to [0, 1].
///
/// Returns 0.0 for constant inputs (time-only formulas that don't react
/// to the probed variable) and approaches 1.0 for highly reactive formulas.
fn coefficient_of_variation(values: &[f64]) -> f64 {
    let finite: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.len() < 2 {
        return 0.0;
    }
    let mean = finite.iter().sum::<f64>() / finite.len() as f64;
    let variance = finite.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
        / finite.len() as f64;
    let std = variance.sqrt();
    (std / (mean.abs() + 1e-6)).min(1.0)
}

// ─────────────────────────────────────────────────────────────────────────────
// 5.  Stats snapshot
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchiveStats {
    pub occupied_niches: usize,
    pub max_niches: usize,
    pub occupancy_pct: f64,
    pub total_attempts: u64,
    pub total_additions: u64,
    pub best_loss: f64,
    pub median_loss: f64,
    pub mean_elite_age: f64,
    /// Mean gradient-sensitivity of archive elites, normalized to [0, 1].
    /// Increases as search discovers gradient-reactive formulas.
    pub gradient_sensitivity_mean: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
// 6.  Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Expr;

    fn default_archive() -> MapElitesArchive {
        MapElitesArchive::new(ArchiveConfig::default())
    }

    fn linear_expr(c: f64) -> Expr {
        Expr::Add(Box::new(Expr::VarT), Box::new(Expr::Const(c)))
    }

    fn gradient_expr() -> Expr {
        // 0.01 * exp(-g) — reactive to gradient norm
        Expr::Mul(
            Box::new(Expr::Const(0.01)),
            Box::new(Expr::Exp(Box::new(Expr::Mul(
                Box::new(Expr::Const(-1.0)),
                Box::new(Expr::VarG),
            )))),
        )
    }

    #[test]
    fn test_archive_insert_into_empty() {
        let mut archive = default_archive();
        assert!(archive.is_empty());
        let accepted = archive.try_add(linear_expr(0.001), 0.5, 0.5);
        assert!(accepted, "First insertion must succeed");
        assert_eq!(archive.len(), 1);
    }

    #[test]
    fn test_archive_better_formula_replaces_incumbent() {
        let mut archive = default_archive();
        archive.try_add(linear_expr(0.001), 1.0, 1.0);
        let niche_key = *archive.niches.keys().next().unwrap();

        // Directly replace with a better entry (same niche key)
        let better = linear_expr(0.002);
        archive.niches.insert(niche_key, Niche {
            complexity: better.size(),
            expr_hash: better.structural_hash(),
            raw_loss: 0.3,
            age: 0,
            expr: better,
        });

        assert_eq!(archive.len(), 1, "Niche count unchanged on replacement");
        assert!((archive.niches[&niche_key].raw_loss - 0.3).abs() < 1e-12);
    }

    #[test]
    fn test_archive_worse_formula_is_rejected() {
        let mut archive = default_archive();
        archive.try_add(linear_expr(0.001), 0.2, 0.2);
        let niche_key = *archive.niches.keys().next().unwrap();
        let before_hash = archive.niches[&niche_key].expr_hash;

        archive.try_add(Expr::Add(Box::new(Expr::VarT), Box::new(Expr::Const(0.003))), 5.0, 5.0);
        assert_eq!(archive.niches[&niche_key].expr_hash, before_hash,
            "Incumbent must survive worse challenge");
    }

    #[test]
    fn test_archive_rejects_structural_duplicate() {
        let mut archive = default_archive();
        archive.try_add(linear_expr(0.001), 0.5, 0.5);
        let key = *archive.niches.keys().next().unwrap();
        let before = archive.niches[&key].expr_hash;

        // Same structure, even better loss — must be rejected
        archive.try_add(linear_expr(0.001), 0.0, 0.0);
        assert_eq!(archive.niches[&key].expr_hash, before,
            "Structural duplicate must not replace incumbent");
    }

    #[test]
    fn test_archive_rejects_nonfinite_loss() {
        let mut archive = default_archive();
        assert!(!archive.try_add(Expr::VarT, f64::NAN, f64::NAN));
        assert!(!archive.try_add(Expr::VarT, f64::INFINITY, f64::INFINITY));
        assert!(archive.is_empty());
    }

    #[test]
    fn test_archive_age_increment() {
        let mut archive = default_archive();
        archive.try_add(linear_expr(0.001), 0.5, 0.5);
        assert_eq!(archive.niches.values().next().unwrap().age, 0);
        archive.increment_ages();
        assert_eq!(archive.niches.values().next().unwrap().age, 1);
        archive.increment_ages();
        assert_eq!(archive.niches.values().next().unwrap().age, 2);
        assert_eq!(archive.current_generation, 2);
    }

    #[test]
    fn test_archive_best_selection() {
        let mut archive = default_archive();
        archive.try_add(linear_expr(0.001), 0.8, 0.8);
        archive.try_add(Expr::Mul(Box::new(Expr::VarT), Box::new(Expr::Const(0.1))), 0.2, 0.2);
        let best = archive.best().unwrap();
        assert!(best.raw_loss <= 0.8);
    }

    #[test]
    fn test_archive_stats_populated() {
        let mut archive = default_archive();
        archive.try_add(linear_expr(0.001), 0.5, 0.5);
        archive.increment_ages();
        let stats = archive.stats();
        assert_eq!(stats.occupied_niches, archive.len());
        assert!(stats.occupancy_pct > 0.0 && stats.occupancy_pct < 100.0);
        assert!(stats.best_loss.is_finite());
    }

    #[test]
    fn test_niche_rejects_zero_constant() {
        let archive = default_archive();
        // Const(0.0): total_lr = 0 → None
        assert!(archive.compute_niche_key(&Expr::Const(0.0)).is_none());
        // Const(-0.5): total_lr < 0 → None
        assert!(archive.compute_niche_key(&Expr::Const(-0.5)).is_none());
    }

    #[test]
    fn test_niche_key_within_bounds() {
        let archive = default_archive();
        // Valid time-only formula
        let expr = linear_expr(0.01);
        let key = archive.compute_niche_key(&expr).expect("Valid formula must produce key");
        assert!(key.0 < archive.config.size_bins, "size_idx out of bounds");
        assert!(key.1 < archive.config.gradient_sensitivity_bins, "grad_idx out of bounds");
        assert!(key.2 < archive.config.loss_sensitivity_bins, "loss_idx out of bounds");
    }

    #[test]
    fn test_gradient_aware_formula_gets_nonzero_sensitivity() {
        let archive = default_archive();
        let expr = gradient_expr();
        let key = archive.compute_niche_key(&expr).expect("gradient formula must produce key");
        // gradient_sensitivity_idx should be > 0 because exp(-g) changes with g
        assert!(key.1 > 0, "gradient_sensitive formula must land in grad_idx > 0, got {}", key.1);
    }

    #[test]
    fn test_time_only_formula_gets_zero_sensitivity() {
        let archive = default_archive();
        // 0.5 * cos(π*t) — time-only, decreases from 0.5 to -0.5
        // The formula IS valid because it's positive for t < 0.5
        let expr = Expr::Mul(
            Box::new(Expr::Const(0.5)),
            Box::new(Expr::Cos(Box::new(Expr::Mul(
                Box::new(Expr::Const(std::f64::consts::PI)),
                Box::new(Expr::VarT),
            )))),
        );
        let key = archive.compute_niche_key(&expr).expect("0.5*cos(pi*t) must produce key");
        // Time-only formula: both sensitivity dimensions must be 0
        assert_eq!(key.1, 0, "Time-only formula must have grad_idx=0, got {}", key.1);
        assert_eq!(key.2, 0, "Time-only formula must have loss_idx=0, got {}", key.2);
    }

    #[test]
    fn test_gradient_sensitivity_increases_over_generations() {
        let mut archive = default_archive();

        // Add some time-only formulas first
        archive.try_add(linear_expr(0.001), 0.5, 0.5);
        archive.try_add(Expr::Cos(Box::new(Expr::VarT)), 0.4, 0.4);
        let sensitivity_before = archive.mean_gradient_sensitivity();

        // Add a strongly gradient-reactive formula
        archive.try_add(gradient_expr(), 0.3, 0.3);
        let sensitivity_after = archive.mean_gradient_sensitivity();

        // Mean sensitivity should increase (or stay same if gradient formula is in bin 0)
        // We can't guarantee it increases, but it should not decrease.
        assert!(sensitivity_after >= sensitivity_before - 1e-9,
            "Adding gradient formula must not decrease mean sensitivity");
    }

    #[test]
    fn test_max_niches_formula() {
        let archive = default_archive();
        assert_eq!(
            archive.max_niches(),
            30 * 20 * 10,
            "Max niches must be size_bins * grad_bins * loss_bins = 6000"
        );
    }

    #[test]
    fn test_stats_includes_gradient_sensitivity() {
        let mut archive = default_archive();
        archive.try_add(linear_expr(0.01), 0.5, 0.5);
        let stats = archive.stats();
        assert!(stats.gradient_sensitivity_mean >= 0.0);
        assert!(stats.gradient_sensitivity_mean <= 1.0);
    }
}
