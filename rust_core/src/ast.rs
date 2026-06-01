//! ast.rs — Native Rust Abstract Syntax Tree for SymboLR.
//!
//! ## Phase 2 Extension: Multi-Variable Formulas
//!
//! The AST now supports three input variables:
//!   • `VarT`  — normalized training time t ∈ [0, 1]
//!   • `VarG`  — log-normalized gradient norm, bounded ≈ [-2, 2]
//!   • `VarDL` — tanh-normalized loss slope, bounded [-1, 1]
//!
//! Backward compatibility is maintained via `eval_schedule_time_only()` which
//! evaluates with g=0.0, dl=0.0, keeping the `SyntheticEvaluator` path unchanged.
//!
//! Design invariants:
//!   • MAX_NODES  = 15  — hard cap enforced by `try_enforce_cap`
//!   • MAX_DEPTH  = 7   — matches Python GP max_tree_depth_limit
//!   • All arithmetic is protected (div-by-zero, NaN, overflow) inline.

use std::fmt;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use serde::{Serialize, Deserialize};

// ─────────────────────────────────────────────────────────────────────────────
// 1.  Hard Structural Caps
// ─────────────────────────────────────────────────────────────────────────────

pub const MAX_NODES: usize = 15;
pub const MAX_DEPTH: usize = 7;

// ─────────────────────────────────────────────────────────────────────────────
// 2.  Expr Enum — the core AST node
// ─────────────────────────────────────────────────────────────────────────────

/// Every node in a symbolic formula tree.
///
/// Leaf variants own no heap allocation.
/// Binary variants box both children; unary variants box one child.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub enum Expr {
    // ── Terminals ────────────────────────────────────────────────────────────
    /// Normalized training time t ∈ [0, 1].
    VarT,
    /// Log-normalized gradient norm (log(||∇||)), typically ≈ [-2, 2].
    VarG,
    /// Tanh-normalized loss slope Δl = tanh((loss_t - loss_{t-k}) / σ).
    VarDL,
    /// A float literal (e.g. `0.5`, `-1.0`).
    Const(f64),

    // ── Binary Operators ─────────────────────────────────────────────────────
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    /// Protected division: returns `a * sign(b)` when `|b| < 1e-6`.
    Div(Box<Expr>, Box<Expr>),

    // ── Unary Functions ──────────────────────────────────────────────────────
    Sin(Box<Expr>),
    Cos(Box<Expr>),
    /// Protected exp: clamps exponent to `[-10, 5]`.
    Exp(Box<Expr>),
    /// Protected log: operates on `|a|`, floored at `1e-6`.
    Log(Box<Expr>),
    /// Protected sqrt: operates on `|a|`.
    Sqrt(Box<Expr>),
    Abs(Box<Expr>),
}

// ─────────────────────────────────────────────────────────────────────────────
// 3.  Structural Metrics
// ─────────────────────────────────────────────────────────────────────────────

impl Expr {
    /// Total node count (bloat proxy).
    pub fn size(&self) -> usize {
        match self {
            Expr::VarT | Expr::VarG | Expr::VarDL | Expr::Const(_) => 1,
            Expr::Sin(a) | Expr::Cos(a) | Expr::Exp(a)
            | Expr::Log(a) | Expr::Sqrt(a) | Expr::Abs(a) => 1 + a.size(),
            Expr::Add(a, b) | Expr::Sub(a, b)
            | Expr::Mul(a, b) | Expr::Div(a, b) => 1 + a.size() + b.size(),
        }
    }

    /// Maximum depth from this node (1-indexed leaf = 1).
    pub fn depth(&self) -> usize {
        match self {
            Expr::VarT | Expr::VarG | Expr::VarDL | Expr::Const(_) => 1,
            Expr::Sin(a) | Expr::Cos(a) | Expr::Exp(a)
            | Expr::Log(a) | Expr::Sqrt(a) | Expr::Abs(a) => 1 + a.depth(),
            Expr::Add(a, b) | Expr::Sub(a, b)
            | Expr::Mul(a, b) | Expr::Div(a, b) => 1 + a.depth().max(b.depth()),
        }
    }

    #[inline]
    pub fn is_within_cap(&self) -> bool {
        self.size() <= MAX_NODES && self.depth() <= MAX_DEPTH
    }

    /// True if this expression uses VarG or VarDL — i.e., is gradient-aware.
    pub fn is_gradient_aware(&self) -> bool {
        match self {
            Expr::VarG | Expr::VarDL => true,
            Expr::VarT | Expr::Const(_) => false,
            Expr::Sin(a) | Expr::Cos(a) | Expr::Exp(a)
            | Expr::Log(a) | Expr::Sqrt(a) | Expr::Abs(a) => a.is_gradient_aware(),
            Expr::Add(a, b) | Expr::Sub(a, b)
            | Expr::Mul(a, b) | Expr::Div(a, b) => a.is_gradient_aware() || b.is_gradient_aware(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 4.  Protected Scalar Evaluation
// ─────────────────────────────────────────────────────────────────────────────

impl Expr {
    /// Evaluate the formula at a single (t, g, dl) point.
    ///
    /// All operations are numerically protected; the result is always finite.
    #[inline]
    pub fn eval(&self, t: f64, g: f64, dl: f64) -> f64 {
        let raw = self.eval_raw(t, g, dl);
        if raw.is_finite() { raw.clamp(-1e6, 1e6) } else { 1.0 }
    }

    fn eval_raw(&self, t: f64, g: f64, dl: f64) -> f64 {
        match self {
            Expr::VarT       => t,
            Expr::VarG       => g,
            Expr::VarDL      => dl,
            Expr::Const(c)   => *c,
            Expr::Add(a, b)  => a.eval_raw(t, g, dl) + b.eval_raw(t, g, dl),
            Expr::Sub(a, b)  => a.eval_raw(t, g, dl) - b.eval_raw(t, g, dl),
            Expr::Mul(a, b)  => {
                (a.eval_raw(t, g, dl) * b.eval_raw(t, g, dl)).clamp(-100.0, 100.0)
            }
            Expr::Div(a, b)  => {
                let den = b.eval_raw(t, g, dl);
                let num = a.eval_raw(t, g, dl);
                if den.abs() < 1e-6 {
                    num * (den + 1e-30_f64).signum()
                } else {
                    num / den
                }
            }
            Expr::Sin(a)  => a.eval_raw(t, g, dl).sin(),
            Expr::Cos(a)  => a.eval_raw(t, g, dl).cos(),
            Expr::Exp(a)  => a.eval_raw(t, g, dl).clamp(-10.0, 5.0).exp(),
            Expr::Log(a)  => {
                let v = a.eval_raw(t, g, dl).abs().max(1e-6);
                v.ln()
            }
            Expr::Sqrt(a) => a.eval_raw(t, g, dl).abs().sqrt(),
            Expr::Abs(a)  => a.eval_raw(t, g, dl).abs(),
        }
    }

    /// Evaluate over a multi-variable input slice `(t, g, dl)`.
    ///
    /// All values are clamped to `[1e-7, 10.0]` — the valid LR range.
    pub fn eval_schedule(&self, inputs: &[(f64, f64, f64)]) -> Vec<f64> {
        inputs.iter().map(|&(t, g, dl)| self.eval(t, g, dl).clamp(1e-7, 10.0)).collect()
    }

    /// Backward-compatible time-only evaluation (g=0.0, dl=0.0).
    ///
    /// Used by `SyntheticEvaluator` and `evaluate_batch` legacy path.
    pub fn eval_schedule_time_only(&self, t_array: &[f64]) -> Vec<f64> {
        t_array.iter().map(|&t| self.eval(t, 0.0, 0.0).clamp(1e-7, 10.0)).collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 5.  Prefix Notation (FFI contract & Rust ↔ Python parity)
// ─────────────────────────────────────────────────────────────────────────────

impl Expr {
    /// Serializes the tree into space-separated prefix (Polish) notation.
    ///
    /// Canonical interchange format with Python:
    /// `(t + g) * exp(-dl)` → `"* + t g exp * -1 dl"`
    pub fn to_prefix(&self) -> String {
        match self {
            Expr::VarT       => "t".to_owned(),
            Expr::VarG       => "g".to_owned(),
            Expr::VarDL      => "dl".to_owned(),
            Expr::Const(c)   => format_const(*c),
            Expr::Add(a, b)  => format!("+ {} {}", a.to_prefix(), b.to_prefix()),
            Expr::Sub(a, b)  => format!("- {} {}", a.to_prefix(), b.to_prefix()),
            Expr::Mul(a, b)  => format!("* {} {}", a.to_prefix(), b.to_prefix()),
            Expr::Div(a, b)  => format!("/ {} {}", a.to_prefix(), b.to_prefix()),
            Expr::Sin(a)     => format!("sin {}", a.to_prefix()),
            Expr::Cos(a)     => format!("cos {}", a.to_prefix()),
            Expr::Exp(a)     => format!("exp {}", a.to_prefix()),
            Expr::Log(a)     => format!("log {}", a.to_prefix()),
            Expr::Sqrt(a)    => format!("sqrt {}", a.to_prefix()),
            Expr::Abs(a)     => format!("abs {}", a.to_prefix()),
        }
    }
}

fn format_const(v: f64) -> String {
    if v.fract() == 0.0 && v.abs() < 1e15 {
        format!("{:.0}", v)
    } else {
        format!("{}", v)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 6.  LaTeX Display
// ─────────────────────────────────────────────────────────────────────────────

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::VarT       => write!(f, "t"),
            Expr::VarG       => write!(f, r"\|g\|"),
            Expr::VarDL      => write!(f, r"\Delta\ell"),
            Expr::Const(c)   => write!(f, "{}", format_const(*c)),
            Expr::Add(a, b)  => write!(f, "({} + {})", a, b),
            Expr::Sub(a, b)  => write!(f, "({} - {})", a, b),
            Expr::Mul(a, b)  => write!(f, "({} \\cdot {})", a, b),
            Expr::Div(a, b)  => write!(f, "\\frac{{{}}}{{{}}}", a, b),
            Expr::Sin(a)     => write!(f, "\\sin({})", a),
            Expr::Cos(a)     => write!(f, "\\cos({})", a),
            Expr::Exp(a)     => write!(f, "e^{{{}}}", a),
            Expr::Log(a)     => write!(f, "\\ln({})", a),
            Expr::Sqrt(a)    => write!(f, "\\sqrt{{{}}}", a),
            Expr::Abs(a)     => write!(f, "|{}|", a),
        }
    }
}

impl fmt::Debug for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Expr({})", self)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 7.  Structural Hashing
// ─────────────────────────────────────────────────────────────────────────────

impl Expr {
    pub fn structural_hash(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.hash_into(&mut hasher);
        hasher.finish()
    }

    fn hash_into(&self, h: &mut DefaultHasher) {
        // Tags 0-11 match original; 13 and 14 are new (skip 12 to avoid future clash).
        let tag: u8 = match self {
            Expr::VarT       => 0,
            Expr::Const(_)   => 1,
            Expr::Add(_, _)  => 2,
            Expr::Sub(_, _)  => 3,
            Expr::Mul(_, _)  => 4,
            Expr::Div(_, _)  => 5,
            Expr::Sin(_)     => 6,
            Expr::Cos(_)     => 7,
            Expr::Exp(_)     => 8,
            Expr::Log(_)     => 9,
            Expr::Sqrt(_)    => 10,
            Expr::Abs(_)     => 11,
            Expr::VarG       => 13,
            Expr::VarDL      => 14,
        };
        tag.hash(h);

        match self {
            Expr::VarT | Expr::VarG | Expr::VarDL => {}
            Expr::Const(c) => { c.to_bits().hash(h); }
            Expr::Sin(a) | Expr::Cos(a) | Expr::Exp(a)
            | Expr::Log(a) | Expr::Sqrt(a) | Expr::Abs(a) => {
                a.hash_into(h);
            }
            Expr::Add(a, b) | Expr::Sub(a, b)
            | Expr::Mul(a, b) | Expr::Div(a, b) => {
                a.hash_into(h);
                b.hash_into(h);
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 8.  Prefix Parser
// ─────────────────────────────────────────────────────────────────────────────

pub fn parse_prefix(src: &str) -> Result<Expr, String> {
    let tokens: Vec<&str> = src.split_whitespace().collect();
    let mut idx = 0usize;
    let expr = parse_tokens(&tokens, &mut idx)?;
    Ok(expr)
}

fn parse_tokens(tokens: &[&str], idx: &mut usize) -> Result<Expr, String> {
    if *idx >= tokens.len() {
        return Err("Unexpected end of token stream".to_owned());
    }
    let tok = tokens[*idx];
    *idx += 1;

    match tok {
        // Variables — all three forms
        "t"    => Ok(Expr::VarT),
        "g"    => Ok(Expr::VarG),
        "dl"   => Ok(Expr::VarDL),
        // Binary operators
        "+"    => Ok(Expr::Add(  Box::new(parse_tokens(tokens, idx)?), Box::new(parse_tokens(tokens, idx)?))),
        "-"    => Ok(Expr::Sub(  Box::new(parse_tokens(tokens, idx)?), Box::new(parse_tokens(tokens, idx)?))),
        "*"    => Ok(Expr::Mul(  Box::new(parse_tokens(tokens, idx)?), Box::new(parse_tokens(tokens, idx)?))),
        "/"    => Ok(Expr::Div(  Box::new(parse_tokens(tokens, idx)?), Box::new(parse_tokens(tokens, idx)?))),
        // Unary functions
        "sin"  => Ok(Expr::Sin(  Box::new(parse_tokens(tokens, idx)?))),
        "cos"  => Ok(Expr::Cos(  Box::new(parse_tokens(tokens, idx)?))),
        "exp"  => Ok(Expr::Exp(  Box::new(parse_tokens(tokens, idx)?))),
        "log"  => Ok(Expr::Log(  Box::new(parse_tokens(tokens, idx)?))),
        "sqrt" => Ok(Expr::Sqrt( Box::new(parse_tokens(tokens, idx)?))),
        "abs"  => Ok(Expr::Abs(  Box::new(parse_tokens(tokens, idx)?))),
        // Numeric literal
        _      => tok.parse::<f64>()
                      .map(Expr::Const)
                      .map_err(|_| format!("Unknown token: `{}`", tok)),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 9.  Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── 9.1  Construction & Structural Metrics ────────────────────────────────

    #[test]
    fn test_ast_construction_and_metrics() {
        // (t + 0.5) * exp(t)   →  6 nodes, depth 3
        let tree = Expr::Mul(
            Box::new(Expr::Add(Box::new(Expr::VarT), Box::new(Expr::Const(0.5)))),
            Box::new(Expr::Exp(Box::new(Expr::VarT))),
        );
        assert_eq!(tree.size(), 6);
        assert_eq!(tree.depth(), 3);
        assert!(tree.is_within_cap());
    }

    #[test]
    fn test_leaf_metrics() {
        assert_eq!(Expr::VarT.size(), 1);
        assert_eq!(Expr::VarT.depth(), 1);
        assert_eq!(Expr::VarG.size(), 1);
        assert_eq!(Expr::VarDL.size(), 1);
        assert_eq!(Expr::Const(3.14).size(), 1);
        assert_eq!(Expr::Const(3.14).depth(), 1);
    }

    #[test]
    fn test_cap_enforcement() {
        let mut tree = Expr::VarT;
        for _ in 0..14 {
            tree = Expr::Abs(Box::new(tree));
        }
        assert_eq!(tree.size(), 15);
        assert!(tree.size() <= MAX_NODES);
        assert!(!tree.is_within_cap(), "Linear 15-deep chain violates MAX_DEPTH");

        let over = Expr::Abs(Box::new(tree));
        assert!(over.size() > MAX_NODES);
        assert!(!over.is_within_cap());

        let balanced = Expr::Mul(
            Box::new(Expr::Add(Box::new(Expr::VarT), Box::new(Expr::Const(0.5)))),
            Box::new(Expr::Exp(Box::new(Expr::VarT))),
        );
        assert_eq!(balanced.size(), 6);
        assert!(balanced.is_within_cap());
    }

    // ── 9.2  Multi-Variable Evaluation ───────────────────────────────────────

    #[test]
    fn test_var_t_evaluation() {
        for &t in &[0.0, 0.25, 0.5, 0.75, 1.0] {
            assert_eq!(Expr::VarT.eval(t, 0.5, -0.3), t);
        }
    }

    #[test]
    fn test_var_g_evaluation() {
        for &g in &[-2.0, 0.0, 1.5] {
            assert_eq!(Expr::VarG.eval(0.5, g, 0.1), g);
        }
    }

    #[test]
    fn test_var_dl_evaluation() {
        for &dl in &[-1.0, 0.0, 0.8] {
            assert_eq!(Expr::VarDL.eval(0.5, 0.0, dl), dl);
        }
    }

    #[test]
    fn test_const_is_independent_of_all_inputs() {
        let c = Expr::Const(0.001);
        assert_eq!(c.eval(0.0, 0.0, 0.0), 0.001);
        assert_eq!(c.eval(1.0, 2.0, -1.0), 0.001);
    }

    #[test]
    fn test_gradient_aware_flag() {
        assert!(!Expr::VarT.is_gradient_aware());
        assert!(Expr::VarG.is_gradient_aware());
        assert!(Expr::VarDL.is_gradient_aware());
        let formula = Expr::Mul(Box::new(Expr::VarT), Box::new(Expr::VarG));
        assert!(formula.is_gradient_aware());
        let time_only = Expr::Cos(Box::new(Expr::VarT));
        assert!(!time_only.is_gradient_aware());
    }

    #[test]
    fn test_protected_division() {
        let expr = Expr::Div(Box::new(Expr::Const(1.0)), Box::new(Expr::Const(0.0)));
        assert!(expr.eval(0.5, 0.0, 0.0).is_finite());
    }

    #[test]
    fn test_protected_exp() {
        let expr = Expr::Exp(Box::new(Expr::Mul(
            Box::new(Expr::Const(1000.0)),
            Box::new(Expr::VarT),
        )));
        let result = expr.eval(1.0, 0.0, 0.0);
        assert!(result.is_finite());
        assert!(result <= 1e6);
    }

    #[test]
    fn test_protected_log() {
        let expr = Expr::Log(Box::new(Expr::Const(0.0)));
        assert!(expr.eval(0.0, 0.0, 0.0).is_finite());
    }

    #[test]
    fn test_eval_schedule_clamping() {
        let expr = Expr::Exp(Box::new(Expr::Mul(
            Box::new(Expr::Const(50.0)),
            Box::new(Expr::VarT),
        )));
        // Use time-only path for this test
        let t_array: Vec<f64> = (0..=100).map(|i| i as f64 / 100.0).collect();
        let schedule = expr.eval_schedule_time_only(&t_array);
        assert_eq!(schedule.len(), t_array.len());
        for &v in &schedule {
            assert!(v >= 1e-7 && v <= 10.0, "Schedule value {} out of [1e-7, 10.0]", v);
        }
    }

    #[test]
    fn test_eval_schedule_multi_variable() {
        // Formula: t * exp(-g)  — a gradient-aware schedule
        let expr = Expr::Mul(
            Box::new(Expr::VarT),
            Box::new(Expr::Exp(Box::new(Expr::Mul(
                Box::new(Expr::Const(-1.0)),
                Box::new(Expr::VarG),
            )))),
        );
        let inputs: Vec<(f64, f64, f64)> = vec![
            (0.5, 0.0, 0.0),   // mid-time, stable gradient
            (0.5, 2.0, 0.0),   // mid-time, large gradient
            (0.5, -2.0, 0.0),  // mid-time, small gradient
        ];
        let schedule = expr.eval_schedule(&inputs);
        assert_eq!(schedule.len(), 3);
        // At g=2.0 LR should be less than at g=0.0 (exp(-2) < exp(0))
        assert!(schedule[1] < schedule[0], "LR at large gradient should be lower");
        // At g=-2.0 LR should be greater than at g=0.0
        assert!(schedule[2] > schedule[0], "LR at negative log-gradient should be higher");
        for &v in &schedule {
            assert!(v >= 1e-7 && v <= 10.0, "Value {} out of bounds", v);
        }
    }

    // ── 9.3  Prefix Serialization ─────────────────────────────────────────────

    #[test]
    fn test_prefix_roundtrip_time_formula() {
        let original = Expr::Mul(
            Box::new(Expr::Add(Box::new(Expr::VarT), Box::new(Expr::Const(0.5)))),
            Box::new(Expr::Exp(Box::new(Expr::VarT))),
        );
        let prefix = original.to_prefix();
        assert_eq!(prefix, "* + t 0.5 exp t");
        let parsed = parse_prefix(&prefix).expect("prefix should parse");
        assert_eq!(original.structural_hash(), parsed.structural_hash());
    }

    #[test]
    fn test_prefix_roundtrip_gradient_formula() {
        // t * exp(-g)  →  "* t exp * -1 g"
        let original = Expr::Mul(
            Box::new(Expr::VarT),
            Box::new(Expr::Exp(Box::new(Expr::Mul(
                Box::new(Expr::Const(-1.0)),
                Box::new(Expr::VarG),
            )))),
        );
        let prefix = original.to_prefix();
        let parsed = parse_prefix(&prefix).expect("gradient formula should parse");
        assert_eq!(original.structural_hash(), parsed.structural_hash());
    }

    #[test]
    fn test_prefix_roundtrip_all_variables() {
        // t + g + dl
        let original = Expr::Add(
            Box::new(Expr::Add(Box::new(Expr::VarT), Box::new(Expr::VarG))),
            Box::new(Expr::VarDL),
        );
        let prefix = original.to_prefix();
        assert_eq!(prefix, "+ + t g dl");
        let parsed = parse_prefix(&prefix).unwrap();
        assert_eq!(original.structural_hash(), parsed.structural_hash());
    }

    #[test]
    fn test_prefix_parse_unknown_token() {
        assert!(parse_prefix("tanh t").is_err());
    }

    // ── 9.4  LaTeX Display ────────────────────────────────────────────────────

    #[test]
    fn test_latex_var_t() {
        assert_eq!(format!("{}", Expr::VarT), "t");
    }

    #[test]
    fn test_latex_var_g() {
        assert_eq!(format!("{}", Expr::VarG), r"\|g\|");
    }

    #[test]
    fn test_latex_var_dl() {
        assert_eq!(format!("{}", Expr::VarDL), r"\Delta\ell");
    }

    #[test]
    fn test_latex_display_add() {
        let e = Expr::Add(Box::new(Expr::VarT), Box::new(Expr::Const(1.0)));
        assert_eq!(format!("{}", e), "(t + 1)");
    }

    #[test]
    fn test_latex_display_div() {
        let e = Expr::Div(Box::new(Expr::VarT), Box::new(Expr::Const(2.0)));
        assert_eq!(format!("{}", e), r"\frac{t}{2}");
    }

    // ── 9.5  Structural Hashing ───────────────────────────────────────────────

    #[test]
    fn test_hash_determinism() {
        let a = Expr::Add(Box::new(Expr::VarT), Box::new(Expr::Const(0.5)));
        let b = Expr::Add(Box::new(Expr::VarT), Box::new(Expr::Const(0.5)));
        assert_eq!(a.structural_hash(), b.structural_hash());
    }

    #[test]
    fn test_new_variables_hash_distinctly() {
        let ht = Expr::VarT.structural_hash();
        let hg = Expr::VarG.structural_hash();
        let hdl = Expr::VarDL.structural_hash();
        assert_ne!(ht, hg, "VarT and VarG must hash differently");
        assert_ne!(ht, hdl, "VarT and VarDL must hash differently");
        assert_ne!(hg, hdl, "VarG and VarDL must hash differently");
    }

    #[test]
    fn test_hash_distinguishes_distinct_trees() {
        let a = Expr::Add(Box::new(Expr::VarT), Box::new(Expr::Const(0.5)));
        let b = Expr::Add(Box::new(Expr::VarT), Box::new(Expr::Const(0.6)));
        assert_ne!(a.structural_hash(), b.structural_hash());
    }
}
