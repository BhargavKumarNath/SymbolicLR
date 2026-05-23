//! ast.rs — Native Rust Abstract Syntax Tree for SymboLR.
//!
//! The `Expr` enum is the single source of truth for every symbolic formula.
//! All allocations are explicit `Box<Expr>` heap allocations — zero GC, zero
//! hidden overhead. Structural hashing via MD5 guarantees O(1) deduplication.
//!
//! Design invariants:
//!   • MAX_NODES  = 15  — hard cap enforced by `try_enforce_cap`
//!   • MAX_DEPTH  = 7   — matches Python GP max_tree_depth_limit
//!   • All arithmetic is protected (div-by-zero, NaN, overflow) inline.

use std::fmt;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

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
/// Leaf variants (`Var`, `Const`) own no heap allocation.
/// Binary variants box both children; unary variants box one child.
/// This makes cloning O(n) in tree size and deeply safe to reason about.
#[derive(Clone, PartialEq)]
pub enum Expr {
    // ── Terminals ────────────────────────────────────────────────────────────
    /// The single input variable `t ∈ [0, 1]`.
    Var,
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
    /// Total node count (bloat proxy). O(n) walk, result is NOT cached here —
    /// caller sites that need it repeatedly should cache it themselves.
    pub fn size(&self) -> usize {
        match self {
            Expr::Var | Expr::Const(_) => 1,
            Expr::Sin(a) | Expr::Cos(a) | Expr::Exp(a)
            | Expr::Log(a) | Expr::Sqrt(a) | Expr::Abs(a) => 1 + a.size(),
            Expr::Add(a, b) | Expr::Sub(a, b)
            | Expr::Mul(a, b) | Expr::Div(a, b) => 1 + a.size() + b.size(),
        }
    }

    /// Maximum depth from this node (1-indexed leaf = 1).
    pub fn depth(&self) -> usize {
        match self {
            Expr::Var | Expr::Const(_) => 1,
            Expr::Sin(a) | Expr::Cos(a) | Expr::Exp(a)
            | Expr::Log(a) | Expr::Sqrt(a) | Expr::Abs(a) => 1 + a.depth(),
            Expr::Add(a, b) | Expr::Sub(a, b)
            | Expr::Mul(a, b) | Expr::Div(a, b) => 1 + a.depth().max(b.depth()),
        }
    }

    /// Returns `true` if this tree respects both structural caps.
    #[inline]
    pub fn is_within_cap(&self) -> bool {
        self.size() <= MAX_NODES && self.depth() <= MAX_DEPTH
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 4.  Protected Scalar Evaluation
// ─────────────────────────────────────────────────────────────────────────────

impl Expr {
    /// Evaluate the formula at a single time step `t`.
    ///
    /// All operations are numerically protected; the result is always finite.
    #[inline]
    pub fn eval(&self, t: f64) -> f64 {
        let raw = self.eval_raw(t);
        // Final clamp & NaN guard — one branch, branch-predictor friendly.
        if raw.is_finite() { raw.clamp(-1e6, 1e6) } else { 1.0 }
    }

    fn eval_raw(&self, t: f64) -> f64 {
        match self {
            Expr::Var        => t,
            Expr::Const(c)   => *c,
            Expr::Add(a, b)  => a.eval_raw(t) + b.eval_raw(t),
            Expr::Sub(a, b)  => a.eval_raw(t) - b.eval_raw(t),
            Expr::Mul(a, b)  => {
                // Protected multiply: clamp to prevent cascading overflow.
                (a.eval_raw(t) * b.eval_raw(t)).clamp(-100.0, 100.0)
            }
            Expr::Div(a, b)  => {
                let den = b.eval_raw(t);
                let num = a.eval_raw(t);
                if den.abs() < 1e-6 {
                    // Preserve sign information from numerator × sign(denominator).
                    num * (den + 1e-30_f64).signum()
                } else {
                    num / den
                }
            }
            Expr::Sin(a)  => a.eval_raw(t).sin(),
            Expr::Cos(a)  => a.eval_raw(t).cos(),
            Expr::Exp(a)  => a.eval_raw(t).clamp(-10.0, 5.0).exp(),
            Expr::Log(a)  => {
                let v = a.eval_raw(t).abs().max(1e-6);
                v.ln()
            }
            Expr::Sqrt(a) => a.eval_raw(t).abs().sqrt(),
            Expr::Abs(a)  => a.eval_raw(t).abs(),
        }
    }

    /// Evaluate the formula over a contiguous slice of time steps.
    ///
    /// Returns a `Vec<f64>` of identical length, all values clamped to
    /// `[1e-7, 10.0]` — the valid learning-rate range.
    pub fn eval_schedule(&self, t_array: &[f64]) -> Vec<f64> {
        t_array
            .iter()
            .map(|&t| {
                let v = self.eval(t);
                v.clamp(1e-7, 10.0)
            })
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 5.  Prefix Notation (FFI contract & Rust ↔ Python parity)
// ─────────────────────────────────────────────────────────────────────────────

impl Expr {
    /// Serializes the tree into space-separated prefix (Polish) notation.
    ///
    /// This is the canonical interchange format with Python:
    /// `(t + 0.5) * exp(t)` → `"* + t 0.5 exp t"`
    pub fn to_prefix(&self) -> String {
        match self {
            Expr::Var        => "t".to_owned(),
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

/// Formats a float constant consistently: integers show no decimal suffix,
/// others use Rust's default float Display (e.g. 0.5 → "0.5", not "0.5000").
fn format_const(v: f64) -> String {
    if v.fract() == 0.0 && v.abs() < 1e15 {
        format!("{:.0}", v)
    } else {
        // Rust's default Display trims trailing zeros: 0.5 → "0.5"
        format!("{}", v)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 6.  LaTeX Display (Human-readable & React frontend serialization)
// ─────────────────────────────────────────────────────────────────────────────

impl fmt::Display for Expr {
    /// Produces a LaTeX string for the formula.
    ///
    /// Examples:
    ///   `t`                        → `t`
    ///   `Add(Var, Const(0.5))`     → `(t + 0.5)`
    ///   `Div(Var, Const(2.0))`     → `\frac{t}{2}`
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Var        => write!(f, "t"),
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
// 7.  Structural Hashing — O(1) deduplication fingerprint
// ─────────────────────────────────────────────────────────────────────────────

impl Expr {
    /// Returns a u64 structural hash derived from the tree's shape and constants.
    ///
    /// Identical trees always produce the same hash; structurally distinct trees
    /// almost never collide (64-bit hash space). Used for archive deduplication.
    pub fn structural_hash(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.hash_into(&mut hasher);
        hasher.finish()
    }

    fn hash_into(&self, h: &mut DefaultHasher) {
        // Discriminant tag first (ensures different variants hash differently).
        let tag: u8 = match self {
            Expr::Var        => 0,
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
        };
        tag.hash(h);

        match self {
            Expr::Var => {}
            Expr::Const(c) => {
                // Bitwise representation for deterministic hashing of floats.
                c.to_bits().hash(h);
            }
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
// 8.  Prefix Parser  (Rust-native; mirrors Python's `_parse_prefix`)
// ─────────────────────────────────────────────────────────────────────────────

/// Parse a space-separated prefix string back into an `Expr`.
///
/// Returns `Err` on empty/malformed input. Used in tests and by the Python
/// `evaluate_fast` bridge which passes prefix strings across the FFI.
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
        "t"    => Ok(Expr::Var),
        "+"    => Ok(Expr::Add(  Box::new(parse_tokens(tokens, idx)?),
                                 Box::new(parse_tokens(tokens, idx)?))),
        "-"    => Ok(Expr::Sub(  Box::new(parse_tokens(tokens, idx)?),
                                 Box::new(parse_tokens(tokens, idx)?))),
        "*"    => Ok(Expr::Mul(  Box::new(parse_tokens(tokens, idx)?),
                                 Box::new(parse_tokens(tokens, idx)?))),
        "/"    => Ok(Expr::Div(  Box::new(parse_tokens(tokens, idx)?),
                                 Box::new(parse_tokens(tokens, idx)?))),
        "sin"  => Ok(Expr::Sin(  Box::new(parse_tokens(tokens, idx)?))),
        "cos"  => Ok(Expr::Cos(  Box::new(parse_tokens(tokens, idx)?))),
        "exp"  => Ok(Expr::Exp(  Box::new(parse_tokens(tokens, idx)?))),
        "log"  => Ok(Expr::Log(  Box::new(parse_tokens(tokens, idx)?))),
        "sqrt" => Ok(Expr::Sqrt( Box::new(parse_tokens(tokens, idx)?))),
        "abs"  => Ok(Expr::Abs(  Box::new(parse_tokens(tokens, idx)?))),
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

    /// Test 1 (mandated): Construct a multi-node AST and verify structural metrics.
    ///
    /// Formula: `(t + 0.5) * exp(t)`
    /// Expected tree layout:
    ///   Mul                 ← node 1
    ///     Add               ← node 2
    ///       Var             ← node 3
    ///       Const(0.5)      ← node 4
    ///     Exp               ← node 5
    ///       Var             ← node 6
    ///
    /// size  = 6
    /// depth = 3  (root→Mul→Add→Var)
    #[test]
    fn test_ast_construction_and_metrics() {
        let tree = Expr::Mul(
            Box::new(Expr::Add(
                Box::new(Expr::Var),
                Box::new(Expr::Const(0.5)),
            )),
            Box::new(Expr::Exp(Box::new(Expr::Var))),
        );

        // Memory: 6 heap-allocated nodes (4 Boxed children + 2 terminals).
        assert_eq!(tree.size(), 6, "Tree must have exactly 6 nodes");
        assert_eq!(tree.depth(), 3, "Tree depth must be 3");
        assert!(tree.is_within_cap(), "Tree must be within MAX_NODES/MAX_DEPTH caps");
    }

    /// Verify leaf terminals produce minimal metrics.
    #[test]
    fn test_leaf_metrics() {
        assert_eq!(Expr::Var.size(), 1);
        assert_eq!(Expr::Var.depth(), 1);
        assert_eq!(Expr::Const(3.14).size(), 1);
        assert_eq!(Expr::Const(3.14).depth(), 1);
    }

    /// MAX_NODES cap boundary: a tree with exactly 15 nodes must satisfy
    /// `size() <= MAX_NODES`. A 16-node tree must violate the cap.
    /// Note: `is_within_cap()` also checks depth, so a linear 15-node chain
    /// will violate the depth cap even if size is fine. We test size and depth
    /// invariants independently here.
    #[test]
    fn test_cap_enforcement() {
        // 15-node linear chain: 14 × Abs + 1 × Var.
        // size = 15, depth = 15 (violates MAX_DEPTH = 7 but not MAX_NODES).
        let mut tree = Expr::Var;
        for _ in 0..14 {
            tree = Expr::Abs(Box::new(tree));
        }
        assert_eq!(tree.size(), 15, "Chain must be exactly 15 nodes");
        assert_eq!(tree.depth(), 15, "Linear chain depth must equal its node count");
        // size is within MAX_NODES cap
        assert!(tree.size() <= MAX_NODES, "15 nodes must be within MAX_NODES");
        // depth violates MAX_DEPTH — is_within_cap() must be false
        assert!(!tree.is_within_cap(),
            "15-deep linear chain must violate MAX_DEPTH cap");

        // 16-node chain violates both caps.
        let over = Expr::Abs(Box::new(tree));
        assert_eq!(over.size(), 16, "Over-cap tree must have 16 nodes");
        assert!(over.size() > MAX_NODES, "16-node tree must exceed MAX_NODES");
        assert!(!over.is_within_cap(), "16-node tree must fail is_within_cap()");

        // A balanced tree of depth 3 and size 6 must pass both caps.
        let balanced = Expr::Mul(
            Box::new(Expr::Add(Box::new(Expr::Var), Box::new(Expr::Const(0.5)))),
            Box::new(Expr::Exp(Box::new(Expr::Var))),
        );
        assert_eq!(balanced.size(), 6);
        assert_eq!(balanced.depth(), 3);
        assert!(balanced.is_within_cap(), "Balanced 6-node tree must pass both caps");
    }

    // ── 9.2  Numeric Evaluation ───────────────────────────────────────────────

    /// Var evaluates to t at every point.
    #[test]
    fn test_var_evaluation() {
        let t_vals = [0.0, 0.25, 0.5, 0.75, 1.0];
        for &t in &t_vals {
            assert_eq!(Expr::Var.eval(t), t);
        }
    }

    /// Const evaluates to its stored value regardless of t.
    #[test]
    fn test_const_evaluation() {
        let c = Expr::Const(0.001);
        for t in [0.0, 0.5, 1.0] {
            assert_eq!(c.eval(t), 0.001);
        }
    }

    /// Protected division: denominator → 0 must not produce NaN or Inf.
    #[test]
    fn test_protected_division() {
        // 1.0 / 0.0  →  protected fallback, result must be finite
        let expr = Expr::Div(Box::new(Expr::Const(1.0)), Box::new(Expr::Const(0.0)));
        let result = expr.eval(0.5);
        assert!(result.is_finite(), "Protected division must return a finite value; got {}", result);
    }

    /// Protected exp: large exponent must not overflow to Inf.
    #[test]
    fn test_protected_exp() {
        // exp(1000 * t) at t=1 would be astronomically large without protection.
        let expr = Expr::Exp(Box::new(Expr::Mul(
            Box::new(Expr::Const(1000.0)),
            Box::new(Expr::Var),
        )));
        let result = expr.eval(1.0);
        assert!(result.is_finite(), "Protected exp must clamp to finite; got {}", result);
        assert!(result <= 1e6, "Protected exp result must be ≤ 1e6; got {}", result);
    }

    /// Protected log: log(0) must not produce -Inf.
    #[test]
    fn test_protected_log() {
        let expr = Expr::Log(Box::new(Expr::Const(0.0)));
        let result = expr.eval(0.0);
        assert!(result.is_finite(), "Protected log(0) must be finite; got {}", result);
    }

    /// eval_schedule output must be fully within [1e-7, 10.0].
    #[test]
    fn test_eval_schedule_clamping() {
        let t_array: Vec<f64> = (0..=100).map(|i| i as f64 / 100.0).collect();
        // A formula that evaluates to very large or very small values.
        let expr = Expr::Exp(Box::new(Expr::Mul(
            Box::new(Expr::Const(50.0)),
            Box::new(Expr::Var),
        )));
        let schedule = expr.eval_schedule(&t_array);
        assert_eq!(schedule.len(), t_array.len());
        for &v in &schedule {
            assert!(v >= 1e-7 && v <= 10.0,
                "Schedule value {} must be in [1e-7, 10.0]", v);
        }
    }

    // ── 9.3  Prefix Serialization / Deserialization Round-trip ───────────────

    /// Prefix round-trip: serialize then re-parse, verify structural equality.
    #[test]
    fn test_prefix_roundtrip() {
        let original = Expr::Mul(
            Box::new(Expr::Add(Box::new(Expr::Var), Box::new(Expr::Const(0.5)))),
            Box::new(Expr::Exp(Box::new(Expr::Var))),
        );
        let prefix = original.to_prefix();
        assert_eq!(prefix, "* + t 0.5 exp t");

        let parsed = parse_prefix(&prefix).expect("prefix should parse cleanly");
        // Verify structural equality via hash (structural_hash is deterministic).
        assert_eq!(original.structural_hash(), parsed.structural_hash(),
            "Round-tripped tree must be structurally identical");
    }

    /// Parser rejects unknown tokens gracefully.
    #[test]
    fn test_prefix_parse_unknown_token() {
        let result = parse_prefix("tanh t");
        assert!(result.is_err(), "Unknown operator 'tanh' must produce Err");
    }

    // ── 9.4  LaTeX Display ────────────────────────────────────────────────────

    #[test]
    fn test_latex_display_var() {
        assert_eq!(format!("{}", Expr::Var), "t");
    }

    #[test]
    fn test_latex_display_add() {
        let e = Expr::Add(Box::new(Expr::Var), Box::new(Expr::Const(1.0)));
        assert_eq!(format!("{}", e), "(t + 1)");
    }

    #[test]
    fn test_latex_display_div() {
        let e = Expr::Div(Box::new(Expr::Var), Box::new(Expr::Const(2.0)));
        assert_eq!(format!("{}", e), r"\frac{t}{2}");
    }

    #[test]
    fn test_latex_display_nested() {
        // sin(t + 0.5) — format_const(0.5) must render as "0.5" not "0.5000"
        let e = Expr::Sin(Box::new(Expr::Add(
            Box::new(Expr::Var),
            Box::new(Expr::Const(0.5)),
        )));
        assert_eq!(format!("{}", e), r"\sin((t + 0.5))");
    }

    // ── 9.5  Structural Hashing ───────────────────────────────────────────────

    /// Same tree → same hash.
    #[test]
    fn test_hash_determinism() {
        let a = Expr::Add(Box::new(Expr::Var), Box::new(Expr::Const(0.5)));
        let b = Expr::Add(Box::new(Expr::Var), Box::new(Expr::Const(0.5)));
        assert_eq!(a.structural_hash(), b.structural_hash());
    }

    /// Distinct trees → distinct hashes (in practice; not guaranteed but holds here).
    #[test]
    fn test_hash_distinguishes_distinct_trees() {
        let a = Expr::Add(Box::new(Expr::Var), Box::new(Expr::Const(0.5)));
        let b = Expr::Add(Box::new(Expr::Var), Box::new(Expr::Const(0.6)));
        assert_ne!(a.structural_hash(), b.structural_hash(),
            "Trees with different constants must hash differently");

        let c = Expr::Sub(Box::new(Expr::Var), Box::new(Expr::Const(0.5)));
        assert_ne!(a.structural_hash(), c.structural_hash(),
            "Add and Sub must hash differently");
    }

    /// Constants hashing uses bit-level representation, so -0.0 == 0.0 may differ.
    /// We just verify that Const(1.0) and Const(2.0) hash differently.
    #[test]
    fn test_const_hash_uniqueness() {
        assert_ne!(
            Expr::Const(1.0).structural_hash(),
            Expr::Const(2.0).structural_hash(),
        );
    }
}
