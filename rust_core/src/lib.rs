//! lib.rs — SymboLR Rust Core: root crate module.
//!
//! Architecture:
//!   ast.rs     — Native Expr enum (Phase 1)
//!   archive.rs — Native MapElites archive (Phase 1)
//!   [Phase 2+] operators.rs, evolution.rs, PyO3 streaming bridge
//!
//! The legacy `evaluate_fast` / `evaluate_batch` PyO3 functions are preserved
//! to guarantee zero regressions against the existing Python test suite
//! (`tests/unit/test_rust_core.py`) until the Phase 3 streaming bridge
//! supersedes them.

// ─────────────────────────────────────────────────────────────────────────────
// Module declarations
// ─────────────────────────────────────────────────────────────────────────────

pub mod ast;
pub mod archive;

// ─────────────────────────────────────────────────────────────────────────────
// Legacy PyO3 Bridge  (Phase 1 compat — will be replaced in Phase 3)
// ─────────────────────────────────────────────────────────────────────────────

use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1};
use rayon::prelude::*;
use dashmap::DashMap;
use once_cell::sync::Lazy;

use ast::{Expr, parse_prefix};

// Global memoization cache: prefix string → evaluated schedule Vec<f64>.
static CACHE: Lazy<DashMap<String, Vec<f64>>> = Lazy::new(DashMap::new);

// ─────────────────────────────────────────────────────────────────────────────
// Internal computation
// ─────────────────────────────────────────────────────────────────────────────

fn compute_single(prefix_expr: &str, t_view: &[f64]) -> Result<Vec<f64>, String> {
    if let Some(cached) = CACHE.get(prefix_expr) {
        return Ok(cached.clone());
    }

    let ast: Expr = parse_prefix(prefix_expr)
        .map_err(|e| format!("parse_prefix error: {e}"))?;

    let mut result = Vec::with_capacity(t_view.len());
    for &t_val in t_view.iter() {
        let mut val = ast.eval(t_val);
        if !val.is_finite() { val = 1.0; }
        result.push(val);
    }

    CACHE.insert(prefix_expr.to_string(), result.clone());
    Ok(result)
}

// ─────────────────────────────────────────────────────────────────────────────
// PyO3 exports (preserved for Python test compatibility)
// ─────────────────────────────────────────────────────────────────────────────

#[pyfunction]
fn evaluate_fast<'py>(
    py: Python<'py>,
    prefix_expr: &str,
    t_array: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let t_view = t_array.as_slice()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let result_vec = compute_single(prefix_expr, t_view)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
    Ok(PyArray1::from_slice_bound(py, &result_vec))
}

#[pyfunction]
fn evaluate_batch<'py>(
    py: Python<'py>,
    prefix_exprs: Vec<String>,
    t_array: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let t_view = t_array.as_slice()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let _n = prefix_exprs.len();
    let _m = t_view.len();

    let results: Result<Vec<Vec<f64>>, String> = prefix_exprs
        .par_iter()
        .map(|expr| compute_single(expr, t_view))
        .collect();

    let results = results
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;

    Ok(PyArray2::from_vec2_bound(py, &results)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?)
}

// ─────────────────────────────────────────────────────────────────────────────
// PyO3 module registration
// ─────────────────────────────────────────────────────────────────────────────

#[pymodule]
fn symbolr_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(evaluate_fast, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_batch, m)?)?;
    Ok(())
}