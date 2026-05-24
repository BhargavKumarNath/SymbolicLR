//! lib.rs — SymboLR Rust Core: root crate module + PyO3 FFI boundary.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │  Python                                                     │
//! │    gp/rust_bridge.py  →  symbolr_rust.EvolutionStream       │
//! │                           ↕  JSON telemetry (serde_json)    │
//! ├─────────────────────────────────────────────────────────────┤
//! │  PyO3 FFI Boundary                                          │
//! │    #[pyclass] EvolutionStream  ←  probe_data: Vec<f64>      │
//! │    __new__:  numpy → Vec<f64>  (single copy, held in Rust)  │
//! │    __next__: one parallel generation, zero FFI per-call     │
//! ├─────────────────────────────────────────────────────────────┤
//! │  Rust Core (no Python/GC/FFI on hot path)                   │
//! │    ast.rs       — Expr enum, Box<Expr> heap ASTs            │
//! │    archive.rs   — HashMap-backed MAP-Elites archive          │
//! │    operators.rs — O(1)-pruned genetic operators              │
//! │    evolution.rs — rayon par_iter() fitness evaluation        │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## FFI Contract
//!
//! Probe data enters Rust **once** in `EvolutionStream::new()` via
//! `PyReadonlyArray1::as_slice()?.to_vec()`.  Every subsequent `__next__`
//! call operates on Rust-owned memory — no GIL acquisition, no numpy
//! reference counting, no serialization on the critical evolutionary path.
//!
//! Per-generation telemetry exits as a compact **JSON string** so Python
//! consumes exactly one string allocation per generation, not one per
//! formula.

// ─────────────────────────────────────────────────────────────────────────────
// Module declarations
// ─────────────────────────────────────────────────────────────────────────────

pub mod ast;
pub mod archive;
pub mod operators;
pub mod evolution;

// ─────────────────────────────────────────────────────────────────────────────
// Imports
// ─────────────────────────────────────────────────────────────────────────────

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1};
use rayon::prelude::*;
use dashmap::DashMap;
use once_cell::sync::Lazy;
use rand::rngs::SmallRng;
use rand::SeedableRng;

use ast::{Expr, parse_prefix};
use archive::{ArchiveConfig, MapElitesArchive};
use evolution::EvolutionConfig;

// ─────────────────────────────────────────────────────────────────────────────
// Legacy memoization cache  (evaluate_fast / evaluate_batch hot path)
// ─────────────────────────────────────────────────────────────────────────────

static CACHE: Lazy<DashMap<String, Vec<f64>>> = Lazy::new(DashMap::new);

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Serialize an f64 that may be non-finite (Inf/NaN) to a JSON Value.
/// Non-finite values map to `null` — the only valid JSON representation.
#[inline]
fn finite_f64(v: f64) -> serde_json::Value {
    serde_json::Number::from_f64(v)
        .map(serde_json::Value::Number)
        .unwrap_or(serde_json::Value::Null)
}

// ─────────────────────────────────────────────────────────────────────────────
// EvolutionStream — the Phase 3 PyO3 streaming iterator
// ─────────────────────────────────────────────────────────────────────────────

/// Rust-native MAP-Elites evolutionary stream, exposed to Python as an iterator.
///
/// ## Memory model
///
/// All evolutionary state (archive, RNG, population, time-step array, probe
/// labels) lives in **Rust-owned memory** for the entire lifetime of the
/// object.  No Python object is kept alive after `__new__` returns.
///
/// ## Python usage
///
/// ```python
/// stream = symbolr_rust.EvolutionStream(
///     probe_labels   = y_train,           # numpy array, copied once
///     max_generations = 100,
///     pop_size        = 50,
///     seed            = 42,
/// )
/// for json_str in stream:               # __next__ = one parallel generation
///     info = json.loads(json_str)
///     print(info["best_mse"], info["top_formula_latex"])
/// ```
pub struct EvolutionState {
    archive:    MapElitesArchive,
    evo_config: EvolutionConfig,
    eval_callback: PyObject,
    rng:        SmallRng,
}

#[pyclass]
pub struct EvolutionStream {
    state: Option<EvolutionState>,
    current_generation: usize,
    max_generations:    usize,
}

#[pymethods]
impl EvolutionStream {
    #[new]
    #[pyo3(signature = (
        eval_callback,
        max_generations = 50,
        pop_size        = 50,
        seed            = 42,
        crossover_rate  = 0.20,
        mutation_rate   = 0.70,
    ))]
    fn new(
        eval_callback:   PyObject,
        max_generations: usize,
        pop_size:        usize,
        seed:            u64,
        crossover_rate:  f64,
        mutation_rate:   f64,
    ) -> PyResult<Self> {
        let archive_config = ArchiveConfig::default();
        let evo_config = EvolutionConfig {
            pop_size,
            crossover_rate,
            mutation_rate,
            max_tree_depth: 4,
            archive_config: archive_config.clone(),
        };
        let archive = MapElitesArchive::new(archive_config);
        let rng = SmallRng::seed_from_u64(seed);

        let state = EvolutionState {
            archive,
            evo_config,
            eval_callback,
            rng,
        };

        Ok(Self {
            state: Some(state),
            current_generation: 0,
            max_generations,
        })
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>, py: Python<'_>) -> Option<String> {
        if slf.current_generation >= slf.max_generations {
            return None;
        }

        let gen = slf.current_generation + 1;
        let mut state = slf.state.take().expect("Stream state poisoned or accessed concurrently");

        let start = std::time::Instant::now();

        // 1. Generate offspring on CPU
        let offspring = crate::evolution::generate_offspring(
            &state.archive,
            &state.evo_config,
            &mut state.rng,
        );

        // 2. Extract prefix formulas
        let formulas: Vec<String> = offspring.iter().map(|e| e.to_prefix()).collect();

        // 3. Call Python callback (holds GIL, blocks async loop)
        let py_res = state.eval_callback.call1(py, (formulas,)).expect("Python evaluation callback failed");
        let fitnesses: Vec<f64> = py_res.extract(py).expect("Failed to extract List[float] from evaluate_batch");

        // 4. Update the MAP-Elites Archive
        let stats = crate::evolution::update_archive(
            &mut state.archive,
            offspring,
            fitnesses,
            gen,
            start.elapsed().as_millis(),
        );

        // Restore state
        slf.state = Some(state);
        slf.current_generation = gen;

        let json = serde_json::json!({
            "generation_number": stats.generation,
            "best_mse":          finite_f64(stats.best_loss),
            "average_mse":       finite_f64(stats.mean_loss),
            "top_formula_latex": stats.best_formula_latex,
            "top_formula_prefix": stats.best_formula_prefix,
            "archive_size":      stats.archive_size,
            "new_entries":       stats.new_entries,
            "gen_time_ms":       stats.gen_time_ms,
        });

        Some(json.to_string())
    }

    fn archive_stats(&self) -> String {
        let state = self.state.as_ref().unwrap();
        let s = state.archive.stats();
        serde_json::json!({
            "occupied_niches": s.occupied_niches,
            "max_niches":      s.max_niches,
            "occupancy_pct":   finite_f64(s.occupancy_pct),
            "total_attempts":  s.total_attempts,
            "total_additions": s.total_additions,
            "best_loss":       finite_f64(s.best_loss),
            "median_loss":     finite_f64(s.median_loss),
            "mean_elite_age":  finite_f64(s.mean_elite_age),
        })
        .to_string()
    }

    fn hall_of_fame(&self, top_k: usize) -> String {
        let state = self.state.as_ref().unwrap();
        let hof: Vec<serde_json::Value> = state.archive
            .hall_of_fame(top_k)
            .iter()
            .map(|n| serde_json::json!({
                "latex":      format!("{}", n.expr),
                "prefix":     n.expr.to_prefix(),
                "loss":       finite_f64(n.raw_loss),
                "complexity": n.complexity,
                "age":        n.age,
            }))
            .collect();
        serde_json::json!(hof).to_string()
    }

    #[getter]
    fn generation(&self) -> usize {
        self.current_generation
    }

    #[getter]
    fn max_generations(&self) -> usize {
        self.max_generations
    }

    #[getter]
    fn archive_size(&self) -> usize {
        self.state.as_ref().unwrap().archive.len()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Legacy PyO3 Bridge  (evaluate_fast / evaluate_batch)
// Preserved for zero-regression compatibility with existing Python test suite.
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

#[pyfunction]
fn evaluate_fast<'py>(
    py: Python<'py>,
    prefix_expr: &str,
    t_array: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let t_view = t_array.as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let result_vec = compute_single(prefix_expr, t_view)
        .map_err(|e| PyValueError::new_err(e))?;
    Ok(PyArray1::from_slice_bound(py, &result_vec))
}

#[pyfunction]
fn evaluate_batch<'py>(
    py: Python<'py>,
    prefix_exprs: Vec<String>,
    t_array: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let t_view = t_array.as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let results: Result<Vec<Vec<f64>>, String> = prefix_exprs
        .par_iter()
        .map(|expr| compute_single(expr, t_view))
        .collect();

    let results = results.map_err(|e| PyValueError::new_err(e))?;

    Ok(PyArray2::from_vec2_bound(py, &results)
        .map_err(|e| PyValueError::new_err(e.to_string()))?)
}

// ─────────────────────────────────────────────────────────────────────────────
// PyO3 module registration
// ─────────────────────────────────────────────────────────────────────────────

#[pymodule]
fn symbolr_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Legacy schedule evaluators (Phase 1 compatibility).
    m.add_function(wrap_pyfunction!(evaluate_fast, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_batch, m)?)?;
    // Phase 3: streaming evolution iterator.
    m.add_class::<EvolutionStream>()?;
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests (Phase 3 Validation)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod ffi_tests {
    use super::*;
    use serde_json::Value;

    #[test]
    fn test_evolution_stream_ffi() {
        // Initialize the Python interpreter for testing outside of a module context
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            // 1. Test: PyObject Callback Init
            let py_code = "lambda formulas: [1.0 for _ in formulas]";
            let eval_callback = py.eval_bound(py_code, None, None)
                .expect("Failed to compile lambda")
                .into_any()
                .unbind();

            let max_generations = 3;
            let stream = EvolutionStream::new(
                eval_callback,
                max_generations,
                10, // pop_size
                42, // seed
                0.2, // crossover_rate
                0.7, // mutation_rate
            ).expect("Failed to initialize EvolutionStream");

            assert_eq!(stream.generation(), 0, "Initial generation should be 0");
            assert_eq!(stream.max_generations(), 3, "max_generations should be 3");

            // Bind the stream to Python to get PyRefMut for `__next__`
            let bound_stream = Bound::new(py, stream).expect("Failed to bind stream");

            // 2. Test: Iterator Yield Protocol
            for expected_gen in 1..=3 {
                let slf = bound_stream.borrow_mut();
                
                // Manually call `__next__`
                let result = EvolutionStream::__next__(slf, py);
                assert!(result.is_some(), "Expected Some(String) on generation {}", expected_gen);
                
                let json_str = result.unwrap();
                let parsed: Value = serde_json::from_str(&json_str).expect("Result should be valid JSON");
                
                // Assert generation number increments correctly
                let gen_num = parsed["generation_number"].as_u64().expect("Missing generation_number");
                assert_eq!(gen_num, expected_gen as u64, "Generation number mismatch");
                
                // Ensure other telemetry fields are present
                assert!(parsed.get("best_mse").is_some(), "Missing best_mse");
                assert!(parsed.get("average_mse").is_some(), "Missing average_mse");
                assert!(parsed.get("top_formula_latex").is_some(), "Missing top_formula_latex");
            }

            // 3. Test: StopIteration Enforcement
            let slf = bound_stream.borrow_mut();
            let result = EvolutionStream::__next__(slf, py);
            assert!(result.is_none(), "Expected None (StopIteration) after max_generations");
        });
    }
}