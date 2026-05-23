use pyo3::prelude::*;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1};
use ndarray::Array1;
use rayon::prelude::*;
use dashmap::DashMap;
use once_cell::sync::Lazy;

// Global Memoization Cache
static CACHE: Lazy<DashMap<String, Vec<f64>>> = Lazy::new(DashMap::new);

/// Native AST Enum matching the SymboLR Python Node structure
#[derive(Clone)]
enum Expr {
    Var,
    Const(f64),
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Div(Box<Expr>, Box<Expr>),
    Sin(Box<Expr>),
    Cos(Box<Expr>),
    Exp(Box<Expr>),
    Log(Box<Expr>),
    Sqrt(Box<Expr>),
    Abs(Box<Expr>),
}

impl Expr {
    /// Scalar evaluation applied per-element to avoid intermediate array allocations
    fn eval(&self, t: f64) -> f64 {
        match self {
            Expr::Var => t,
            Expr::Const(c) => *c,
            Expr::Add(a, b) => a.eval(t) + b.eval(t),
            Expr::Sub(a, b) => a.eval(t) - b.eval(t),
            Expr::Mul(a, b) => a.eval(t) * b.eval(t),
            Expr::Div(a, b) => {
                let den = b.eval(t);
                // Protected Division
                if den.abs() < 1e-6 { 1.0 } else { a.eval(t) / den }
            },
            Expr::Sin(a) => a.eval(t).sin(),
            Expr::Cos(a) => a.eval(t).cos(),
            Expr::Exp(a) => {
                // Protected Exponential
                a.eval(t).clamp(-100.0, 10.0).exp()
            },
            Expr::Log(a) => {
                // Protected Logarithm
                let val = a.eval(t).abs();
                if val < 1e-6 { 1e-6_f64.ln() } else { val.ln() }
            },
            Expr::Sqrt(a) => {
                // Protected Sqrt
                a.eval(t).abs().sqrt()
            },
            Expr::Abs(a) => a.eval(t).abs(),
        }
    }
}

/// Recursively parses the space-separated prefix string from Python
fn parse_prefix(tokens: &mut std::slice::Iter<&str>) -> Result<Expr, String> {
    let token = tokens.next().ok_or("Unexpected end of expression")?;
    match *token {
        "t" => Ok(Expr::Var),
        "+" => Ok(Expr::Add(Box::new(parse_prefix(tokens)?), Box::new(parse_prefix(tokens)?))),
        "-" => Ok(Expr::Sub(Box::new(parse_prefix(tokens)?), Box::new(parse_prefix(tokens)?))),
        "*" => Ok(Expr::Mul(Box::new(parse_prefix(tokens)?), Box::new(parse_prefix(tokens)?))),
        "/" => Ok(Expr::Div(Box::new(parse_prefix(tokens)?), Box::new(parse_prefix(tokens)?))),
        "sin" => Ok(Expr::Sin(Box::new(parse_prefix(tokens)?))),
        "cos" => Ok(Expr::Cos(Box::new(parse_prefix(tokens)?))),
        "exp" => Ok(Expr::Exp(Box::new(parse_prefix(tokens)?))),
        "log" => Ok(Expr::Log(Box::new(parse_prefix(tokens)?))),
        "sqrt" => Ok(Expr::Sqrt(Box::new(parse_prefix(tokens)?))),
        "abs" => Ok(Expr::Abs(Box::new(parse_prefix(tokens)?))),
        _ => {
            if let Ok(val) = token.parse::<f64>() {
                Ok(Expr::Const(val))
            } else {
                Err(format!("Unknown token: {}", token))
            }
        }
    }
}

fn compute_single(prefix_expr: &str, t_view: &[f64]) -> Result<Vec<f64>, String> {
    // Check global cache
    if let Some(cached) = CACHE.get(prefix_expr) {
        return Ok(cached.clone());
    }

    let tokens: Vec<&str> = prefix_expr.split_whitespace().collect();
    let mut iter = tokens.iter();
    
    // Parse to AST
    let ast = parse_prefix(&mut iter)?;
    
    let mut result = Vec::with_capacity(t_view.len());
    
    // Execute element-wise evaluation
    for &t_val in t_view.iter() {
        let mut val = ast.eval(t_val);
        if val.is_nan() || val.is_infinite() {
            val = 1.0;
        }
        result.push(val);
    }
    
    // Store in cache
    CACHE.insert(prefix_expr.to_string(), result.clone());
    
    Ok(result)
}

#[pyfunction]
fn evaluate_fast<'py>(
    py: Python<'py>,
    prefix_expr: &str,
    t_array: PyReadonlyArray1<f64>
) -> PyResult<&'py PyArray1<f64>> {
    let t_view = t_array.as_slice().unwrap();
    let result_vec = compute_single(prefix_expr, t_view)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
        
    let arr = Array1::from(result_vec);
    Ok(arr.into_pyarray(py))
}

#[pyfunction]
fn evaluate_batch<'py>(
    py: Python<'py>,
    prefix_exprs: Vec<String>,
    t_array: PyReadonlyArray1<f64>
) -> PyResult<&'py PyArray2<f64>> {
    let t_view = t_array.as_slice().unwrap();
    let n = prefix_exprs.len();
    let m = t_view.len();
    
    // Parallelize with Rayon!
    let results: Result<Vec<Vec<f64>>, String> = prefix_exprs
        .par_iter()
        .map(|expr| compute_single(expr, t_view))
        .collect();
        
    let results = results.map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
    
    // Flatten into a 2D numpy array natively
    let mut flat = Vec::with_capacity(n * m);
    for mut row in results {
        flat.append(&mut row);
    }
    
    let arr2 = ndarray::Array2::from_shape_vec((n, m), flat)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        
    Ok(arr2.into_pyarray(py))
}

#[pymodule]
fn symbolr_rust(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(evaluate_fast, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_batch, m)?)?;
    Ok(())
}