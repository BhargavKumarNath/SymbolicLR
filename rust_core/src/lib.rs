use pyo3::prelude::*;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use ndarray::Array1;

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

#[pyfunction]
fn evaluate_fast<'py>(
    py: Python<'py>,
    prefix_expr: &str,
    t_array: PyReadonlyArray1<f64>
) -> PyResult<&'py PyArray1<f64>> {
    let tokens: Vec<&str> = prefix_expr.split_whitespace().collect();
    let mut iter = tokens.iter();
    
    // Parse to AST
    let ast = parse_prefix(&mut iter).map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
    
    let t_view = t_array.as_array();
    let mut result = Array1::<f64>::zeros(t_view.raw_dim());
    
    // Execute element-wise evaluation
    for (i, &t_val) in t_view.iter().enumerate() {
        let mut val = ast.eval(t_val);
        // Nan/Inf guard identical to Python's nan_to_num fallback
        if val.is_nan() || val.is_infinite() {
            val = 1.0;
        }
        result[i] = val;
    }
    
    Ok(result.into_pyarray(py))
}

#[pymodule]
fn symbolr_rust(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(evaluate_fast, m)?)?;
    Ok(())
}