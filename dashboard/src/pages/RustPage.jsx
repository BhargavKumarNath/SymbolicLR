import React from 'react';
import { BarChart, Bar, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import SectionHeader from '../components/SectionHeader';
import ChartCard from '../components/ChartCard';
import CodeBlock from '../components/CodeBlock';

const RustPage = () => {
    const benchmarkData = [
        { label: "1-node constant", python: 12, rust: 12, speedup: 1 },
        { label: "3-node linear", python: 45, rust: 14, speedup: 3.2 },
        { label: "5-node cosine", python: 120, rust: 18, speedup: 6.7 },
        { label: "10-node complex", python: 380, rust: 22, speedup: 17.3 },
        { label: "15-node nested", python: 820, rust: 28, speedup: 29.3 },
        { label: "20-node deep", python: 1600, rust: 35, speedup: 45.7 },
        { label: "30-node max", python: 3200, rust: 64, speedup: 50.0 },
    ];

    return (
        <div style={{ padding: "60px" }}>
            <SectionHeader
                tag="Performance"
                title="The Rust Evaluation Engine"
                subtitle="The inner evaluation loop is the computational bottleneck. SymboLR replaces Python's recursive AST traversal with a zero-allocation Rust extension via PyO3, achieving up to 50× speedup."
            />

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24, marginBottom: 32 }}>
                {/* Benchmark chart */}
                <ChartCard title="Evaluation Time per Formula" subtitle="Microseconds for 100-step t-array evaluation">
                    <ResponsiveContainer width="100%" height={280}>
                        <BarChart data={benchmarkData} margin={{ top: 10, right: 10, bottom: 0, left: 0 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                            <XAxis dataKey="label" stroke="rgba(255,255,255,0.3)" fontSize={9} tick={{ angle: -30, textAnchor: "end" }} height={60} />
                            <YAxis stroke="rgba(255,255,255,0.3)" fontSize={10} tickFormatter={v => `${v}µs`} width={50} />
                            <Tooltip
                                contentStyle={{ background: "rgba(0,0,0,0.9)", border: "1px solid rgba(255,255,255,0.15)", borderRadius: 8 }}
                                formatter={(v, name) => [`${v}µs`, name]}
                            />
                            <Bar dataKey="python" name="Python + NumPy" fill="rgba(255,159,10,0.6)" radius={[4, 4, 0, 0]} />
                            <Bar dataKey="rust" name="Rust Extension" fill="rgba(48,209,88,0.8)" radius={[4, 4, 0, 0]} />
                        </BarChart>
                    </ResponsiveContainer>
                </ChartCard>

                {/* Speedup chart */}
                <ChartCard title="Speedup Factor" subtitle="Rust vs Python performance ratio by tree depth">
                    <ResponsiveContainer width="100%" height={280}>
                        <AreaChart data={benchmarkData} margin={{ top: 10, right: 10, bottom: 0, left: 0 }}>
                            <defs>
                                <linearGradient id="speedGrad" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor="#30d158" stopOpacity={0.3} />
                                    <stop offset="95%" stopColor="#30d158" stopOpacity={0} />
                                </linearGradient>
                            </defs>
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                            <XAxis dataKey="label" stroke="rgba(255,255,255,0.3)" fontSize={9} tick={{ angle: -30, textAnchor: "end" }} height={60} />
                            <YAxis stroke="rgba(255,255,255,0.3)" fontSize={10} tickFormatter={v => `${v}×`} width={40} />
                            <Tooltip
                                contentStyle={{ background: "rgba(0,0,0,0.9)", border: "1px solid rgba(255,255,255,0.15)", borderRadius: 8 }}
                                formatter={v => [`${v}×`, "Speedup"]}
                            />
                            <Area type="monotone" dataKey="speedup" name="Speedup" stroke="#30d158" strokeWidth={2} fill="url(#speedGrad)" dot={{ fill: "#30d158", r: 3 }} />
                        </AreaChart>
                    </ResponsiveContainer>
                </ChartCard>
            </div>

            {/* How it works */}
            <div className="glass" style={{ padding: "36px", marginBottom: 24 }}>
                <h2 style={{ fontSize: 20, fontWeight: 700, marginBottom: 24, letterSpacing: "-0.02em" }}>How the Bridge Works</h2>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24 }}>
                    <div>
                        <h3 style={{ fontSize: 14, fontWeight: 600, color: "var(--orange)", marginBottom: 12 }}>Python Side</h3>
                        <CodeBlock lang="Python" code={`# gp/rust_bridge.py
def evaluate_schedule(tree, t_array):
    # Serialize AST to prefix notation
    prefix = tree.to_prefix()
    # "* 0.067 cos t"
    
    # Call into compiled Rust
    result = symbolr_rust.evaluate_fast(prefix, t_array)
    
    # Sanitize output
    result = np.nan_to_num(result, nan=1e-4)
    return np.clip(result, 1e-7, 10.0)`} />
                    </div>
                    <div>
                        <h3 style={{ fontSize: 14, fontWeight: 600, color: "var(--green)", marginBottom: 12 }}>Rust Side</h3>
                        <CodeBlock lang="Rust" code={`// rust_core/src/lib.rs
#[pyfunction]
fn evaluate_fast(prefix_expr: &str, t_array: PyReadonlyArray1<f64>)
    -> PyResult<&PyArray1<f64>>
{
    let tokens: Vec<&str> = prefix_expr.split_whitespace().collect();
    let ast = parse_prefix(&mut tokens.iter())?;
    
    let mut result = Array1::<f64>::zeros(t_view.raw_dim());
    for (i, &t) in t_view.iter().enumerate() {
        result[i] = ast.eval(t);  // Scalar, no alloc
    }
    Ok(result.into_pyarray(py))  // Zero-copy
}`} />
                    </div>
                </div>
            </div>

            {/* Key optimizations */}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 16 }}>
                {[
                    {
                        title: "Zero Allocations",
                        icon: "◎",
                        color: "var(--green)",
                        desc: "Scalar evaluation per time step. No intermediate Vec<f64> allocations inside the loop. The result array is pre-allocated once."
                    },
                    {
                        title: "Zero-Copy Return",
                        icon: "◈",
                        color: "var(--blue)",
                        desc: "PyO3's into_pyarray() transfers ownership to NumPy without copying. Python sees a direct view of Rust-owned memory."
                    },
                    {
                        title: "Protected Math",
                        icon: "⬡",
                        color: "var(--orange)",
                        desc: "All safety guards (protected div, log, exp clipping) are mirrored exactly in Rust. Test suite verifies 1e-7 parity with Python."
                    },
                    {
                        title: "Prefix Parsing",
                        icon: "≋",
                        color: "var(--purple)",
                        desc: "AST serializes to a space-separated prefix string in O(n). Rust parses this recursively into a native Expr enum — no serde overhead."
                    },
                    {
                        title: "PyO3 + Maturin",
                        icon: "⚙",
                        color: "var(--teal)",
                        desc: "Compiled via Maturin to a .pyd/.so extension module. Import exactly like Python — no subprocess, no FFI boilerplate."
                    },
                    {
                        title: "Graceful Fallback",
                        icon: "△",
                        color: "var(--red)",
                        desc: "rust_bridge.py detects RUST_AVAILABLE at import time. Cloud deployments silently fall back to Node.evaluate() with identical numerical behavior."
                    },
                ].map(({ title, icon, color, desc }, i) => (
                    <div key={i} className="glass" style={{ padding: "24px", borderColor: `${color}33`, animation: `fadeUp ${0.4 + i * 0.08}s ease both` }}>
                        <div style={{ fontSize: 24, color, marginBottom: 12 }}>{icon}</div>
                        <h3 style={{ fontSize: 14, fontWeight: 600, marginBottom: 10 }}>{title}</h3>
                        <p style={{ fontSize: 12, color: "var(--text-2)", lineHeight: 1.7 }}>{desc}</p>
                    </div>
                ))}
            </div>
        </div>
    );
};

export default RustPage;
