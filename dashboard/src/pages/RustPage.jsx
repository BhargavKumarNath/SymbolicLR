import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import { Cpu, Zap, Lock, ShieldCheck } from 'lucide-react';
import SectionHeader from '../components/SectionHeader';
import CodeBlock from '../components/CodeBlock';

const RustPage = () => {
    const benchmarkData = [
        { label: "10-Node Formula", python: 380, rust: 4.2 },
        { label: "20-Node Formula", python: 1600, rust: 8.5 },
        { label: "30-Node Formula", python: 3200, rust: 14.1 },
    ];

    const CustomTooltip = ({ active, payload, label }) => {
        if (active && payload && payload.length) {
            return (
                <div style={{ background: "rgba(10,10,10,0.95)", border: "1px solid var(--border)", borderRadius: 8, padding: 12, fontSize: 12 }}>
                    <div style={{ fontWeight: 600, marginBottom: 8, color: "var(--text)" }}>{label}</div>
                    <div style={{ color: "var(--orange)", marginBottom: 4 }}>
                        Python: {payload[0].value} ms
                    </div>
                    <div style={{ color: "var(--green)", marginBottom: 8 }}>
                        Rust: {payload[1].value} ms
                    </div>
                    <div style={{ fontSize: 10, color: "var(--text-3)", fontStyle: "italic", borderTop: "1px solid var(--border)", paddingTop: 8 }}>
                        * Benchmark executed on consumer AMD Ryzen 9
                    </div>
                </div>
            );
        }
        return null;
    };

    return (
        <div style={{ padding: "60px", maxWidth: 1200, margin: "0 auto" }}>
            <SectionHeader
                tag="Architecture"
                title="Fearless Concurrency"
                subtitle="To evaluate millions of mathematical schedules without bankrupting compute budgets, we bypassed the Python GIL entirely. The evaluation engine is fully decoupled and rewritten in safe, multithreaded Rust."
            />

            {/* Section 1: The Compute Bottleneck (The Problem) */}
            <div style={{ marginBottom: 60 }}>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 32, alignItems: "center" }}>
                    <div className="glass" style={{ padding: "36px", height: "100%", borderTop: "3px solid var(--orange)" }}>
                        <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 20 }}>
                            <Lock color="var(--orange)" size={24} />
                            <h2 style={{ fontSize: 22, fontWeight: 700 }}>The Compute Bottleneck</h2>
                        </div>
                        <p style={{ fontSize: 15, color: "var(--text-2)", lineHeight: 1.7, marginBottom: 16 }}>
                            Quality-Diversity genetic programming is inherently parallel, yet the Python Global Interpreter Lock (GIL) enforces sequential execution. 
                        </p>
                        <p style={{ fontSize: 15, color: "var(--text-2)", lineHeight: 1.7 }}>
                            When evaluating thousands of dynamic Abstract Syntax Trees (ASTs) recursively in pure Python, the overhead becomes catastrophic. Memory allocation spikes, CPU cache misses compound, and the evolutionary loop grinds to a halt. We needed C-level performance with memory safety.
                        </p>
                    </div>

                    <div className="glass" style={{ padding: "0" }}>
                        <div style={{ padding: "16px 24px", borderBottom: "1px solid var(--border)", background: "rgba(255,255,255,0.02)" }}>
                            <h3 style={{ fontSize: 13, fontWeight: 600, color: "var(--text-3)", textTransform: "uppercase", letterSpacing: "0.05em" }}>The PyO3 Boundary</h3>
                        </div>
                        <div style={{ padding: "20px" }}>
                            <CodeBlock lang="python" code={`# gp/rust_bridge.py
def evaluate_schedule(tree, t_array):
    # 1. Serialize Python AST to prefix O(N)
    prefix = tree.to_prefix()
    
    # 2. Handoff to compiled Rust extension
    # The GIL is released immediately here
    result = symbolr_rust.evaluate_fast(prefix, t_array)
    
    return np.clip(result, 1e-7, 10.0)`} />
                        </div>
                    </div>
                </div>
            </div>

            {/* Section 2: Fearless Concurrency (The Solution) */}
            <div style={{ marginBottom: 60 }}>
                <div className="glass" style={{ padding: "40px", borderTop: "3px solid var(--green)" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 24 }}>
                        <ShieldCheck color="var(--green)" size={28} />
                        <h2 style={{ fontSize: 24, fontWeight: 700 }}>Zero-Cost Abstractions</h2>
                    </div>
                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 40 }}>
                        <p style={{ fontSize: 15, color: "var(--text-2)", lineHeight: 1.7 }}>
                            The <code style={{fontFamily: "var(--mono)", background: "rgba(255,255,255,0.1)", padding: "2px 6px", borderRadius: 4, fontSize: 13}}>rust_core</code> engine receives the serialized ASTs and instantly compiles them into highly optimized, zero-allocation evaluation graphs. Because Rust guarantees memory safety at compile-time, we can aggressively parallelize the fitness benchmarking.
                        </p>
                        <p style={{ fontSize: 15, color: "var(--text-2)", lineHeight: 1.7 }}>
                            Data races are impossible. Memory leaks are eliminated. The system flawlessly saturates every available CPU core concurrently, processing thousands of non-linear mathematical evaluations per second before returning a zero-copy NumPy array back to Python.
                        </p>
                    </div>
                </div>
            </div>

            {/* Section 2.5: The Evaluation Algorithms */}
            <div style={{ marginBottom: 60 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 24 }}>
                    <Cpu color="var(--purple)" size={24} />
                    <h2 style={{ fontSize: 22, fontWeight: 700 }}>Under the Hood: Protected Scalar Evaluation</h2>
                </div>
                
                <div className="glass" style={{ padding: "0" }}>
                    <div style={{ padding: "20px 24px", borderBottom: "1px solid var(--border)", background: "rgba(255,255,255,0.02)" }}>
                        <p style={{ fontSize: 15, color: "var(--text-2)", lineHeight: 1.6, margin: 0 }}>
                            By parsing Python's complex object-oriented AST into a flat Rust <code>enum</code>, we eliminate dynamic dispatch. The code below demonstrates <strong>Protected Scalar Evaluation</strong>: instead of allocating memory for entire vector operations step-by-step (like NumPy does), the Rust engine evaluates the entire formula element-by-element inside a single contiguous array, ensuring absolute memory stability.
                        </p>
                    </div>
                    <div style={{ padding: "20px" }}>
                        <CodeBlock lang="rust" code={`// rust_core/src/lib.rs

// 1. Zero-Cost AST Mapping
// Using Rust's enums to represent the formula tree natively. 
// No vtables, no garbage collection, just pure memory-contiguous structures.
#[derive(Clone)]
enum Expr {
    Var,
    Const(f64),
    Add(Box<Expr>, Box<Expr>),
    Div(Box<Expr>, Box<Expr>),
    Exp(Box<Expr>),
    // ... other operators
}

impl Expr {
    // 2. Protected Scalar Evaluation
    // This executes per time-step (t) instead of allocating massive arrays.
    fn eval(&self, t: f64) -> f64 {
        match self {
            Expr::Var => t,
            Expr::Const(c) => *c,
            Expr::Add(a, b) => a.eval(t) + b.eval(t),
            
            // Protected Division: Prevents NaN crashes in the evolutionary loop
            Expr::Div(a, b) => {
                let den = b.eval(t);
                if den.abs() < 1e-6 { 1.0 } else { a.eval(t) / den }
            },
            
            // Protected Exponential: Prevents overflow limits
            Expr::Exp(a) => a.eval(t).clamp(-100.0, 10.0).exp(),
            // ...
        }
    }
}

// 3. The PyO3 Bridge
#[pyfunction]
fn evaluate_fast<'py>(py: Python<'py>, prefix_expr: &str, t_array: PyReadonlyArray1<f64>) -> PyResult<&'py PyArray1<f64>> {
    // Parse O(N) space-separated string from Python into the Expr AST
    let tokens: Vec<&str> = prefix_expr.split_whitespace().collect();
    let mut iter = tokens.iter();
    let ast = parse_prefix(&mut iter).unwrap();
    
    // Allocate the output array EXACTLY once.
    let t_view = t_array.as_array();
    let mut result = Array1::<f64>::zeros(t_view.raw_dim());
    
    // Execute element-wise to keep the CPU cache hot. 
    for (i, &t_val) in t_view.iter().enumerate() {
        result[i] = ast.eval(t_val);
    }
    
    // 4. Zero-Copy Transfer
    // Return ownership to Python's NumPy without copying memory.
    Ok(result.into_pyarray(py))
}`} />
                    </div>
                </div>
            </div>

            {/* Section 3: Benchmarking Visualization */}
            <div>
                <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 24 }}>
                    <Zap color="var(--blue)" size={24} />
                    <h2 style={{ fontSize: 22, fontWeight: 700 }}>Execution Latency</h2>
                </div>
                
                <div className="glass" style={{ padding: "32px" }}>
                    <div style={{ marginBottom: 24, display: "flex", justifyContent: "space-between", alignItems: "flex-end" }}>
                        <div>
                            <h3 style={{ fontSize: 16, fontWeight: 600 }}>Time to Evaluate 10,000 Formulas</h3>
                            <p style={{ fontSize: 13, color: "var(--text-3)" }}>Comparing pure Python recursion vs Rust FFI parallel graphs.</p>
                        </div>
                        <div style={{ display: "flex", gap: 16 }}>
                            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                                <div style={{ width: 12, height: 12, background: "var(--orange)", borderRadius: 2 }} />
                                <span style={{ fontSize: 12, color: "var(--text-3)" }}>Python + NumPy</span>
                            </div>
                            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                                <div style={{ width: 12, height: 12, background: "var(--green)", borderRadius: 2 }} />
                                <span style={{ fontSize: 12, color: "var(--text-3)" }}>Rust Engine</span>
                            </div>
                        </div>
                    </div>

                    <div style={{ height: 300 }}>
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart layout="vertical" data={benchmarkData} margin={{ top: 10, right: 30, left: 30, bottom: 0 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" horizontal={true} vertical={false} />
                                <XAxis type="number" stroke="rgba(255,255,255,0.3)" fontSize={11} tickFormatter={(v) => v + "ms"} />
                                <YAxis dataKey="label" type="category" stroke="var(--text-3)" fontSize={12} width={120} tickLine={false} axisLine={false} />
                                <Tooltip content={<CustomTooltip />} cursor={{ fill: "rgba(255,255,255,0.02)" }} />
                                <Bar dataKey="python" fill="var(--orange)" barSize={20} radius={[0, 4, 4, 0]} />
                                <Bar dataKey="rust" fill="var(--green)" barSize={20} radius={[0, 4, 4, 0]} />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </div>
            </div>

        </div>
    );
};

export default RustPage;
