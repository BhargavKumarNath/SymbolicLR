import React, { useState, useEffect, useCallback } from 'react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import SectionHeader from '../components/SectionHeader';

const PlaygroundPage = () => {
    const [formula, setFormula] = useState("0.5 * (1 + cos(PI * t))");
    const [evalResult, setEvalResult] = useState(null);
    const [customConst, setCustomConst] = useState(0.5);
    const [preset, setPreset] = useState(null);

    const presets = [
        { name: "Cosine Annealing", expr: "0.01 * (1 + cos(PI * t)) / 2", color: "var(--blue)" },
        { name: "Warmup + Decay", expr: "t < 0.1 ? 10*t*0.01 : 0.01*(1-t)/0.9", color: "var(--orange)" },
        { name: "Exponential", expr: "0.01 * exp(-5 * t)", color: "var(--purple)" },
        { name: "Linear Decay", expr: "0.01 * (1 - t)", color: "var(--green)" },
        { name: "Oscillating", expr: "0.005 * (1 + sin(10 * PI * t))", color: "var(--teal)" },
        { name: "Step-like", expr: "t < 0.33 ? 0.01 : t < 0.66 ? 0.005 : 0.001", color: "var(--red)" },
        { name: "Warmup Cosine", expr: "t < 0.05 ? t/0.05*0.01 : 0.01*cos(PI*(t-0.05)/1.9)", color: "var(--orange)" },
        { name: "Custom Sqrt", expr: `${customConst.toFixed(3)} * sqrt(1 - t)`, color: "var(--teal)" },
    ];

    // Evaluate formula
    const evaluateFormula = useCallback((expr) => {
        const steps = 100;
        const t_array = Array.from({ length: steps }, (_, i) => i / (steps - 1));
        try {
            const safeExpr = expr
                .replace(/PI/g, Math.PI.toString())
                .replace(/cos/g, "Math.cos")
                .replace(/sin/g, "Math.sin")
                .replace(/exp/g, "Math.exp")
                .replace(/sqrt/g, "Math.sqrt")
                .replace(/log/g, "Math.log")
                .replace(/abs/g, "Math.abs");

            const fn = new Function("t", `return ${safeExpr}`);
            const values = t_array.map(t => {
                try {
                    const v = fn(t);
                    return isFinite(v) ? Math.max(1e-7, Math.min(10, v)) : null;
                } catch { return null; }
            });

            if (values.some(v => v === null)) throw new Error("Invalid expression");

            setEvalResult({
                data: t_array.map((t, i) => ({ t: parseFloat(t.toFixed(3)), lr: parseFloat((values[i] || 0).toFixed(6)) })),
                mean: values.reduce((a, b) => a + (b || 0), 0) / values.length,
                max: Math.max(...values.filter(Boolean)),
                min: Math.min(...values.filter(Boolean)),
                valid: true
            });
        } catch (e) {
            setEvalResult({ valid: false, error: e.message });
        }
    }, []);

    useEffect(() => { evaluateFormula(formula); }, [formula, evaluateFormula]);

    return (
        <div style={{ padding: "60px" }}>
            <SectionHeader
                tag="Interactive"
                title="Formula Lab"
                subtitle="Explore learning rate schedules interactively. Type any mathematical expression using t ∈ [0,1] and see the schedule instantly. Use built-in functions: cos, sin, exp, sqrt, log, abs, PI."
            />

            {/* Presets */}
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 32 }}>
                {presets.map((p, i) => (
                    <button
                        key={i}
                        onClick={() => { setFormula(p.expr); setPreset(i); }}
                        style={{
                            padding: "7px 14px", borderRadius: 8, fontSize: 12, fontWeight: 500,
                            background: preset === i ? `${p.color}22` : "var(--bg2)",
                            border: `1px solid ${preset === i ? p.color : "var(--border)"}`,
                            color: preset === i ? p.color : "var(--text-2)",
                            transition: "all var(--transition)", cursor: "pointer"
                        }}
                    >{p.name}</button>
                ))}
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 300px", gap: 24 }}>
                {/* Main chart */}
                <div>
                    {/* Formula input */}
                    <div className="glass" style={{ padding: "20px", marginBottom: 20 }}>
                        <label style={{ fontSize: 11, color: "var(--text-3)", letterSpacing: "0.08em", textTransform: "uppercase", display: "block", marginBottom: 10 }}>
                            η(t) =
                        </label>
                        <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
                            <input
                                value={formula}
                                onChange={e => { setFormula(e.target.value); setPreset(null); }}
                                style={{
                                    flex: 1, padding: "12px 16px", borderRadius: 8,
                                    background: "rgba(0,0,0,0.4)", border: "1px solid var(--border)",
                                    color: "var(--text)", fontSize: 14, fontFamily: "var(--mono)",
                                    outline: "none", transition: "border-color var(--transition)"
                                }}
                                onFocus={e => e.target.style.borderColor = "var(--blue)"}
                                onBlur={e => e.target.style.borderColor = "var(--border)"}
                                placeholder="0.01 * cos(PI * t)"
                            />
                            {evalResult?.valid === false && (
                                <span style={{ fontSize: 12, color: "var(--red)", flexShrink: 0 }}>Invalid ✗</span>
                            )}
                            {evalResult?.valid === true && (
                                <span style={{ fontSize: 12, color: "var(--green)", flexShrink: 0 }}>Valid ✓</span>
                            )}
                        </div>
                    </div>

                    {/* Chart */}
                    <div className="glass" style={{ padding: "24px" }}>
                        <h3 style={{ fontSize: 14, fontWeight: 600, marginBottom: 16 }}>Learning Rate Schedule</h3>
                        {evalResult?.valid && (
                            <ResponsiveContainer width="100%" height={280}>
                                <AreaChart data={evalResult.data} margin={{ top: 10, right: 10, bottom: 0, left: 0 }}>
                                    <defs>
                                        <linearGradient id="lrGrad" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="5%" stopColor="#0a84ff" stopOpacity={0.3} />
                                            <stop offset="95%" stopColor="#0a84ff" stopOpacity={0} />
                                        </linearGradient>
                                    </defs>
                                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                                    <XAxis dataKey="t" stroke="rgba(255,255,255,0.3)" fontSize={10} tickFormatter={v => v.toFixed(1)} />
                                    <YAxis stroke="rgba(255,255,255,0.3)" fontSize={10} tickFormatter={v => v.toFixed(4)} width={60} />
                                    <Tooltip
                                        contentStyle={{ background: "rgba(0,0,0,0.9)", border: "1px solid rgba(255,255,255,0.15)", borderRadius: 8 }}
                                        labelStyle={{ color: "rgba(255,255,255,0.5)", fontSize: 11 }}
                                        itemStyle={{ color: "#0a84ff", fontSize: 12 }}
                                        formatter={v => [Number(v).toFixed(6), "η(t)"]}
                                        labelFormatter={v => `t = ${v}`}
                                    />
                                    <Area type="monotone" dataKey="lr" stroke="#0a84ff" strokeWidth={2} fill="url(#lrGrad)" dot={false} />
                                </AreaChart>
                            </ResponsiveContainer>
                        )}
                        {evalResult?.valid === false && (
                            <div style={{ height: 280, display: "flex", alignItems: "center", justifyContent: "center", color: "var(--red)", fontSize: 14 }}>
                                ✗ Invalid formula — check syntax
                            </div>
                        )}
                    </div>
                </div>

                {/* Side metrics */}
                <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
                    {evalResult?.valid && (
                        <>
                            {[
                                { label: "Mean LR", value: evalResult.mean.toFixed(6), color: "var(--blue)" },
                                { label: "Peak LR", value: evalResult.max.toFixed(6), color: "var(--orange)" },
                                { label: "Final LR", value: (evalResult.data[evalResult.data.length - 1]?.lr || 0).toFixed(6), color: "var(--green)" },
                                { label: "Min LR", value: evalResult.min.toFixed(6), color: "var(--red)" },
                            ].map(({ label, value, color }, i) => (
                                <div key={i} className="glass" style={{ padding: "18px 20px" }}>
                                    <div style={{ fontSize: 11, color: "var(--text-3)", letterSpacing: "0.06em", textTransform: "uppercase", marginBottom: 6 }}>{label}</div>
                                    <div style={{ fontFamily: "var(--mono)", fontSize: 18, fontWeight: 700, color }}>{value}</div>
                                </div>
                            ))}

                            {/* Schedule quality score */}
                            <div className="glass" style={{ padding: "18px 20px" }}>
                                <div style={{ fontSize: 11, color: "var(--text-3)", letterSpacing: "0.06em", textTransform: "uppercase", marginBottom: 12 }}>Quality Signals</div>
                                {[
                                    { label: "Has warmup", check: evalResult.data[0]?.lr < evalResult.max * 0.5 },
                                    { label: "Decays over time", check: evalResult.data[99]?.lr < evalResult.data[0]?.lr },
                                    { label: "Non-trivial variance", check: (evalResult.max - evalResult.min) > 1e-5 },
                                    { label: "Stable (no NaN)", check: true },
                                ].map(({ label, check }, i) => (
                                    <div key={i} style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
                                        <span style={{ fontSize: 12, color: "var(--text-2)" }}>{label}</span>
                                        <span style={{ fontSize: 12, color: check ? "var(--green)" : "var(--red)" }}>
                                            {check ? "✓" : "✗"}
                                        </span>
                                    </div>
                                ))}
                            </div>
                        </>
                    )}

                    {/* Custom constant slider */}
                    <div className="glass" style={{ padding: "18px 20px" }}>
                        <div style={{ fontSize: 11, color: "var(--text-3)", letterSpacing: "0.06em", textTransform: "uppercase", marginBottom: 12 }}>Custom Constant</div>
                        <input
                            type="range" min={0.001} max={1} step={0.001}
                            value={customConst}
                            onChange={e => setCustomConst(parseFloat(e.target.value))}
                            style={{ width: "100%", marginBottom: 8 }}
                        />
                        <div style={{ display: "flex", justifyContent: "space-between" }}>
                            <span style={{ fontSize: 11, color: "var(--text-3)" }}>0.001</span>
                            <span style={{ fontFamily: "var(--mono)", fontSize: 13, color: "var(--blue)" }}>{customConst.toFixed(3)}</span>
                            <span style={{ fontSize: 11, color: "var(--text-3)" }}>1.000</span>
                        </div>
                        <button
                            onClick={() => setFormula(`${customConst.toFixed(3)} * sqrt(1 - t)`)}
                            style={{
                                width: "100%", marginTop: 12, padding: "8px", borderRadius: 6,
                                background: "var(--bg2)", border: "1px solid var(--border)",
                                color: "var(--text-2)", fontSize: 12, cursor: "pointer"
                            }}
                        >Apply to Sqrt Preset</button>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default PlaygroundPage;
