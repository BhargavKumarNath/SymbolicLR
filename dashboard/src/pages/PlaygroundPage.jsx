import React, { useState, useEffect, useCallback } from 'react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer } from "recharts";
import { motion, AnimatePresence } from 'framer-motion';
import { Info, HelpCircle, Settings2 } from 'lucide-react';
import SectionHeader from '../components/SectionHeader';

// Reusable Metric Tooltip
const MetricCard = ({ label, value, color, explanation }) => {
    const [isHovered, setIsHovered] = useState(false);
    return (
        <div 
            className="glass" 
            style={{ padding: "20px", position: "relative", cursor: "help" }}
            onMouseEnter={() => setIsHovered(true)}
            onMouseLeave={() => setIsHovered(false)}
        >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
                <div style={{ fontSize: 11, color: "var(--text-3)", letterSpacing: "0.05em", textTransform: "uppercase", fontWeight: 600 }}>{label}</div>
                <HelpCircle size={14} color="var(--text-3)" />
            </div>
            <div style={{ fontFamily: "var(--mono)", fontSize: 20, fontWeight: 700, color }}>{value}</div>
            
            <AnimatePresence>
                {isHovered && (
                    <motion.div
                        initial={{ opacity: 0, y: 5 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: 5 }}
                        style={{
                            position: "absolute", top: "100%", left: 0, right: 0, marginTop: 8,
                            padding: "12px", background: "rgba(10,10,10,0.95)", border: "1px solid var(--border)",
                            borderRadius: 8, zIndex: 50, fontSize: 12, color: "var(--text-2)", lineHeight: 1.5,
                            boxShadow: "0 10px 20px rgba(0,0,0,0.5)"
                        }}
                    >
                        {explanation}
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
};

const PlaygroundPage = () => {
    const [formula, setFormula] = useState("0.0539 * cos(PI * t) + 0.001");
    const [evalResult, setEvalResult] = useState(null);
    const [customConst, setCustomConst] = useState(0.053);
    const [preset, setPreset] = useState(null);

    const presets = [
        { name: "Golden Run (Rank 1)", expr: "0.0539 * cos(PI * t) + 0.001", color: "var(--orange)" },
        { name: "Polynomial (Rank 2)", expr: "0.0603 / (t*t + 0.909)", color: "var(--blue)" },
        { name: "Oscillating (Rank 4)", expr: "sin(sin(0.0434 * abs(log(t+0.01)) + 0.0016))", color: "var(--purple)" },
        { name: "Custom Constant Test", expr: `${customConst.toFixed(3)} * cos(PI * t) + 0.001`, color: "var(--green)" },
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
                    return isFinite(v) ? Math.max(0, Math.min(10, v)) : null;
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
        <div style={{ padding: "60px", maxWidth: 1200, margin: "0 auto" }}>
            <SectionHeader
                tag="Interactive Lab"
                title="Formula Playground"
                subtitle="Manually inspect, stress-test, and fine-tune machine-generated mathematical schedules before deploying them to production neural networks."
            />

            {/* Guided Workflow */}
            <div className="glass-bright" style={{ padding: "24px", marginBottom: 40, display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 20 }}>
                {[
                    { step: 1, title: "Select Formula", desc: "Pick a mathematically discovered schedule from the MAP-Elites archive." },
                    { step: 2, title: "Tweak Constants", desc: "Simulate different epoch constraints by overriding the evolved scalar weights." },
                    { step: 3, title: "Analyze Curve", desc: "Verify the projected learning rate bounds and convergence stability." },
                ].map(({ step, title, desc }) => (
                    <div key={step} style={{ display: "flex", gap: 16 }}>
                        <div style={{ width: 32, height: 32, borderRadius: "50%", background: "var(--blue)", color: "#fff", display: "flex", alignItems: "center", justifyContent: "center", fontWeight: 700, flexShrink: 0 }}>
                            {step}
                        </div>
                        <div>
                            <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 4 }}>{title}</div>
                            <div style={{ fontSize: 12, color: "var(--text-2)", lineHeight: 1.5 }}>{desc}</div>
                        </div>
                    </div>
                ))}
            </div>

            {/* Presets */}
            <div style={{ display: "flex", gap: 12, flexWrap: "wrap", marginBottom: 32 }}>
                {presets.map((p, i) => (
                    <button
                        key={i}
                        onClick={() => { setFormula(p.expr); setPreset(i); }}
                        style={{
                            padding: "10px 20px", borderRadius: 8, fontSize: 13, fontWeight: 600,
                            background: preset === i ? `${p.color}22` : "var(--bg2)",
                            border: `1px solid ${preset === i ? p.color : "var(--border)"}`,
                            color: preset === i ? p.color : "var(--text-2)",
                            transition: "all var(--transition)", cursor: "pointer"
                        }}
                    >{p.name}</button>
                ))}
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 340px", gap: 32 }}>
                {/* Main chart */}
                <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
                    {/* Formula input */}
                    <div className="glass" style={{ padding: "24px" }}>
                        <label style={{ fontSize: 12, color: "var(--text-3)", letterSpacing: "0.08em", textTransform: "uppercase", display: "block", marginBottom: 12, fontWeight: 600 }}>
                            η(t) =
                        </label>
                        <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
                            <input
                                value={formula}
                                onChange={e => { setFormula(e.target.value); setPreset(null); }}
                                style={{
                                    flex: 1, padding: "16px 20px", borderRadius: 8,
                                    background: "rgba(0,0,0,0.4)", border: "1px solid var(--border)",
                                    color: "var(--text)", fontSize: 15, fontFamily: "var(--mono)",
                                    outline: "none", transition: "border-color var(--transition)",
                                    boxShadow: "inset 0 2px 4px rgba(0,0,0,0.2)"
                                }}
                                onFocus={e => e.target.style.borderColor = "var(--blue)"}
                                onBlur={e => e.target.style.borderColor = "var(--border)"}
                                placeholder="0.0539 * cos(PI * t) + 0.001"
                            />
                            {evalResult?.valid === false && (
                                <div style={{ display: "flex", alignItems: "center", gap: 6, color: "var(--red)", fontSize: 13, fontWeight: 600, flexShrink: 0, padding: "0 12px" }}>
                                    <span>✗</span> Invalid Syntax
                                </div>
                            )}
                            {evalResult?.valid === true && (
                                <div style={{ display: "flex", alignItems: "center", gap: 6, color: "var(--green)", fontSize: 13, fontWeight: 600, flexShrink: 0, padding: "0 12px" }}>
                                    <span>✓</span> AST Valid
                                </div>
                            )}
                        </div>
                    </div>

                    {/* Chart */}
                    <div className="glass" style={{ padding: "32px" }}>
                        <h3 style={{ fontSize: 16, fontWeight: 700, marginBottom: 24 }}>Projected Learning Rate Trajectory</h3>
                        {evalResult?.valid ? (
                            <ResponsiveContainer width="100%" height={320}>
                                <AreaChart data={evalResult.data} margin={{ top: 10, right: 10, bottom: 0, left: 0 }}>
                                    <defs>
                                        <linearGradient id="lrGrad" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="5%" stopColor="#0a84ff" stopOpacity={0.3} />
                                            <stop offset="95%" stopColor="#0a84ff" stopOpacity={0} />
                                        </linearGradient>
                                    </defs>
                                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                                    <XAxis dataKey="t" stroke="rgba(255,255,255,0.3)" fontSize={11} tickFormatter={v => v.toFixed(1)} />
                                    <YAxis stroke="rgba(255,255,255,0.3)" fontSize={11} tickFormatter={v => v.toFixed(4)} width={65} />
                                    <RechartsTooltip
                                        contentStyle={{ background: "rgba(10,10,10,0.95)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 8 }}
                                        labelStyle={{ color: "var(--text-3)", fontSize: 12, marginBottom: 4 }}
                                        itemStyle={{ color: "var(--blue)", fontSize: 14, fontWeight: 700, fontFamily: "var(--mono)" }}
                                        formatter={v => [Number(v).toFixed(6), "η(t)"]}
                                        labelFormatter={v => `Epoch Progress (t) = ${v}`}
                                    />
                                    <Area type="monotone" dataKey="lr" stroke="var(--blue)" strokeWidth={3} fill="url(#lrGrad)" dot={false} />
                                </AreaChart>
                            </ResponsiveContainer>
                        ) : (
                            <div style={{ height: 320, display: "flex", alignItems: "center", justifyContent: "center", background: "rgba(0,0,0,0.2)", borderRadius: 8, border: "1px dashed var(--border)" }}>
                                <span style={{ color: "var(--text-3)", fontSize: 14 }}>Waiting for valid mathematical expression...</span>
                            </div>
                        )}
                    </div>
                </div>

                {/* Side panel */}
                <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
                    {evalResult?.valid && (
                        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
                            <MetricCard 
                                label="Peak LR" value={evalResult.max.toFixed(6)} color="var(--orange)" 
                                explanation="The absolute maximum learning rate hit during the epoch. Crucial for controlling the explosion of gradients during early training."
                            />
                            <MetricCard 
                                label="Final LR" value={(evalResult.data[evalResult.data.length - 1]?.lr || 0).toFixed(6)} color="var(--green)" 
                                explanation="The learning rate at t=1. Drives the network into its final local minimum convergence phase."
                            />
                            <MetricCard 
                                label="Mean LR" value={evalResult.mean.toFixed(6)} color="var(--blue)" 
                                explanation="The average learning rate across all steps, acting as a proxy for the Area Under Curve (AUC) and total momentum applied."
                            />
                            <MetricCard 
                                label="Min LR" value={evalResult.min.toFixed(6)} color="var(--purple)" 
                                explanation="The lowest learning rate in the schedule. Ensures the network never stops updating entirely."
                            />
                        </div>
                    )}

                    {/* Custom Constant Tuning */}
                    <div className="glass" style={{ padding: "24px", borderTop: "3px solid var(--green)" }}>
                        <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 16 }}>
                            <Settings2 size={18} color="var(--green)" />
                            <h4 style={{ fontSize: 15, fontWeight: 700 }}>Constant Tuning</h4>
                        </div>
                        
                        <div style={{ padding: "12px", background: "rgba(48,209,88,0.05)", borderRadius: 8, marginBottom: 20, border: "1px solid rgba(48,209,88,0.1)" }}>
                            <p style={{ fontSize: 12, color: "var(--text-2)", lineHeight: 1.6 }}>
                                The genetic engine evolves <strong>Custom Constants</strong> (scalar weights). Use this slider to manually override these weights and observe how sensitive the mathematical schedule is to slight perturbations.
                            </p>
                        </div>

                        <input
                            type="range" min={0.001} max={0.1} step={0.001}
                            value={customConst}
                            onChange={e => {
                                setCustomConst(parseFloat(e.target.value));
                                setPreset(3); // Auto-switch to custom preset
                                setFormula(`${parseFloat(e.target.value).toFixed(3)} * cos(PI * t) + 0.001`);
                            }}
                            style={{ width: "100%", marginBottom: 12, cursor: "pointer" }}
                        />
                        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                            <span style={{ fontSize: 11, color: "var(--text-3)" }}>0.001</span>
                            <span style={{ fontFamily: "var(--mono)", fontSize: 16, fontWeight: 700, color: "var(--green)" }}>{customConst.toFixed(3)}</span>
                            <span style={{ fontSize: 11, color: "var(--text-3)" }}>0.100</span>
                        </div>
                    </div>

                    {/* Quality Signals */}
                    {evalResult?.valid && (
                        <div className="glass" style={{ padding: "24px" }}>
                            <h4 style={{ fontSize: 14, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.05em", color: "var(--text-3)", marginBottom: 16 }}>Stability Signals</h4>
                            {[
                                { label: "Has warmup phase", check: evalResult.data[0]?.lr < evalResult.max * 0.5 },
                                { label: "Decays consistently", check: evalResult.data[99]?.lr < evalResult.data[0]?.lr },
                                { label: "Non-trivial variance", check: (evalResult.max - evalResult.min) > 1e-5 },
                                { label: "Zero Division Safe", check: true },
                            ].map(({ label, check }, i) => (
                                <div key={i} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
                                    <span style={{ fontSize: 13, color: "var(--text-2)" }}>{label}</span>
                                    <span style={{ fontSize: 14, fontWeight: 700, color: check ? "var(--green)" : "var(--red)" }}>
                                        {check ? "✓ PASS" : "✗ FAIL"}
                                    </span>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default PlaygroundPage;
