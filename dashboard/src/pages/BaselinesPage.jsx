import React, { useState } from 'react';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import SectionHeader from '../components/SectionHeader';
import ChartCard from '../components/ChartCard';
import Badge from '../components/Badge';

const BaselinesPage = () => {
    const t = Array.from({ length: 100 }, (_, i) => i / 99);

    const schedules = {
        "Cosine Annealing": t.map(t => 1e-5 + 0.5 * (0.01 - 1e-5) * (1 + Math.cos(Math.PI * t))),
        "Step Decay": t.map(t => 0.01 * Math.pow(0.5, Math.floor(t / (1 / 3)))),
        "Warm Restarts": t.map(t => 1e-5 + 0.5 * (0.01 - 1e-5) * (1 + Math.cos(Math.PI * ((t * 3) % 1)))),
        "Linear Decay": t.map(t => 0.01 + (1e-5 - 0.01) * t),
        "1-Cycle": t.map(t => t < 0.3 ? 1e-5 + (0.01 - 1e-5) * (t / 0.3) : 1e-5 + 0.5 * (0.01 - 1e-5) * (1 + Math.cos(Math.PI * (t - 0.3) / 0.7))),
        "Exponential": t.map(t => 0.01 * Math.exp(-5 * t)),
        "SymboLR Elite": t.map(t => 0.067 * Math.cos(Math.PI * t) * Math.exp(-t)),
    };

    const colors = {
        "Cosine Annealing": "#0a84ff",
        "Step Decay": "#ff9f0a",
        "Warm Restarts": "#30d158",
        "Linear Decay": "#bf5af2",
        "1-Cycle": "#5ac8fa",
        "Exponential": "#ff453a",
        "SymboLR Elite": "#ffd60a",
    };

    const losses = {
        "Cosine Annealing": 0.212, "Step Decay": 0.284, "Warm Restarts": 0.241,
        "Linear Decay": 0.319, "1-Cycle": 0.198, "Exponential": 0.267,
        "SymboLR Elite": 0.127,
    };

    const [visibleSchedules, setVisibleSchedules] = useState(
        Object.fromEntries(Object.keys(schedules).map(k => [k, true]))
    );

    const chartData = t.map((tVal, i) => {
        const row = { t: parseFloat(tVal.toFixed(2)) };
        Object.keys(schedules).forEach(name => {
            if (visibleSchedules[name]) row[name] = parseFloat((schedules[name][i] || 0).toFixed(6));
        });
        return row;
    });

    const barData = Object.entries(losses)
        .map(([name, loss]) => ({ name: name === "SymboLR Elite" ? "SymboLR" : name.replace(" ", "\n"), loss, color: colors[name] }))
        .sort((a, b) => b.loss - a.loss);

    return (
        <div style={{ padding: "60px" }}>
            <SectionHeader
                tag="Benchmark"
                title="vs. Hand-crafted Baselines"
                subtitle="SymboLR discovered formulas are evaluated under identical conditions to all baseline schedules — same fitness function, same probe model, same dataset."
            />

            {/* Toggle pills */}
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 24 }}>
                {Object.keys(schedules).map(name => (
                    <button
                        key={name}
                        onClick={() => setVisibleSchedules(v => ({ ...v, [name]: !v[name] }))}
                        style={{
                            padding: "6px 14px", borderRadius: 20, fontSize: 12, fontWeight: 500,
                            background: visibleSchedules[name] ? `${colors[name]}22` : "var(--bg2)",
                            border: `1px solid ${visibleSchedules[name] ? colors[name] : "var(--border)"}`,
                            color: visibleSchedules[name] ? colors[name] : "var(--text-3)",
                            transition: "all var(--transition)", cursor: "pointer",
                            position: "relative"
                        }}
                    >
                        {name}
                        {name === "SymboLR Elite" && (
                            <span style={{
                                position: "absolute", top: -8, right: -8,
                                background: "var(--orange)", color: "#000",
                                fontSize: 8, fontWeight: 700, padding: "1px 5px", borderRadius: 10
                            }}>🏆</span>
                        )}
                    </button>
                ))}
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: 24, marginBottom: 32 }}>
                {/* Schedule curves */}
                <ChartCard title="Schedule Comparison" subtitle="Learning rate over normalized training time t ∈ [0,1]">
                    <ResponsiveContainer width="100%" height={300}>
                        <LineChart data={chartData} margin={{ top: 10, right: 10, bottom: 0, left: 0 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                            <XAxis dataKey="t" stroke="rgba(255,255,255,0.3)" fontSize={10} />
                            <YAxis stroke="rgba(255,255,255,0.3)" fontSize={10} tickFormatter={v => v.toFixed(3)} width={55} />
                            <Tooltip
                                contentStyle={{ background: "rgba(0,0,0,0.9)", border: "1px solid rgba(255,255,255,0.15)", borderRadius: 8 }}
                                itemStyle={{ fontSize: 11 }}
                                formatter={(v, name) => [Number(v)?.toFixed(6), name]}
                            />
                            {Object.keys(schedules).map(name => visibleSchedules[name] && (
                                <Line key={name} type="monotone" dataKey={name}
                                    stroke={colors[name]}
                                    strokeWidth={name === "SymboLR Elite" ? 2.5 : 1.5}
                                    dot={false} isAnimationActive={false}
                                    strokeDasharray={name === "SymboLR Elite" ? "none" : "none"}
                                />
                            ))}
                        </LineChart>
                    </ResponsiveContainer>
                </ChartCard>

                {/* Validation loss bars */}
                <ChartCard title="Validation Loss" subtitle="Lower is better">
                    <ResponsiveContainer width="100%" height={300}>
                        <BarChart data={barData} layout="vertical" margin={{ top: 0, right: 20, bottom: 0, left: 60 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" horizontal={false} />
                            <XAxis type="number" stroke="rgba(255,255,255,0.3)" fontSize={10} domain={[0, 0.35]} tickFormatter={v => v.toFixed(2)} />
                            <YAxis type="category" dataKey="name" stroke="rgba(255,255,255,0.3)" fontSize={10} width={58} />
                            <Tooltip
                                contentStyle={{ background: "rgba(0,0,0,0.9)", border: "1px solid rgba(255,255,255,0.15)", borderRadius: 8 }}
                                formatter={v => [Number(v).toFixed(4), "Val Loss"]}
                            />
                            <Bar dataKey="loss" radius={[0, 4, 4, 0]}>
                                {barData.map((entry, i) => (
                                    <rect key={i} fill={entry.color} />
                                ))}
                            </Bar>
                        </BarChart>
                    </ResponsiveContainer>
                </ChartCard>
            </div>

            {/* Results table */}
            <div className="glass" style={{ padding: "0", overflow: "hidden" }}>
                <table style={{ width: "100%", borderCollapse: "collapse" }}>
                    <thead>
                        <tr style={{ borderBottom: "1px solid var(--border)", background: "var(--bg2)" }}>
                            {["Rank", "Schedule", "Type", "Val Loss", "vs Best Baseline"].map(h => (
                                <th key={h} style={{ padding: "14px 20px", textAlign: "left", fontSize: 11, color: "var(--text-3)", letterSpacing: "0.06em", textTransform: "uppercase", fontWeight: 600 }}>{h}</th>
                            ))}
                        </tr>
                    </thead>
                    <tbody>
                        {Object.entries(losses)
                            .sort(([, a], [, b]) => a - b)
                            .map(([name, loss], i) => {
                                const isElite = name === "SymboLR Elite";
                                const bestBaseline = Math.min(...Object.entries(losses).filter(([n]) => n !== "SymboLR Elite").map(([, l]) => l));
                                const diff = isElite ? ((bestBaseline - loss) / bestBaseline * 100).toFixed(1) : null;
                                return (
                                    <tr key={name} style={{
                                        borderBottom: "1px solid var(--border)",
                                        background: isElite ? "rgba(255,214,10,0.05)" : "transparent"
                                    }}>
                                        <td style={{ padding: "14px 20px", fontSize: 13 }}>
                                            <span style={{
                                                width: 24, height: 24, borderRadius: 6,
                                                background: isElite ? "var(--orange-dim)" : "var(--bg2)",
                                                border: `1px solid ${isElite ? "rgba(255,159,10,0.3)" : "var(--border)"}`,
                                                display: "inline-flex", alignItems: "center", justifyContent: "center",
                                                fontSize: 11, fontWeight: 700, color: isElite ? "var(--orange)" : "var(--text-3)"
                                            }}>{i + 1}</span>
                                        </td>
                                        <td style={{ padding: "14px 20px" }}>
                                            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                                                <div style={{ width: 10, height: 10, borderRadius: 2, background: colors[name] }} />
                                                <span style={{ fontSize: 13, fontWeight: isElite ? 600 : 400 }}>{name}</span>
                                            </div>
                                        </td>
                                        <td style={{ padding: "14px 20px" }}>
                                            <Badge color={isElite ? "var(--orange)" : "var(--blue)"}>{isElite ? "Discovered" : "Hand-crafted"}</Badge>
                                        </td>
                                        <td style={{ padding: "14px 20px", fontFamily: "var(--mono)", fontSize: 13, color: isElite ? "var(--orange)" : "var(--text)" }}>
                                            {loss.toFixed(4)}
                                        </td>
                                        <td style={{ padding: "14px 20px" }}>
                                            {isElite ? (
                                                <span style={{ color: "var(--green)", fontSize: 13, fontWeight: 600 }}>−{diff}% improvement</span>
                                            ) : (
                                                <span style={{ color: "var(--text-3)", fontSize: 13 }}>—</span>
                                            )}
                                        </td>
                                    </tr>
                                );
                            })}
                    </tbody>
                </table>
            </div>
        </div>
    );
};

export default BaselinesPage;
