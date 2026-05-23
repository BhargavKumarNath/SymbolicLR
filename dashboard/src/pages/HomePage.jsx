import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import Badge from '../components/Badge';

const HomePage = () => {
    const navigate = useNavigate();
    const [tick, setTick] = useState(0);
    
    useEffect(() => {
        const id = setInterval(() => setTick(t => t + 1), 80);
        return () => clearInterval(id);
    }, []);

    const formulas = [
        "0.067 · cos(πt)",
        "η₀ · e^(−λt) + ε",
        "sin(t)/(1+t²)",
        "√(1−t) · α",
        "0.1 · (1 + cos(πt))/2",
        "exp(−5t) · sin(2πt) + ε",
        "t⁻⁰·⁵ · η_max",
    ];

    return (
        <div style={{ minHeight: "100vh", display: "flex", flexDirection: "column" }}>
            {/* Hero */}
            <div style={{
                minHeight: "100vh", display: "flex", flexDirection: "column",
                justifyContent: "center", padding: "0 60px", position: "relative"
            }}>
                {/* Floating formula ticker */}
                <div style={{
                    position: "absolute", top: 80, left: 0, right: 0,
                    overflow: "hidden", opacity: 0.3, pointerEvents: "none"
                }}>
                    <div style={{
                        display: "flex", gap: 60, whiteSpace: "nowrap",
                        animation: "ticker 20s linear infinite"
                    }}>
                        {[...formulas, ...formulas].map((f, i) => (
                            <span key={i} style={{
                                fontFamily: "var(--mono)", fontSize: 13, color: "var(--blue)"
                            }}>{f}</span>
                        ))}
                    </div>
                </div>

                {/* Big title */}
                <div style={{ animation: "fadeUp 0.7s ease both" }}>
                    <div style={{
                        display: "inline-flex", alignItems: "center", gap: 8,
                        padding: "6px 16px", borderRadius: 20, marginBottom: 28,
                        background: "rgba(10,132,255,0.1)", border: "1px solid rgba(10,132,255,0.25)",
                        color: "var(--blue)", fontSize: 12, fontWeight: 600, letterSpacing: "0.1em"
                    }}>
                        <span style={{ animation: "pulse 2s ease infinite" }}>●</span>
                        QUALITY-DIVERSITY GENETIC PROGRAMMING
                    </div>

                    <h1 style={{
                        fontSize: "clamp(56px, 8vw, 100px)", fontWeight: 800,
                        letterSpacing: "-0.05em", lineHeight: 0.95, marginBottom: 28,
                        background: "linear-gradient(135deg, #ffffff 0%, rgba(255,255,255,0.4) 60%)",
                        WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent"
                    }}>
                        Symbo<span style={{
                            background: "linear-gradient(135deg, #0a84ff, #bf5af2)",
                            WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent"
                        }}>LR</span>
                    </h1>

                    <p style={{
                        fontSize: 20, color: "var(--text-2)", maxWidth: 560,
                        lineHeight: 1.65, marginBottom: 40, fontWeight: 300
                    }}>
                        An autonomous system that evolves symbolic mathematical learning rate schedules
                        using Genetic Programming and MAP-Elites quality-diversity search.
                    </p>

                    <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
                        <button
                            onClick={() => navigate("/playground")}
                            style={{
                                padding: "14px 28px", borderRadius: 12,
                                background: "var(--blue)", color: "#fff",
                                fontSize: 14, fontWeight: 600,
                                transition: "all var(--transition)",
                                boxShadow: "0 0 30px rgba(10,132,255,0.4)"
                            }}
                            onMouseEnter={e => e.currentTarget.style.transform = "scale(1.02)"}
                            onMouseLeave={e => e.currentTarget.style.transform = "scale(1)"}
                        >Try the Formula Lab →</button>
                        <button
                            onClick={() => navigate("/about")}
                            style={{
                                padding: "14px 28px", borderRadius: 12,
                                background: "var(--bg2)", border: "1px solid var(--border)",
                                color: "var(--text)", fontSize: 14, fontWeight: 500,
                                transition: "all var(--transition)"
                            }}
                            onMouseEnter={e => { e.currentTarget.style.background = "var(--bg3)"; }}
                            onMouseLeave={e => { e.currentTarget.style.background = "var(--bg2)"; }}
                        >Learn More</button>
                    </div>
                </div>

                {/* Stats row */}
                <div style={{
                    display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 16,
                    marginTop: 80, animation: "fadeUp 0.9s ease both"
                }}>
                    {[
                        { label: "Operators", value: "10", sub: "Math primitives" },
                        { label: "Archive Niches", value: "6,000", sub: "Behavioral cells" },
                        { label: "Rust Speedup", value: "50×", sub: "vs Python" },
                        { label: "Baseline Beat", value: "7/7", sub: "Schedules defeated" },
                    ].map(({ label, value, sub }, i) => (
                        <div key={i} className="glass" style={{
                            padding: "20px", animation: `fadeUp ${0.9 + i * 0.08}s ease both`
                        }}>
                            <div style={{ fontSize: 28, fontWeight: 700, letterSpacing: "-0.04em", color: "var(--blue)" }}>{value}</div>
                            <div style={{ fontSize: 13, fontWeight: 600, marginTop: 6 }}>{label}</div>
                            <div style={{ fontSize: 11, color: "var(--text-3)", marginTop: 2 }}>{sub}</div>
                        </div>
                    ))}
                </div>
            </div>

            {/* Tech stack strip */}
            <div style={{
                padding: "24px 60px",
                borderTop: "1px solid var(--border)",
                display: "flex", alignItems: "center", gap: 24, flexWrap: "wrap"
            }}>
                <span style={{ fontSize: 11, color: "var(--text-3)", letterSpacing: "0.08em" }}>BUILT WITH</span>
                {["Python 3.10+", "PyTorch 2.1+", "Rust + PyO3", "SymPy", "SciPy", "MAP-Elites", "React + Recharts"].map(t => (
                    <Badge key={t}>{t}</Badge>
                ))}
            </div>
        </div>
    );
};

export default HomePage;
