import React from 'react';
import SectionHeader from '../components/SectionHeader';
import Badge from '../components/Badge';

const AboutPage = () => {
    const problems = [
        {
            icon: "◎",
            title: "The Search Space is Vast",
            body: "Every possible combination of math operations, constants, and the time variable 't' is a candidate schedule. There are infinitely many. No human can explore this space manually.",
            color: "var(--orange)"
        },
        {
            icon: "◈",
            title: "Hand-crafted Schedules Are Guesses",
            body: "Cosine annealing, step decay, 1-cycle — these were invented by researchers based on intuition and empirical testing. They're good, but they may not be optimal for your specific dataset or architecture.",
            color: "var(--purple)"
        },
        {
            icon: "⬡",
            title: "Standard AutoML Doesn't Apply",
            body: "Traditional hyperparameter search (Bayesian optimization, grid search) works on fixed parametric forms. It cannot discover entirely new mathematical structures.",
            color: "var(--blue)"
        },
        {
            icon: "◉",
            title: "Diversity Gets Trapped",
            body: "Single-objective evolutionary methods collapse to one solution. But many structurally different formulas can achieve the same loss — the archive should contain all of them.",
            color: "var(--green)"
        }
    ];

    return (
        <div style={{ padding: "60px" }}>
            <SectionHeader
                tag="The Problem"
                title="Why Symbolic LR Discovery?"
                subtitle="The learning rate schedule is one of the most impactful hyperparameters in neural network training — yet we've barely scratched the surface of what's mathematically possible."
            />

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 48 }}>
                {problems.map((p, i) => (
                    <div key={i} className="glass" style={{
                        padding: "28px", animation: `fadeUp ${0.3 + i * 0.1}s ease both`,
                        borderColor: `${p.color}33`
                    }}>
                        <div style={{ fontSize: 28, marginBottom: 16 }}>{p.icon}</div>
                        <h3 style={{ fontSize: 16, fontWeight: 600, marginBottom: 10, color: p.color }}>{p.title}</h3>
                        <p style={{ fontSize: 13, color: "var(--text-2)", lineHeight: 1.7 }}>{p.body}</p>
                    </div>
                ))}
            </div>

            {/* The Solution */}
            <div className="glass-bright" style={{ padding: "40px", marginBottom: 48, animation: "fadeUp 0.7s ease both" }}>
                <Badge color="var(--green)">The Solution</Badge>
                <h2 style={{ fontSize: 28, fontWeight: 700, letterSpacing: "-0.03em", marginTop: 16, marginBottom: 16 }}>
                    Genetic Programming + Quality-Diversity
                </h2>
                <p style={{ fontSize: 15, color: "var(--text-2)", lineHeight: 1.8, maxWidth: 700 }}>
                    SymboLR treats the learning rate schedule <code style={{ fontFamily: "var(--mono)", color: "var(--blue)", fontSize: 13 }}>η(t)</code> not as a parametric family,
                    but as an <em style={{ color: "var(--text)" }}>unknown mathematical expression</em> built from primitives.
                    It uses Genetic Programming to evolve ASTs (Abstract Syntax Trees) and MAP-Elites
                    to maintain a behaviorally diverse archive of solutions — ensuring we discover
                    many structurally different schedules, not just local optima.
                </p>
                <div style={{ display: "flex", gap: 12, marginTop: 24, flexWrap: "wrap" }}>
                    {["Structural search over expression space", "Real GPU fitness via probe training", "Rust-accelerated evaluation", "SymPy algebraic simplification", "L-BFGS-B constant refinement"].map(f => (
                        <div key={f} style={{
                            padding: "6px 14px", borderRadius: 8,
                            background: "rgba(48,209,88,0.1)", border: "1px solid rgba(48,209,88,0.2)",
                            color: "var(--green)", fontSize: 12, fontWeight: 500
                        }}>✓ {f}</div>
                    ))}
                </div>
            </div>

            {/* Proxy-to-Production pipeline */}
            <div style={{ animation: "fadeUp 0.8s ease both" }}>
                <h2 style={{ fontSize: 22, fontWeight: 700, marginBottom: 24, letterSpacing: "-0.02em" }}>
                    The Proxy-to-Production Pipeline
                </h2>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24 }}>
                    <div className="glass" style={{ padding: "32px" }}>
                        <div style={{
                            width: 36, height: 36, borderRadius: "50%",
                            background: "var(--blue-dim)", border: "1px solid rgba(10,132,255,0.3)",
                            display: "flex", alignItems: "center", justifyContent: "center",
                            fontSize: 16, marginBottom: 16
                        }}>1</div>
                        <h3 style={{ fontSize: 16, fontWeight: 600, marginBottom: 10 }}>Proxy Search Phase</h3>
                        <p style={{ fontSize: 13, color: "var(--text-2)", lineHeight: 1.7 }}>
                            Run SymboLR overnight on a lightweight proxy task — 5% of MNIST on a tiny FastConvNet.
                            Tens of thousands of formulas are evaluated via real GPU training at minimal cost.
                        </p>
                        <div style={{ marginTop: 16, padding: "12px 16px", borderRadius: 8, background: "rgba(0,0,0,0.4)", border: "1px solid var(--border)" }}>
                            <code style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--blue)" }}>
                                symbolr benchmark --generations 30 --pop-size 100 --epochs 5
                            </code>
                        </div>
                    </div>
                    <div className="glass" style={{ padding: "32px" }}>
                        <div style={{
                            width: 36, height: 36, borderRadius: "50%",
                            background: "var(--purple-dim)", border: "1px solid rgba(191,90,242,0.3)",
                            display: "flex", alignItems: "center", justifyContent: "center",
                            fontSize: 16, marginBottom: 16
                        }}>2</div>
                        <h3 style={{ fontSize: 16, fontWeight: 600, marginBottom: 10 }}>Production Deployment</h3>
                        <p style={{ fontSize: 13, color: "var(--text-2)", lineHeight: 1.7 }}>
                            Extract the winning formula from the Hall of Fame. Drop it directly into your production training loop.
                            Because <code style={{ fontFamily: "var(--mono)", color: "var(--purple)", fontSize: 11 }}>t ∈ [0,1]</code>, deployment is trivial.
                        </p>
                        <div style={{ marginTop: 16, padding: "12px 16px", borderRadius: 8, background: "rgba(0,0,0,0.4)", border: "1px solid var(--border)" }}>
                            <code style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--purple)" }}>
                                {`lr = 0.067 * math.cos(t)  # Discovered`}
                            </code>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default AboutPage;
