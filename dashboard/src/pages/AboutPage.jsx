import React from 'react';
import SectionHeader from '../components/SectionHeader';
import Badge from '../components/Badge';

const AboutPage = () => {
    return (
        <div style={{ padding: "60px", maxWidth: 1200, margin: "0 auto" }}>
            <SectionHeader
                tag="The Core Bottleneck"
                title="The Fragility of Manual Tuning"
                subtitle="Learning rate schedules shouldn't be guesswork. It's time to stop relying on intuition and start discovering mathematically optimal structures."
            />

            {/* The Problem Statement Grid */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 32, marginBottom: 60 }}>
                {/* Left Side: The Old Way (Manual) */}
                <div className="glass" style={{ 
                    padding: "40px", 
                    borderTop: "3px solid var(--red)",
                    background: "linear-gradient(180deg, rgba(255,69,58,0.05) 0%, rgba(0,0,0,0) 100%)" 
                }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 24 }}>
                        <div style={{ width: 12, height: 12, borderRadius: "50%", background: "var(--red)" }} />
                        <h2 style={{ fontSize: 20, fontWeight: 700, color: "var(--red)", letterSpacing: "-0.02em" }}>The Unoptimized Bottleneck</h2>
                    </div>
                    <ul style={{ listStyle: "none", padding: 0, margin: 0, display: "flex", flexDirection: "column", gap: 20 }}>
                        {[
                            "Researchers waste days manually tuning hyperparameter schedules.",
                            "Hand-crafted forms like 'Cosine Annealing' are based on human intuition, not dataset realities.",
                            "Standard AutoML (like Bayesian Optimization) cannot invent entirely new mathematical functions.",
                            "Results are brittle: what works for ResNet may fail on Transformers."
                        ].map((text, i) => (
                            <li key={i} style={{ display: "flex", gap: 12, alignItems: "flex-start" }}>
                                <span style={{ color: "var(--text-3)", fontSize: 16 }}>✗</span>
                                <span style={{ fontSize: 15, color: "var(--text-2)", lineHeight: 1.6 }}>{text}</span>
                            </li>
                        ))}
                    </ul>
                </div>

                {/* Right Side: The SymboLR Way */}
                <div className="glass" style={{ 
                    padding: "40px", 
                    borderTop: "3px solid var(--green)",
                    background: "linear-gradient(180deg, rgba(48,209,88,0.05) 0%, rgba(0,0,0,0) 100%)" 
                }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 24 }}>
                        <div style={{ width: 12, height: 12, borderRadius: "50%", background: "var(--green)" }} />
                        <h2 style={{ fontSize: 20, fontWeight: 700, color: "var(--green)", letterSpacing: "-0.02em" }}>The SymboLR Advantage</h2>
                    </div>
                    <ul style={{ listStyle: "none", padding: 0, margin: 0, display: "flex", flexDirection: "column", gap: 20 }}>
                        {[
                            "Discovers robust, mathematically interpretable schedules completely autonomously.",
                            "Treats learning rates not as parameters, but as unknown expressions to be evolved.",
                            "Maintains an archive of structurally diverse formulas using Quality-Diversity search.",
                            "Outputs simple math that can be dropped into any production training loop."
                        ].map((text, i) => (
                            <li key={i} style={{ display: "flex", gap: 12, alignItems: "flex-start" }}>
                                <span style={{ color: "var(--green)", fontSize: 16 }}>✓</span>
                                <span style={{ fontSize: 15, color: "var(--text-2)", lineHeight: 1.6 }}>{text}</span>
                            </li>
                        ))}
                    </ul>
                </div>
            </div>

            {/* Feature Callouts (Card Layout) */}
            <h3 style={{ fontSize: 22, fontWeight: 700, marginBottom: 32, letterSpacing: "-0.02em" }}>Under The Hood</h3>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 20, marginBottom: 60 }}>
                {[
                    {
                        icon: "◈",
                        title: "AST Evolution",
                        body: "Expressions are modeled as Abstract Syntax Trees and mutated to explore vast mathematical search spaces.",
                        color: "var(--blue)"
                    },
                    {
                        icon: "⬡",
                        title: "GPU Proxy Training",
                        body: "Fitness is measured by training thousands of micro-networks on the GPU, evaluating true convergence speed.",
                        color: "var(--purple)"
                    },
                    {
                        icon: "◎",
                        title: "Algebraic Simplification",
                        body: "A SymPy pipeline automatically collapses complex mutations into their simplest mathematical forms.",
                        color: "var(--orange)"
                    }
                ].map((card, i) => (
                    <div key={i} className="glass" style={{ padding: "32px", borderColor: `${card.color}22` }}>
                        <div style={{ fontSize: 24, color: card.color, marginBottom: 16 }}>{card.icon}</div>
                        <h4 style={{ fontSize: 16, fontWeight: 600, marginBottom: 12 }}>{card.title}</h4>
                        <p style={{ fontSize: 14, color: "var(--text-2)", lineHeight: 1.6 }}>{card.body}</p>
                    </div>
                ))}
            </div>

            {/* The Solution Banner */}
            <div className="glass-bright" style={{ padding: "40px 60px", display: "flex", alignItems: "center", justifyContent: "space-between", gap: 40 }}>
                <div>
                    <Badge color="var(--blue)">Seamless Integration</Badge>
                    <h2 style={{ fontSize: 28, fontWeight: 700, letterSpacing: "-0.03em", marginTop: 20, marginBottom: 16 }}>
                        From Proxy to Production
                    </h2>
                    <p style={{ fontSize: 16, color: "var(--text-2)", lineHeight: 1.6, maxWidth: 600 }}>
                        Run the discovery pipeline offline on a lightweight task. Extract the winning expression, and inject it directly into your deep learning codebase. No heavy dependencies required at runtime.
                    </p>
                </div>
                <div style={{ padding: "20px", borderRadius: 12, background: "rgba(0,0,0,0.5)", border: "1px solid var(--border)", minWidth: 320 }}>
                    <code style={{ fontFamily: "var(--mono)", fontSize: 13, color: "var(--text)" }}>
                        <span style={{ color: "var(--purple)" }}># Drop directly into PyTorch</span><br/>
                        <span style={{ color: "var(--blue)" }}>def</span> lr_schedule(t):<br/>
                        &nbsp;&nbsp;&nbsp;&nbsp;<span style={{ color: "var(--orange)" }}>return</span> 0.067 * math.cos(t)
                    </code>
                </div>
            </div>
        </div>
    );
};

export default AboutPage;
