import React, { useState } from 'react';
import SectionHeader from '../components/SectionHeader';

const SystemPage = () => {
    const [hovered, setHovered] = useState(null);

    const nodes = [
        { id: "init", label: "Ramped H&H\nInitialization", x: 340, y: 40, color: "var(--blue)", icon: "⬡" },
        { id: "eval", label: "Parallel Fitness\nEvaluation", x: 340, y: 140, color: "var(--orange)", icon: "⚡" },
        { id: "rust", label: "Rust AST\nEvaluator", x: 560, y: 140, color: "var(--red)", icon: "⚙" },
        { id: "gpu", label: "GPU Probe\nTrainer", x: 120, y: 140, color: "var(--purple)", icon: "◈" },
        { id: "archive", label: "MAP-Elites\nArchive", x: 340, y: 260, color: "var(--green)", icon: "▦" },
        { id: "novelty", label: "Novelty\nSearch", x: 560, y: 260, color: "var(--teal)", icon: "◎" },
        { id: "surrogate", label: "Surrogate\nTriage", x: 120, y: 260, color: "var(--orange)", icon: "≋" },
        { id: "ops", label: "Genetic\nOperators", x: 340, y: 380, color: "var(--blue)", icon: "◉" },
        { id: "simplify", label: "SymPy\nSimplification", x: 560, y: 380, color: "var(--purple)", icon: "∑" },
        { id: "meta", label: "Meta-Controller\nPhase FSM", x: 120, y: 380, color: "var(--teal)", icon: "⬡" },
        { id: "lbfgs", label: "L-BFGS-B\nConstant Refinement", x: 340, y: 480, color: "var(--green)", icon: "△" },
    ];

    const descriptions = {
        init: "Ramped half-and-half initialization creates a diverse starting population. Trees range from depth 2–5 using both 'grow' and 'full' methods, ensuring maximum structural diversity at generation 0.",
        eval: "ThreadPoolExecutor parallelizes fitness evaluation across the population. Integrates with the surrogate triage layer to skip 30% of low-confidence candidates, focusing compute on promising individuals.",
        rust: "PyO3-compiled Rust extension that parses prefix-notation AST strings and evaluates them over 100-step time arrays. Zero intermediate allocations — 10–50× faster than Python NumPy.",
        gpu: "FastConvNet probe trainer with AMP, custom VRAM-resident DataLoader, torch.compile, and patience-based early stopping. Evaluates each formula by training a lightweight CNN from scratch.",
        archive: "3D behavioral grid: AST complexity × center-of-mass × smoothness. Each cell holds one elite. Selection is uniform over occupied niches, preventing fitness-dominated collapse.",
        novelty: "k-NN behavioral fingerprint archive. 6-dimensional schedule descriptors (mean, std, min, max, CoM, total variation). Augments fitness to reward genuinely novel schedule behaviors.",
        surrogate: "Online Ridge Regression predictor on 10 schedule features. Activates after 50 real evaluations. Filters 30% of candidates, always preserving structurally new and high-novelty individuals.",
        ops: "Five operators with EMA-based bandit selection: subtree crossover (structural recombination), subtree mutation (novelty injection), hoist mutation (anti-bloat), point mutation, constant perturbation.",
        simplify: "SymPy CAS pipeline: constant folding → nsimplify → algebraic reduction → AST reconstruction. Eliminates (t+0), (t*1), (t-t) etc. Skips trees >30 nodes for performance.",
        meta: "3-phase state machine: EXPLOIT → EXPLORE → DIVERSIFY. Monitors stagnation, diversity collapse. Adjusts mutation boost, novelty weight, immigrant injection, and crossover rate in real-time.",
        lbfgs: "SciPy L-BFGS-B gradient descent on scalar constants in the top-5 Hall of Fame formulas. Typically converges in <1 second per formula, improving fitness by ~3–8% without changing structure.",
    };

    const connections = [
        ["init", "eval"], ["eval", "rust"], ["eval", "gpu"], ["eval", "surrogate"],
        ["eval", "archive"], ["archive", "novelty"], ["archive", "ops"],
        ["ops", "simplify"], ["ops", "meta"], ["ops", "eval"],
        ["archive", "lbfgs"]
    ];

    return (
        <div style={{ padding: "60px" }}>
            <SectionHeader
                tag="Architecture"
                title="System Design"
                subtitle="A multi-layer evolutionary pipeline combining GPU probe training, Rust evaluation, MAP-Elites quality-diversity, and adaptive control systems."
            />

            <div style={{ display: "grid", gridTemplateColumns: "1fr 300px", gap: 32 }}>
                {/* Architecture diagram */}
                <div className="glass" style={{ padding: "32px", position: "relative" }}>
                    <svg viewBox="0 0 680 560" style={{ width: "100%", height: "auto" }}>
                        {/* Connection lines */}
                        {connections.map(([from, to], i) => {
                            const f = nodes.find(n => n.id === from);
                            const t = nodes.find(n => n.id === to);
                            if (!f || !t) return null;
                            return (
                                <line key={i}
                                    x1={f.x} y1={f.y + 30}
                                    x2={t.x} y2={t.y - 10}
                                    stroke="rgba(255,255,255,0.1)"
                                    strokeWidth={1.5}
                                    strokeDasharray={from === "ops" && to === "eval" ? "5 5" : "none"}
                                />
                            );
                        })}

                        {/* Nodes */}
                        {nodes.map(node => {
                            const isHovered = hovered === node.id;
                            return (
                                <g key={node.id}
                                    transform={`translate(${node.x - 70}, ${node.y - 20})`}
                                    style={{ cursor: "pointer" }}
                                    onMouseEnter={() => setHovered(node.id)}
                                    onMouseLeave={() => setHovered(null)}
                                >
                                    <rect
                                        width={140} height={52}
                                        rx={10}
                                        fill={isHovered ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.5)"}
                                        stroke={isHovered ? node.color : "rgba(255,255,255,0.12)"}
                                        strokeWidth={isHovered ? 1.5 : 0.5}
                                        style={{ transition: "all 0.2s" }}
                                    />
                                    <text
                                        x={70} y={18}
                                        textAnchor="middle"
                                        fontSize={16}
                                        fill={node.color}
                                        dominantBaseline="middle"
                                    >{node.icon}</text>
                                    <text
                                        x={70} y={30}
                                        textAnchor="middle"
                                        fontSize={9}
                                        fill={isHovered ? "white" : "rgba(255,255,255,0.7)"}
                                        dominantBaseline="middle"
                                    >{node.label.split("\n")[0]}</text>
                                    <text
                                        x={70} y={42}
                                        textAnchor="middle"
                                        fontSize={8.5}
                                        fill="rgba(255,255,255,0.45)"
                                        dominantBaseline="middle"
                                    >{node.label.split("\n")[1]}</text>
                                </g>
                            );
                        })}

                        {/* Return arrow for evolution loop */}
                        <path
                            d="M 620 410 Q 670 350 670 220 Q 670 160 620 150"
                            fill="none" stroke="rgba(10,132,255,0.3)" strokeWidth={1.5} strokeDasharray="4 4"
                        />
                        <text x={680} y={280} fontSize={9} fill="rgba(10,132,255,0.6)" textAnchor="middle" transform="rotate(90 680 280)">
                            Evolution Loop
                        </text>
                    </svg>
                </div>

                {/* Detail panel */}
                <div>
                    <div className="glass" style={{ padding: "24px", minHeight: 200 }}>
                        {hovered ? (
                            <div className="fade-in">
                                <div style={{
                                    fontSize: 22, marginBottom: 8,
                                    color: nodes.find(n => n.id === hovered)?.color
                                }}>
                                    {nodes.find(n => n.id === hovered)?.icon}
                                </div>
                                <h3 style={{ fontSize: 15, fontWeight: 600, marginBottom: 12 }}>
                                    {nodes.find(n => n.id === hovered)?.label.replace("\n", " ")}
                                </h3>
                                <p style={{ fontSize: 12, color: "var(--text-2)", lineHeight: 1.7 }}>
                                    {descriptions[hovered]}
                                </p>
                            </div>
                        ) : (
                            <div style={{ color: "var(--text-3)", fontSize: 13 }}>
                                <div style={{ fontSize: 24, marginBottom: 12 }}>👆</div>
                                Hover over any component to see its description.
                            </div>
                        )}
                    </div>

                    {/* Component list */}
                    <div style={{ marginTop: 16 }}>
                        {[
                            { label: "Runtime Modes", value: "Cloud CPU / Local CPU / Local GPU" },
                            { label: "Archive Dims", value: "30 × 20 × 10 = 6,000 niches" },
                            { label: "Time Steps", value: "100 normalized steps t ∈ [0,1]" },
                            { label: "Operators", value: "+, −, ×, ÷, sin, cos, exp, log, √, |x|" },
                        ].map(({ label, value }, i) => (
                            <div key={i} style={{
                                padding: "12px 16px", marginBottom: 8, borderRadius: 8,
                                background: "var(--bg2)", border: "1px solid var(--border)",
                                animation: `fadeUp ${0.4 + i * 0.08}s ease both`
                            }}>
                                <div style={{ fontSize: 10, color: "var(--text-3)", letterSpacing: "0.06em", textTransform: "uppercase" }}>{label}</div>
                                <div style={{ fontSize: 12, fontFamily: "var(--mono)", color: "var(--text-2)", marginTop: 4 }}>{value}</div>
                            </div>
                        ))}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default SystemPage;
