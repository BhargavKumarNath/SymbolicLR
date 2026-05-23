import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Terminal, Server, Database, Cloud, LayoutTemplate, Settings2, BrainCircuit, Network, Cpu, ArrowRightLeft, Blocks } from 'lucide-react';
import SectionHeader from '../components/SectionHeader';

const SystemPage = () => {
    const [view, setView] = useState("infrastructure"); // "infrastructure" | "internals"
    const [activeNode, setActiveNode] = useState(null);

    const handleHover = (id) => setActiveNode(id);
    const handleLeave = () => setActiveNode(null);

    // Reusable Node Component
    const Node = ({ id, label, icon: Icon, color, x, y, width = 200, height = 70, subcomponents = [] }) => {
        const isHovered = activeNode === id;
        return (
            <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1, y: isHovered ? -4 : 0 }}
                exit={{ opacity: 0, scale: 0.9 }}
                transition={{ duration: 0.4, type: "spring", bounce: 0.3 }}
                onMouseEnter={() => handleHover(id)}
                onMouseLeave={handleLeave}
                style={{
                    position: "absolute", left: x, top: y, width, height,
                    background: isHovered ? `linear-gradient(180deg, ${color}22, rgba(0,0,0,0.8))` : `linear-gradient(180deg, rgba(255,255,255,0.05), rgba(0,0,0,0.6))`,
                    border: `1px solid ${isHovered ? color : "var(--border)"}`,
                    borderRadius: 12, cursor: "pointer", zIndex: isHovered ? 20 : 10,
                    boxShadow: isHovered ? `0 0 30px ${color}33` : "none",
                    display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center"
                }}
            >
                <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                    <Icon color={isHovered ? color : "var(--text-3)"} size={22} />
                    <span style={{ fontSize: 13, fontWeight: 600, color: isHovered ? "#fff" : "var(--text)" }}>{label}</span>
                </div>
                
                <AnimatePresence>
                    {isHovered && subcomponents.length > 0 && (
                        <motion.div
                            initial={{ opacity: 0, height: 0, marginTop: 0 }}
                            animate={{ opacity: 1, height: "auto", marginTop: 12 }}
                            exit={{ opacity: 0, height: 0, marginTop: 0 }}
                            style={{ width: "100%", padding: "0 16px", display: "flex", flexDirection: "column", gap: 6, overflow: "hidden" }}
                        >
                            <div style={{ width: "100%", height: 1, background: `${color}44`, marginBottom: 6 }} />
                            {subcomponents.map((sub, i) => (
                                <div key={i} style={{ display: "flex", alignItems: "center", gap: 8 }}>
                                    <div style={{ width: 4, height: 4, borderRadius: "50%", background: color }} />
                                    <span style={{ fontSize: 11, color: "var(--text-2)", fontFamily: "var(--mono)", whiteSpace: "nowrap" }}>{sub}</span>
                                </div>
                            ))}
                        </motion.div>
                    )}
                </AnimatePresence>
            </motion.div>
        );
    };

    // Reusable Edge Component
    const Edge = ({ startX, startY, endX, endY, label, color = "var(--border)", activeColor = "var(--blue)", isCyclical = false }) => {
        const path = isCyclical 
            ? `M ${startX} ${startY} C ${startX} ${startY + 60}, ${endX} ${endY + 60}, ${endX} ${endY}`
            : `M ${startX} ${startY} C ${startX + 60} ${startY}, ${endX - 60} ${endY}, ${endX} ${endY}`;
        
        const midX = (startX + endX) / 2;
        const midY = isCyclical ? Math.max(startY, endY) + 40 : (startY + endY) / 2;

        return (
            <motion.svg 
                initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", pointerEvents: "none", zIndex: 1 }}
            >
                {/* Base track */}
                <path d={path} fill="none" stroke="var(--border)" strokeWidth={2} />
                {/* Animated dash */}
                <path d={path} fill="none" stroke={activeColor} strokeWidth={2} strokeDasharray="6 8" strokeOpacity={0.6}>
                    <animate attributeName="stroke-dashoffset" from="14" to="0" dur="0.8s" repeatCount="indefinite" />
                </path>
                {/* Label background */}
                {label && (
                    <g transform={`translate(${midX}, ${midY})`}>
                        <rect x="-60" y="-12" width="120" height="24" rx="12" fill="var(--bg)" stroke="var(--border)" strokeWidth="1" />
                        <text x="0" y="4" textAnchor="middle" fill="var(--text-3)" fontSize="10" fontFamily="var(--mono)" letterSpacing="0.05em">
                            {label}
                        </text>
                    </g>
                )}
            </motion.svg>
        );
    };

    return (
        <div style={{ padding: "60px", maxWidth: 1200, margin: "0 auto", position: "relative" }}>
            <SectionHeader
                tag="Technical Deep Dive"
                title="System Architecture"
                subtitle="Explore the dual nature of SymboLR: the decoupling of the heavy optimization backend from the lightning-fast web presentation, and the inner workings of the algorithmic loop."
            />

            {/* View Toggle */}
            <div style={{ display: "flex", justifyContent: "center", marginBottom: 60 }}>
                <div style={{ 
                    display: "flex", background: "rgba(0,0,0,0.4)", borderRadius: 30, 
                    border: "1px solid var(--border)", padding: 4 
                }}>
                    <button
                        onClick={() => setView("infrastructure")}
                        style={{
                            padding: "10px 24px", borderRadius: 26, fontSize: 13, fontWeight: 600,
                            background: view === "infrastructure" ? "var(--blue)" : "transparent",
                            color: view === "infrastructure" ? "#fff" : "var(--text-3)",
                            border: "none", cursor: "pointer", transition: "all 0.2s"
                        }}
                    >
                        View A: Infrastructure Pipeline
                    </button>
                    <button
                        onClick={() => setView("internals")}
                        style={{
                            padding: "10px 24px", borderRadius: 26, fontSize: 13, fontWeight: 600,
                            background: view === "internals" ? "var(--purple)" : "transparent",
                            color: view === "internals" ? "#fff" : "var(--text-3)",
                            border: "none", cursor: "pointer", transition: "all 0.2s"
                        }}
                    >
                        View B: Algorithmic Core
                    </button>
                </div>
            </div>

            {/* Interactive Graph Canvas */}
            <div className="glass" style={{ 
                height: 600, position: "relative", overflow: "hidden", 
                background: "radial-gradient(circle at center, rgba(255,255,255,0.02) 0%, transparent 80%)" 
            }}>
                <AnimatePresence mode="wait">
                    {/* VIEW A: INFRASTRUCTURE */}
                    {view === "infrastructure" && (
                        <motion.div key="infra" style={{ width: "100%", height: "100%", position: "absolute" }}>
                            <Edge startX={260} startY={200} endX={380} endY={200} label="Offline Export" activeColor="var(--orange)" />
                            <Edge startX={580} startY={200} endX={700} endY={200} label="Pre-computed JSON" activeColor="var(--blue)" />
                            <Edge startX={800} startY={235} endX={800} endY={330} label="Edge Hydration" activeColor="var(--green)" />
                            
                            <Node id="compute" label="Dedicated Compute Node" icon={Server} color="var(--orange)" x={60} y={165} width={200}
                                subcomponents={["RTX 4070 / Ryzen 9", "Python Orchestrator", "Rust Evaluation Engine"]} />
                            
                            <Node id="artifact-gen" label="Artifact Generation" icon={Blocks} color="var(--orange)" x={380} y={165} width={200}
                                subcomponents={["JSON Serialization", "Metric Aggregation", "LaTeX Formula Export"]} />

                            <Node id="artifact-store" label="results/*.json Artifact" icon={Database} color="var(--blue)" x={700} y={165} width={200}
                                subcomponents={["Immutable State", "Zero Latency Source", "Git LFS Managed"]} />

                            <Node id="edge-pipeline" label="Static Edge Pipeline" icon={Cloud} color="var(--blue)" x={700} y={330} width={200}
                                subcomponents={["Source Control Sync", "Vercel Edge Network", "Global CDN Caching"]} />

                            <Node id="presentation" label="Client Presentation" icon={LayoutTemplate} color="var(--green)" x={380} y={330} width={200}
                                subcomponents={["React/Vite Frontend", "Zero-latency State", "Interactive Visualization"]} />
                            
                            {/* Return arrow visually connecting Edge Pipeline to Client */}
                            <Edge startX={700} startY={365} endX={580} endY={365} label="Hydrate" activeColor="var(--green)" />
                        </motion.div>
                    )}

                    {/* VIEW B: ALGORITHMIC CORE */}
                    {view === "internals" && (
                        <motion.div key="internals" style={{ width: "100%", height: "100%", position: "absolute" }}>
                            {/* Forward Edges */}
                            <Edge startX={320} startY={120} endX={250} endY={220} label="Configured Spec" activeColor="var(--purple)" />
                            <Edge startX={150} startY={290} endX={150} endY={380} label="Hyperparameters" activeColor="var(--purple)" />
                            <Edge startX={250} startY={415} endX={400} endY={415} label="Serialized ASTs" activeColor="var(--teal)" />
                            <Edge startX={600} startY={415} endX={700} endY={415} label="Parallel Batches" activeColor="var(--red)" />
                            
                            {/* Feedback Loop Edge */}
                            <Edge startX={800} startY={450} endX={150} endY={450} label="Fitness Scores & MSE" activeColor="var(--green)" isCyclical={true} />

                            <Node id="cli" label="CLI Orchestrator" icon={Terminal} color="var(--text)" x={320} y={50} width={200}
                                subcomponents={["cli/main.py", "Configuration Parsing", "Experiment Initialization", "Baseline Loading"]} />

                            <Node id="meta-controller" label="Python Meta-Controller" icon={Settings2} color="var(--purple)" x={50} y={220} width={200}
                                subcomponents={["Population Initialization", "Evolutionary Loop Control", "State Management"]} />

                            <Node id="qd-core" label="Quality-Diversity Core" icon={Network} color="var(--purple)" x={50} y={380} width={200}
                                subcomponents={["gp/map_elites.py", "Map-Elites Archive", "Mutation/Crossover", "Novelty Scoring"]} />

                            <Node id="pyo3-bridge" label="PyO3 FFI Bridge" icon={ArrowRightLeft} color="var(--teal)" x={400} y={380} width={200}
                                subcomponents={["gp/rust_bridge.py", "Memory-safe Serialization", "Cross-language Types"]} />

                            <Node id="rust-evaluator" label="Rust Parallel Evaluator" icon={Cpu} color="var(--red)" x={700} y={380} width={200}
                                subcomponents={["rust_core/", "Parallel Batched Eval", "MSE Calculation", "High-Speed Benchmarks"]} />
                        </motion.div>
                    )}
                </AnimatePresence>
            </div>
        </div>
    );
};

export default SystemPage;
