import React, { useState } from 'react';
import SectionHeader from '../components/SectionHeader';

const SystemPage = () => {
    const [view, setView] = useState("infrastructure"); // "infrastructure" | "internals"
    const [activeTooltip, setActiveTooltip] = useState(null);

    const handleHover = (id) => setActiveTooltip(id);
    const handleLeave = () => setActiveTooltip(null);

    // Reusable diagram node
    const DiagramNode = ({ id, label, color, x, y, width = 160, height = 60, icon }) => {
        const isHovered = activeTooltip === id;
        return (
            <div
                onMouseEnter={() => handleHover(id)}
                onMouseLeave={handleLeave}
                style={{
                    position: "absolute",
                    left: x,
                    top: y,
                    width,
                    height,
                    background: `linear-gradient(180deg, rgba(255,255,255,0.05), rgba(0,0,0,0.5))`,
                    border: `1px solid ${isHovered ? color : "var(--border)"}`,
                    borderRadius: 12,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    gap: 10,
                    cursor: "pointer",
                    boxShadow: isHovered ? `0 0 20px ${color}44` : "none",
                    transform: isHovered ? "translateY(-4px)" : "translateY(0)",
                    transition: "all 0.3s cubic-bezier(0.16, 1, 0.3, 1)",
                    zIndex: isHovered ? 10 : 2
                }}
            >
                {icon && <span style={{ color, fontSize: 18 }}>{icon}</span>}
                <span style={{ fontSize: 13, fontWeight: 600, color: isHovered ? "#fff" : "var(--text-2)" }}>
                    {label}
                </span>
            </div>
        );
    };

    // Reusable connecting arrow (simplified SVG)
    const Arrow = ({ startX, startY, endX, endY, active }) => {
        const strokeColor = active ? "var(--text)" : "var(--border)";
        const path = `M ${startX} ${startY} C ${startX + 50} ${startY}, ${endX - 50} ${endY}, ${endX} ${endY}`;
        return (
            <svg style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", pointerEvents: "none", zIndex: 1 }}>
                <path d={path} fill="none" stroke={strokeColor} strokeWidth={2} strokeDasharray={active ? "4 4" : "none"} />
                {active && (
                    <circle r="4" fill="var(--text)">
                        <animateMotion dur="2s" repeatCount="indefinite" path={path} />
                    </circle>
                )}
            </svg>
        );
    };

    // Tooltip data mapped to actual codebase files
    const tooltips = {
        "offline-compute": {
            title: "Heavy Offline Compute",
            file: "cli/main.py",
            body: "The core hybrid engine (Python & Rust) runs entirely offline. It evaluates millions of learning rate schedules asynchronously using local GPUs, completely disconnected from the web dashboard.",
            color: "var(--blue)"
        },
        "json-artifact": {
            title: "Static JSON Artifact",
            file: "run.json",
            body: "Evolution telemetry, including the archive grids, Hall of Fame formulas, and diversity metrics, are serialized into a lightweight JSON file. This acts as the single source of truth.",
            color: "var(--orange)"
        },
        "vercel-edge": {
            title: "Vercel Edge Network",
            file: "dashboard/src/main.jsx",
            body: "The dashboard is statically built and deployed globally via Vercel. By fetching the static JSON artifact, it achieves zero-latency renders and zero backend maintenance.",
            color: "var(--text)"
        },
        "meta-controller": {
            title: "Python Meta-Controller",
            file: "gp/meta_controller.py",
            body: "A 3-phase rule-based state machine that dynamically adjusts mutation intensities and triggers exploration/diversification phases based on real-time stagnation metrics.",
            color: "var(--purple)"
        },
        "map-elites": {
            title: "MAP-Elites Archive",
            file: "gp/map_elites.py",
            body: "Maintains behavioral diversity. Solutions are binned by structural complexity and temporal center-of-mass. An incumbent is only replaced if a new formula achieves strictly lower validation loss.",
            color: "var(--green)"
        },
        "pyo3-bridge": {
            title: "PyO3 Interop Bridge",
            file: "gp/rust_bridge.py",
            body: "Serializes ASTs into prefix mathematical notation and bridges Python to Rust, bypassing standard Python recursive overhead while maintaining safety guarantees.",
            color: "var(--teal)"
        },
        "rust-evaluator": {
            title: "Rust Evaluator",
            file: "rust_core/src/lib.rs",
            body: "High-speed fitness benchmarking. Parses the prefix AST into native enums and performs zero-allocation scalar evaluation across the 100-step time array, returning a zero-copy NumPy view.",
            color: "var(--red)"
        }
    };

    return (
        <div style={{ padding: "60px", maxWidth: 1200, margin: "0 auto", position: "relative" }}>
            <SectionHeader
                tag="Technical Deep Dive"
                title="System Architecture"
                subtitle="Explore the dual nature of SymboLR: the decoupling of the heavy optimization backend from the lightning-fast web dashboard."
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
                            padding: "10px 24px", borderRadius: 26, fontSize: 14, fontWeight: 600,
                            background: view === "infrastructure" ? "rgba(255,255,255,0.1)" : "transparent",
                            color: view === "infrastructure" ? "#fff" : "var(--text-3)",
                            border: "none", cursor: "pointer", transition: "all 0.2s"
                        }}
                    >
                        View A: Infrastructure
                    </button>
                    <button
                        onClick={() => setView("internals")}
                        style={{
                            padding: "10px 24px", borderRadius: 26, fontSize: 14, fontWeight: 600,
                            background: view === "internals" ? "rgba(255,255,255,0.1)" : "transparent",
                            color: view === "internals" ? "#fff" : "var(--text-3)",
                            border: "none", cursor: "pointer", transition: "all 0.2s"
                        }}
                    >
                        View B: Engine Internals
                    </button>
                </div>
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 320px", gap: 40, alignItems: "start" }}>
                
                {/* Interactive Diagram Area */}
                <div className="glass" style={{ 
                    height: 500, position: "relative", overflow: "hidden", 
                    background: "radial-gradient(circle at center, rgba(255,255,255,0.03) 0%, transparent 70%)" 
                }}>
                    {/* View A: Infrastructure */}
                    {view === "infrastructure" && (
                        <div style={{ width: "100%", height: "100%", position: "absolute", animation: "fadeUp 0.4s ease both" }}>
                            <DiagramNode id="offline-compute" label="Heavy Offline Compute" icon="⚙" color="var(--blue)" x={60} y={220} width={200} />
                            <Arrow startX={260} startY={250} endX={400} endY={250} active={activeTooltip === "offline-compute"} />
                            
                            <DiagramNode id="json-artifact" label="Static JSON Artifact" icon="📄" color="var(--orange)" x={400} y={220} />
                            <Arrow startX={560} startY={250} endX={700} endY={250} active={activeTooltip === "json-artifact"} />
                            
                            <DiagramNode id="vercel-edge" label="Vercel Edge Network" icon="⚡" color="var(--text)" x={700} y={220} />
                        </div>
                    )}

                    {/* View B: Engine Internals */}
                    {view === "internals" && (
                        <div style={{ width: "100%", height: "100%", position: "absolute", animation: "fadeUp 0.4s ease both" }}>
                            <DiagramNode id="meta-controller" label="Python Meta-Controller" icon="🧠" color="var(--purple)" x={60} y={100} width={200} />
                            <Arrow startX={160} startY={160} endX={160} endY={240} active={activeTooltip === "meta-controller"} />
                            
                            <DiagramNode id="map-elites" label="MAP-Elites Archive" icon="▦" color="var(--green)" x={60} y={240} width={200} />
                            <Arrow startX={260} startY={270} endX={420} endY={270} active={activeTooltip === "map-elites"} />
                            
                            <DiagramNode id="pyo3-bridge" label="PyO3 Interop Bridge" icon="🌉" color="var(--teal)" x={420} y={240} width={180} />
                            <Arrow startX={510} startY={240} endX={650} endY={140} active={activeTooltip === "pyo3-bridge"} />
                            
                            <DiagramNode id="rust-evaluator" label="Rust Evaluation Engine" icon="🦀" color="var(--red)" x={580} y={80} width={220} />
                            
                            {/* Feedback loop arrow */}
                            <Arrow startX={690} startY={140} endX={260} endY={270} active={activeTooltip === "rust-evaluator"} />
                        </div>
                    )}
                </div>

                {/* Dynamic Tooltip Panel */}
                <div className="glass" style={{ padding: "32px", minHeight: 320, display: "flex", flexDirection: "column" }}>
                    {activeTooltip && tooltips[activeTooltip] ? (
                        <div style={{ animation: "fadeUp 0.2s ease both" }}>
                            <div style={{ 
                                display: "inline-flex", alignItems: "center", gap: 6,
                                padding: "4px 10px", borderRadius: 6, marginBottom: 20,
                                background: "rgba(255,255,255,0.05)", border: "1px solid rgba(255,255,255,0.1)",
                                color: "var(--text-3)", fontSize: 11, fontFamily: "var(--mono)"
                            }}>
                                <span>📄</span> {tooltips[activeTooltip].file}
                            </div>
                            <h3 style={{ fontSize: 20, fontWeight: 700, color: tooltips[activeTooltip].color, marginBottom: 16 }}>
                                {tooltips[activeTooltip].title}
                            </h3>
                            <p style={{ fontSize: 14, color: "var(--text-2)", lineHeight: 1.7 }}>
                                {tooltips[activeTooltip].body}
                            </p>
                        </div>
                    ) : (
                        <div style={{ 
                            flex: 1, display: "flex", flexDirection: "column", 
                            alignItems: "center", justifyContent: "center", textAlign: "center" 
                        }}>
                            <div style={{ fontSize: 32, marginBottom: 16, opacity: 0.3 }}>👆</div>
                            <h4 style={{ fontSize: 15, fontWeight: 600, color: "var(--text-3)", marginBottom: 8 }}>Explore the Code</h4>
                            <p style={{ fontSize: 13, color: "var(--text-3)", opacity: 0.7 }}>
                                Hover over any block in the diagram to reveal its architectural purpose and the exact file that implements it.
                            </p>
                        </div>
                    )}
                </div>

            </div>
        </div>
    );
};

export default SystemPage;
