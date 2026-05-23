import React, { useState, useRef, useMemo } from 'react';
import SectionHeader from '../components/SectionHeader';
import Badge from '../components/Badge';

const DiagnosticsPage = () => {
    const [uploadedData, setUploadedData] = useState(null);
    const fileRef = useRef(null);

    // Demo data
    const demoData = useMemo(() => {
        const gens = 30;
        let bestLoss = 1.8;
        return Array.from({ length: gens }, (_, i) => {
            bestLoss = Math.max(0.12, bestLoss - (Math.random() * 0.08 + 0.01) * (1 - i / gens * 0.5));
            const noiseRange = 0.3;
            const phase = i < 8 ? "EXPLOIT" : i < 18 ? "EXPLORE" : "DIVERSIFY";
            return {
                generation: i + 1,
                best_loss: parseFloat(bestLoss.toFixed(4)),
                median_loss: parseFloat((bestLoss + 0.2 + Math.random() * noiseRange).toFixed(4)),
                archive_size: Math.min(800, Math.floor(40 + i * 28 + Math.random() * 20)),
                structural_diversity: parseFloat((0.3 + 0.5 * Math.random() + (i > 15 ? 0.1 : 0)).toFixed(3)),
                behavioral_diversity: parseFloat((0.4 + 0.4 * Math.random()).toFixed(3)),
                novelty_mean: parseFloat((0.05 + Math.random() * 0.3).toFixed(3)),
                controller_phase: phase,
                operator_probs: {
                    crossover: parseFloat((0.35 + Math.random() * 0.1).toFixed(3)),
                    subtree_mutation: parseFloat((0.25 + Math.random() * 0.08).toFixed(3)),
                    hoist_mutation: parseFloat((0.15 + Math.random() * 0.05).toFixed(3)),
                    point_mutation: parseFloat((0.12 + Math.random() * 0.05).toFixed(3)),
                    constant_perturbation: parseFloat((0.10 + Math.random() * 0.04).toFixed(3)),
                },
                gen_time_s: parseFloat((2.5 + Math.random() * 3).toFixed(1)),
                surrogate_rmse: parseFloat((0.08 + Math.random() * 0.15).toFixed(4)),
            };
        });
    }, []);

    const activeData = uploadedData || demoData;
    const phaseColors = { EXPLOIT: "var(--blue)", EXPLORE: "var(--orange)", DIVERSIFY: "var(--purple)" };

    const handleUpload = (e) => {
        const file = e.target.files[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = ev => {
            try {
                const json = JSON.parse(ev.target.result);
                setUploadedData(json.generations || null);
            } catch { alert("Invalid JSON format"); }
        };
        reader.readAsText(file);
    };

    return (
        <div style={{ padding: "60px" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 40 }}>
                <SectionHeader
                    tag="Run Analytics"
                    title="Diagnostics Viewer"
                    subtitle="Visualize evolutionary dynamics from a saved run.json file, or explore the demo data below."
                />
                <div style={{ flexShrink: 0 }}>
                    <input type="file" ref={fileRef} onChange={handleUpload} style={{ display: "none" }} accept=".json" />
                    <button
                        onClick={() => fileRef.current?.click()}
                        style={{
                            padding: "12px 20px", borderRadius: 10,
                            background: "var(--bg2)", border: "1px solid var(--border)",
                            color: "var(--text)", fontSize: 13, fontWeight: 500,
                            cursor: "pointer", transition: "all var(--transition)"
                        }}
                        onMouseEnter={e => e.currentTarget.style.background = "var(--bg3)"}
                        onMouseLeave={e => e.currentTarget.style.background = "var(--bg2)"}
                    >⬆ Upload run.json</button>
                    {uploadedData && (
                        <button
                            onClick={() => setUploadedData(null)}
                            style={{
                                marginLeft: 8, padding: "12px 20px", borderRadius: 10,
                                background: "transparent", border: "1px solid var(--border)",
                                color: "var(--text-3)", fontSize: 12, cursor: "pointer"
                            }}
                        >Use Demo</button>
                    )}
                </div>
            </div>

            {!uploadedData && (
                <div style={{
                    padding: "10px 16px", borderRadius: 8, marginBottom: 24,
                    background: "rgba(10,132,255,0.08)", border: "1px solid rgba(10,132,255,0.2)",
                    color: "var(--blue)", fontSize: 12
                }}>
                    Showing demo data — upload a real run.json from symbolr benchmark to see your actual results.
                </div>
            )}

            {/* Generation table */}
            <div className="glass" style={{ overflow: "hidden" }}>
                <div style={{ padding: "20px 24px", borderBottom: "1px solid var(--border)", display: "flex", justifyContent: "space-between" }}>
                    <h3 style={{ fontSize: 14, fontWeight: 600 }}>Generation Log</h3>
                    <span style={{ fontSize: 12, color: "var(--text-3)" }}>Last 10 generations</span>
                </div>
                <div style={{ overflowX: "auto" }}>
                    <table style={{ width: "100%", borderCollapse: "collapse" }}>
                        <thead>
                            <tr style={{ background: "var(--bg2)" }}>
                                {["Gen", "Phase", "Best Loss", "Archive", "Struct. Div", "Novelty", "Time (s)"].map(h => (
                                    <th key={h} style={{ padding: "10px 16px", textAlign: "left", fontSize: 10, color: "var(--text-3)", letterSpacing: "0.06em", textTransform: "uppercase", fontWeight: 600, whiteSpace: "nowrap" }}>{h}</th>
                                ))}
                            </tr>
                        </thead>
                        <tbody>
                            {activeData.slice(-10).reverse().map((g, i) => (
                                <tr key={i} style={{ borderBottom: "1px solid var(--border)" }}>
                                    <td style={{ padding: "10px 16px", fontSize: 13, fontFamily: "var(--mono)", color: "var(--text-3)" }}>{g.generation}</td>
                                    <td style={{ padding: "10px 16px" }}>
                                        <Badge color={phaseColors[g.controller_phase]}>{g.controller_phase}</Badge>
                                    </td>
                                    <td style={{ padding: "10px 16px", fontSize: 13, fontFamily: "var(--mono)", color: "var(--green)" }}>{g.best_loss.toFixed(4)}</td>
                                    <td style={{ padding: "10px 16px", fontSize: 13, fontFamily: "var(--mono)" }}>{g.archive_size}</td>
                                    <td style={{ padding: "10px 16px", fontSize: 13, fontFamily: "var(--mono)" }}>{g.structural_diversity.toFixed(3)}</td>
                                    <td style={{ padding: "10px 16px", fontSize: 13, fontFamily: "var(--mono)" }}>{g.novelty_mean.toFixed(3)}</td>
                                    <td style={{ padding: "10px 16px", fontSize: 13, fontFamily: "var(--mono)", color: "var(--text-3)" }}>{g.gen_time_s}s</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    );
};

export default DiagnosticsPage;
