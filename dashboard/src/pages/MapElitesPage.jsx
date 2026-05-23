import React, { useState, useMemo } from 'react';
import SectionHeader from '../components/SectionHeader';
import Badge from '../components/Badge';

const MapElitesPage = () => {
    const [selectedNiche, setSelectedNiche] = useState(null);
    const [hoveredCell, setHoveredCell] = useState(null);

    // Simulate an archive
    const archiveData = useMemo(() => {
        const data = [];
        for (let size = 0; size < 20; size++) {
            for (let com = 0; com < 15; com++) {
                if (Math.random() > 0.55) {
                    const loss = 0.05 + Math.random() * 0.8 + (size / 20) * 0.1 + (com / 15) * 0.1;
                    const families = ["cyclical", "exponential", "linear", "inverse", "polynomial"];
                    data.push({
                        size, com,
                        loss: parseFloat(loss.toFixed(4)),
                        family: families[Math.floor(Math.random() * families.length)],
                        formula: [
                            "0.067 · cos(πt)", "0.1 · e^(−5t)", "0.5 · (1−t)",
                            "sin(t)/(1+t)", "√(1−t) · α"
                        ][Math.floor(Math.random() * 5)],
                        size_actual: 1 + size * 0.5,
                        age: Math.floor(Math.random() * 20)
                    });
                }
            }
        }
        return data;
    }, []);

    const maxLoss = Math.max(...archiveData.map(d => d.loss));
    const minLoss = Math.min(...archiveData.map(d => d.loss));

    const getLossColor = (loss) => {
        const t = (loss - minLoss) / (maxLoss - minLoss);
        const r = Math.round(48 + (255 - 48) * t);
        const g = Math.round(209 - 209 * t);
        const b = Math.round(88 - 88 * t);
        return `rgb(${r},${g},${b})`;
    };

    const familyColors = {
        cyclical: "var(--blue)",
        exponential: "var(--orange)",
        linear: "var(--green)",
        inverse: "var(--purple)",
        polynomial: "var(--teal)"
    };

    const topFormulas = archiveData
        .sort((a, b) => a.loss - b.loss)
        .slice(0, 5);

    return (
        <div style={{ padding: "60px" }}>
            <SectionHeader
                tag="QD Archive"
                title="MAP-Elites Archive"
                subtitle="A 3-dimensional quality-diversity grid that maintains behaviorally diverse solutions simultaneously. Each cell holds the best formula for its unique combination of complexity, temporal focus, and smoothness."
            />

            <div style={{ display: "grid", gridTemplateColumns: "1fr 280px", gap: 32, marginBottom: 40 }}>
                {/* Archive heatmap */}
                <div className="glass" style={{ padding: "28px", position: "relative" }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end", marginBottom: 20 }}>
                        <div>
                            <h3 style={{ fontSize: 15, fontWeight: 600 }}>Archive Heatmap</h3>
                            <p style={{ fontSize: 12, color: "var(--text-3)", marginTop: 4 }}>Size bins × Center-of-mass bins. Color = validation loss.</p>
                        </div>
                        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                            <div style={{ width: 60, height: 6, borderRadius: 3, background: "linear-gradient(90deg, #30d158, #ff453a)" }} />
                            <span style={{ fontSize: 10, color: "var(--text-3)" }}>low → high loss</span>
                        </div>
                    </div>

                    {/* Y axis label */}
                    <div style={{ display: "flex", gap: 0 }}>
                        <div style={{ display: "flex", flexDirection: "column", justifyContent: "space-between", marginRight: 8, paddingBottom: 20 }}>
                            <span style={{ fontSize: 9, color: "var(--text-3)", writingMode: "vertical-rl", transform: "rotate(180deg)", textAlign: "center" }}>Center of Mass →</span>
                        </div>

                        <div style={{ flex: 1 }}>
                            {/* Grid */}
                            <div style={{
                                display: "grid",
                                gridTemplateColumns: `repeat(20, 1fr)`,
                                gap: 2
                            }}>
                                {Array.from({ length: 20 }, (_, size) =>
                                    Array.from({ length: 15 }, (_, com) => {
                                        const cell = archiveData.find(d => d.size === size && d.com === com);
                                        const isHovered = hoveredCell && hoveredCell.size === size && hoveredCell.com === com;
                                        return (
                                            <div
                                                key={`${size}-${com}`}
                                                style={{
                                                    width: "100%", paddingBottom: "100%",
                                                    borderRadius: 2, position: "relative",
                                                    background: cell ? getLossColor(cell.loss) : "rgba(255,255,255,0.04)",
                                                    border: isHovered ? "1px solid white" : "1px solid transparent",
                                                    cursor: cell ? "pointer" : "default",
                                                    opacity: cell ? 1 : 0.3,
                                                    transition: "transform 0.1s",
                                                    transform: isHovered ? "scale(1.2)" : "scale(1)",
                                                    zIndex: isHovered ? 2 : 1
                                                }}
                                                onMouseEnter={() => cell && setHoveredCell(cell)}
                                                onMouseLeave={() => setHoveredCell(null)}
                                                onClick={() => cell && setSelectedNiche(cell)}
                                            />
                                        );
                                    })
                                ).flat()}
                            </div>
                            <div style={{ display: "flex", justifyContent: "space-between", marginTop: 6 }}>
                                <span style={{ fontSize: 9, color: "var(--text-3)" }}>Simple (sz=1)</span>
                                <span style={{ fontSize: 9, color: "var(--text-3)", textAlign: "center" }}>Formula Complexity →</span>
                                <span style={{ fontSize: 9, color: "var(--text-3)" }}>Complex (sz=30)</span>
                            </div>
                        </div>
                    </div>

                    {/* Hover tooltip */}
                    {hoveredCell && (
                        <div className="glass" style={{
                            position: "absolute", padding: "10px 14px", borderRadius: 8,
                            pointerEvents: "none", zIndex: 100,
                            fontSize: 12, background: "rgba(0,0,0,0.9)",
                            top: 40, right: 40
                        }}>
                            <div style={{ fontFamily: "var(--mono)", color: "var(--blue)" }}>{hoveredCell.formula}</div>
                            <div style={{ color: "var(--text-3)", marginTop: 4 }}>Loss: {hoveredCell.loss.toFixed(4)}</div>
                        </div>
                    )}
                </div>

                {/* Side panel */}
                <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
                    <div className="glass" style={{ padding: "20px" }}>
                        <h4 style={{ fontSize: 13, fontWeight: 600, marginBottom: 16 }}>Archive Stats</h4>
                        {[
                            { label: "Occupied Niches", value: archiveData.length.toString() },
                            { label: "Occupancy", value: `${((archiveData.length / (20 * 15)) * 100).toFixed(1)}%` },
                            { label: "Best Loss", value: minLoss.toFixed(4) },
                            { label: "Archive Dims", value: "20 × 15" },
                        ].map(({ label, value }) => (
                            <div key={label} style={{ display: "flex", justifyContent: "space-between", marginBottom: 12 }}>
                                <span style={{ fontSize: 12, color: "var(--text-3)" }}>{label}</span>
                                <span style={{ fontSize: 12, fontFamily: "var(--mono)", color: "var(--text)" }}>{value}</span>
                            </div>
                        ))}
                    </div>

                    {/* Family legend */}
                    <div className="glass" style={{ padding: "20px" }}>
                        <h4 style={{ fontSize: 13, fontWeight: 600, marginBottom: 14 }}>Formula Families</h4>
                        {Object.entries(familyColors).map(([family, color]) => {
                            const count = archiveData.filter(d => d.family === family).length;
                            return (
                                <div key={family} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 10 }}>
                                    <div style={{ width: 8, height: 8, borderRadius: 2, background: color, flexShrink: 0 }} />
                                    <span style={{ fontSize: 12, flex: 1, textTransform: "capitalize" }}>{family}</span>
                                    <span style={{ fontSize: 11, color: "var(--text-3)", fontFamily: "var(--mono)" }}>{count}</span>
                                </div>
                            );
                        })}
                    </div>
                </div>
            </div>

            {/* Hall of Fame */}
            <div>
                <h2 style={{ fontSize: 20, fontWeight: 700, marginBottom: 20, letterSpacing: "-0.02em" }}>Hall of Fame</h2>
                <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                    {topFormulas.map((formula, i) => (
                        <div key={i} className="glass" style={{
                            padding: "16px 24px",
                            display: "grid", gridTemplateColumns: "32px 1fr 80px 80px 100px",
                            alignItems: "center", gap: 16,
                            borderColor: i === 0 ? "rgba(255,159,10,0.3)" : "var(--border)",
                            animation: `fadeUp ${0.3 + i * 0.08}s ease both`
                        }}>
                            <div style={{
                                width: 28, height: 28, borderRadius: 8,
                                background: i === 0 ? "var(--orange-dim)" : "var(--bg2)",
                                border: `1px solid ${i === 0 ? "rgba(255,159,10,0.3)" : "var(--border)"}`,
                                display: "flex", alignItems: "center", justifyContent: "center",
                                fontSize: 12, fontWeight: 700, color: i === 0 ? "var(--orange)" : "var(--text-3)"
                            }}>{i + 1}</div>
                            <code style={{ fontFamily: "var(--mono)", fontSize: 13, color: "var(--text)" }}>{formula.formula}</code>
                            <div style={{ textAlign: "right" }}>
                                <div style={{ fontSize: 14, fontWeight: 600, color: "var(--green)" }}>{formula.loss.toFixed(4)}</div>
                                <div style={{ fontSize: 10, color: "var(--text-3)" }}>val loss</div>
                            </div>
                            <div style={{ textAlign: "right" }}>
                                <div style={{ fontSize: 14, fontFamily: "var(--mono)" }}>{Math.floor(formula.size_actual)}</div>
                                <div style={{ fontSize: 10, color: "var(--text-3)" }}>nodes</div>
                            </div>
                            <Badge color={familyColors[formula.family]}>{formula.family}</Badge>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
};

export default MapElitesPage;
