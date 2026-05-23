import React, { useState, useEffect, useRef, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
    LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer,
    BarChart, Bar
} from 'recharts';
import { UploadCloud, FileJson, CheckCircle, ChevronDown, ChevronUp, BarChart3, Database, TrendingUp, Cpu } from 'lucide-react';
import SectionHeader from '../components/SectionHeader';
import Badge from '../components/Badge';
import Latex from '../components/Latex';

// Reusable Metric Card
const StatCard = ({ title, value, icon: Icon, color, delay }) => (
    <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay, duration: 0.4 }}
        className="glass"
        style={{ padding: "24px", display: "flex", flexDirection: "column", gap: 12, position: "relative", overflow: "hidden" }}
    >
        <div style={{ position: "absolute", top: -20, right: -20, opacity: 0.05, color }}>
            <Icon size={120} />
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 8, color: "var(--text-3)", fontSize: 13, textTransform: "uppercase", letterSpacing: "0.05em", fontWeight: 600 }}>
            <Icon size={16} color={color} />
            {title}
        </div>
        <div style={{ fontSize: 32, fontWeight: 700, fontFamily: "var(--mono)", color: "var(--text)" }}>
            {value}
        </div>
    </motion.div>
);

const GenerationRow = ({ gen, prevGen, index }) => {
    const [isExpanded, setIsExpanded] = useState(false);
    
    const delta = prevGen ? (prevGen.best_loss - gen.best_loss) : 0;
    const isImprovement = delta > 0.0001;

    const phaseColors = { exploit: "var(--blue)", explore: "var(--orange)", diversify: "var(--purple)" };
    const phaseColor = phaseColors[gen.controller_phase.toLowerCase()] || "var(--text-3)";

    // Prepare operator data for the mini chart
    const operatorData = useMemo(() => {
        if (!gen.operator_probs) return [];
        return Object.entries(gen.operator_probs).map(([key, val]) => ({
            name: key.replace("_", " "),
            prob: val * 100
        })).sort((a, b) => b.prob - a.prob);
    }, [gen]);

    return (
        <div style={{ marginBottom: 12 }}>
            <div 
                className="glass" 
                style={{ 
                    padding: "20px 24px", display: "grid", gridTemplateColumns: "80px 1fr 140px 100px 40px", 
                    alignItems: "center", gap: 24, cursor: "pointer",
                    border: isExpanded ? "1px solid var(--border)" : "1px solid transparent",
                    background: isExpanded ? "var(--bg3)" : "rgba(255,255,255,0.01)",
                    transition: "all var(--transition)"
                }}
                onClick={() => setIsExpanded(!isExpanded)}
            >
                <div>
                    <div style={{ fontSize: 11, color: "var(--text-3)", textTransform: "uppercase" }}>Generation</div>
                    <div style={{ fontSize: 18, fontWeight: 700, fontFamily: "var(--mono)" }}>{gen.generation}</div>
                </div>
                
                <div style={{ overflowX: "auto" }}>
                    <div style={{ fontSize: 11, color: "var(--text-3)", textTransform: "uppercase", marginBottom: 6 }}>Discovered Elite Formula</div>
                    <div style={{ fontSize: 15, color: "var(--text)" }}>
                        {gen.top_formula_latex ? (
                            <Latex expression={gen.top_formula_latex} block={false} />
                        ) : (
                            <span style={{ color: "var(--text-3)", fontStyle: "italic" }}>No formula logged</span>
                        )}
                    </div>
                </div>

                <div style={{ textAlign: "right" }}>
                    <div style={{ fontSize: 11, color: "var(--text-3)", textTransform: "uppercase", marginBottom: 4 }}>Best Fitness (MSE)</div>
                    <div style={{ display: "flex", alignItems: "baseline", justifyContent: "flex-end", gap: 8 }}>
                        <span style={{ fontSize: 16, fontWeight: 700, fontFamily: "var(--mono)", color: "var(--green)" }}>{gen.best_loss.toFixed(4)}</span>
                        {isImprovement && (
                            <span style={{ fontSize: 12, color: "var(--green)", fontFamily: "var(--mono)" }}>-{delta.toFixed(4)}</span>
                        )}
                    </div>
                </div>

                <div style={{ textAlign: "right" }}>
                    <Badge color={phaseColor}>{gen.controller_phase}</Badge>
                </div>

                <div style={{ display: "flex", justifyContent: "flex-end", color: "var(--text-3)" }}>
                    {isExpanded ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
                </div>
            </div>

            <AnimatePresence>
                {isExpanded && (
                    <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: "auto", opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        style={{ overflow: "hidden", marginLeft: 20, marginRight: 20 }}
                    >
                        <div style={{ padding: "24px", background: "rgba(10,10,10,0.4)", borderLeft: "1px solid var(--border)", borderRight: "1px solid var(--border)", borderBottom: "1px solid var(--border)", borderBottomLeftRadius: 12, borderBottomRightRadius: 12, display: "grid", gridTemplateColumns: "1fr 1fr", gap: 40 }}>
                            
                            <div>
                                <h4 style={{ fontSize: 13, color: "var(--text-2)", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: 16 }}>Evolutionary Drivers</h4>
                                <div style={{ height: 160 }}>
                                    <ResponsiveContainer width="100%" height="100%">
                                        <BarChart data={operatorData} layout="vertical" margin={{ top: 0, right: 30, left: 40, bottom: 0 }}>
                                            <XAxis type="number" hide />
                                            <YAxis dataKey="name" type="category" axisLine={false} tickLine={false} tick={{ fontSize: 11, fill: "var(--text-3)" }} width={120} />
                                            <RechartsTooltip 
                                                cursor={{ fill: "rgba(255,255,255,0.05)" }}
                                                contentStyle={{ background: "var(--bg3)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 12 }}
                                                formatter={(val) => [`${val.toFixed(1)}%`, 'Probability']}
                                            />
                                            <Bar dataKey="prob" fill="var(--blue)" radius={[0, 4, 4, 0]} barSize={12} />
                                        </BarChart>
                                    </ResponsiveContainer>
                                </div>
                            </div>

                            <div>
                                <h4 style={{ fontSize: 13, color: "var(--text-2)", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: 16 }}>Niche Dynamics</h4>
                                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
                                    {[
                                        { label: "New Niches Filled", value: `+${gen.new_niches}`, color: "var(--green)" },
                                        { label: "Total Archive Size", value: gen.archive_size, color: "var(--text)" },
                                        { label: "Structural Diversity", value: gen.structural_diversity.toFixed(2), color: "var(--purple)" },
                                        { label: "Compute Time", value: `${gen.gen_time_s}s`, color: "var(--text-3)" },
                                    ].map((stat, i) => (
                                        <div key={i} style={{ padding: "16px", background: "rgba(255,255,255,0.02)", borderRadius: 8 }}>
                                            <div style={{ fontSize: 11, color: "var(--text-3)", marginBottom: 4 }}>{stat.label}</div>
                                            <div style={{ fontSize: 18, fontWeight: 700, fontFamily: "var(--mono)", color: stat.color }}>{stat.value}</div>
                                        </div>
                                    ))}
                                </div>
                            </div>

                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
};


const DiagnosticsPage = () => {
    const [mode, setMode] = useState('golden'); // 'golden' | 'local'
    const [runData, setRunData] = useState(null);
    const [isLoading, setIsLoading] = useState(true);
    const [isDragging, setIsDragging] = useState(false);
    const fileRef = useRef(null);

    const fetchGoldenRun = () => {
        setIsLoading(true);
        fetch('/results.json')
            .then(res => res.json())
            .then(data => {
                setRunData(data);
                setIsLoading(false);
            })
            .catch(err => {
                console.error("Failed to fetch golden run", err);
                setIsLoading(false);
            });
    };

    useEffect(() => {
        if (mode === 'golden') {
            fetchGoldenRun();
        } else {
            setRunData(null); // Clear data when switching to local until they upload
        }
    }, [mode]);

    const handleFileUpload = (e) => {
        const file = e.target.files?.[0];
        if (!file) return;
        processFile(file);
    };

    const handleDrop = (e) => {
        e.preventDefault();
        setIsDragging(false);
        const file = e.dataTransfer.files?.[0];
        if (file) processFile(file);
    };

    const processFile = (file) => {
        const reader = new FileReader();
        reader.onload = ev => {
            try {
                const json = JSON.parse(ev.target.result);
                setRunData(json);
            } catch (err) {
                alert("Invalid JSON format. Please upload a valid symbolr diagnostics artifact.");
            }
        };
        reader.readAsText(file);
    };

    const summary = runData?.summary || {};
    const generations = runData?.generations || [];

    return (
        <div style={{ padding: "60px", maxWidth: 1200, margin: "0 auto" }}>
            
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end", marginBottom: 40 }}>
                <SectionHeader
                    tag="Observability"
                    title="Telemetry Dashboard"
                    subtitle="Deep inspect the evolutionary health of the run. Analyze search space dynamics, algorithmic convergence, and operator distributions."
                />

                {/* Dual-Mode Toggle */}
                <div style={{ display: "flex", background: "var(--bg3)", padding: "4px", borderRadius: 12, border: "1px solid var(--border)" }}>
                    <button 
                        onClick={() => setMode('golden')}
                        style={{ 
                            padding: "8px 16px", borderRadius: 8, fontSize: 13, fontWeight: 600, border: "none", cursor: "pointer",
                            background: mode === 'golden' ? "var(--bg)" : "transparent",
                            color: mode === 'golden' ? "var(--text)" : "var(--text-3)",
                            boxShadow: mode === 'golden' ? "0 2px 8px rgba(0,0,0,0.2)" : "none",
                            transition: "all 0.2s ease"
                        }}
                    >
                        Golden Run
                    </button>
                    <button 
                        onClick={() => setMode('local')}
                        style={{ 
                            padding: "8px 16px", borderRadius: 8, fontSize: 13, fontWeight: 600, border: "none", cursor: "pointer",
                            background: mode === 'local' ? "var(--bg)" : "transparent",
                            color: mode === 'local' ? "var(--text)" : "var(--text-3)",
                            boxShadow: mode === 'local' ? "0 2px 8px rgba(0,0,0,0.2)" : "none",
                            transition: "all 0.2s ease"
                        }}
                    >
                        Local Artifact
                    </button>
                </div>
            </div>

            {/* Ingestion Zone for Local Mode */}
            {mode === 'local' && !runData && (
                <motion.div 
                    initial={{ opacity: 0, scale: 0.98 }} animate={{ opacity: 1, scale: 1 }}
                    className="glass"
                    style={{ 
                        padding: "60px", textAlign: "center", border: `2px dashed ${isDragging ? "var(--blue)" : "var(--border)"}`,
                        background: isDragging ? "rgba(10, 132, 255, 0.05)" : "var(--bg2)",
                        transition: "all 0.2s ease", cursor: "pointer", marginBottom: 40
                    }}
                    onDragOver={e => { e.preventDefault(); setIsDragging(true); }}
                    onDragLeave={() => setIsDragging(false)}
                    onDrop={handleDrop}
                    onClick={() => fileRef.current?.click()}
                >
                    <input type="file" ref={fileRef} onChange={handleFileUpload} style={{ display: "none" }} accept=".json" />
                    <UploadCloud size={48} color={isDragging ? "var(--blue)" : "var(--text-3)"} style={{ marginBottom: 16 }} />
                    <h3 style={{ fontSize: 18, fontWeight: 600, marginBottom: 8 }}>Inspect Local CLI Artifact</h3>
                    <p style={{ color: "var(--text-2)", fontSize: 14 }}>Drag and drop a <code style={{fontFamily: "var(--mono)", background: "rgba(255,255,255,0.1)", padding: "2px 6px", borderRadius: 4}}>run_seed*.json</code> file here, or click to browse.</p>
                </motion.div>
            )}

            {runData && (
                <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.5 }}>
                    
                    {/* KPI Ribbon */}
                    <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 24, marginBottom: 40 }}>
                        <StatCard title="Total Generations" value={summary.total_generations || 0} icon={Cpu} color="var(--blue)" delay={0.1} />
                        <StatCard title="Unique Formulas" value={summary.final_archive_size || 0} icon={Database} color="var(--purple)" delay={0.2} />
                        <StatCard title="Global Best MSE" value={summary.final_best_loss ? summary.final_best_loss.toFixed(4) : "0.0000"} icon={TrendingUp} color="var(--green)" delay={0.3} />
                        <StatCard title="Diversity Score" value={summary.final_behavioral_diversity ? summary.final_behavioral_diversity.toFixed(3) : "1.000"} icon={BarChart3} color="var(--orange)" delay={0.4} />
                    </div>

                    {/* Telemetry Charts */}
                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24, marginBottom: 40 }}>
                        {/* Convergence Chart */}
                        <div className="glass" style={{ padding: "24px" }}>
                            <div style={{ marginBottom: 20 }}>
                                <h3 style={{ fontSize: 16, fontWeight: 600 }}>Convergence Trajectory</h3>
                                <p style={{ fontSize: 13, color: "var(--text-3)" }}>Best and Median MSE mapping the compression of the search space.</p>
                            </div>
                            <div style={{ height: 260 }}>
                                <ResponsiveContainer width="100%" height="100%">
                                    <LineChart data={generations} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                                        <XAxis dataKey="generation" tick={{ fontSize: 11, fill: "var(--text-3)" }} tickLine={false} axisLine={false} />
                                        <YAxis tick={{ fontSize: 11, fill: "var(--text-3)" }} tickLine={false} axisLine={false} domain={['auto', 'auto']} />
                                        <RechartsTooltip 
                                            contentStyle={{ background: "rgba(10,10,10,0.9)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 12, fontFamily: "var(--mono)" }}
                                            itemStyle={{ padding: "4px 0" }}
                                        />
                                        <Line type="monotone" dataKey="best_loss" name="Best MSE" stroke="var(--green)" strokeWidth={3} dot={false} />
                                        <Line type="monotone" dataKey="median_loss" name="Median MSE" stroke="var(--text-3)" strokeWidth={2} dot={false} strokeDasharray="5 5" />
                                    </LineChart>
                                </ResponsiveContainer>
                            </div>
                        </div>

                        {/* Complexity Drift Chart */}
                        <div className="glass" style={{ padding: "24px" }}>
                            <div style={{ marginBottom: 20 }}>
                                <h3 style={{ fontSize: 16, fontWeight: 600 }}>Mathematical Complexity Drift</h3>
                                <p style={{ fontSize: 13, color: "var(--text-3)" }}>Tracking structural diversity and novelty drift over time.</p>
                            </div>
                            <div style={{ height: 260 }}>
                                <ResponsiveContainer width="100%" height="100%">
                                    <LineChart data={generations} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                                        <XAxis dataKey="generation" tick={{ fontSize: 11, fill: "var(--text-3)" }} tickLine={false} axisLine={false} />
                                        <YAxis yAxisId="left" tick={{ fontSize: 11, fill: "var(--text-3)" }} tickLine={false} axisLine={false} />
                                        <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 11, fill: "var(--text-3)" }} tickLine={false} axisLine={false} />
                                        <RechartsTooltip 
                                            contentStyle={{ background: "rgba(10,10,10,0.9)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 12, fontFamily: "var(--mono)" }}
                                        />
                                        <Line yAxisId="left" type="monotone" dataKey="novelty_mean" name="Mean Novelty" stroke="var(--purple)" strokeWidth={3} dot={false} />
                                        <Line yAxisId="right" type="monotone" dataKey="structural_diversity" name="Struct. Diversity" stroke="var(--blue)" strokeWidth={2} dot={false} />
                                    </LineChart>
                                </ResponsiveContainer>
                            </div>
                        </div>
                    </div>

                    {/* Deep-Dive Generation Log */}
                    <div>
                        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 24 }}>
                            <h2 style={{ fontSize: 20, fontWeight: 700 }}>Deep-Dive Generation Log</h2>
                            <Badge color="var(--blue)">Interactive Drilldown</Badge>
                        </div>
                        <div style={{ display: "flex", flexDirection: "column" }}>
                            {generations.map((gen, idx) => (
                                <GenerationRow 
                                    key={gen.generation} 
                                    gen={gen} 
                                    prevGen={idx > 0 ? generations[idx - 1] : null}
                                    index={idx}
                                />
                            ))}
                        </div>
                    </div>

                </motion.div>
            )}
        </div>
    );
};

export default DiagnosticsPage;
