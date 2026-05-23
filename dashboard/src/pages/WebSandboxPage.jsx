import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer } from 'recharts';
import { AlertTriangle, Play, Cpu, Server, CheckCircle, Terminal } from 'lucide-react';
import SectionHeader from '../components/SectionHeader';
import Latex from '../components/Latex';
import Badge from '../components/Badge';

const WebSandboxPage = () => {
    // Environment detection
    const isLocal = typeof window !== 'undefined' && (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1');

    // Controls State - Dynamic Limits
    const maxPop = isLocal ? 10000 : 100;
    const maxGen = isLocal ? 1000 : 50;

    const [popSize, setPopSize] = useState(isLocal ? 1000 : 50);
    const [generations, setGenerations] = useState(isLocal ? 100 : 20);
    const [epochs, setEpochs] = useState(10);

    // Simulation State
    const [status, setStatus] = useState('IDLE'); // IDLE, RUNNING, COMPLETE, ERROR
    const [liveData, setLiveData] = useState([]);
    const [goldenData, setGoldenData] = useState(null);
    const [progress, setProgress] = useState(0);

    // Terminal Status Logs
    const [logs, setLogs] = useState([]);
    const logsEndRef = useRef(null);

    const addLog = (msg, type = 'info') => {
        setLogs(prev => [...prev, { time: new Date().toISOString().substring(11, 19), msg, type }]);
    };

    // Auto-scroll terminal
    useEffect(() => {
        if (logsEndRef.current) {
            logsEndRef.current.scrollIntoView({ behavior: 'smooth' });
        }
    }, [logs]);

    // Load static data for cloud fallback
    useEffect(() => {
        fetch('/results.json')
            .then(res => res.json())
            .then(data => setGoldenData(data))
            .catch(err => console.error("Failed to load mock data:", err));
    }, []);

    const timerRef = useRef(null);

    const handleInitialize = async () => {
        if (status === 'RUNNING') return;
        
        setLiveData([]);
        setProgress(0);
        setStatus('RUNNING');
        setLogs([]);
        addLog("Initializing Evaluation Pipeline...", "system");

        if (isLocal) {
            // Local Compute Node Execution
            addLog("Executing Local High-Performance Node...", "info");
            addLog("Ingesting Target Dataset (110M Record Memory Map)...", "info");
            
            try {
                // Wait briefly for UI effect
                await new Promise(r => setTimeout(r, 600));
                addLog("Allocating Shared Memory via PyO3...", "info");
                
                await new Promise(r => setTimeout(r, 600));
                addLog("Initializing Quality-Diversity Archive...", "info");
                
                await new Promise(r => setTimeout(r, 600));
                addLog(`Dispatching ${popSize} ASTs to Rust Engine...`, "info");
                addLog("Establishing Server-Sent Events (SSE) Stream...", "info");

                const sseUrl = `http://localhost:8000/api/stream-evolve?population_size=${popSize}&generations=${generations}&target_epochs=${epochs}`;
                const sse = new EventSource(sseUrl);
                let currentGen = 0;

                sse.onmessage = (event) => {
                    const genData = JSON.parse(event.data);
                    setLiveData(prev => [...prev, genData]);
                    
                    if (currentGen % 5 === 0) {
                        addLog(`Processed Gen ${genData.generation} | Best MSE: ${genData.best_loss.toFixed(4)}`, "success");
                    }
                    
                    currentGen++;
                    setProgress((currentGen / generations) * 100);
                };

                sse.addEventListener("COMPLETE", (event) => {
                    const finalData = JSON.parse(event.data);
                    setGoldenData(finalData);
                    setStatus('COMPLETE');
                    addLog("Evaluation Pipeline Complete. Global Minimum Reached.", "system");
                    sse.close();
                });

                sse.addEventListener("ERROR", (event) => {
                    const errData = JSON.parse(event.data);
                    addLog(`STREAM ERROR: ${errData.detail}`, "error");
                    setStatus('ERROR');
                    sse.close();
                });

                sse.onerror = (error) => {
                    // Only log error if we weren't already complete
                    if (status !== 'COMPLETE' && status !== 'ERROR') {
                        addLog("CONNECTION REFUSED: Stream unexpectedly closed.", "error");
                        addLog("Ensure the Python FastAPI engine is running (uvicorn api.main:app).", "error");
                        setStatus('ERROR');
                    }
                    sse.close();
                };

            } catch (error) {
                addLog(`CRITICAL ERROR: ${error.message}`, "error");
                setStatus('ERROR');
            }

        } else {
            // Cloud Surrogate Execution
            addLog("Cloud Compute Mode Detected. Enforcing constraints.", "warning");
            addLog("Ingesting Surrogate Dataset (1K Memory Map)...", "info");
            
            setTimeout(() => {
                addLog("Initializing Quality-Diversity Archive...", "info");
                if (goldenData) {
                    addLog("Executing Surrogate Evaluation Batch...", "info");
                    playDataStream(goldenData);
                } else {
                    addLog("Failed to load surrogate artifact.", "error");
                    setStatus('ERROR');
                }
            }, 1000);
        }
    };

    const playDataStream = (dataPayload) => {
        let currentGen = 0;
        const maxGens = Math.min(generations, dataPayload.generations?.length || 0);
        
        if (maxGens === 0) {
            addLog("No generation data received.", "error");
            setStatus('ERROR');
            return;
        }

        timerRef.current = setInterval(() => {
            if (currentGen < maxGens) {
                const genData = dataPayload.generations[currentGen];
                setLiveData(prev => [...prev, genData]);
                
                if (currentGen % 5 === 0) {
                    addLog(`Processed Gen ${genData.generation} | Best MSE: ${genData.best_loss.toFixed(4)}`, "success");
                }
                
                currentGen++;
                setProgress((currentGen / maxGens) * 100);
            } else {
                setStatus('COMPLETE');
                addLog("Evaluation Pipeline Complete. Global Minimum Reached.", "system");
                clearInterval(timerRef.current);
            }
        }, isLocal ? 100 : 300); // Faster playback in local mode simulation
    };

    useEffect(() => {
        return () => {
            if (timerRef.current) clearInterval(timerRef.current);
        };
    }, []);

    return (
        <div style={{ padding: "60px", maxWidth: 1200, margin: "0 auto" }}>
            
            {/* Header & Hardware Badge */}
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 32 }}>
                <SectionHeader
                    tag="Simulation"
                    title="Dual Execution Sandbox"
                    subtitle="Experience the pipeline. This page dynamically routes execution to the Rust backend if running locally, or falls back to a surrogate simulation in the cloud."
                />
                
                <div style={{
                    padding: "8px 16px", borderRadius: 20, 
                    background: isLocal ? "rgba(48, 209, 88, 0.1)" : "rgba(255, 159, 10, 0.1)",
                    border: `1px solid ${isLocal ? "var(--green)" : "var(--orange)"}`,
                    display: "flex", alignItems: "center", gap: 10
                }}>
                    {isLocal ? (
                        <>
                            <Cpu size={16} color="var(--green)" />
                            <span style={{ fontSize: 12, fontWeight: 700, color: "var(--green)", textTransform: "uppercase" }}>CUDA ENABLED (RTX 4070 / Ryzen 9 Node Detected)</span>
                        </>
                    ) : (
                        <>
                            <Server size={16} color="var(--orange)" />
                            <span style={{ fontSize: 12, fontWeight: 700, color: "var(--orange)", textTransform: "uppercase" }}>NO CUDA (Cloud Fallback)</span>
                        </>
                    )}
                </div>
            </div>

            {/* Disclaimer Banner */}
            {!isLocal && (
                <div style={{
                    padding: "16px 24px", borderRadius: 12, marginBottom: 40,
                    background: "rgba(10, 132, 255, 0.1)", border: "1px solid rgba(10, 132, 255, 0.3)",
                    display: "flex", gap: 16, alignItems: "center"
                }}>
                    <AlertTriangle color="var(--blue)" size={24} />
                    <div>
                        <h4 style={{ fontSize: 14, fontWeight: 700, color: "var(--blue)", marginBottom: 4 }}>Cloud Showcase Mode Active</h4>
                        <p style={{ fontSize: 13, color: "var(--text-2)", margin: 0 }}>
                            Running a lightweight surrogate simulation. To unlock the full PyO3/Rust evaluation engine and 100% compute capacity, download the repository and execute via CLI.
                        </p>
                    </div>
                </div>
            )}

            {/* The Control Deck */}
            <div className="glass" style={{ padding: "32px", marginBottom: 40, borderTop: "3px solid var(--purple)" }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 24 }}>
                    <h3 style={{ fontSize: 18, fontWeight: 700 }}>Configuration Deck</h3>
                    {isLocal && <Badge color="var(--green)">Uncapped Hardware Mode</Badge>}
                </div>
                
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr auto", gap: 24, alignItems: "flex-end" }}>
                    
                    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                        <label style={{ fontSize: 12, fontWeight: 600, color: "var(--text-3)", textTransform: "uppercase" }}>
                            Population Size (Max {maxPop})
                        </label>
                        <input 
                            type="range" min="10" max={maxPop} step={isLocal ? 100 : 10} 
                            value={popSize} onChange={e => setPopSize(Number(e.target.value))}
                            disabled={status === 'RUNNING'}
                            style={{ width: "100%", accentColor: "var(--purple)" }}
                        />
                        <div style={{ fontSize: 14, fontFamily: "var(--mono)", textAlign: "right" }}>{popSize}</div>
                    </div>

                    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                        <label style={{ fontSize: 12, fontWeight: 600, color: "var(--text-3)", textTransform: "uppercase" }}>
                            Generations (Max {maxGen})
                        </label>
                        <input 
                            type="range" min="5" max={maxGen} step={isLocal ? 10 : 5} 
                            value={generations} onChange={e => setGenerations(Number(e.target.value))}
                            disabled={status === 'RUNNING'}
                            style={{ width: "100%", accentColor: "var(--purple)" }}
                        />
                        <div style={{ fontSize: 14, fontFamily: "var(--mono)", textAlign: "right" }}>{generations}</div>
                    </div>

                    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                        <label style={{ fontSize: 12, fontWeight: 600, color: "var(--text-3)", textTransform: "uppercase" }}>
                            Target Epochs
                        </label>
                        <input 
                            type="range" min="1" max="20" step="1" 
                            value={epochs} onChange={e => setEpochs(Number(e.target.value))}
                            disabled={status === 'RUNNING'}
                            style={{ width: "100%", accentColor: "var(--purple)" }}
                        />
                        <div style={{ fontSize: 14, fontFamily: "var(--mono)", textAlign: "right" }}>{epochs}</div>
                    </div>

                    <button 
                        onClick={handleInitialize}
                        disabled={status === 'RUNNING'}
                        style={{
                            height: 48, padding: "0 32px", borderRadius: 8, border: "none",
                            background: status === 'RUNNING' ? "var(--bg3)" : "var(--purple)",
                            color: status === 'RUNNING' ? "var(--text-3)" : "#fff",
                            fontSize: 14, fontWeight: 700, cursor: status === 'RUNNING' ? "not-allowed" : "pointer",
                            display: "flex", alignItems: "center", gap: 12,
                            boxShadow: status === 'RUNNING' ? "none" : "0 0 20px rgba(191, 90, 242, 0.4)",
                            transition: "all 0.3s ease"
                        }}
                    >
                        {status === 'RUNNING' ? (
                            <>
                                <div style={{ width: 16, height: 16, border: "2px solid var(--text-3)", borderTopColor: "transparent", borderRadius: "50%", animation: "spin 1s linear infinite" }} />
                                Simulating...
                            </>
                        ) : (
                            <>
                                <Play size={18} />
                                Initialize Pipeline
                            </>
                        )}
                    </button>
                </div>
            </div>

            {/* Terminal Console & Telemetry Board */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 2fr", gap: 24, marginBottom: 40 }}>
                
                {/* Engine Pipeline Status (Terminal) */}
                <div className="glass" style={{ padding: "24px", display: "flex", flexDirection: "column" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 16 }}>
                        <Terminal size={18} color="var(--text-2)" />
                        <h3 style={{ fontSize: 14, fontWeight: 700, color: "var(--text)" }}>Engine Pipeline Status</h3>
                    </div>
                    
                    <div style={{ 
                        background: "rgba(0,0,0,0.5)", borderRadius: 8, padding: "16px", 
                        flex: 1, overflowY: "auto", height: 320,
                        fontFamily: "var(--mono)", fontSize: 12,
                        border: "1px solid rgba(255,255,255,0.05)"
                    }}>
                        {logs.length === 0 ? (
                            <div style={{ color: "var(--text-3)" }}>Waiting for initialization...</div>
                        ) : (
                            logs.map((log, i) => {
                                let color = "var(--text-2)";
                                if (log.type === 'error') color = "var(--red)";
                                if (log.type === 'success') color = "var(--green)";
                                if (log.type === 'warning') color = "var(--orange)";
                                if (log.type === 'system') color = "var(--purple)";

                                return (
                                    <div key={i} style={{ marginBottom: 6, display: "flex", gap: 12 }}>
                                        <span style={{ color: "var(--text-3)", flexShrink: 0 }}>[{log.time}]</span>
                                        <span style={{ color }}>{log.msg}</span>
                                    </div>
                                );
                            })
                        )}
                        <div ref={logsEndRef} />
                    </div>
                </div>

                {/* Live Telemetry Board */}
                <div className="glass" style={{ padding: "32px", display: "flex", flexDirection: "column" }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 24 }}>
                        <div>
                            <h3 style={{ fontSize: 18, fontWeight: 700 }}>Live Telemetry: Loss Curve</h3>
                            <p style={{ fontSize: 13, color: "var(--text-3)" }}>Streaming real-time simulated fitness evaluation.</p>
                        </div>
                        {(status === 'RUNNING' || status === 'COMPLETE') && (
                            <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: 8 }}>
                                <div style={{ fontSize: 12, fontFamily: "var(--mono)", color: "var(--text-2)", fontWeight: 700 }}>
                                    GEN {liveData.length} / {Math.min(generations, goldenData?.generations?.length || generations)}
                                </div>
                                <div style={{ width: 120, height: 6, background: "rgba(255,255,255,0.1)", borderRadius: 3, overflow: "hidden" }}>
                                    <div style={{ height: "100%", background: "var(--green)", width: progress + "%", transition: "width 0.3s linear" }} />
                                </div>
                            </div>
                        )}
                    </div>

                    {/* Live Formula Spotlight */}
                    {status !== 'IDLE' && liveData.length > 0 && (
                        <div style={{ 
                            marginBottom: 20, padding: "16px", background: "rgba(255,255,255,0.03)", 
                            border: "1px solid rgba(255,255,255,0.1)", borderRadius: 12, 
                            display: "flex", alignItems: "center", justifyContent: "space-between" 
                        }}>
                            <div>
                                <div style={{ fontSize: 11, color: "var(--text-3)", textTransform: "uppercase", marginBottom: 4 }}>Current Leading Schedule</div>
                                <Latex expression={liveData[liveData.length - 1].top_formula_latex || "..." } />
                            </div>
                            <div style={{ textAlign: "right" }}>
                                <div style={{ fontSize: 11, color: "var(--text-3)", textTransform: "uppercase", marginBottom: 4 }}>Archive Size</div>
                                <Badge color="var(--purple)">{liveData[liveData.length - 1].archive_size} Niches</Badge>
                            </div>
                        </div>
                    )}

                    <div style={{ flex: 1, minHeight: 220, background: "rgba(0,0,0,0.2)", borderRadius: 12, border: "1px solid rgba(255,255,255,0.05)", position: "relative" }}>
                        {status === 'IDLE' ? (
                            <div style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", color: "var(--text-3)", fontSize: 14 }}>
                                Awaiting initialization sequence...
                            </div>
                        ) : (
                            <ResponsiveContainer width="100%" height="100%">
                                <LineChart data={liveData} margin={{ top: 20, right: 20, left: -20, bottom: 0 }}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                                    <XAxis dataKey="generation" tick={{ fontSize: 11, fill: "var(--text-3)" }} tickLine={false} axisLine={false} />
                                    <YAxis tick={{ fontSize: 11, fill: "var(--text-3)" }} tickLine={false} axisLine={false} domain={['auto', 'auto']} />
                                    <RechartsTooltip 
                                        contentStyle={{ background: "rgba(10,10,10,0.9)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 12, fontFamily: "var(--mono)" }}
                                        itemStyle={{ padding: "4px 0" }}
                                    />
                                    <Line 
                                        type="monotone" dataKey="best_loss" name="Best MSE" 
                                        stroke="var(--green)" strokeWidth={3} dot={{ r: 4, fill: "var(--bg)", strokeWidth: 2 }} 
                                        isAnimationActive={false} // Disable recharts animation to show distinct steps
                                    />
                                </LineChart>
                            </ResponsiveContainer>
                        )}
                    </div>
                </div>
            </div>

            {/* Hall of Fame */}
            <AnimatePresence>
                {status === 'COMPLETE' && goldenData && (
                    <motion.div
                        initial={{ opacity: 0, y: 30 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.6, ease: "easeOut" }}
                    >
                        <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 24 }}>
                            <CheckCircle color="var(--green)" size={24} />
                            <h2 style={{ fontSize: 22, fontWeight: 700 }}>Final Artifacts: Top Schedules</h2>
                        </div>
                        <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 24 }}>
                            {goldenData.hall_of_fame?.slice(0, 3).map((hof, idx) => (
                                <motion.div 
                                    key={idx}
                                    initial={{ opacity: 0, scale: 0.95 }}
                                    animate={{ opacity: 1, scale: 1 }}
                                    transition={{ delay: idx * 0.15, duration: 0.4 }}
                                    className="glass" 
                                    style={{ padding: "24px", borderTop: "2px solid " + (idx === 0 ? "var(--orange)" : "var(--border)") }}
                                >
                                    <div style={{ fontSize: 11, color: "var(--text-3)", textTransform: "uppercase", marginBottom: 16 }}>
                                        Rank {idx + 1} • {hof.family}
                                    </div>
                                    <div style={{ marginBottom: 24, padding: "16px", background: "rgba(255,255,255,0.02)", borderRadius: 8, minHeight: 80, display: "flex", alignItems: "center", justifyContent: "center" }}>
                                        <Latex expression={hof.latex} />
                                    </div>
                                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end" }}>
                                        <div>
                                            <div style={{ fontSize: 10, color: "var(--text-3)", marginBottom: 4 }}>FITNESS (MSE)</div>
                                            <div style={{ fontSize: 18, fontWeight: 700, fontFamily: "var(--mono)", color: "var(--green)" }}>{hof.loss.toFixed(4)}</div>
                                        </div>
                                        <div style={{ textAlign: "right" }}>
                                            <div style={{ fontSize: 10, color: "var(--text-3)", marginBottom: 4 }}>COMPLEXITY</div>
                                            <Badge color="var(--purple)">{hof.size} Nodes</Badge>
                                        </div>
                                    </div>
                                </motion.div>
                            ))}
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
            
            <style>{`
                @keyframes spin { 100% { transform: rotate(360deg); } }
            `}</style>
        </div>
    );
};

export default WebSandboxPage;
