import React from 'react';
import { NavLink } from 'react-router-dom';

const NAV = [
    { path: "/", label: "Welcome", icon: "⌘", desc: "Introduction" },
    { path: "/about", label: "The Problem", icon: "◈", desc: "Motivation" },
    { path: "/system", label: "Architecture", icon: "⬡", desc: "System Design" },
    { path: "/evolution", label: "Evolution", icon: "◉", desc: "How GP Works" },
    { path: "/map", label: "MAP-Elites", icon: "▦", desc: "QD Archive" },
    { path: "/playground", label: "Formula Lab", icon: "⚗", desc: "Interactive" },
    { path: "/sandbox", label: "Web Sandbox", icon: "⏵", desc: "Live Sim" },
    { path: "/schedules", label: "Baselines", icon: "≋", desc: "Comparison" },
    { path: "/diagnostics", label: "Analytics", icon: "⏣", desc: "Run Data" },
    { path: "/rust", label: "Rust Engine", icon: "⚙", desc: "Performance" },
];

const Sidebar = ({ collapsed, setCollapsed }) => {
    return (
        <aside style={{
            position: "fixed", left: 0, top: 0, bottom: 0,
            width: collapsed ? 72 : 220,
            zIndex: 100,
            transition: "width 0.3s cubic-bezier(0.25,0.46,0.45,0.94)",
            background: "rgba(0,0,0,0.8)",
            backdropFilter: "blur(40px)",
            borderRight: "1px solid var(--border)",
            display: "flex", flexDirection: "column",
            overflow: "hidden"
        }}>
            {/* Logo */}
            <div style={{
                padding: collapsed ? "24px 0" : "24px 20px",
                borderBottom: "1px solid var(--border)",
                display: "flex", alignItems: "center",
                justifyContent: collapsed ? "center" : "space-between"
            }}>
                {!collapsed && (
                    <div>
                        <div style={{
                            fontSize: 18, fontWeight: 700, letterSpacing: "-0.03em",
                            background: "linear-gradient(135deg, #fff 0%, rgba(255,255,255,0.6) 100%)",
                            WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent"
                        }}>SymboLR</div>
                        <div style={{ fontSize: 10, color: "var(--text-3)", letterSpacing: "0.1em", marginTop: 2 }}>
                            SYMBOLIC GP SYSTEM
                        </div>
                    </div>
                )}
                <button
                    onClick={() => setCollapsed(!collapsed)}
                    style={{
                        background: "var(--bg2)", border: "1px solid var(--border)",
                        borderRadius: 8, width: 28, height: 28, color: "var(--text-3)",
                        display: "flex", alignItems: "center", justifyContent: "center",
                        fontSize: 12, flexShrink: 0,
                        transition: "all var(--transition)"
                    }}
                >{collapsed ? "→" : "←"}</button>
            </div>

            {/* Nav items */}
            <nav style={{ flex: 1, padding: "12px 8px", overflowY: "auto", display: "flex", flexDirection: "column", gap: "2px" }}>
                {NAV.map((item, i) => (
                    <NavLink
                        key={item.path}
                        to={item.path}
                        style={({ isActive }) => ({
                            width: "100%", textAlign: "left",
                            padding: collapsed ? "10px 0" : "10px 12px",
                            borderRadius: 10,
                            textDecoration: "none",
                            background: isActive ? "rgba(10,132,255,0.15)" : "transparent",
                            border: `1px solid ${isActive ? "rgba(10,132,255,0.3)" : "transparent"}`,
                            color: isActive ? "var(--blue)" : "var(--text-2)",
                            transition: "all var(--transition)",
                            display: "flex", alignItems: "center",
                            gap: 10, cursor: "pointer",
                            animation: `fadeUp ${0.3 + i * 0.04}s ease both`
                        })}
                    >
                        {({ isActive }) => (
                            <>
                                <span style={{
                                    fontSize: 16, width: 20, textAlign: "center", flexShrink: 0,
                                    ...(collapsed ? { margin: "0 auto" } : {})
                                }}>{item.icon}</span>
                                {!collapsed && (
                                    <div style={{ overflow: "hidden" }}>
                                        <div style={{ fontSize: 13, fontWeight: 500, whiteSpace: "nowrap" }}>{item.label}</div>
                                        <div style={{ fontSize: 10, color: "var(--text-3)", whiteSpace: "nowrap" }}>{item.desc}</div>
                                    </div>
                                )}
                            </>
                        )}
                    </NavLink>
                ))}
            </nav>

            {/* Status indicator */}
            <div style={{
                padding: collapsed ? "16px 0" : "16px 20px",
                borderTop: "1px solid var(--border)",
                display: "flex", alignItems: "center", gap: 8,
                justifyContent: collapsed ? "center" : "flex-start"
            }}>
                <div style={{
                    width: 8, height: 8, borderRadius: "50%",
                    background: "var(--green)",
                    animation: "pulse 2s ease infinite",
                    boxShadow: "0 0 8px var(--green)", flexShrink: 0
                }} />
                {!collapsed && <span style={{ fontSize: 11, color: "var(--text-3)" }}>System Online</span>}
            </div>
        </aside>
    );
};

export default Sidebar;
