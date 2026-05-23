import React from 'react';

const AmbientBg = () => (
    <div style={{ position: "fixed", inset: 0, zIndex: 0, overflow: "hidden", pointerEvents: "none" }}>
        <div style={{
            position: "absolute", width: 800, height: 800,
            borderRadius: "50%", top: "-20%", left: "-10%",
            background: "radial-gradient(circle, rgba(10,132,255,0.12) 0%, transparent 70%)",
            animation: "orb-1 18s ease-in-out infinite"
        }} />
        <div style={{
            position: "absolute", width: 600, height: 600,
            borderRadius: "50%", bottom: "-10%", right: "5%",
            background: "radial-gradient(circle, rgba(191,90,242,0.1) 0%, transparent 70%)",
            animation: "orb-2 22s ease-in-out infinite"
        }} />
        <div style={{
            position: "absolute", width: 400, height: 400,
            borderRadius: "50%", top: "50%", left: "50%", transform: "translate(-50%,-50%)",
            background: "radial-gradient(circle, rgba(48,209,88,0.06) 0%, transparent 70%)"
        }} />
        {/* Grid overlay */}
        <div style={{
            position: "absolute", inset: 0,
            backgroundImage: `linear-gradient(rgba(255,255,255,0.02) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.02) 1px, transparent 1px)`,
            backgroundSize: "60px 60px"
        }} />
        {/* Scan line */}
        <div style={{
            position: "absolute", left: 0, right: 0, height: "1px",
            background: "linear-gradient(90deg, transparent, rgba(10,132,255,0.3), transparent)",
            animation: "scan 8s linear infinite", top: 0
        }} />
    </div>
);

export default AmbientBg;
