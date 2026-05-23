import React from 'react';

const StatCard = ({ label, value, unit = "", color = "var(--blue)", sublabel, delay = 0 }) => (
    <div className="glass" style={{
        padding: "20px 24px",
        animation: `fadeUp 0.5s ease ${delay}s both`,
        borderColor: `${color}33`
    }}>
        <div style={{ fontSize: 11, color: "var(--text-3)", letterSpacing: "0.08em", marginBottom: 8, textTransform: "uppercase" }}>
            {label}
        </div>
        <div style={{
            fontSize: 32, fontWeight: 700, letterSpacing: "-0.04em",
            color, fontVariantNumeric: "tabular-nums"
        }}>
            {value}<span style={{ fontSize: 16, fontWeight: 400, marginLeft: 4, color: "var(--text-3)" }}>{unit}</span>
        </div>
        {sublabel && <div style={{ fontSize: 11, color: "var(--text-3)", marginTop: 6 }}>{sublabel}</div>}
    </div>
);

export default StatCard;
