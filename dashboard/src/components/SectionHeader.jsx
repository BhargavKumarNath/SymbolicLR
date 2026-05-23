import React from 'react';

const SectionHeader = ({ title, subtitle, tag }) => (
    <div style={{ marginBottom: 40, animation: "fadeUp 0.5s ease both" }}>
        {tag && (
            <div style={{
                display: "inline-flex", alignItems: "center", gap: 6,
                padding: "4px 12px", borderRadius: 20,
                background: "var(--blue-dim)", border: "1px solid rgba(10,132,255,0.2)",
                color: "var(--blue)", fontSize: 11, fontWeight: 600,
                letterSpacing: "0.06em", textTransform: "uppercase", marginBottom: 16
            }}>{tag}</div>
        )}
        <h1 style={{
            fontSize: 42, fontWeight: 700, letterSpacing: "-0.04em", lineHeight: 1.1,
            background: "linear-gradient(135deg, #fff 0%, rgba(255,255,255,0.5) 100%)",
            WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent"
        }}>{title}</h1>
        {subtitle && (
            <p style={{ fontSize: 16, color: "var(--text-2)", marginTop: 12, lineHeight: 1.7, maxWidth: 600 }}>
                {subtitle}
            </p>
        )}
    </div>
);

export default SectionHeader;
