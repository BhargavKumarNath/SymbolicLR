import React from 'react';

const ChartCard = ({ title, subtitle, children, style = {} }) => (
    <div className="glass" style={{ padding: "24px", ...style }}>
        <div style={{ marginBottom: 20 }}>
            <h3 style={{ fontSize: 15, fontWeight: 600, letterSpacing: "-0.02em" }}>{title}</h3>
            {subtitle && <p style={{ fontSize: 12, color: "var(--text-3)", marginTop: 4 }}>{subtitle}</p>}
        </div>
        {children}
    </div>
);

export default ChartCard;
