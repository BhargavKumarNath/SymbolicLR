import React from 'react';

const Badge = ({ children, color = "var(--blue)" }) => (
    <span style={{
        display: "inline-flex", alignItems: "center",
        padding: "3px 10px", borderRadius: 20,
        background: `${color}22`, border: `1px solid ${color}44`,
        color, fontSize: 11, fontWeight: 600
    }}>{children}</span>
);

export default Badge;
