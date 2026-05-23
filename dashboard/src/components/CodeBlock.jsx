import React from 'react';

const CodeBlock = ({ code, lang = "" }) => (
    <div style={{
        background: "rgba(0,0,0,0.5)", border: "1px solid var(--border)",
        borderRadius: 10, padding: "16px 20px", overflowX: "auto"
    }}>
        {lang && <div style={{ fontSize: 10, color: "var(--text-3)", marginBottom: 10, letterSpacing: "0.08em", textTransform: "uppercase" }}>{lang}</div>}
        <pre style={{
            fontFamily: "var(--mono)", fontSize: 12, lineHeight: 1.7,
            color: "var(--text-2)", margin: 0
        }}>{code}</pre>
    </div>
);

export default CodeBlock;
