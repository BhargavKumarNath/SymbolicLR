import React from 'react';

export default function Diagnostics({ data }) {
  return (
    <div>
      <h1 className="page-title">Diagnostics & Raw Data</h1>
      <p className="page-subtitle">Underlying JSON payload for the current evolutionary run.</p>

      <div className="glass-panel" style={{ overflow: 'hidden', padding: 0 }}>
        <div style={{ padding: '16px 24px', borderBottom: '1px solid var(--panel-border)', backgroundColor: 'rgba(255,255,255,0.02)' }}>
          <h2 className="section-title" style={{ margin: 0 }}>Payload Viewer</h2>
        </div>
        <div style={{ padding: '24px', maxHeight: '70vh', overflowY: 'auto' }}>
          <pre style={{ 
            fontFamily: 'monospace', 
            fontSize: '0.85rem', 
            color: 'var(--accent-green)',
            margin: 0
          }}>
            {JSON.stringify(data, null, 2)}
          </pre>
        </div>
      </div>
    </div>
  );
}
