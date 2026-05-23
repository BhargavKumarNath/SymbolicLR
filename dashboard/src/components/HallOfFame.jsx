import React from 'react';
import katex from 'katex';
import 'katex/dist/katex.min.css';

export default function HallOfFame({ data }) {
  if (!data || data.length === 0) return <p style={{ color: 'var(--text-secondary)' }}>No Hall of Fame data available.</p>;

  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
        <thead>
          <tr style={{ borderBottom: '1px solid var(--panel-border)', color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
            <th style={{ padding: '12px 16px', fontWeight: 500 }}>Rank</th>
            <th style={{ padding: '12px 16px', fontWeight: 500 }}>Val Loss</th>
            <th style={{ padding: '12px 16px', fontWeight: 500 }}>Size</th>
            <th style={{ padding: '12px 16px', fontWeight: 500 }}>Family</th>
            <th style={{ padding: '12px 16px', fontWeight: 500 }}>Mathematical Formula</th>
          </tr>
        </thead>
        <tbody>
          {data.map((item, idx) => {
            const rawHtml = katex.renderToString(item.latex, { throwOnError: false, displayMode: true });
            return (
              <tr 
                key={idx} 
                style={{ 
                  borderBottom: '1px solid rgba(255,255,255,0.05)',
                  transition: 'background-color 0.2s',
                }}
                onMouseOver={(e) => e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.02)'}
                onMouseOut={(e) => e.currentTarget.style.backgroundColor = 'transparent'}
              >
                <td style={{ padding: '16px', fontWeight: 600 }}>{item.rank}</td>
                <td style={{ padding: '16px', color: 'var(--accent-green)', fontWeight: 600 }}>{item.loss.toFixed(4)}</td>
                <td style={{ padding: '16px' }}>{item.size}</td>
                <td style={{ padding: '16px' }}>
                  <span style={{ 
                    padding: '4px 10px', 
                    borderRadius: '12px', 
                    fontSize: '0.8rem', 
                    fontWeight: 600,
                    backgroundColor: 'rgba(255,255,255,0.1)',
                    color: 'var(--text-primary)'
                  }}>
                    {item.family}
                  </span>
                </td>
                <td style={{ padding: '8px 16px', overflowX: 'auto', minWidth: '400px' }}>
                  <div dangerouslySetInnerHTML={{ __html: rawHtml }} style={{ fontSize: '1.1rem' }} />
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
