import React from 'react';
import OperatorChart from '../components/OperatorChart';

export default function Operators({ data }) {
  const { generations } = data;
  
  // Extract final operator probabilities
  const finalProbs = generations[generations.length - 1]?.operator_probs || {};
  const operators = Object.keys(finalProbs).sort((a, b) => finalProbs[b] - finalProbs[a]);

  return (
    <div>
      <h1 className="page-title">Operator Intelligence</h1>
      <p className="page-subtitle">Tracking the Multi-Armed Bandit's probability allocations across the evolutionary search.</p>

      <div className="glass-panel" style={{ minHeight: '400px', padding: '32px', marginBottom: '24px' }}>
        <h2 className="section-title">Bandit Probability Shifts Over Time</h2>
        <div style={{ height: '350px' }}>
          <OperatorChart data={generations} />
        </div>
      </div>

      <h2 className="section-title" style={{ marginTop: '32px' }}>Final Operator Dominance</h2>
      <div className="grid-4">
        {operators.map(op => (
          <div key={op} className="glass-panel" style={{ padding: '16px' }}>
            <h3 style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', marginBottom: '8px', wordBreak: 'break-word' }}>{op}</h3>
            <span style={{ fontSize: '1.8rem', fontWeight: 700, color: 'var(--text-primary)' }}>
              {(finalProbs[op] * 100).toFixed(1)}%
            </span>
            <div style={{ width: '100%', height: '4px', backgroundColor: 'rgba(255,255,255,0.1)', borderRadius: '2px', marginTop: '12px', overflow: 'hidden' }}>
              <div style={{ width: `${finalProbs[op] * 100}%`, height: '100%', backgroundColor: 'var(--accent-purple)' }} />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
