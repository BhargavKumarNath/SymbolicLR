import React from 'react';
import EvolutionChart from '../components/EvolutionChart';

export default function Evolution({ data }) {
  return (
    <div>
      <h1 className="page-title">Evolution Dynamics</h1>
      <p className="page-subtitle">Tracking the optimization loss landscape across generations, overlaid with Meta-Controller phase shifts.</p>

      <div className="glass-panel" style={{ minHeight: '500px', padding: '32px' }}>
        <h2 className="section-title">Validation Loss Trajectory</h2>
        <div style={{ height: '400px' }}>
          <EvolutionChart data={data.generations} />
        </div>
      </div>
    </div>
  );
}
