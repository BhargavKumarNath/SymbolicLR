import React from 'react';
import DiversityChart from '../components/DiversityChart';

export default function Diversity({ data }) {
  const { generations } = data;

  return (
    <div>
      <h1 className="page-title">Diversity & Novelty</h1>
      <p className="page-subtitle">Monitoring population collapse thresholds and semantic exploration.</p>

      <div className="glass-panel" style={{ minHeight: '500px', padding: '32px' }}>
        <h2 className="section-title">Structural & Behavioral Divergence</h2>
        <div style={{ height: '400px' }}>
          <DiversityChart data={generations} />
        </div>
      </div>
    </div>
  );
}
