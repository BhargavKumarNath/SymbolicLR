import React from 'react';
import EvolutionChart from './EvolutionChart';
import OperatorChart from './OperatorChart';
import DiversityChart from './DiversityChart';
import HallOfFame from './HallOfFame';
import { ChevronLeft } from 'lucide-react';

export default function Dashboard({ data, onReset }) {
  const { summary, generations, hall_of_fame } = data;

  return (
    <div className="fade-in" style={{ padding: '32px', maxWidth: '1400px', margin: '0 auto' }}>
      
      {/* Header */}
      <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', marginBottom: '32px' }}>
        <div>
          <button 
            onClick={onReset}
            style={{ 
              background: 'none', border: 'none', color: 'var(--text-secondary)', 
              cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '4px',
              padding: 0, marginBottom: '16px', fontSize: '0.9rem'
            }}>
            <ChevronLeft size={16} /> Load different run
          </button>
          <h1 style={{ fontSize: '2.5rem', fontWeight: 700, letterSpacing: '-0.02em' }}>Evolution Dynamics</h1>
          <p style={{ color: 'var(--text-secondary)', fontSize: '1.1rem' }}>
            Final Loss: <span style={{ color: 'var(--text-primary)', fontWeight: 600 }}>{summary.final_best_loss.toFixed(4)}</span> • 
            Generations: <span style={{ color: 'var(--text-primary)', fontWeight: 600 }}>{summary.total_generations}</span> •
            Dominant Operator: <span style={{ color: 'var(--accent-blue)', fontWeight: 600 }}>{summary.dominant_operator}</span>
          </p>
        </div>
      </header>

      {/* Grid Layout */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(600px, 1fr))', gap: '24px', marginBottom: '24px' }}>
        <div className="glass-panel" style={{ minHeight: '350px' }}>
          <h2 style={{ fontSize: '1.2rem', marginBottom: '16px', fontWeight: 600 }}>Loss & Meta-Controller Phases</h2>
          <EvolutionChart data={generations} />
        </div>
        
        <div className="glass-panel" style={{ minHeight: '350px' }}>
          <h2 style={{ fontSize: '1.2rem', marginBottom: '16px', fontWeight: 600 }}>Operator Bandit Probabilities</h2>
          <OperatorChart data={generations} />
        </div>
        
        <div className="glass-panel" style={{ minHeight: '350px' }}>
          <h2 style={{ fontSize: '1.2rem', marginBottom: '16px', fontWeight: 600 }}>Structural & Behavioral Diversity</h2>
          <DiversityChart data={generations} />
        </div>
      </div>

      <div className="glass-panel" style={{ minHeight: '350px' }}>
        <h2 style={{ fontSize: '1.2rem', marginBottom: '16px', fontWeight: 600 }}>Hall of Fame (Pareto Front)</h2>
        <HallOfFame data={hall_of_fame} />
      </div>

    </div>
  );
}
