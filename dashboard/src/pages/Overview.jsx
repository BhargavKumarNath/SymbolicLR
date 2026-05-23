import React from 'react';
import { Target, GitMerge, Cpu, Network } from 'lucide-react';
import { ResponsiveContainer, LineChart, Line } from 'recharts';

function MetricCard({ title, value, icon: Icon, color, data, dataKey }) {
  return (
    <div className="glass-panel" style={{ display: 'flex', flexDirection: 'column', padding: '20px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '16px' }}>
        <div>
          <h3 style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', fontWeight: 500, marginBottom: '4px' }}>{title}</h3>
          <span style={{ fontSize: '2rem', fontWeight: 700, color: 'var(--text-primary)' }}>{value}</span>
        </div>
        <div style={{ padding: '10px', backgroundColor: `var(--accent-${color}20)`, borderRadius: '12px' }}>
          <Icon size={24} color={`var(--accent-${color})`} />
        </div>
      </div>
      
      {data && (
        <div style={{ height: '60px', marginTop: 'auto' }}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data}>
              <Line type="monotone" dataKey={dataKey} stroke={`var(--accent-${color})`} strokeWidth={2} dot={false} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}

export default function Overview({ data }) {
  const { summary, generations } = data;

  return (
    <div>
      <h1 className="page-title">Research Overview</h1>
      <p className="page-subtitle">High-level summary of the evolutionary search dynamics.</p>

      <div className="grid-4">
        <MetricCard 
          title="Final Best Loss" 
          value={summary.final_best_loss.toFixed(4)} 
          icon={Target} 
          color="blue"
          data={generations}
          dataKey="best_loss"
        />
        <MetricCard 
          title="Total Generations" 
          value={summary.total_generations} 
          icon={Cpu} 
          color="green" 
        />
        <MetricCard 
          title="Archive Size (Niches)" 
          value={summary.final_archive_size || summary.hall_of_fame_size || 0} 
          icon={Network} 
          color="purple" 
          data={generations}
          dataKey="archive_size"
        />
        <MetricCard 
          title="Dominant Operator" 
          value={summary.dominant_operator.split('_')[0]} 
          icon={GitMerge} 
          color="orange" 
        />
      </div>

      <div className="glass-panel">
        <h2 className="section-title">Meta-Controller Phase Timeline</h2>
        <div style={{ display: 'flex', height: '40px', borderRadius: '8px', overflow: 'hidden' }}>
          {generations.map((g, i) => {
            let color = 'transparent';
            if (g.controller_phase === 'EXPLOIT') color = 'var(--accent-blue)';
            if (g.controller_phase === 'EXPLORE') color = 'var(--accent-green)';
            if (g.controller_phase === 'DIVERSIFY') color = 'var(--accent-orange)';
            
            return (
              <div 
                key={i} 
                style={{ flex: 1, backgroundColor: color, opacity: 0.8 }} 
                title={`Gen ${g.generation}: ${g.controller_phase}`}
              />
            );
          })}
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '8px', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
          <span>Generation 0</span>
          <span>Generation {generations.length}</span>
        </div>
        <div style={{ display: 'flex', gap: '16px', marginTop: '16px', fontSize: '0.9rem' }}>
          <span style={{ display: 'flex', alignItems: 'center', gap: '8px' }}><div style={{ width: '12px', height: '12px', borderRadius: '50%', backgroundColor: 'var(--accent-blue)' }} /> Exploit</span>
          <span style={{ display: 'flex', alignItems: 'center', gap: '8px' }}><div style={{ width: '12px', height: '12px', borderRadius: '50%', backgroundColor: 'var(--accent-green)' }} /> Explore</span>
          <span style={{ display: 'flex', alignItems: 'center', gap: '8px' }}><div style={{ width: '12px', height: '12px', borderRadius: '50%', backgroundColor: 'var(--accent-orange)' }} /> Diversify</span>
        </div>
      </div>
    </div>
  );
}
