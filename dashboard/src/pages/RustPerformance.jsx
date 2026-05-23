import React from 'react';
import { Cpu, Zap, Activity } from 'lucide-react';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid, Cell } from 'recharts';

export default function RustPerformance({ data }) {
  const { summary } = data;
  
  // Graceful degradation if rust metrics are missing
  const hasRustMetrics = summary.rust_accelerated !== undefined || summary.throughput_evals_per_sec !== undefined;
  
  const isAccelerated = summary.rust_accelerated === true;
  const throughput = summary.throughput_evals_per_sec || 0;
  const runtime = summary.total_runtime || 0;

  const barData = [
    { name: 'This Run', time: runtime, color: isAccelerated ? 'var(--accent-blue)' : 'var(--accent-orange)' }
  ];

  if (!hasRustMetrics) {
    return (
      <div>
        <h1 className="page-title">Rust Backend Performance</h1>
        <p className="page-subtitle">Analyze evaluation throughput and latency.</p>
        <div className="glass-panel" style={{ textAlign: 'center', padding: '64px 24px' }}>
          <Cpu size={48} color="var(--text-secondary)" style={{ margin: '0 auto 16px' }} />
          <h2 style={{ fontSize: '1.2rem', marginBottom: '8px' }}>No Performance Metrics Found</h2>
          <p style={{ color: 'var(--text-secondary)' }}>The loaded diagnostics file does not contain Rust acceleration telemetry. Ensure you are running SymboLR with the Rust core enabled.</p>
        </div>
      </div>
    );
  }

  return (
    <div>
      <h1 className="page-title">Rust Backend Performance</h1>
      <p className="page-subtitle">Analyze evaluation throughput and execution acceleration.</p>

      <div className="grid-3">
        <div className="glass-panel" style={{ padding: '24px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '16px' }}>
            <div>
              <h3 style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', fontWeight: 500, marginBottom: '4px' }}>Backend Status</h3>
              <span style={{ fontSize: '1.5rem', fontWeight: 700, color: isAccelerated ? 'var(--accent-blue)' : 'var(--accent-orange)' }}>
                {isAccelerated ? 'Rust Accelerated' : 'Python Fallback'}
              </span>
            </div>
            <div style={{ padding: '10px', backgroundColor: isAccelerated ? 'var(--accent-blue)20' : 'var(--accent-orange)20', borderRadius: '12px' }}>
              <Cpu size={24} color={isAccelerated ? 'var(--accent-blue)' : 'var(--accent-orange)'} />
            </div>
          </div>
        </div>

        <div className="glass-panel" style={{ padding: '24px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '16px' }}>
            <div>
              <h3 style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', fontWeight: 500, marginBottom: '4px' }}>Throughput</h3>
              <span style={{ fontSize: '2rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                {throughput.toFixed(0)} <span style={{ fontSize: '1rem', color: 'var(--text-secondary)', fontWeight: 500 }}>evals/s</span>
              </span>
            </div>
            <div style={{ padding: '10px', backgroundColor: 'var(--accent-green)20', borderRadius: '12px' }}>
              <Zap size={24} color="var(--accent-green)" />
            </div>
          </div>
        </div>

        <div className="glass-panel" style={{ padding: '24px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '16px' }}>
            <div>
              <h3 style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', fontWeight: 500, marginBottom: '4px' }}>Total Execution Time</h3>
              <span style={{ fontSize: '2rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                {runtime.toFixed(1)} <span style={{ fontSize: '1rem', color: 'var(--text-secondary)', fontWeight: 500 }}>s</span>
              </span>
            </div>
            <div style={{ padding: '10px', backgroundColor: 'var(--accent-purple)20', borderRadius: '12px' }}>
              <Activity size={24} color="var(--accent-purple)" />
            </div>
          </div>
        </div>
      </div>

      <div className="glass-panel" style={{ minHeight: '400px' }}>
        <h2 className="section-title">Execution Profile</h2>
        <div style={{ height: '300px' }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={barData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }} barSize={60}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--panel-border)" vertical={false} />
              <XAxis dataKey="name" stroke="var(--text-secondary)" tickLine={false} axisLine={false} />
              <YAxis stroke="var(--text-secondary)" tickLine={false} axisLine={false} label={{ value: 'Seconds', angle: -90, position: 'insideLeft', fill: 'var(--text-secondary)' }} />
              <Tooltip 
                contentStyle={{ backgroundColor: 'var(--panel-bg)', borderColor: 'var(--panel-border)', borderRadius: '8px' }}
                itemStyle={{ color: 'var(--text-primary)' }}
                cursor={{ fill: 'rgba(255,255,255,0.05)' }}
              />
              <Bar dataKey="time" radius={[8, 8, 0, 0]}>
                {barData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
