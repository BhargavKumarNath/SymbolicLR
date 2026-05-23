import React, { useMemo } from 'react';
import HallOfFame from '../components/HallOfFame';
import { ResponsiveContainer, PieChart, Pie, Cell, Tooltip } from 'recharts';

export default function Archive({ data }) {
  const { hall_of_fame } = data;

  const familyData = useMemo(() => {
    if (!hall_of_fame) return [];
    const counts = {};
    hall_of_fame.forEach(item => {
      counts[item.family] = (counts[item.family] || 0) + 1;
    });
    return Object.keys(counts).map(key => ({ name: key, value: counts[key] }));
  }, [hall_of_fame]);

  const COLORS = ['var(--accent-blue)', 'var(--accent-green)', 'var(--accent-orange)', 'var(--accent-purple)', '#FF375F', '#64D2FF'];

  return (
    <div>
      <h1 className="page-title">Archive Explorer</h1>
      <p className="page-subtitle">The final MAP-Elites Pareto Front and symbolic discoveries.</p>

      <div className="grid-2">
        <div className="glass-panel" style={{ display: 'flex', flexDirection: 'column' }}>
          <h2 className="section-title">Schedule Family Distribution</h2>
          <div style={{ flex: 1, minHeight: '300px' }}>
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={familyData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={100}
                  paddingAngle={5}
                  dataKey="value"
                  stroke="none"
                >
                  {familyData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip 
                  contentStyle={{ backgroundColor: 'var(--panel-bg)', borderColor: 'var(--panel-border)', borderRadius: '8px', backdropFilter: 'blur(10px)' }}
                  itemStyle={{ color: 'var(--text-primary)' }}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>
        
        <div className="glass-panel" style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
          <h2 className="section-title">Archive Stats</h2>
          <div style={{ display: 'flex', justifyContent: 'space-between', padding: '16px 0', borderBottom: '1px solid var(--panel-border)' }}>
            <span style={{ color: 'var(--text-secondary)' }}>Total Niches Filled</span>
            <span style={{ fontWeight: 600 }}>{hall_of_fame?.length || 0}</span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', padding: '16px 0', borderBottom: '1px solid var(--panel-border)' }}>
            <span style={{ color: 'var(--text-secondary)' }}>Most Complex Formula Size</span>
            <span style={{ fontWeight: 600 }}>{Math.max(...(hall_of_fame?.map(h => h.size) || [0]))}</span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', padding: '16px 0' }}>
            <span style={{ color: 'var(--text-secondary)' }}>Simplest Formula Size</span>
            <span style={{ fontWeight: 600 }}>{Math.min(...(hall_of_fame?.map(h => h.size) || [0]))}</span>
          </div>
        </div>
      </div>

      <div className="glass-panel">
        <h2 className="section-title">Hall of Fame (Pareto Front)</h2>
        <HallOfFame data={hall_of_fame} />
      </div>
    </div>
  );
}
