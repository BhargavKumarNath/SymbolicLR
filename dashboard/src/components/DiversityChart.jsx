import React from 'react';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts';

export default function DiversityChart({ data }) {
  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--panel-border)" vertical={false} />
        <XAxis 
          dataKey="generation" 
          stroke="var(--text-secondary)" 
          fontSize={12} 
          tickLine={false} 
          axisLine={false} 
        />
        <YAxis 
          stroke="var(--text-secondary)" 
          fontSize={12} 
          tickLine={false} 
          axisLine={false} 
          domain={[0, 1]}
          tickFormatter={(val) => val.toFixed(1)}
        />
        <Tooltip 
          contentStyle={{ backgroundColor: 'var(--panel-bg)', borderColor: 'var(--panel-border)', borderRadius: '8px', backdropFilter: 'blur(10px)' }}
          itemStyle={{ color: 'var(--text-primary)' }}
          labelStyle={{ color: 'var(--text-secondary)' }}
        />
        <Line 
          type="monotone" 
          dataKey="structural_diversity" 
          stroke="var(--accent-green)" 
          strokeWidth={2}
          dot={false} 
          isAnimationActive={false}
          name="Structural Diversity"
        />
        <Line 
          type="monotone" 
          dataKey="behavioral_diversity" 
          stroke="var(--accent-purple)" 
          strokeWidth={2}
          dot={false} 
          isAnimationActive={false}
          name="Behavioral Diversity"
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
