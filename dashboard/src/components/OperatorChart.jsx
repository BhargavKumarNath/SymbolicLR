import React, { useMemo } from 'react';
import { ResponsiveContainer, AreaChart, Area, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts';

export default function OperatorChart({ data }) {
  // Flatten operator_probs into the main object for Recharts
  const flatData = useMemo(() => {
    return data.map(d => ({
      generation: d.generation,
      ...d.operator_probs
    }));
  }, [data]);

  // Extract all unique operators to map them to Areas
  const operators = useMemo(() => {
    if (data.length === 0) return [];
    return Object.keys(data[0].operator_probs || {});
  }, [data]);

  // Generate a distinct color for each operator
  const colors = [
    'var(--accent-blue)', 
    'var(--accent-green)', 
    'var(--accent-orange)', 
    'var(--accent-purple)', 
    '#FF375F', // Pink
    '#64D2FF'  // Light blue
  ];

  return (
    <ResponsiveContainer width="100%" height={300}>
      <AreaChart data={flatData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
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
          tickFormatter={(val) => `${(val * 100).toFixed(0)}%`}
        />
        <Tooltip 
          contentStyle={{ backgroundColor: 'var(--panel-bg)', borderColor: 'var(--panel-border)', borderRadius: '8px', backdropFilter: 'blur(10px)' }}
          itemStyle={{ color: 'var(--text-primary)' }}
          labelStyle={{ color: 'var(--text-secondary)' }}
          formatter={(val) => `${(val * 100).toFixed(1)}%`}
        />
        {operators.map((op, idx) => (
          <Area 
            key={op}
            type="monotone" 
            dataKey={op} 
            stackId="1" 
            stroke={colors[idx % colors.length]} 
            fill={colors[idx % colors.length]} 
            fillOpacity={0.7}
            isAnimationActive={false}
          />
        ))}
      </AreaChart>
    </ResponsiveContainer>
  );
}
