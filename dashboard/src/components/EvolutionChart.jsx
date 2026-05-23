import React, { useMemo } from 'react';
import { ResponsiveContainer, ComposedChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ReferenceArea } from 'recharts';

export default function EvolutionChart({ data }) {
  // Extract phase boundaries for ReferenceAreas
  const phaseAreas = useMemo(() => {
    const areas = [];
    if (!data || data.length === 0) return areas;
    
    let currentPhase = data[0].controller_phase;
    let startGen = data[0].generation;
    
    for (let i = 1; i < data.length; i++) {
      if (data[i].controller_phase !== currentPhase || i === data.length - 1) {
        const endGen = i === data.length - 1 ? data[i].generation : data[i - 1].generation;
        areas.push({
          start: startGen,
          end: endGen,
          phase: currentPhase
        });
        currentPhase = data[i].controller_phase;
        startGen = data[i].generation;
      }
    }
    return areas;
  }, [data]);

  const getPhaseColor = (phase) => {
    switch(phase) {
      case 'EXPLOIT': return 'rgba(10, 132, 255, 0.1)'; // accent-blue
      case 'EXPLORE': return 'rgba(50, 215, 75, 0.1)'; // accent-green
      case 'DIVERSIFY': return 'rgba(255, 159, 10, 0.1)'; // accent-orange
      default: return 'transparent';
    }
  };

  return (
    <ResponsiveContainer width="100%" height={300}>
      <ComposedChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--panel-border)" vertical={false} />
        
        {phaseAreas.map((area, idx) => (
          <ReferenceArea 
            key={idx} 
            x1={area.start} 
            x2={area.end} 
            fill={getPhaseColor(area.phase)} 
            strokeOpacity={0} 
          />
        ))}

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
          domain={['auto', 'auto']}
          tickFormatter={(val) => val.toFixed(2)}
        />
        <Tooltip 
          contentStyle={{ backgroundColor: 'var(--panel-bg)', borderColor: 'var(--panel-border)', borderRadius: '8px', backdropFilter: 'blur(10px)' }}
          itemStyle={{ color: 'var(--text-primary)' }}
          labelStyle={{ color: 'var(--text-secondary)' }}
        />
        <defs>
          <linearGradient id="colorLoss" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="var(--accent-blue)" stopOpacity={0.3}/>
            <stop offset="95%" stopColor="var(--accent-blue)" stopOpacity={0}/>
          </linearGradient>
        </defs>
        <Area 
          type="monotone" 
          dataKey="best_loss" 
          stroke="var(--accent-blue)" 
          fillOpacity={1} 
          fill="url(#colorLoss)" 
          strokeWidth={3}
          isAnimationActive={false}
          name="Best Loss"
        />
        <Line 
          type="monotone" 
          dataKey="median_loss" 
          stroke="var(--text-secondary)" 
          strokeWidth={2}
          strokeDasharray="5 5"
          dot={false} 
          isAnimationActive={false}
          name="Median Loss"
        />
      </ComposedChart>
    </ResponsiveContainer>
  );
}
