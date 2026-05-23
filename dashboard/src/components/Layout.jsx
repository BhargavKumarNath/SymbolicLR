import React, { useState } from 'react';
import { 
  Activity, 
  LineChart as ChartLine, 
  GitMerge, 
  Target, 
  Database, 
  TerminalSquare, 
  Cpu
} from 'lucide-react';
import Overview from '../pages/Overview';
import Evolution from '../pages/Evolution';
import Operators from '../pages/Operators';
import Diversity from '../pages/Diversity';
import Archive from '../pages/Archive';
import Diagnostics from '../pages/Diagnostics';
import RustPerformance from '../pages/RustPerformance';

const NAV_ITEMS = [
  { id: 'overview', label: 'Overview', icon: Activity },
  { id: 'evolution', label: 'Evolution Dynamics', icon: ChartLine },
  { id: 'operators', label: 'Operator Intelligence', icon: Target },
  { id: 'diversity', label: 'Diversity & Novelty', icon: GitMerge },
  { id: 'archive', label: 'Archive Explorer', icon: Database },
  { id: 'rust', label: 'Rust Performance', icon: Cpu },
  { id: 'diagnostics', label: 'Diagnostics', icon: TerminalSquare },
];

export default function Layout({ data, onReset }) {
  const [activeTab, setActiveTab] = useState('overview');

  const renderContent = () => {
    switch (activeTab) {
      case 'overview': return <Overview data={data} />;
      case 'evolution': return <Evolution data={data} />;
      case 'operators': return <Operators data={data} />;
      case 'diversity': return <Diversity data={data} />;
      case 'archive': return <Archive data={data} />;
      case 'rust': return <RustPerformance data={data} />;
      case 'diagnostics': return <Diagnostics data={data} />;
      default: return <Overview data={data} />;
    }
  };

  return (
    <div className="layout-container fade-in">
      {/* Sidebar Navigation */}
      <aside className="sidebar glass-panel">
        <div className="sidebar-header">
          <h1 className="logo-text">SymboLR</h1>
          <p className="logo-subtext">Research Visualization</p>
        </div>

        <nav className="nav-menu">
          {NAV_ITEMS.map(item => (
            <button
              key={item.id}
              className={`nav-button ${activeTab === item.id ? 'active' : ''}`}
              onClick={() => setActiveTab(item.id)}
            >
              <item.icon size={18} />
              <span>{item.label}</span>
            </button>
          ))}
        </nav>

        <div className="sidebar-footer">
          <button className="reset-button" onClick={onReset}>
            Load New Dataset
          </button>
        </div>
      </aside>

      {/* Main Content Area */}
      <main className="main-content">
        <div key={activeTab} className="page-transition">
          {renderContent()}
        </div>
      </main>
    </div>
  );
}
