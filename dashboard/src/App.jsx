import React, { useState } from 'react';
import Layout from './components/Layout';
import { UploadCloud } from 'lucide-react';
import './App.css';

function App() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const json = JSON.parse(e.target.result);
          if (!json.summary || !json.generations) {
            throw new Error("Invalid SymboLR JSON format");
          }
          setData(json);
          setError(null);
        } catch (err) {
          setError(err.message);
        }
      };
      reader.readAsText(file);
    }
  };

  if (data) {
    return <Layout data={data} onReset={() => setData(null)} />;
  }

  return (
    <div className="upload-container fade-in" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100vh', padding: '24px' }}>
      <div className="glass-panel" style={{ textAlign: 'center', maxWidth: '500px', width: '100%' }}>
        <h1 style={{ fontSize: '2rem', marginBottom: '8px', fontWeight: 600 }}>SymboLR</h1>
        <p style={{ color: 'var(--text-secondary)', marginBottom: '32px' }}>Research Visualization Interface</p>
        
        <label style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          border: '2px dashed var(--panel-border)',
          borderRadius: '12px',
          padding: '48px',
          cursor: 'pointer',
          transition: 'all 0.2s ease',
          backgroundColor: 'rgba(255,255,255,0.02)'
        }}>
          <UploadCloud size={48} color="var(--accent-blue)" style={{ marginBottom: '16px' }} />
          <span style={{ fontWeight: 500, fontSize: '1.1rem' }}>Drop diagnostics JSON here</span>
          <span style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginTop: '8px' }}>or click to browse</span>
          <input type="file" accept=".json" onChange={handleFileUpload} style={{ display: 'none' }} />
        </label>
        
        {error && <p style={{ color: 'var(--accent-orange)', marginTop: '16px' }}>Error: {error}</p>}
      </div>
    </div>
  );
}

export default App;
