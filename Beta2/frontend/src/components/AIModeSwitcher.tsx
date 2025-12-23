import React, { useState, useEffect } from 'react';
import { getAIStatus, switchMode } from '../api/client';
import type { AIStatus } from '../types';

export const AIModeSwitcher: React.FC = () => {
  const [status, setStatus] = useState<AIStatus | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    loadStatus();
  }, []);

  const loadStatus = async () => {
    const data = await getAIStatus();
    setStatus(data);
  };

  const handleSwitch = async (newMode: 'api' | 'ollama') => {
    setLoading(true);
    try {
      await switchMode(newMode);
      await loadStatus();
      alert(`Switched to ${newMode} mode!`);
    } catch (error) {
      alert('Failed to switch mode');
    } finally {
      setLoading(false);
    }
  };

  if (!status) return <div>Loading...</div>;

  return (
    <div className="ai-mode-switcher">
      <h2>🤖 AI Mode</h2>

      <div className="button-group">
        <button
          onClick={() => handleSwitch('api')}
          disabled={loading || status.mode === 'api'}
          className={status.mode === 'api' ? 'active' : ''}
        >
          ☁️ API Mode
        </button>

        <button
          onClick={() => handleSwitch('ollama')}
          disabled={loading || status.mode === 'ollama' || !status.ollama_available}
          className={status.mode === 'ollama' ? 'active' : ''}
        >
          🖥️ Local Mode (Ollama)
        </button>
      </div>

      <div className="status-panel">
        <p>Current: <strong>{status.mode}</strong></p>
        <p>Ollama: {status.ollama_available ? '✅ Available' : '❌ Unavailable'}</p>

        <details>
          <summary>Model Details</summary>
          <ul>
            <li>Triplet: {status.models.triplet_extraction}</li>
            <li>Quality: {status.models.quality_check}</li>
            <li>Self-RAG: {status.models.self_rag}</li>
          </ul>
        </details>
      </div>
    </div>
  );
};