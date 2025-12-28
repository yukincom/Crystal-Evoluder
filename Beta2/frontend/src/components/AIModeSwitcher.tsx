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
          🌐 API Mode
        </button>

        <button
          onClick={() => handleSwitch('ollama')}
          disabled={loading || status.mode === 'ollama' || !status.ollama_available}
          className={status.mode === 'ollama' ? 'active' : ''}
        >
          🏠 Local Mode (Ollama)
        </button>
      </div>

      <div className="status-panel">
        <p>Current: <strong>{status.mode}</strong></p>
        <p>Ollama: {status.ollama_available ? '✅ Available' : '❌ Unavailable'}</p>

        <details>
          <summary>Model Details</summary>
          <ul>
            <li>Triplet: {status.task_models.triplet_extraction}</li>
            <li>Quality (Critic): {status.task_models.quality_check}</li>
            <li>Self-RAG Critic: {status.task_models.self_rag_critic}</li>
            <li>Self-RAG Refiner: {status.task_models.self_rag_refiner}</li>
          </ul>
        </details>
      </div>
    </div>
  );
};