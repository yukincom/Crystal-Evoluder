import React, { useState, useEffect } from 'react';
import { getConfig, saveConfig } from '../api/client';
import type { Config } from '../types';
import './AdvancedSettings.css';

export const AdvancedSettings: React.FC = () => {
  const [config, setConfig] = useState<Config | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    loadConfig();
  }, []);

  const loadConfig = async () => {
    try {
      const data = await getConfig();
      setConfig(data);
    } catch (error) {
      console.error('Failed to load config:', error);
    }
  };

  const handleSave = async () => {
    if (!config) return;
    setLoading(true);
    try {
      await saveConfig(config);
      alert('設定を保存しました');
    } catch (error) {
      alert('保存に失敗しました');
    } finally {
      setLoading(false);
    }
  };

  const handleInputChange = (section: keyof Config, field: string, value: any) => {
    if (!config) return;
    setConfig({
      ...config,
      [section]: {
        ...config[section],
        [field]: value
      }
    });
  };

  if (!config) return <div>Loading...</div>;

  return (
    <div className="advanced-settings">
      <h2>詳細設定</h2>

      <div className="settings-section">
        <h3>チャンク設定</h3>
        <div className="form-group">
          <label>Graph:</label>
          <input
            type="number"
            value={config.advanced.graph_chunk_size}
            onChange={(e) => handleInputChange('advanced', 'graph_chunk_size', parseInt(e.target.value))}
            min="384"
            max="640"
          />
        </div>
        <div className="form-group">
          <label>Retrieval:</label>
          <input
            type="number"
            value={config.advanced.retrieval_chunk_size}
            onChange={(e) => handleInputChange('advanced', 'retrieval_chunk_size', parseInt(e.target.value))}
            min="256"
            max="768"
          />
        </div>
      </div>

      <div className="settings-section">
        <h3>Self-RAG</h3>
        <div className="checkbox-group">
          <label>
            <input
              type="checkbox"
              checked={config.advanced.self_rag_enabled}
              onChange={(e) => handleInputChange('advanced', 'self_rag_enabled', e.target.checked)}
            />
            有効化
          </label>
        </div>
        <div className="form-group">
          <label>閾値:</label>
          <input
            type="number"
            value={config.advanced.self_rag_threshold}
            onChange={(e) => handleInputChange('advanced', 'self_rag_threshold', parseFloat(e.target.value))}
            step="0.01"
            min="0"
            max="1"
            disabled={!config.advanced.self_rag_enabled}
          />
        </div>
      </div>

      <div className="actions">
        <button onClick={handleSave} disabled={loading}>
          {loading ? '保存中...' : '保存'}
        </button>
      </div>
    </div>
  );
};