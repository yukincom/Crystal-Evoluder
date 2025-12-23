import React, { useState, useEffect } from 'react';
import { getConfig, saveConfig, testAIConnection } from '../api/client';
import type { Config } from '../types';
import './AISettings.css';

export const AISettings: React.FC = () => {
  const [config, setConfig] = useState<Config | null>(null);
  const [loading, setLoading] = useState(false);
  const [testingAI, setTestingAI] = useState(false);

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

  const handleTestAI = async () => {
    if (!config) return;
    setTestingAI(true);
    try {
      const result = await testAIConnection(config.ai);
      alert(result.message);
    } catch (error) {
      alert('検証に失敗しました');
    } finally {
      setTestingAI(false);
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
    <div className="ai-settings">
      <h2>AI設定</h2>

      <div className="settings-section">
        <h3>モデル選択</h3>

        <div className="radio-group">
          <label>
            <input
              type="radio"
              name="ai-mode"
              value="local"
              checked={config.ai.mode === 'local'}
              onChange={(e) => handleInputChange('ai', 'mode', e.target.value)}
            />
            Local_AI
          </label>
          <select
            value={config.ai.model}
            onChange={(e) => handleInputChange('ai', 'model', e.target.value)}
            disabled={config.ai.mode !== 'local'}
          >
            <option value="llama3.1:70b">llama3.1:70b</option>
            <option value="llama3.1:8b" disabled>llama3.1:8b (能力不足)</option>
            <option value="mistral:7b" disabled>mistral:7b (能力不足)</option>
          </select>
        </div>

        <div className="radio-group">
          <label>
            <input
              type="radio"
              name="ai-mode"
              value="api"
              checked={config.ai.mode === 'api'}
              onChange={(e) => handleInputChange('ai', 'mode', e.target.value)}
            />
            API_AI
          </label>
          <input
            type="text"
            value={config.ai.model}
            onChange={(e) => handleInputChange('ai', 'model', e.target.value)}
            placeholder="gpt-4o-mini"
            disabled={config.ai.mode !== 'api'}
          />
          <small>GPT-4o-mini以上を推奨</small>
        </div>

        <div className="form-group">
          <label>Your_API_Key:</label>
          <input
            type="password"
            value={config.ai.api_key}
            onChange={(e) => handleInputChange('ai', 'api_key', e.target.value)}
            disabled={config.ai.mode !== 'api'}
          />
        </div>

        <button onClick={handleTestAI} disabled={testingAI}>
          {testingAI ? '検証中...' : '検証'}
        </button>
      </div>

      <div className="settings-section">
        <h3>図表解析</h3>
        <div className="form-group">
          <label>モデル:</label>
          <select
            value={config.ai.figure_model}
            onChange={(e) => handleInputChange('ai', 'figure_model', e.target.value)}
          >
            <option value="granite">granite</option>
            <option value="other">other</option>
          </select>
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