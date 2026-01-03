import React, { useState, useEffect } from 'react';
import { getConfig, saveConfig, testNeo4jConnection } from '../api/client';
import type { Config } from '../types';
import './BasicSettings.css';

export const BasicSettings: React.FC = () => {
  const [config, setConfig] = useState<Config | null>(null);
  const [loading, setLoading] = useState(false);
  const [testingNeo4j, setTestingNeo4j] = useState(false);

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
      alert('✅ 設定を保存しました');
    } catch (error) {
      alert('❌ 保存に失敗しました');
    } finally {
      setLoading(false);
    }
  };

  const handleTestNeo4j = async () => {
    if (!config) return;
    setTestingNeo4j(true);
    try {
      const result = await testNeo4jConnection(config.neo4j);
      alert(result.message);
    } catch (error: any) {
      alert(`❌ 接続失敗: ${error.message || '不明なエラー'}`);
    } finally {
      setTestingNeo4j(false);
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
    <div className="basic-settings">
      {/* Neo4j設定 */}
      <div className="settings-section">
        <h3 className="settings-section-title">Neo4j 設定</h3>
        
        <div className="form-group">
          <label className="form-label">URL:</label>
          <input
            type="text"
            value={config.neo4j.url}
            onChange={(e) => handleInputChange('neo4j', 'url', e.target.value)}
            placeholder="bolt://localhost:7687"
            className="form-input"
          />
        </div>

        <div className="form-group">
          <label className="form-label">UserName:</label>
          <input
            type="text"
            value={config.neo4j.username}
            onChange={(e) => handleInputChange('neo4j', 'username', e.target.value)}
            placeholder="neo4j"
            className="form-input"
          />
        </div>

        <div className="form-group">
          <label className="form-label">Password:</label>
          <input
            type="password"
            value={config.neo4j.password}
            onChange={(e) => handleInputChange('neo4j', 'password', e.target.value)}
            className="form-input"
          />
        </div>

        <div className="form-group">
          <label className="form-label">Database:</label>
          <select
            value={config.neo4j.database}
            onChange={(e) => handleInputChange('neo4j', 'database', e.target.value)}
            className="form-select"
          >
            <option value="neo4j">neo4j</option>
            <option value="system">system</option>
          </select>
        </div>

        <button 
          className="btn btn-test btn-full"
          onClick={handleTestNeo4j} 
          disabled={testingNeo4j}
        >
          {testingNeo4j ? '確認中...' : '🔐 接続テスト'}
        </button>
      </div>

      {/* 保存ボタン */}
      <div className="actions">
        <button 
          className="btn btn-primary btn-full"
          onClick={handleSave} 
          disabled={loading}
        >
          {loading ? '保存中...' : '設定を保存'}
        </button>
      </div>
    </div>
  );
};