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
      alert('設定を保存しました');
    } catch (error) {
      alert('保存に失敗しました');
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
    } catch (error) {
      alert('接続テストに失敗しました');
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

  const handleDirSelect = async (field: 'input_dir' | 'output_dir') => {
    try {
      // ディレクトリ選択ダイアログ（ブラウザの制限でファイル選択に）
      const input = document.createElement('input');
      input.type = 'file';
      input.webkitdirectory = true;
      input.onchange = (e) => {
        const files = (e.target as HTMLInputElement).files;
        if (files && files.length > 0) {
          const dirPath = files[0].webkitRelativePath.split('/')[0];
          handleInputChange('processing', field, dirPath);
        }
      };
      input.click();
    } catch (error) {
      console.error('Directory selection failed:', error);
    }
  };

  if (!config) return <div>Loading...</div>;

  return (
    <div className="basic-settings">
      <h2> GraphRAG 設定</h2>

      <div className="settings-section">
        <h3>Neo4j 設定</h3>
        <div className="form-group">
          <label>URL:</label>
          <input
            type="text"
            value={config.neo4j.url}
            onChange={(e) => handleInputChange('neo4j', 'url', e.target.value)}
            placeholder="bolt://localhost:7687"
          />
        </div>
        <div className="form-group">
          <label>UserName:</label>
          <input
            type="text"
            value={config.neo4j.username}
            onChange={(e) => handleInputChange('neo4j', 'username', e.target.value)}
            placeholder="neo4j"
          />
        </div>
        <div className="form-group">
          <label>Password:</label>
          <input
            type="password"
            value={config.neo4j.password}
            onChange={(e) => handleInputChange('neo4j', 'password', e.target.value)}
          />
        </div>
        <div className="form-group">
          <label>Database:</label>
          <select
            value={config.neo4j.database}
            onChange={(e) => handleInputChange('neo4j', 'database', e.target.value)}
          >
            <option value="neo4j">neo4j</option>
            <option value="system">system</option>
          </select>
        </div>
        <button onClick={handleTestNeo4j} disabled={testingNeo4j}>
          {testingNeo4j ? 'テスト中...' : '接続テスト'}
        </button>
      </div>

      <div className="settings-section">
        <h3>GraphRAG 構築対象データ</h3>
        <div className="form-group">
          <label>元データフォルダ:</label>
          <div className="input-with-button">
            <input
              type="text"
              value={config.processing.input_dir}
              onChange={(e) => handleInputChange('processing', 'input_dir', e.target.value)}
              placeholder="フォルダを選択"
              readOnly
            />
            <button onClick={() => handleDirSelect('input_dir')}>選択</button>
          </div>
        </div>
        <div className="form-group">
          <label>構築結果フォルダ:</label>
          <div className="input-with-button">
            <input
              type="text"
              value={config.processing.output_dir}
              onChange={(e) => handleInputChange('processing', 'output_dir', e.target.value)}
              placeholder="結果フォルダを選択"
              readOnly
            />
            <button onClick={() => handleDirSelect('output_dir')}>選択</button>
          </div>
        </div>
        <button>実行</button>
      </div>

      <div className="actions">
        <button onClick={handleSave} disabled={loading}>
          {loading ? '保存中...' : '保存'}
        </button>
      </div>
    </div>
  );
};