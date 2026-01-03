import React, { useState, useEffect } from 'react';
import { getConfig, saveConfig } from '../api/client';
import type { Config } from '../types';
import './AdvancedSettings.css';

export const AdvancedSettings: React.FC = () => {
  const [config, setConfig] = useState<Config | null>(null);
  const [loading, setLoading] = useState(false);
  
  // アコーディオン開閉状態
  const [openSections, setOpenSections] = useState({
    chunk: false,
    entity: false,
    selfrag: false
  });

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

  const toggleSection = (section: keyof typeof openSections) => {
    setOpenSections(prev => ({
      ...prev,
      [section]: !prev[section]
    }));
  };

  if (!config) return <div>Loading...</div>;

  return (
    <div className="advanced-settings">
      <h2>パラメータ詳細設定</h2>

      {/* チャンク設定 */}
      <div className="accordion">
        <div 
          className="accordion-header" 
          onClick={() => toggleSection('chunk')}
        >
          <span className="accordion-title">▶ チャンク設定</span>
          <span className={`accordion-icon ${openSections.chunk ? 'open' : ''}`}>
            ▶
          </span>
        </div>
        
        {openSections.chunk && (
          <div className="accordion-content">
            <div className="info-box">
              <strong>Graph Chunk Size（グラフ用文章の長さ）</strong><br/>
              値を大きくすると、1つのチャンクに含まれる文章量が増え、より多くの関係性を抽出しやすくなりますが、処理時間は長くなります。<br/>
              目安: 論文なら512、短い記事なら384が適切です。
            </div>

            <div className="form-group">
              <label>Graph Chunk Size</label>
              <input
                type="number"
                value={config.advanced.graph_chunk_size}
                onChange={(e) => handleInputChange('advanced', 'graph_chunk_size', parseInt(e.target.value))}
                min="384"
                max="640"
                className="form-input"
              />
              <small className="form-hint">推奨: 512</small>
            </div>

            <div className="form-group">
              <label>Graph Chunk Overlap</label>
              <input
                type="number"
                value={config.advanced.graph_chunk_overlap}
                onChange={(e) => handleInputChange('advanced', 'graph_chunk_overlap', parseInt(e.target.value))}
                min="0"
                max="200"
                className="form-input"
              />
              <small className="form-hint">推奨: 50</small>
            </div>

            <div className="form-group">
              <label>Retrieval Chunk Size</label>
              <input
                type="number"
                value={config.advanced.retrieval_chunk_size}
                onChange={(e) => handleInputChange('advanced', 'retrieval_chunk_size', parseInt(e.target.value))}
                min="256"
                max="768"
                className="form-input"
              />
              <small className="form-hint">推奨: 320</small>
            </div>

            <div className="form-group">
              <label>Retrieval Chunk Overlap</label>
              <input
                type="number"
                value={config.advanced.retrieval_chunk_overlap}
                onChange={(e) => handleInputChange('advanced', 'retrieval_chunk_overlap', parseInt(e.target.value))}
                min="0"
                max="200"
                className="form-input"
              />
              <small className="form-hint">推奨: 120</small>
            </div>
          </div>
        )}
      </div>

      {/* エンティティリンキング */}
      <div className="accordion">
        <div 
          className="accordion-header" 
          onClick={() => toggleSection('entity')}
        >
          <span className="accordion-title">▶ エンティティリンキング</span>
          <span className={`accordion-icon ${openSections.entity ? 'open' : ''}`}>
            ▶
          </span>
        </div>
        
        {openSections.entity && (
          <div className="accordion-content">
            <div className="info-box">
              <strong>Entity Linking Threshold（同一判定の厳しさ）</strong><br/>
              値を高くすると、「同一である」と強く判断できる場合のみ統合されます。精度は向上しますが、つながりの数は少なくなります。<br/>
              例: 0.95 = 「田中太郎」と「田中太郎氏」は別人扱い<br/>
              例: 0.88（推奨）= 「田中太郎」と「田中太郎氏」は同一人物扱い
            </div>

            <div className="form-group">
              <label>Entity Linking Threshold</label>
              <input
                type="number"
                step="0.01"
                value={config.advanced.entity_linking_threshold}
                onChange={(e) => handleInputChange('advanced', 'entity_linking_threshold', parseFloat(e.target.value))}
                min="0.7"
                max="0.95"
                className="form-input"
              />
              <small className="form-hint">推奨: 0.88</small>
            </div>

            <div className="form-group">
              <label>Relation Compat Threshold</label>
              <input
                type="number"
                step="0.01"
                value={config.advanced.relation_compat_threshold}
                onChange={(e) => handleInputChange('advanced', 'relation_compat_threshold', parseFloat(e.target.value))}
                min="0.05"
                max="0.3"
                className="form-input"
              />
              <small className="form-hint">推奨: 0.11</small>
            </div>

            <div className="form-group">
              <label>Final Weight Cutoff</label>
              <input
                type="number"
                step="0.001"
                value={config.advanced.final_weight_cutoff}
                onChange={(e) => handleInputChange('advanced', 'final_weight_cutoff', parseFloat(e.target.value))}
                min="0.01"
                max="0.1"
                className="form-input"
              />
              <small className="form-hint">推奨: 0.035</small>
            </div>

            <div className="form-group">
              <label>Max Triplets per Chunk</label>
              <input
                type="number"
                value={config.advanced.max_triplets_per_chunk}
                onChange={(e) => handleInputChange('advanced', 'max_triplets_per_chunk', parseInt(e.target.value))}
                min="5"
                max="30"
                className="form-input"
              />
              <small className="form-hint">推奨: 15</small>
            </div>
          </div>
        )}
      </div>

      {/* Self-RAG */}
      <div className="accordion">
        <div 
          className="accordion-header" 
          onClick={() => toggleSection('selfrag')}
        >
          <span className="accordion-title">▶ Self-RAG（自動修正機能）</span>
          <span className={`accordion-icon ${openSections.selfrag ? 'open' : ''}`}>
            ▶
          </span>
        </div>
        
        {openSections.selfrag && (
          <div className="accordion-content">
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
              <label>閾値</label>
              <input
                type="number"
                step="0.01"
                value={config.advanced.self_rag_threshold}
                onChange={(e) => handleInputChange('advanced', 'self_rag_threshold', parseFloat(e.target.value))}
                min="0"
                max="1"
                disabled={!config.advanced.self_rag_enabled}
                className="form-input"
              />
              <small className="form-hint">推奨: 0.75</small>
            </div>

            {config.advanced.self_rag_enabled && (
              <>
                <div className="form-group">
                  <label>Max Retries</label>
                  <input
                    type="number"
                    value={config.advanced.self_rag_max_retries}
                    onChange={(e) => handleInputChange('advanced', 'self_rag_max_retries', parseInt(e.target.value))}
                    min="0"
                    max="3"
                    className="form-input"
                  />
                  <small className="form-hint">推奨: 1</small>
                </div>

                <div className="form-group">
                  <label>Token Budget</label>
                  <input
                    type="number"
                    value={config.advanced.self_rag_token_budget}
                    onChange={(e) => handleInputChange('advanced', 'self_rag_token_budget', parseInt(e.target.value))}
                    min="10000"
                    max="200000"
                    step="10000"
                    className="form-input"
                  />
                  <small className="form-hint">推奨: 100000</small>
                </div>
              </>
            )}
          </div>
        )}
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