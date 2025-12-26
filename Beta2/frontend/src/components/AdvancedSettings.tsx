import React, { useState, useEffect } from 'react';
import { getConfig, saveConfig } from '../api/client';
import type { Config } from '../types';
import './AdvancedSettings.css';

export const AdvancedSettings: React.FC = () => {
  const [config, setConfig] = useState<Config | null>(null);
  const [loading, setLoading] = useState(false);
  // 各セクションの開閉状態
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

  const toggleSection = (section: keyof typeof openSections) => {
    setOpenSections(prev => ({
      ...prev,
      [section]: !prev[section]
    }));
  };

  if (!config) return <div>Loading...</div>;

  return (
    <div className="advanced-settings">
      <h2>各種パラメータの詳細設定</h2>

      {/* チャンク設定 */}
      <div className="settings-section">
        <h3 onClick={() => toggleSection('chunk')} className="collapsible-header">
          {openSections.chunk ? '▼' : '▶'}チャンク設定</h3>
                    {openSections.chunk && (
          <>
            <div className="section-description">
              <strong>Graph Chunk Size（グラフ用文章の長さ）</strong><br/>
              数値を大きくすると、1つのチャンクに含まれる文章量が増えます。<br/>
              より多くの関係性を抽出しやすくなりますが、処理時間は長くなります。<br/>
              🔹 目安: 論文なら512、短い記事なら384が適切です。<br/>
<br/>
              <strong>Graph Chunk Overlap（グラフ用重なり量）</strong><br/>
              重なりを増やすことで、文章の切れ目による関係性の断絶を防ぎやすくなります。<br/>
              一方で、処理負荷は増加します。<br/>
              🔹 目安: Chunk Sizeの10%程度（512なら50）が標準的です。<br/>
<br/>
              <strong>Retrieval Chunk Size（検索用文章の長さ）</strong><br/>
              数値を大きくすると、検索結果として返される文章が長くなります。<br/>
              取得できる情報量は増えますが、余計な情報も混ざりやすくなります。<br/>
              🔹 目安: 質問応答には320、要約には512が向いています。<br/>
<br/>
              <strong>Retrieval Chunk Overlap（検索用重なり量）</strong><br/>
              重なりを多く設定すると、検索時の情報の取りこぼしを抑えられます。<br/>
              ただし、処理速度はやや低下します。<br/>
              🔹 目安: Chunk Sizeの30-40%程度が一般的です。<br/><br/>
            </div>
                  </>
        )}
                <div className="form-group">
          <label>
            Graph Chunk Size</label>
          <input
            type="number"
            value={config.advanced.graph_chunk_size}
            onChange={(e) => handleInputChange('advanced', 'graph_chunk_size', parseInt(e.target.value))}
            min="384"
            max="640"
          />
          <span className="hint">推奨: 512</span>
        </div>
        
        <div className="form-group">
          <label>Graph Chunk Overlap:</label>
          <input
            type="number"
            value={config.advanced.graph_chunk_overlap}
            onChange={(e) => handleInputChange('advanced', 'graph_chunk_overlap', parseInt(e.target.value))}
            min="0"
            max="200"
          />
          <span className="hint">推奨: 50</span>
        </div>
        
        <div className="form-group">
          <label>Retrieval Chunk Size:</label>
          <input
            type="number"
            value={config.advanced.retrieval_chunk_size}
            onChange={(e) => handleInputChange('advanced', 'retrieval_chunk_size', parseInt(e.target.value))}
            min="256"
            max="768"
          />
          <span className="hint">推奨: 320</span>
        </div>
        
        <div className="form-group">
          <label>Retrieval Chunk Overlap:</label>
          <input
            type="number"
            value={config.advanced.retrieval_chunk_overlap}
            onChange={(e) => handleInputChange('advanced', 'retrieval_chunk_overlap', parseInt(e.target.value))}
            min="0"
            max="200"
          />
          <span className="hint">推奨: 120</span>
        </div>
      </div>

      {/* エンティティリンキング */}
      <div className="settings-section">
        <h3 onClick={() => toggleSection('entity')} className="collapsible-header">
          {openSections.entity ? '▼' : '▶'}エンティティリンキング（同一概念の統合設定）</h3>

                    {openSections.entity && (
          <>
            <div className="section-description">
              <strong>Entity Linking Threshold（同一判定の厳しさ）</strong><br/>
              数値を高くすると、「同一である」と強く判断できる場合のみ統合されます<br/>
              精度は向上しますが、つながりの数は少なくなります。<br/>
              数値を低くすると、類似度が高いものも統合されやすくなりますが、誤統合のリスクが高まります。<br/>
              🔹 例 ：<br/>
              ・0.92以上: 「田中太郎」と「T.Tanaka」は別人扱い（厳格）<br/>
              ・0.88（推奨）: 「田中太郎」と「田中太郎氏」は同一人物扱い<br/>
              ・0.85以下: 「田中太郎」と「田中」も同一人物扱い（緩い）<br/>
              <br/>
              <strong>Relation Compatibility Threshold（関係の類似度の許容範囲）</strong><br/>
              数値を低く設定すると、似た意味の関係も同じものとして扱われ、関係性の数が増えます。<br/>
              その分、関係の精度はやや下がる可能性があります。<br/>
              🔹 例: 「所属する」と「勤務する」を同じ関係と見なすかどうか<br/>
              <br/>
              <strong>Final Weight Cutoff（重要度の足切りライン）</strong><br/>
              数値を高くすると、影響の小さい関係は除外され、グラフが簡潔になります。<br/>
              処理速度も向上しますが、詳細な関係は省略されます。<br/>
              数値を低くすると、弱い関係も保持され、より詳細なグラフになります。<br/>
              🔹 例: 0.05 = 影響度5%未満の関係は除外<br/>
              <br/>
              <strong>Max Triplets per Chunk（チャンクあたりの関係数上限）</strong><br/>
              値を大きくすると、1つの文章から抽出される関係数が増えます。<br/>
              情報量は豊富になりますが、処理時間は増加します。<br/>
              🔹 目安: Chunk Sizeの30-40%程度が一般的です。<br/>
              ・簡潔なグラフ: 10<br/>
              ・標準: 15（推奨）<br/>
              ・詳細なグラフ: 20-30<br/><br/>
            </div>
                  </>
        )}        
        <div className="form-group">
          <label>Entity Linking Threshold:</label>
          <input
            type="number"
            step="0.01"
            value={config.advanced.entity_linking_threshold}
            onChange={(e) => handleInputChange('advanced', 'entity_linking_threshold', parseFloat(e.target.value))}
            min="0.7"
            max="0.95"
          />
          <span className="hint">推奨: 0.88</span>
        </div>
        
        <div className="form-group">
          <label>Relation Compat Threshold:</label>
          <input
            type="number"
            step="0.01"
            value={config.advanced.relation_compat_threshold}
            onChange={(e) => handleInputChange('advanced', 'relation_compat_threshold', parseFloat(e.target.value))}
            min="0.05"
            max="0.3"
          />
          <span className="hint">推奨: 0.11</span>
        </div>
        
        <div className="form-group">
          <label>Final Weight Cutoff:</label>
          <input
            type="number"
            step="0.001"
            value={config.advanced.final_weight_cutoff}
            onChange={(e) => handleInputChange('advanced', 'final_weight_cutoff', parseFloat(e.target.value))}
            min="0.01"
            max="0.1"
          />
          <span className="hint">推奨: 0.035</span>
        </div>
        
        <div className="form-group">
          <label>Max Triplets per Chunk:</label>
          <input
            type="number"
            value={config.advanced.max_triplets_per_chunk}
            onChange={(e) => handleInputChange('advanced', 'max_triplets_per_chunk', parseInt(e.target.value))}
            min="5"
            max="30"
          />
          <span className="hint">推奨: 15</span>
        </div>
      </div>

      {/* Self-RAG */}
      <div className="settings-section">
        <h3 onClick={() => toggleSection('selfrag')} className="collapsible-header">
          {openSections.selfrag ? '▼' : '▶'}Self-RAG（自動修正機能）</h3>

            {openSections.selfrag && (
          <>
            <div className="section-description">
              <strong>Self-RAG 有効化</strong><br/>
              有効にすると、不自然または不適切な関係を自動的に修正しようとします。<br/>
              結果の品質は向上しますが、処理時間は長くなります。<br/>
              🔹 推奨: 初めての処理では有効化を推奨します。<br/>
            <br/>
              <strong>閾値（修正判定基準）</strong><br/>
              数値を高くすると、明確な問題がある場合のみ修正が行われます。<br/>
              処理は高速になりますが、修正回数は少なくなります。<br/>
              数値を低くすると、軽微な違和感でも修正が試みられます。<br/>
              ⚠️ 注意: 低すぎると「正しい関係」も修正されることがあります。<br/>
            <br/>
              <strong>Max Retries（修正試行回数の上限）</strong><br/>
              回数を増やすことで、より丁寧な修正が行われますが、処理時間は大幅に増加します。<br/>
              🔹 目安:<br/>
              ・0回: 修正なし（高速）<br/>
              ・1回: 標準（推奨）<br/>
              ・2-3回: 品質重視（処理時間3-5倍）<br/>
              <br/>
              <strong>Token Budget（修正に使えるAIの計算量）</strong><br/>
              値を大きくすると、修正により多くの計算資源を割り当てられます。<br/>
              品質向上が期待できますが、コストや処理時間が増加します。<br/>
              💰 注意: 値を大きくするとAPI料金が増えます！<br/>
              🔹 目安:<br/>
              ・50,000: 小規模文書（10-20ページ）<br/>
              ・100,000: 標準（推奨）<br/>
              ・200,000: 大規模文書（100ページ以上）<br/><br/>              
            </div>
                  </>
        )}                
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
          <span className="hint">推奨: 0.75</span>
        </div>
        
        {config.advanced.self_rag_enabled && (
          <>
            <div className="form-group">
              <label>Max Retries:</label>
              <input
                type="number"
                value={config.advanced.self_rag_max_retries}
                onChange={(e) => handleInputChange('advanced', 'self_rag_max_retries', parseInt(e.target.value))}
                min="0"
                max="3"
              />
              <span className="hint">推奨: 1</span>
            </div>
            
            <div className="form-group">
              <label>Token Budget:</label>
              <input
                type="number"
                value={config.advanced.self_rag_token_budget}
                onChange={(e) => handleInputChange('advanced', 'self_rag_token_budget', parseInt(e.target.value))}
                min="10000"
                max="200000"
                step="10000"
              />
              <span className="hint">推奨: 100000</span>
            </div>
          </>
        )}
      </div>

      {/* 保存ボタン */}
      <div className="actions">
        <button onClick={handleSave} disabled={loading}>
          {loading ? '保存中...' : '💾 保存'}
        </button>
      </div>
    </div>
  );
};