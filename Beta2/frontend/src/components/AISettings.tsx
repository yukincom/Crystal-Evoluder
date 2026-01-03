import React, { useState, useEffect } from 'react';
import { getConfig, saveConfig, testAIConnection } from '../api/client';
import type { Config } from '../types';
import './AISettings.css';

interface OllamaModel {
  name: string;
  size: number;
  capable: boolean;
  is_vision: boolean;
  recommended_for_base?: boolean;      // Base用推奨フラグ
  recommended_for_quality?: boolean;   // Quality用推奨フラグ
}

export const AISettings: React.FC = () => {
  const [config, setConfig] = useState<Config | null>(null);
  const [loading, setLoading] = useState(false);
  const [testingAI, setTestingAI] = useState(false);
  const [testingQualityCheck, setTestingQualityCheck] = useState(false);
  const [testingRefiner, setTestingRefiner] = useState(false);
  const [customRefiner, setCustomRefiner] = useState(false);
  const [customQualityCheck, setCustomQualityCheck] = useState(false);

  // Ollamaモデル一覧
  const [ollamaModels, setOllamaModels] = useState<OllamaModel[]>([]);
  const [ollamaAvailable, setOllamaAvailable] = useState(false);
  const [loadingModels, setLoadingModels] = useState(false);

  // LLM用モデル（Visionを除外）
  const llmModels = ollamaModels.filter(m => !m.is_vision);

  useEffect(() => {
    loadOllamaModels();
    loadConfig();
  }, []);

  useEffect(() => {
    if (config) {
      // Refinerのカスタム判定
      setCustomRefiner(config.ai.refiner_mode !== null);
      
      // 品質チェックのカスタム判定
      setCustomQualityCheck(config.ai.quality_mode !== null);
    }
  }, [config]);

  useEffect(() => {
    // Ollamaモデル読み込み後、configがあれば検証
    if (!config || !ollamaModels.length) return;

    if (config.ai.mode === 'ollama') {
      const currentModel = config.ai.ollama_model;
      
      if (!currentModel || currentModel === '') {
        const firstValidModel = llmModels.find(m => m.capable);
        if (firstValidModel) {
          setConfig({
            ...config,
            ai: {
              ...config.ai,
              ollama_model: firstValidModel.name
            }
          });
          console.log('✅ Auto-selected Ollama model:', firstValidModel.name);
        }
      }
    }
  }, [config, ollamaModels, llmModels]);

  const loadConfig = async () => {
    try {
      const data = await getConfig();
      setConfig(data);
    } catch (error) {
      console.error('Failed to load config:', error);
    }
  };

  const loadOllamaModels = async () => {
    setLoadingModels(true);
    try {
      const response = await fetch('http://localhost:8000/config/ollama/models');
      const data = await response.json();
      
      if (data.available) {
        setOllamaAvailable(true);
        setOllamaModels(data.models);
      } else {
        setOllamaAvailable(false);
        setOllamaModels([]);
      }
    } catch (error) {
      console.error('Failed to load Ollama models:', error);
      setOllamaAvailable(false);
    } finally {
      setLoadingModels(false);
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

  const handleTestAI = async () => {
    if (!config) return;
    setTestingAI(true);
    try {
      const testConfig = {
        mode: config.ai.mode,
        api_key: config.ai.api_key,
        ollama_url: config.ai.ollama_url,
        api_model: config.ai.api_model,      
        ollama_model: config.ai.ollama_model, 
      };

      const result = await testAIConnection(testConfig);
      alert(result.message || '✅ 接続成功！');
    } catch (error: any) {
      alert(`❌ 接続失敗: ${error.message || '不明なエラー'}`);
    } finally {
      setTestingAI(false);
    }
  };

  const handleTestQualityCheck = async () => {
    if (!config) return;
    setTestingQualityCheck(true);

    try {
      const qualityCheckConfig = {
        mode: config.ai.quality_mode || config.ai.mode,
        api_key: config.ai.quality_check_api_key || config.ai.api_key,
        ollama_url: config.ai.quality_check_ollama_url || config.ai.ollama_url,
        api_model: config.ai.quality_check_api_model || config.ai.api_model,
        ollama_model: config.ai.quality_check_ollama_model || config.ai.ollama_model,
      };

      const result = await testAIConnection(qualityCheckConfig);
      alert(result.message || '✅ 品質チェックモデル接続成功');
    } catch (error: any) {
      alert(`❌ 品質チェック接続失敗: ${error.message || '不明なエラー'}`);
    } finally {
      setTestingQualityCheck(false);
    }
  };

  const handleTestRefiner = async () => {
    if (!config) return;
    setTestingRefiner(true);
    try {
      const currentModel = config.ai.mode === 'api' 
        ? config.ai.api_model 
        : config.ai.ollama_model;
      
      const refinerConfig = {
        mode: config.ai.refiner_mode || config.ai.mode,
        api_key: config.ai.refiner_api_key || config.ai.api_key,
        ollama_url: config.ai.refiner_ollama_url || config.ai.ollama_url,
        api_model: config.ai.refiner_api_model || currentModel,
        ollama_model: config.ai.refiner_ollama_model || currentModel,
      };
      
      const result = await testAIConnection(refinerConfig);
      alert(result.message || '✅ Refiner接続成功！');
    } catch (error: any) {
      alert(`❌ Refiner接続失敗: ${error.message || '不明なエラー'}`);
    } finally {
      setTestingRefiner(false);
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

  const getCurrentModel = (): string => {
    if (!config) return '(未設定)';
    return config.ai.mode === 'api' 
      ? config.ai.api_model || '(未指定)' 
      : config.ai.ollama_model || '(未指定)';
  };

  const handleCustomQualityCheckToggle = (enabled: boolean) => {
    if (!config) return;
    
    setCustomQualityCheck(enabled);
    
    if (enabled) {
      // カスタム設定を有効化
      setConfig({
        ...config,
        ai: {
          ...config.ai,
          quality_check_api_model: config.ai.mode === 'api' ? 'gpt-4o-mini' : config.ai.quality_check_api_model,
          quality_check_ollama_model: config.ai.mode === 'ollama' ? (llmModels.find(m => m.capable)?.name || '') : config.ai.quality_check_ollama_model
         }
      });
    } else {
      // カスタム設定を無効化
      setConfig({
        ...config,
        ai: {
          ...config.ai,
          quality_mode: null,
          quality_check_api_model: undefined,
          quality_check_ollama_model: undefined,
          quality_check_api_key: null,
          quality_check_ollama_url: null
        }
      });
    }
  };

  const handleCustomRefinerToggle = (enabled: boolean) => {
    if (!config) return;
    
    setCustomRefiner(enabled);
    
    if (enabled) {
      let currentModel = config.ai.mode === 'api' 
        ? config.ai.api_model 
        : config.ai.ollama_model;
      
      if (config.ai.mode === 'ollama') {
        const isValidModel = llmModels.some(m => m.name === currentModel && m.capable);
        
        if (!isValidModel || !currentModel) {
          const firstValidModel = llmModels.find(m => m.capable);
          currentModel = firstValidModel?.name || '';
        }
      }
      
      if (config.ai.mode === 'api' && !currentModel) {
        currentModel = 'gpt-4o-mini';
      }

      setConfig({
        ...config,
        ai: {
          ...config.ai,
          refiner_mode: config.ai.mode,
          refiner_api_model: config.ai.mode === 'api' ? config.ai.api_model : config.ai.refiner_api_model,
          refiner_ollama_model: config.ai.mode === 'ollama' ? config.ai.ollama_model : config.ai.refiner_ollama_model
        }
      });
    } else {
      setConfig({
        ...config,
        ai: {
          ...config.ai,
          refiner_mode: null,
          refiner_api_model: undefined,
          refiner_ollama_model: undefined,
          refiner_api_key: null,
          refiner_ollama_url: null
        }
      });
    }
  };

  if (!config) return <div className="ai-settings">Loading...</div>;

  return (
    <div className="ai-settings">
      <h2>AI 設定</h2>

      {/* Ollama接続状態 */}
      {loadingModels && (
        <div className="info-box">⏳ Ollamaモデルを読み込み中...</div>
      )}
      
      {!loadingModels && !ollamaAvailable && (
        <div className="warning-box">
          ⚠️ Ollamaが接続できません。Local AIモードを使用する場合は、Ollamaを起動してください。
          <button onClick={loadOllamaModels} className="btn-secondary" style={{marginLeft: '12px'}}>🔄 再読込</button>
        </div>
      )}

      {/* 基本モデル設定 */}
      <div className="settings-section">
        <h3>基本モデル選択</h3>
        <small style={{display: 'block', marginBottom: '12px', color: '#6b7280'}}>
          推奨：14B〜32Bクラス（トリプレット抽出用）
        </small>

        {/* Local AI */}
        <div className="radio-group">
          <label>
            <input
              type="radio"
              name="ai-mode"
              value="ollama"
              checked={config.ai.mode === 'ollama'}
              onChange={(e) => handleInputChange('ai', 'mode', e.target.value)}
              disabled={!ollamaAvailable || llmModels.length === 0}
            />
            🏠 Local AI
          </label>

          {ollamaAvailable && llmModels.length > 0 ? (
            <select
              value={config.ai.ollama_model || ''}
              onChange={(e) => handleInputChange('ai', 'ollama_model', e.target.value)}
              disabled={config.ai.mode !== 'ollama'}
            >
              {!config.ai.ollama_model && <option value="">モデルを選択...</option>}
              {llmModels.map(model => {
              // Base用の表示ロジック
              const isRecommended = model.recommended_for_base;
              const displayText = isRecommended 
                ? `${model.name} (${model.size}GB) ✅ 推奨`
                : `${model.name} (${model.size}GB) ⚠️ 性能不足`;
    
              return (
                <option 
                  key={model.name} 
                  value={model.name}
                  disabled={!model.recommended_for_base}
                >
                  {displayText}
                </option>
                  );
              })}
            </select>
          ) : (
            <select disabled>
              <option>❌ Ollamaモデルが見つかりません</option>
            </select>
          )}
        </div>

        {/* API */}
        <div className="radio-group">
          <label>
            <input
              type="radio"
              name="ai-mode"
              value="api"
              checked={config.ai.mode === 'api'}
              onChange={(e) => handleInputChange('ai', 'mode', e.target.value)}
            />
            🌐 API
          </label>
          <input
            type="text"
            value={config.ai.api_model || ''}
            onChange={(e) => handleInputChange('ai', 'api_model', e.target.value)}
            placeholder="gpt-4o-mini"
            disabled={config.ai.mode !== 'api'}
          />
        </div>

        <div className="form-group">
          <label>🔑 APIキー:</label>
          <input
            type="password"
            value={config.ai.api_key || ''}
            onChange={(e) => handleInputChange('ai', 'api_key', e.target.value)}
            disabled={config.ai.mode !== 'api'}
            placeholder={config.ai.mode === 'api' ? 'sk-...' : 'Local AIでは不要'}
          />
        </div>

        {/* 接続テスト */}
        {config.ai.mode === 'api' && (
          <div className="test-button-row">
            <button onClick={handleTestAI} disabled={testingAI}>
              {testingAI ? '確認中...' : '🔍 API接続確認'}
            </button>
          </div>
        )}

        {config.ai.mode === 'ollama' && ollamaAvailable && (
          <div className="success-box">
            ✅ Ollama接続済み 🦙
          </div>
        )}
      </div>

      {/* 品質チェック専用モデル設定 */}
      <div className="settings-section">
        <h3>品質チェック専用モデル（推奨：軽量モデル）</h3>
        
        <label className="toggle-label">
          <input 
            type="checkbox"
            checked={customQualityCheck}
            onChange={(e) => handleCustomQualityCheckToggle(e.target.checked)}
          />
          <strong>基本モデルとは別のモデルを使用する</strong>
        </label>
        <small className="hint-text">
          💡 品質チェックは7B〜8Bクラスの軽量モデルで十分です。<br/>
          未設定の場合は基本モデルと同じものを使用します。
        </small>

        {!customQualityCheck ? (
          <div className="readonly-refiner">
            <p className="info-text">📌 基本モデルと同じ設定を使用します</p>
            <div className="info-box">
              <p><strong>モード:</strong> {config.ai.mode === 'api' ? '🌐 API' : '🏠 Local AI'}</p>
              <p><strong>モデル:</strong> {getCurrentModel()}</p>
            </div>
          </div>
        ) : (
          <div className="custom-config">
            {/* Quality Check Local AI */}
            <div className="radio-group">
              <label>
                <input
                  type="radio"
                  name="quality-mode"
                  value="ollama"
                  checked={config.ai.quality_mode === 'ollama'}
                  onChange={(e) => handleInputChange('ai', 'quality_mode', e.target.value)}
                  disabled={!ollamaAvailable || llmModels.length === 0}
                />
                🏠 Local AI
              </label>

                            {ollamaAvailable && llmModels.length > 0 ? (
                <select
                  value={config.ai.quality_check_ollama_model || ''}
                  onChange={(e) => handleInputChange('ai', 'quality_check_ollama_model', e.target.value)}
                  disabled={config.ai.quality_mode !== 'ollama'}
                >
                  {!config.ai.quality_check_ollama_model && <option value="">モデルを選択...</option>}
                  {llmModels.map(model => {
                    const isRecommended = model.recommended_for_quality;
                    const label = isRecommended 
                      ? `${model.name} (${model.size}GB) ✅ 最適`
                      : `${model.name} (${model.size}GB) ⚠️ 性能不足`;
                    
                    return (
                      <option 
                        key={model.name} 
                        value={model.name}
                        disabled={!model.capable}
                      >
                        {label}
                      </option>
                    );
                  })}
                </select>
              ) : (
                <select disabled>
                  <option>❌ Ollamaモデルが見つかりません</option>
                </select>
              )}
            </div>

            {/* Quality Check API */}
            <div className="radio-group">
              <label>
                <input
                  type="radio"
                  name="quality-mode"
                  value="api"
                  checked={config.ai.quality_mode === 'api'}
                  onChange={(e) => handleInputChange('ai', 'quality_mode', e.target.value)}
                />
                🌐 API
              </label>
              <input
                type="text"
                value={config.ai.quality_check_api_model || ''}
                onChange={(e) => handleInputChange('ai', 'quality_check_api_model', e.target.value)}
                placeholder="gpt-4o-mini"
                disabled={config.ai.quality_mode !== 'api'}
              />
            </div>

            {config.ai.quality_mode === 'api' && (
              <div className="form-group">
                <label>🔑 APIキー（オプション）:</label>
                <input
                  type="password"
                  value={config.ai.quality_check_api_key || ''}
                  onChange={(e) => handleInputChange('ai', 'quality_check_api_key', e.target.value)}
                  placeholder="空欄なら基本モデルと同じキーを使用"
                />
              </div>
            )}

            {config.ai.quality_mode === 'api' && (
              <div className="test-button-row">
                <button onClick={handleTestQualityCheck} disabled={testingQualityCheck}>
                  {testingQualityCheck ? '確認中...' : '🔍 品質チェック接続確認'}
                </button>
              </div>
            )}

            {config.ai.quality_mode === 'ollama' && ollamaAvailable && (
              <div className="success-box">
                ✅ Ollama接続済み 🦙
              </div>
            )}
          </div>
        )}
      </div>

      {/* Refiner設定 */}
      <div className="settings-section">
        <h3>Refiner（仕上げ）モデル</h3>
        
        <label className="toggle-label">
          <input 
            type="checkbox"
            checked={customRefiner}
            onChange={(e) => handleCustomRefinerToggle(e.target.checked)}
          />
          <strong>基本モデルとは別のモデルを使用する</strong>
        </label>
        <small className="hint-text">
          基本モデルより下位のAIを指定すると精度が下がります。<br/>
          Criticモデルは常にクオリティチェックモデルと同じものを使用します（変更不可）
        </small>

        {!customRefiner ? (
          <div className="readonly-refiner">
            <p className="info-text">📌 基本モデルと同じ設定を使用します</p>
            <div className="info-box">
              <p><strong>モード:</strong> {config.ai.mode === 'api' ? '🌐 API' : '🏠 Local AI'}</p>
              <p><strong>モデル:</strong> {getCurrentModel()}</p>
            </div>
          </div>
        ) : (
          <div className="custom-config">

            {/* Refiner Local AI */}
            <div className="radio-group">
              <label>
                <input
                  type="radio"
                  name="refiner-mode"
                  value="ollama"
                  checked={config.ai.refiner_mode === 'ollama'}
                  onChange={(e) => handleInputChange('ai', 'refiner_mode', e.target.value)}
                  disabled={!ollamaAvailable || llmModels.length === 0}
                />
                🏠 Local AI
              </label>
              {ollamaAvailable && llmModels.length > 0 ? (
                <select
                  value={config.ai.refiner_ollama_model || ''}
                  onChange={(e) => handleInputChange('ai', 'refiner_ollama_model', e.target.value)}
                  disabled={config.ai.refiner_mode !== 'ollama'}
                >
                  {!config.ai.refiner_ollama_model && <option value="">モデルを選択...</option>}
                  {llmModels.map(model => {
                    const isRecommended = model.recommended_for_base;
                    const label = isRecommended 
                      ? `${model.name} (${model.size}GB) ✅ 推奨`
                      : `${model.name} (${model.size}GB) ⚠️ 性能不足`;
                    
                    return (
                      <option 
                        key={model.name} 
                        value={model.name}
                        disabled={!model.recommended_for_base}
                      >
                        {label}
                      </option>
                    );
                  })}
                </select>
              ) : (
                <select disabled>
                  <option>❌ Ollamaモデルが見つかりません</option>
                </select>
              )}
            </div>

            {/* Refiner API */}
            <div className="radio-group">
              <label>
                <input
                  type="radio"
                  name="refiner-mode"
                  value="api"
                  checked={config.ai.refiner_mode === 'api'}
                  onChange={(e) => handleInputChange('ai', 'refiner_mode', e.target.value)}
                />
                🌐 API
              </label>
              <input
                type="text"
                value={config.ai.refiner_api_model || ''}
                onChange={(e) => handleInputChange('ai', 'refiner_api_model', e.target.value)}
                placeholder="gpt-4o-mini"
                disabled={config.ai.refiner_mode !== 'api'}
              />
            </div>

            {config.ai.refiner_mode === 'api' && (
              <div className="form-group">
                <label>🔑 APIキー（オプション）:</label>
                <input
                  type="password"
                  value={config.ai.refiner_api_key || ''}
                  onChange={(e) => handleInputChange('ai', 'refiner_api_key', e.target.value)}
                  placeholder="空欄なら基本モデルと同じキーを使用"
                />
              </div>
            )}

            {config.ai.refiner_mode === 'api' && (
              <div className="test-button-row">
                <button onClick={handleTestRefiner} disabled={testingRefiner}>
                  {testingRefiner ? '確認中...' : '🔍 Refiner接続確認'}
                </button>
              </div>
            )}

            {config.ai.refiner_mode === 'ollama' && ollamaAvailable && (
              <div className="success-box">
                ✅ Ollama接続済み 🦙
              </div>
            )}
          </div>
        )}
      </div>

      {/* 保存ボタン */}
      <div className="actions">
        <button onClick={handleSave} disabled={loading}>
          {loading ? '保存中...' : '💾 設定を保存'}
        </button>
      </div>
    </div>
  );
};