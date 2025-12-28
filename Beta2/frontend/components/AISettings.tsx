import React, { useState, useEffect } from 'react';
import { getConfig, saveConfig, testAIConnection } from '../api/client';
import type { Config } from '../types';
import './AISettings.css';

interface OllamaModel {
  name: string;
  size: number;
  capable: boolean;
  is_vision: boolean;
}

export const AISettings: React.FC = () => {
  const [config, setConfig] = useState<Config | null>(null);
  const [loading, setLoading] = useState(false);
  const [testingAI, setTestingAI] = useState(false);
  const [testingRefiner, setTestingRefiner] = useState(false);
  const [customRefiner, setCustomRefiner] = useState(false);

  // Ollamaモデル一覧
  const [ollamaModels, setOllamaModels] = useState<OllamaModel[]>([]);
  const [ollamaAvailable, setOllamaAvailable] = useState(false);
  const [loadingModels, setLoadingModels] = useState(false);

  // LLM用モデル（Visionを除外）
  const llmModels = ollamaModels.filter(m => !m.is_vision);
  
  // Vision用モデル
  const visionModels = ollamaModels.filter(m => m.is_vision);

  useEffect(() => {
    loadOllamaModels();
    loadConfig();
  }, []);

  useEffect(() => {
    // configが読み込まれたら、refiner_modeがnullでなければカスタム設定と判定
    if (config && config.ai.refiner_mode !== null) {
      setCustomRefiner(true);
    }
  }, [config]);

  useEffect(() => {
    // Ollamaモデル読み込み後、configがあれば検証
    if (!config || !ollamaModels.length) return;

    // Ollamaモードかつ、ollama_modelが未設定 or 無効な場合
    if (config.ai.mode === 'ollama') {
      const currentModel = config.ai.ollama_model;
      
      if (!currentModel || currentModel === '') {
        // 有効な最初のモデルを自動設定
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
      } else {
        // 設定されているモデルが有効か確認
        const validModel = llmModels.find(m => 
          m.name === currentModel && m.capable
        );
        
        if (!validModel) {
          console.warn('⚠️ 無効なOllamaモデルが設定されています:', currentModel);
          // 有効な最初のモデルに置き換え
          const firstValidModel = llmModels.find(m => m.capable);
          if (firstValidModel) {
            setConfig({
              ...config,
              ai: {
                ...config.ai,
                ollama_model: firstValidModel.name
              }
            });
            console.log('✅ Replaced with valid Ollama model:', firstValidModel.name);
          }
        }
      }
    }
  }, [config, ollamaModels, llmModels]);

  const loadConfig = async () => {
    try {
      const data = await getConfig();
    // 🔧 追加: 設定の検証とサニタイズ
    if (data.ai.mode === 'ollama' && data.ai.ollama_model) {
      // ollama_modelが実在するか確認（llmModelsがまだ空の場合は後でuseEffectが処理）
      console.log('Loaded ollama_model:', data.ai.ollama_model);
    }
    
    // refiner_modelも検証
    if (data.ai.refiner_model && data.ai.refiner_mode === 'ollama') {
      console.log('Loaded refiner_model:', data.ai.refiner_model);
    }
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
        ollama_model: config.ai.refiner_model || currentModel,
        api_model: config.ai.refiner_model || currentModel,
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

  const handleCustomRefinerToggle = (enabled: boolean) => {
    if (!config) return;
    
    setCustomRefiner(enabled);
    
    if (enabled) {
      // カスタム設定を有効化：現在のモデルをコピー
      let currentModel = config.ai.mode === 'api' 
        ? config.ai.api_model 
        : config.ai.ollama_model;
      
    // 🔧 修正: Ollamaモードで無効なモデルの場合、有効なモデルを選択
    if (config.ai.mode === 'ollama') {
      const isValidModel = llmModels.some(m => m.name === currentModel && m.capable);
      
      if (!isValidModel || !currentModel) {
        // 有効な最初のモデルを取得
        const firstValidModel = llmModels.find(m => m.capable);
        currentModel = firstValidModel?.name || '';
        
        console.warn('無効なモデルが検出されたため、自動修正しました:', currentModel);
      }
    }
    
    // 🔧 修正: APIモードで空の場合もデフォルト値を設定
    if (config.ai.mode === 'api' && !currentModel) {
      currentModel = 'gpt-4o-mini';
    }


      setConfig({
        ...config,
        ai: {
        ...config.ai,
        refiner_mode: null,
        refiner_model: null,
        refiner_api_key: null,
        refiner_ollama_url: null
        }
      });
    } else {
      // カスタム設定を無効化
      setConfig({
        ...config,
        ai: {
          ...config.ai,
          refiner_mode: null,
          refiner_model: null,
          refiner_api_key: null,
          refiner_ollama_url: null
        }
      });
    }
  };

  if (!config) return <div>Loading...</div>;

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
          <button onClick={loadOllamaModels} className="btn-small">🔄 再読込</button>
        </div>
      )}

      {/* 基本モデル設定 */}
      <div className="settings-section">
        <h3>基本モデル選択</h3>

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
              {llmModels.map(model => (
                <option 
                  key={model.name} 
                  value={model.name}
                  disabled={!model.capable}
                >
                  {model.name} ({model.size}GB) {model.capable ? '✅' : '⚠️ 能力不足'}
                </option>
              ))}
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
          <small>GPT-4o-mini以上を推奨</small>
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

        {/* 接続テスト（APIモードのみ） */}
        {config.ai.mode === 'api' && (
          <button onClick={handleTestAI} disabled={testingAI}>
            {testingAI ? '確認中...' : '🔐 API接続確認'}
          </button>
        )}

        {/* Ollamaモードの場合は接続済みメッセージ */}
        {config.ai.mode === 'ollama' && ollamaAvailable && (
          <div className="success-box">
            ✅ Ollama接続済み 🦙
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
          Criticモデルは常に基本モデルと同じものを使用します（変更不可）
        </small>

        {!customRefiner ? (
          // 追従モード（読み取り専用表示）
          <div className="readonly-refiner">
            <p className="info-text">📌 基本モデルと同じ設定を使用します</p>
            <div className="info-box">
              <p><strong>モード:</strong> {config.ai.mode === 'api' ? '🌐 API' : '🏠 Local AI'}</p>
              <p><strong>モデル:</strong> {getCurrentModel()}</p>
            </div>
          </div>
        ) : (
          // カスタム設定モード
          <div className="custom-config">
            <div className="warning-box">
            </div>

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
                  value={config.ai.refiner_model || ''}
                  onChange={(e) => handleInputChange('ai', 'refiner_model', e.target.value)}
                  disabled={config.ai.refiner_mode !== 'ollama'}
                >
                  {!config.ai.refiner_model && <option value="">モデルを選択...</option>}
                  {llmModels.map(model => (
                    <option 
                      key={model.name} 
                      value={model.name}
                      disabled={!model.capable}
                    >
                      {model.name} ({model.size}GB) {model.capable ? '✅' : '⚠️'}
                    </option>
                  ))}
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
                value={config.ai.refiner_model || ''}
                onChange={(e) => handleInputChange('ai', 'refiner_model', e.target.value)}
                placeholder="claude-sonnet-4-20250514（上位モデル推奨）"
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

            {/* Refiner接続テスト（APIモードのみ） */}
            {config.ai.refiner_mode === 'api' && (
              <button onClick={handleTestRefiner} disabled={testingRefiner}>
                {testingRefiner ? '確認中...' : '🔐 Refiner接続確認'}
              </button>
            )}

            {/* Ollamaモードの場合は接続済みメッセージ */}
            {config.ai.refiner_mode === 'ollama' && ollamaAvailable && (
              <div className="success-box">
                ✅ Ollama接続済み 🦙
              </div>
            )}
          </div>
        )}
      </div>

      {/* 図表解析 */}
      <div className="settings-section">
        <h3>🖼️ 図表解析モデル</h3>
        
        {ollamaAvailable && visionModels.length > 0 ? (
          <div className="form-group">
            <select
              value={config.ai.vision_model || ''}
              onChange={(e) => handleInputChange('ai', 'vision_model', e.target.value)}
            >
              {visionModels.map(model => (
                <option key={model.name} value={model.name}>
                  {model.name} ({model.size}GB)
                </option>
              ))}
            </select>
            <small>図表の解析に使用</small>
          </div>
        ) : (
          <div className="warning-box">
            ⚠️ Visionモデルが見つかりません。
            <code>ollama pull granite3.2-vision</code> でインストールしてください。
          </div>
        )}
      </div>

      {/* 保存ボタン */}
      <div className="actions">
        <button onClick={handleSave} disabled={loading} className="btn-primary">
          {loading ? '保存中...' : ' 設定を保存'}
        </button>
      </div>
    </div>
  );
};