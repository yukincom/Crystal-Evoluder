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
  const [testingRefiner, setTestingRefiner] = useState(false); // Refiner用  
  const [customRefiner, setCustomRefiner] = useState(false);

    // Ollamaモデル一覧
  const [ollamaModels, setOllamaModels] = useState<OllamaModel[]>([]);
  const [ollamaAvailable, setOllamaAvailable] = useState(false);
  const [loadingModels, setLoadingModels] = useState(false);

  useEffect(() => {
    loadConfig();
    loadOllamaModels(); // Ollamaモデルを取得    
  }, []);

  useEffect(() => {
    // configが読み込まれたら、refiner_modeがnullでなければカスタム設定と判定
    if (config && config.ai.refiner_mode !== null) {
      setCustomRefiner(true);
    }
  }, [config]);

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
      // Refinerの設定を構築
      const refinerConfig = {
        mode: config.ai.refiner_mode || config.ai.mode,
        api_key: config.ai.refiner_api_key || config.ai.api_key,
        ollama_url: config.ai.refiner_ollama_url || config.ai.ollama_url,
        llm_model: config.ai.refiner_model || config.ai.llm_model,
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
        // モード切替時の特別処理
    if (section === 'ai' && field === 'mode') {
      setConfig({
        ...config,
        ai: {
          ...config.ai,
          mode: value,
          // Ollamaに切り替える場合、利用可能なモデルがあればそれを設定
          llm_model: value === 'ollama' && llmModels.length > 0 
            ? llmModels.find(m => m.capable)?.name || llmModels[0].name
            : value === 'api' && config.ai.mode === 'ollama'
              ? 'gpt-4o-mini' // Ollamaから戻す場合のデフォルト
              : config.ai.llm_model
        }
      });
    } else {

    setConfig({
      ...config,
      [section]: {
        ...config[section],
        [field]: value
        }
      });
    }
  };

  const handleCustomRefinerToggle = (enabled: boolean) => {
    setCustomRefiner(enabled);

    if (!config) return;
    
    if (enabled) {
      // カスタム設定を有効化：メインの設定をコピー
      setConfig({
        ...config,
        ai: {
          ...config.ai,
          refiner_mode: config.ai.mode,
          refiner_model: config.ai.llm_model,
          refiner_api_key: config.ai.api_key,
          refiner_ollama_url: config.ai.ollama_url
        }
      });
    } else {
      // カスタム設定を無効化：nullに戻す（メインに追従）
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
  // LLM用モデル（Visionを除外）
  const llmModels = ollamaModels.filter(m => !m.is_vision);
  
  // Vision用モデル
  const visionModels = ollamaModels.filter(m => m.is_vision);
    
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
              disabled={!ollamaAvailable}
            />
            🏠 Local_AI
          </label>

          {ollamaAvailable && llmModels.length > 0 ? (
            <select
              value={config.ai.llm_model}
              onChange={(e) => handleInputChange('ai', 'llm_model', e.target.value)}
              disabled={config.ai.mode !== 'ollama'}
            >
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

            🌐 API_AI
          </label>
          <input
            type="text"
            value={config.ai.llm_model}
            onChange={(e) => handleInputChange('ai', 'llm_model', e.target.value)}
            placeholder="gpt-4o-mini"
            disabled={config.ai.mode !== 'api'}
          />
          <small>GPT-4o-mini以上を推奨</small>
        </div>

        <div className="form-group">
          <label>🔑 APIキー:</label>
          <input
            type="password"
            value={config.ai.api_key}
            onChange={(e) => handleInputChange('ai', 'api_key', e.target.value)}
            disabled={config.ai.mode !== 'api'}
            placeholder={config.ai.mode === 'api' ? 'sk-...' : 'Local AIでは不要'}
          />
        </div>
        {/* 接続テスト（APIモードのみ） */}
        {config.ai.mode === 'api' && (
        <button onClick={handleTestAI} disabled={testingAI}>
          {testingAI ? '確認中...' : '🔐 API 接続確認'}
        </button>
        )}
                {/* Ollamaモードの場合は接続済みメッセージ */}
        {config.ai.mode === 'ollama' && ollamaAvailable && (
          <div className="success-box">
            ✅ Ollama 接続済み 🦙🦙🦙
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
          />モデル変更<small>基本モデルより下位のモデルを指定しないでください。</small>
        </label>

        {!customRefiner ? (
          // 追従モード
          <div className="readonly-info">
            <p>📌 <strong>基本モデルと同じAIを利用</strong></p>

          </div>
        ) : (
          
          // カスタム設定モード
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
                  disabled={!ollamaAvailable}                  
                />
              🏠 Local AI
              </label>
              {ollamaAvailable && llmModels.length > 0 ? (
                <select
                  value={config.ai.refiner_model || config.ai.llm_model}
                  onChange={(e) => handleInputChange('ai', 'refiner_model', e.target.value)}
                  disabled={config.ai.refiner_mode !== 'ollama'}
                >
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
                value={config.ai.refiner_model || config.ai.llm_model}
                onChange={(e) => handleInputChange('ai', 'refiner_model', e.target.value)}
                placeholder="gpt-4o（上位モデル推奨）"
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

        {/* 接続テスト（APIモードのみ） */}
        {config.ai.refiner_mode === 'api' && (            
        <button onClick={handleTestRefiner} disabled={testingRefiner}>
          {testingRefiner ? '確認中...' : '🔐 API接続確認'}
        </button>
        )}

        {/* Ollamaモードの場合は接続済みメッセージ */}
        {config.ai.refiner_mode === 'ollama' && ollamaAvailable && (
          <div className="success-box">
            ✅ Ollama 接続済み 🦙🦙🦙
          </div>
        )}
        </div>)} 
</div>      

      <div className="settings-section">
        <h3>図表解析モデル</h3>

        {ollamaAvailable && visionModels.length > 0 ? (
          <div className="form-group">
            <select
              value={config.ai.vision_model}
              onChange={(e) => handleInputChange('ai', 'vision_model', e.target.value)}
            >
              {visionModels.map(model => (
                <option key={model.name} value={model.name}>
                  {model.name} ({model.size}GB)
                </option>
              ))}
            </select>
          </div>
        ) : (
          <div className="warning-box">
            ⚠️ Visionモデルが見つかりません。
            <code>ollama pull granite3.2-vision</code> でインストールしてください。
          </div>
        )}
      </div>


      <div className="actions">
        <button onClick={handleSave} disabled={loading}>
          {loading ? '保存中...' : '保存'}
        </button>
      </div>
    </div>
  );
};