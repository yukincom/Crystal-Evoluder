import React, { useState } from 'react';
import { BasicSettings } from './components/BasicSettings';
import { AISettings } from './components/AISettings';
import { AdvancedSettings } from './components/AdvancedSettings';
import { DictionaryManagement } from './components/DictionaryManagement';
import { Logs } from './components/Logs';
import { ClusterTab } from './components/ClusterTab';
import { GeodeTab } from './components/GeodeTab';
import './App.css';

type SettingTab = 'neo4j' | 'advanced' | 'ai' | 'dictionary' | 'logs';
type MainTab = 'cluster' | 'geode';

function App() {
  const [mainTab, setMainTab] = useState<MainTab>('cluster');
  const [settingTab, setSettingTab] = useState<SettingTab>('neo4j');

  // 設定コンテンツのマッピング
  const renderSettingContent = () => {
    switch (settingTab) {
      case 'neo4j':
        return <BasicSettings />;
      case 'advanced':
        return <AdvancedSettings />;
      case 'ai':
        return <AISettings />;
      case 'dictionary':
        return <DictionaryManagement />;
      case 'logs':
        return <Logs />;
      default:
        return <BasicSettings />;
    }
  };

  return (
    <div className="app-container">
      {/* サイドバー */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <h2 className="sidebar-title">設定</h2>
        </div>
        <nav className="sidebar-nav">
          <button
            className={`sidebar-button ${settingTab === 'neo4j' ? 'active' : ''}`}
            onClick={() => setSettingTab('neo4j')}
          >
            Neo4j 設定
          </button>
          <button
            className={`sidebar-button ${settingTab === 'advanced' ? 'active' : ''}`}
            onClick={() => setSettingTab('advanced')}
          >
            詳細設定
          </button>
          <button
            className={`sidebar-button ${settingTab === 'ai' ? 'active' : ''}`}
            onClick={() => setSettingTab('ai')}
          >
            AI 設定
          </button>
          <button
            className={`sidebar-button ${settingTab === 'dictionary' ? 'active' : ''}`}
            onClick={() => setSettingTab('dictionary')}
          >
            辞書管理
          </button>
          <button
            className={`sidebar-button ${settingTab === 'logs' ? 'active' : ''}`}
            onClick={() => setSettingTab('logs')}
          >
            ログ
          </button>
        </nav>
      </aside>

      {/* メインエリア */}
      <div className="main-area">
      {/* ヘッダー */}
      <header className="main-header">
        <div>
          <h1 className="main-title">
            💎 Crystal Evoluder
          </h1>
          <p className="main-subtitle">Knowledge Graph RAG System</p>
        </div>
      </header>

      {/* Cluster/Geodeタブ */}
      <div style={{ padding: '24px', background: '#f5f5f5' }}>
        <div className="tabs-container">
          <button
            className={`tab-button ${mainTab === 'cluster' ? 'active' : ''}`}
            onClick={() => setMainTab('cluster')}
          >
            Cluster
          </button>
          <button
            className={`tab-button ${mainTab === 'geode' ? 'active' : ''}`}
            onClick={() => setMainTab('geode')}
          >
            Geode
          </button>
        </div>
      </div>

        {/* メインコンテンツ */}
        <main className="main-content">
          {mainTab === 'cluster' ? (
            <ClusterTab />
          ) : (
            <GeodeTab />
          )}

          {/* 設定エリア（Clusterタブでのみ表示） */}
          {mainTab === 'cluster' && (
            <div className="settings-container">
              {renderSettingContent()}
            </div>
          )}
        </main>
      </div>
    </div>
  );
}

export default App;