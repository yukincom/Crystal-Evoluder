import React from 'react';
import { Tabs } from './components/Tabs';
import { BasicSettings } from './components/BasicSettings';
import { AISettings } from './components/AISettings';
import { AdvancedSettings } from './components/AdvancedSettings';
import { DictionaryManagement } from './components/DictionaryManagement';
import { Logs } from './components/Logs';
import './App.css';

function App() {
  const tabs = [
    {
      id: 'basic',
      label: '基本設定',
      content: <BasicSettings />
    },
    {
      id: 'advanced',
      label: '詳細設定',
      content: <AdvancedSettings />
    },
    {
      id: 'ai',
      label: 'AI設定',
      content: <AISettings />
    },
    {
      id: 'dictionary',
      label: '辞書管理',
      content: <DictionaryManagement />
    },
    {
      id: 'logs',
      label: 'ログ',
      content: <Logs />
    }
  ];

  return (
    <div className="app">
      <header>
        <h1>🔮 Crystal Cluster</h1>
        <p>Knowledge Graph RAG System</p>
      </header>

      <main>
        <Tabs tabs={tabs} defaultActiveTab="basic" />
      </main>
    </div>
  );
}

export default App;