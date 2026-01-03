import React, { useState, useCallback } from 'react';
import './DragDrop.css';

export const ClusterTab: React.FC = () => {
  const [isDragging, setIsDragging] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);

  const handleDragEnter = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const files = e.dataTransfer.files;
    if (files.length > 0 && files[0].name.endsWith('.json')) {
      setSelectedFile(files[0]);
    } else {
      alert('JSONファイルを選択してください');
    }
  }, []);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      setSelectedFile(files[0]);
    }
  }, []);

  const handleExecute = async () => {
    if (!selectedFile) {
      alert('ファイルを選択してください');
      return;
    }

    setIsProcessing(true);
    try {
      // TODO: バックエンドAPIを呼び出す
      console.log('Processing file:', selectedFile.name);
      
      // 仮の処理時間
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      alert('グラフ構築が完了しました！');
    } catch (error) {
      console.error('Error:', error);
      alert('エラーが発生しました');
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="cluster-tab-wrapper">
    <div className="cluster-tab">
      {/* ドラッグ&ドロップエリア */}
      <div
        className={`drag-drop-area ${isDragging ? 'dragging' : ''}`}
        onDragEnter={handleDragEnter}
        onDragLeave={handleDragLeave}
        onDragOver={handleDragOver}
        onDrop={handleDrop}
        onClick={() => document.getElementById('file-input')?.click()}
      >
        <p className="drag-drop-text">
          {selectedFile ? selectedFile.name : 'ドラッグ and ドロップ'}
        </p>
        <p className="drag-drop-hint">又はファイルを選択</p>
        <p className="drag-drop-file-type">JSON only</p>
        
        <input
          id="file-input"
          type="file"
          accept=".json"
          style={{ display: 'none' }}
          onChange={handleFileSelect}
        />
      </div>


      {/* 実行ボタン */}
      <button
        className="execute-button"
        onClick={handleExecute}
        disabled={!selectedFile || isProcessing}
      >
        {isProcessing ? '処理中...' : '実 行'}
      </button>
    </div>
    </div>
  );
};