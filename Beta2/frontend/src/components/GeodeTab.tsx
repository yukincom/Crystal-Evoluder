import React, { useState, useCallback } from 'react';
import './GeodeTab.css';

interface SelectedFile {
  name: string;
  path: string;
}

export const GeodeTab: React.FC = () => {
  const [isDragging, setIsDragging] = useState(false);
  const [selectedFiles, setSelectedFiles] = useState<SelectedFile[]>([]);
  const [outputDir, setOutputDir] = useState('...User/Desktop/paper/rag');
  const [isProcessing, setIsProcessing] = useState(false);

  // ドラッグ&ドロップ処理
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

    const items = Array.from(e.dataTransfer.items);
    const newFiles: SelectedFile[] = [];

    items.forEach((item) => {
      if (item.kind === 'file') {
        const file = item.getAsFile();
        if (file) {
          // 対応形式チェック
          const supportedFormats = ['.pdf', '.txt', '.tei', '.docx', '.html', '.md'];
          const ext = '.' + file.name.split('.').pop()?.toLowerCase();
          
          if (supportedFormats.includes(ext)) {
            newFiles.push({
              name: file.name,
              path: file.webkitRelativePath || file.name
            });
          }
        }
      }
    });

    setSelectedFiles(prev => [...prev, ...newFiles]);
  }, []);

  // ファイル選択
  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files) return;

    const newFiles: SelectedFile[] = Array.from(files).map(file => ({
      name: file.name,
      path: file.webkitRelativePath || file.name
    }));

    setSelectedFiles(prev => [...prev, ...newFiles]);
  }, []);

  // フォルダ選択
  const handleFolderSelect = useCallback(() => {
    const input = document.createElement('input');
    input.type = 'file';
    input.webkitdirectory = true;
    input.multiple = true;
    
    input.onchange = (e) => {
      const files = (e.target as HTMLInputElement).files;
      if (!files) return;

      const newFiles: SelectedFile[] = Array.from(files).map(file => ({
        name: file.name,
        path: file.webkitRelativePath
      }));

      setSelectedFiles(newFiles);
    };
    
    input.click();
  }, []);

  // ファイル削除
  const handleRemoveFile = (index: number) => {
    setSelectedFiles(prev => prev.filter((_, i) => i !== index));
  };

  // フォルダクリア
  const handleClearFolder = () => {
    setSelectedFiles([]);
  };

  // 実行
  const handleExecute = async () => {
    if (selectedFiles.length === 0) {
      alert('ファイルを選択してください');
      return;
    }

    setIsProcessing(true);
    try {
      // TODO: バックエンドAPI呼び出し
      const response = await fetch('http://localhost:8000/geode/parse', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          files: selectedFiles.map(f => f.path),
          output_dir: outputDir
        })
      });

      if (response.ok) {
        const result = await response.json();
        alert(`✅ 変換完了！\n出力: ${result.output}`);
      } else {
        throw new Error('変換に失敗しました');
      }
    } catch (error) {
      console.error('Error:', error);
      alert('❌ エラーが発生しました');
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="geode-tab">
      <div className="geode-container">
        <h2 className="geode-title">データをJSONに変換</h2>

        {/* ドラッグ&ドロップエリア */}
        <div
          className={`geode-drop-area ${isDragging ? 'dragging' : ''}`}
          onDragEnter={handleDragEnter}
          onDragLeave={handleDragLeave}
          onDragOver={handleDragOver}
          onDrop={handleDrop}
          onClick={() => document.getElementById('geode-file-input')?.click()}
        >
          <p className="drop-text">ドラッグandドロップ</p>
          <p className="drop-hint">又はファイルを選択</p>
          <div className="drop-icons">
            📄 📁
          </div>
          <p className="drop-formats">
            対応形式：<br/>
            PDF, TXT, TEI, DOCX, HTML, Markdown
          </p>
          
          <input
            id="geode-file-input"
            type="file"
            accept=".pdf,.txt,.xml,.docx,.html,.md"
            multiple
            style={{ display: 'none' }}
            onChange={handleFileSelect}
          />
        </div>

        {/* ファイルリスト */}
        {selectedFiles.length > 0 && (
          <div className="file-list-container">
            <div className="file-list-header">
              <span className="folder-path">
                📁 /mypaper
                <button 
                  className="btn-clear-folder"
                  onClick={handleClearFolder}
                >
                  ✕
                </button>
              </span>
            </div>
            
            <div className="file-list">
              {selectedFiles.map((file, index) => (
                <div key={index} className="file-item">
                  <span className="file-icon">📄</span>
                  <span className="file-name">{file.name}</span>
                  <button
                    className="btn-remove-file"
                    onClick={() => handleRemoveFile(index)}
                  >
                    ✕
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* 出力先設定 */}
        <div className="output-dir-container">
          <input
            type="text"
            value={outputDir}
            onChange={(e) => setOutputDir(e.target.value)}
            className="output-dir-input"
            placeholder="出力先フォルダ"
          />
          <button className="btn-save-dir">保存先</button>
        </div>

        <div className="upload-status">
          <label>
            <input type="checkbox" defaultChecked />
            上書き保存
          </label>
        </div>

        {/* 実行ボタン */}
        <button
          className="execute-button"
          onClick={handleExecute}
          disabled={selectedFiles.length === 0 || isProcessing}
        >
          {isProcessing ? '処理中...' : '実 行'}
        </button>

      </div>
    </div>
  );
};