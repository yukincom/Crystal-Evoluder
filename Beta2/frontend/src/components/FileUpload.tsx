import React, { useState } from 'react';
import { uploadFiles } from '../api/client';

export const FileUpload: React.FC = () => {
  const [files, setFiles] = useState<File[]>([]);
  const [uploading, setUploading] = useState(false);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setFiles(Array.from(e.target.files));
    }
  };

  const handleUpload = async () => {
    if (files.length === 0) {
      alert('Please select files');
      return;
    }

    setUploading(true);
    try {
      const result = await uploadFiles(files);
      alert(`Uploaded ${result.file_count} files!`);
      setFiles([]);
    } catch (error) {
      alert('Upload failed');
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="file-upload">
      <h2>📂 Upload Documents</h2>

      <input
        type="file"
        multiple
        accept=".pdf,.md,.txt,.docx"
        onChange={handleFileChange}
      />

      {files.length > 0 && (
        <div>
          <p>Selected: {files.length} files</p>
          <ul>
            {files.map((f, i) => <li key={i}>{f.name}</li>)}
          </ul>
        </div>
      )}

      <button onClick={handleUpload} disabled={uploading || files.length === 0}>
        {uploading ? 'Uploading...' : 'Process Files'}
      </button>
    </div>
  );
};