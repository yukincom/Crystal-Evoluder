import axios from 'axios';

const API_BASE_URL = '/api';

export const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
});

// API関数
export const getAIStatus = async () => {
  const response = await api.get('/ai_status');
  return response.data;
};

export const switchMode = async (mode: 'api' | 'ollama') => {
  const response = await api.post('/switch_mode', { mode });
  return response.data;
};

export const uploadFiles = async (files: File[]) => {
  const formData = new FormData();
  files.forEach(file => formData.append('files', file));

  const response = await api.post('/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  });
  return response.data;
};

export const getConfig = async () => {
  const response = await api.get('/config');
  return response.data;
};

export const saveConfig = async (config: any) => {
  const response = await api.post('/config', config);
  return response.data;
};

export const testNeo4jConnection = async (neo4jConfig: any) => {
  const response = await api.post('/test_neo4j', neo4jConfig);
  return response.data;
};

export const testAIConnection = async (aiConfig: any) => {
  const response = await api.post('/test_ai', aiConfig);
  return response.data;
};