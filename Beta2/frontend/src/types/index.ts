// 詳細設定の型定義
export interface AdvancedConfig {
  // チャンク設定
  graph_chunk_size: number;
  graph_chunk_overlap: number;
  retrieval_chunk_size: number;
  retrieval_chunk_overlap: number;
  
  // エンティティリンキング
  entity_linking_threshold: number;
  relation_compat_threshold: number;
  final_weight_cutoff: number;
  max_triplets_per_chunk: number;
  

  // Self-RAG
  self_rag_enabled: boolean;
  self_rag_threshold: number;
  
  self_rag_max_retries: number;
  self_rag_token_budget: number;
}
// メイン設定の型定義
export interface Config {
  neo4j: {
    url: string;
    username: string;
    password: string;
    database: string;
  };
  ai: {
    mode: 'api' | 'ollama';  
    ollama_url: string; 
    // 基本モデル
    api_model: string;
    ollama_model: string;
    api_key: string;

    // クオリティチェックモデル
    quality_mode?: 'api' | 'ollama' | null; // nullなら基本mode追従
    quality_check_api_model?: string;       // API専用モデル名（空なら api_model）
    quality_check_ollama_model?: string;    // Local専用モデル名（空なら ollama_model）
    quality_check_api_key?: string | null;
    quality_check_ollama_url?: string | null;
    // Refinerモデル
    refiner_mode?: 'api' | 'ollama' | null;
    refiner_api_model?: string;             // ← 追加
    refiner_ollama_model?: string;          // ← 追加
    refiner_api_key?: string | null; 
    refiner_ollama_url?: string | null;
  };
    parameters: {
    entity_linking_threshold: number;
    retrieval_chunk_size: number;
    retrieval_chunk_overlap: number;
    graph_chunk_size: number;
    graph_chunk_overlap: number;
    relation_compat_threshold: number;
    final_weight_cutoff: number;
    max_triplets_per_chunk: number;
  };
  
  self_rag: {
    enable: boolean;
    confidence_threshold: number;
    max_retries: number;
    token_budget: number;
  };
  processing: {
    input_dir: string;
    output_dir: string;
    patterns: string[];
  };
  
  advanced: AdvancedConfig;  
}
// その他の型定義
export interface AIStatus {
  mode: 'api' | 'ollama';
  ollama_available: boolean;
  ollama_url: string;
  api_configured: boolean;
  models?: {      
    api: string;
    ollama: string;
  };
  task_models: {
    triplet_extraction: string;
    quality_check: string;
    self_rag_critic: string;
    self_rag_refiner: string;
  };
}

export interface OllamaModel {
  name: string;
  size: number;
  capable: boolean;
  is_vision: boolean;
  recommended_for_base?: boolean;
  recommended_for_quality?: boolean;
}
export interface ProcessingStatus {
  total: number;
  processed: number;
  failed: number;  
  current_file: string;
}