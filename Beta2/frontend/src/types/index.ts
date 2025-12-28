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
  
  // Critic（読み取り専用、メインに追従）
  self_rag_critic_model: string;
  self_rag_critic_mode: string;
  
  // Refiner（編集可能）
  self_rag_refiner_model: string;
  self_rag_refiner_mode: string;
  self_rag_refiner_ollama_url: string | null;
  self_rag_refiner_api_key: string | null;
  
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
    // モード別のモデル名
    api_model: string;
    ollama_model: string;
    // 後方互換性（非推奨）
    llm_model: string;

    api_key: string;
    vision_model: string; 
        
    // Refiner専用設定
    refiner_mode: string | null;
    refiner_ollama_url: string | null;
    refiner_api_key: string | null;
    refiner_model: string | null;
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
  models: {
    llm: string;
    vision: string;
  };
}
export interface OllamaModel {
  name: string;
  size: number;
  capable: boolean;
  is_vision: boolean;
}
export interface ProcessingStatus {
  total: number;
  processed: number;
  failed: number;  
  current_file: string;
}