export interface AIStatus {
  mode: 'api' | 'ollama';
  ollama_available: boolean;
  ollama_url: string;
  models: {
    triplet_extraction: string;
    quality_check: string;
    self_rag: string;
  };
}

export interface ProcessingStatus {
  total: number;
  processed: number;
  current_file: string;
}

export interface Config {
  neo4j: {
    url: string;
    user: string;
    password: string;
    database: string;
  };
  ai: {
    mode: 'api' | 'local';
    api_key: string;
    model: string;
    local_models: string[];
    figure_model: string;
  };
  processing: {
    input_dir: string;
    output_dir: string;
  };
  advanced: {
    graph_chunk_size: number;
    retrieval_chunk_size: number;
    self_rag_enabled: boolean;
    self_rag_threshold: number;
  };
}