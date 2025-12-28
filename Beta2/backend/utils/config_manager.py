"""
設定管理システム
ユーザー設定の保存・読込・検証を担当
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ConfigManager:
    """設定管理クラス"""

    # デフォルト設定
    DEFAULT_CONFIG = {
        # Neo4j接続
        'neo4j': {
            'url': 'bolt://localhost:7687',
            'username': 'neo4j',
            'password': '',
            'database': 'neo4j'
        },

        # API Keys
        'api_keys': {
            'openai': '',
            'anthropic': '',
        },

        # AI設定
        'ai': {
            'mode': 'api',  # 'api' or 'ollama'
            'ollama_url': 'http://localhost:11434',
            'api_key': '',  
            'api_model': 'gpt-4o-mini',  
            'ollama_model': '',           
            'vision_model': 'granite3.2-vision',
            'llm_timeout': 120.0,
        # Refiner専用設定
            'refiner_mode': None,
            'refiner_ollama_url': None,
            'refiner_api_key': None,
            'refiner_model': None,
        },

        # 基本パラメータ
        'parameters': {
            'entity_linking_threshold': 0.88,
            'retrieval_chunk_size': 320,
            'retrieval_chunk_overlap': 120,
            'graph_chunk_size': 512,
            'graph_chunk_overlap': 50,
            'relation_compat_threshold': 0.11,
            'final_weight_cutoff': 0.035,
            'max_triplets_per_chunk': 15,
        },

        # Self-RAG設定
        'self_rag': {
            'enable': True,
            'confidence_threshold': 0.75,
            'critic_model': None,
            'refiner_model': None,
            'max_retries': 1,
            'token_budget': 100000,
        },

        # 図表解析設定
        'figure_analysis': {
            'enable': True,
            'dpi': 200,
            'use_cache': True,
        },

        # 処理設定
        'processing': {
            'enable_duplicate_check': True,
            'enable_provenance': True,
            'log_level': 'INFO',
            'max_workers': 4,
        },

        # Geode設定
        'geode': {
            'input_dir': '',
            'output_dir': './output',
            'patterns': ['*.pdf', '*.md', '*.docx'],
        }
    }

    def __init__(self, config_path: Optional[str] = None):
        """
        初期化

        Args:
            config_path: 設定ファイルパス（省略時はデフォルト）
        """
        if config_path is None:
            # デフォルトパス: backend/config/user_config.json
            backend_dir = Path(__file__).parent.parent.parent / 'backend'
            config_dir = backend_dir / 'config'
            config_dir.mkdir(parents=True, exist_ok=True)
            config_path = config_dir / 'user_config.json'

        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """設定ファイルを読み込み"""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)

                # デフォルト設定とマージ
                config = self._deep_merge(self.DEFAULT_CONFIG.copy(), user_config)
                logger.info(f"Config loaded from {self.config_path}")
                return config

            except Exception as e:
                logger.error(f"Failed to load config: {e}")
                return self.DEFAULT_CONFIG.copy()

        else:
            logger.info("No config file found, using defaults")
            return self.DEFAULT_CONFIG.copy()

    def save_config(self) -> bool:
        """設定ファイルを保存"""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)

            logger.info(f"Config saved to {self.config_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save config: {e}")
            return False

    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """辞書を再帰的にマージ"""
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    # ========================================
    # Getter/Setter
    # ========================================

    def get(self, section: str, key: str = None, default: Any = None) -> Any:
        """
        設定値を取得

        Args:
            section: セクション名（例: 'neo4j'）
            key: キー名（例: 'url'）省略時はセクション全体
            default: デフォルト値

        Returns:
            設定値
        """
        if key is None:
            return self.config.get(section, default)

        section_data = self.config.get(section, {})
        return section_data.get(key, default)

    def set(self, section: str, key: str, value: Any):
        """
        設定値を更新

        Args:
            section: セクション名
            key: キー名
            value: 値
        """
        if section not in self.config:
            self.config[section] = {}

        self.config[section][key] = value

    def get_neo4j_config(self) -> Dict[str, str]:
        """Neo4j設定を取得"""
        return self.config.get('neo4j', {})

    def set_neo4j_config(self, url: str, username: str, password: str, database: str):
        """Neo4j設定を更新"""
        self.config['neo4j'] = {
            'url': url,
            'username': username,
            'password': password,
            'database': database
        }

    def get_api_key(self, provider: str) -> str:
        """APIキーを取得"""
        return self.config.get('api_keys', {}).get(provider, '')

    def set_api_key(self, provider: str, api_key: str):
        """APIキーを設定"""
        if 'api_keys' not in self.config:
            self.config['api_keys'] = {}

        self.config['api_keys'][provider] = api_key

    def get_processing_config(self) -> Dict[str, Any]:
        """
        処理用の完全な設定を取得（バックエンドに渡す用）

        Returns:
            CrystalClusterに渡せる形式の設定辞書
        """
        return {
            # 基本パラメータ
            **self.config['parameters'],

            # LLM設定
            'llm_model': self.config['ai']['llm_model'],
            'llm_timeout': self.config['ai']['llm_timeout'],

            # Self-RAG
            'enable_self_rag': self.config['self_rag']['enable'],
            'self_rag_confidence_threshold': self.config['self_rag']['confidence_threshold'],
            'self_rag_critic_model': self.config['self_rag']['critic_model'],
            'self_rag_refiner_model': self.config['self_rag']['refiner_model'],
            'self_rag_max_retries': self.config['self_rag']['max_retries'],
            'self_rag_token_budget': self.config['self_rag']['token_budget'],

            # 処理設定
            'enable_duplicate_check': self.config['processing']['enable_duplicate_check'],
            'enable_provenance': self.config['processing']['enable_provenance'],
            'max_workers': self.config['processing']['max_workers'],

            # AI Routing
            'ai_routing': {
                'mode': self.config['ai']['mode'],
                'ollama_url': self.config['ai']['ollama_url'],
                'api_key': self.get_api_key('openai'),
            },

            # Neo4j
            'neo4j': self.get_neo4j_config(),
        }

    def reset_to_defaults(self):
        """設定をデフォルトにリセット"""
        self.config = self.DEFAULT_CONFIG.copy()
        logger.info("Config reset to defaults")

    def export_config(self, export_path: str) -> bool:
        """設定をエクスポート"""
        try:
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return False

    def import_config(self, import_path: str) -> bool:
        """設定をインポート"""
        try:
            with open(import_path, 'r', encoding='utf-8') as f:
                imported = json.load(f)

            self.config = self._deep_merge(self.DEFAULT_CONFIG.copy(), imported)
            self.save_config()
            return True

        except Exception as e:
            logger.error(f"Import failed: {e}")
            return False


# シングルトンインスタンス
_global_config_manager = None

def get_config_manager() -> ConfigManager:
    """
    グローバルなConfigManagerインスタンスを取得
    
    Returns:
        ConfigManager: シングルトンインスタンス
    """
    global _global_config_manager
    
    if _global_config_manager is None:
        _global_config_manager = ConfigManager()
    
    return _global_config_manager