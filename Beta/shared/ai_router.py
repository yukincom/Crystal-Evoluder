"""
AI Router - API/Ollama 自動切り替えシステム
"""
import os
import json
import requests
from enum import Enum
from typing import Dict, Any, Optional, Union
from openai import OpenAI

from .logger import setup_logger

class TaskType(Enum):
    """タスクタイプ"""
    TRIPLET_EXTRACTION = "triplet_extraction"
    QUALITY_CHECK = "quality_check"
    SELF_RAG = "self_rag"
    GENERAL = "general"

class AIRouter:
    """AI Router - OpenAI API と Ollama の自動切り替え"""

    def __init__(self, config: Optional[Dict[str, Any]] = None, logger=None):
        self.config = config or {}
        self.logger = logger or setup_logger('AIRouter')

        # デフォルト設定
        self.task_defaults = {
            TaskType.TRIPLET_EXTRACTION: {
                'api_model': 'gpt-4o-mini',
                'ollama_model': 'llama3.1:70b',
                'temperature': 0.3,
                'max_tokens': 1024
            },
            TaskType.QUALITY_CHECK: {
                'api_model': 'gpt-4o-mini',
                'ollama_model': 'qwen2.5:32b',
                'temperature': 0.1,
                'max_tokens': 512
            },
            TaskType.SELF_RAG: {
                'api_model': 'gpt-4o-mini',
                'ollama_model': 'llama3.1:70b',
                'temperature': 0.7,
                'max_tokens': 2048
            },
            TaskType.GENERAL: {
                'api_model': 'gpt-4o-mini',
                'ollama_model': 'llama3.1:8b',
                'temperature': 0.5,
                'max_tokens': 1024
            }
        }

        # AIルーティング設定
        self.ai_routing = self.config.get('ai_routing', {
            'mode': 'api',  # 'api' or 'ollama'
            'ollama_url': 'http://localhost:11434',
            'api_key': os.environ.get('OPENAI_API_KEY')
        })

        # モード設定
        self.mode = self.ai_routing.get('mode', 'api')
        self.ollama_url = self.ai_routing.get('ollama_url', 'http://localhost:11434')
        self.api_key = self.ai_routing.get('api_key') or os.environ.get('OPENAI_API_KEY')

        # Ollama接続確認
        self.ollama_available = self._check_ollama()

        # OpenAIクライアント初期化
        self.openai_client = None
        if self.api_key:
            self.openai_client = OpenAI(api_key=self.api_key)

        self.logger.info(f"✅ AI Router initialized (mode: {self.mode})")
        if self.ollama_available:
            self.logger.info("✅ Ollama available")
        else:
            self.logger.warning("⚠️  Ollama not available (API mode only)")

    def _check_ollama(self) -> bool:
        """Ollamaサーバーが起動しているか確認"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False

    def call(self, task: Union[TaskType, str], prompt: str, system_prompt: str = "", **kwargs) -> str:
        """
        AI呼び出し（自動切り替え）

        Args:
            task: タスクタイプ
            prompt: ユーザープロンプト
            system_prompt: システムプロンプト
            **kwargs: 追加パラメータ

        Returns:
            AI応答
        """
        if isinstance(task, str):
            task = TaskType(task)

        # タスク設定取得
        task_config = self.task_defaults.get(task, self.task_defaults[TaskType.GENERAL])

        # モード決定
        if self.mode == 'ollama' and self.ollama_available:
            return self._call_ollama(task_config, prompt, system_prompt, **kwargs)
        else:
            return self._call_api(task_config, prompt, system_prompt, **kwargs)

    def _call_api(self, task_config: Dict, prompt: str, system_prompt: str, **kwargs) -> str:
        """OpenAI API呼び出し"""
        if not self.openai_client:
            raise ValueError("OpenAI API key not configured")

        model = kwargs.get('model', task_config['api_model'])
        temperature = kwargs.get('temperature', task_config['temperature'])
        max_tokens = kwargs.get('max_tokens', task_config['max_tokens'])

        try:
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=120
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"OpenAI API call failed: {e}")
            raise

    def _call_ollama(self, task_config: Dict, prompt: str, system_prompt: str, **kwargs) -> str:
        """Ollama呼び出し"""
        model = kwargs.get('model', task_config['ollama_model'])
        temperature = kwargs.get('temperature', task_config['temperature'])

        # Ollama API 形式
        data = {
            "model": model,
            "prompt": f"{system_prompt}\n\n{prompt}" if system_prompt else prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": kwargs.get('max_tokens', task_config['max_tokens'])
            }
        }

        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=data,
                timeout=120
            )
            response.raise_for_status()

            result = response.json()
            return result.get('response', '')
        except Exception as e:
            self.logger.error(f"Ollama call failed: {e}")
            raise

    def switch_mode(self, mode: str) -> bool:
        """
        AIモード切り替え

        Args:
            mode: 'api' or 'ollama'

        Returns:
            成功したか
        """
        if mode not in ['api', 'ollama']:
            self.logger.error(f"Invalid mode: {mode}")
            return False

        old_mode = self.mode
        self.mode = mode

        if mode == 'ollama' and not self.ollama_available:
            self.logger.warning("⚠️  Ollama not available, staying in API mode")
            self.mode = 'api'
            return False

        self.logger.info(f"🔀 Switching AI mode: {old_mode} → {self.mode}")
        return True

    def get_status(self) -> Dict[str, Any]:
        """現在のステータス取得"""
        return {
            'mode': self.mode,
            'ollama_available': self.ollama_available,
            'ollama_url': self.ollama_url,
            'api_configured': bool(self.api_key)
        }