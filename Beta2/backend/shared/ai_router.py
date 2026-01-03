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
    TRIPLET = "triplet"
    QUALITY_CHECK = "quality_check"
    SELF_RAG_REFINER = "refiner"

class AIRouter:
    """AI Router - API と Ollama の自動切り替え"""

    def __init__(self, config: Optional[Dict[str, Any]] = None, logger = None):
        self.config = config
        self.mode = config['ai']['mode']
        self.api_model = config['ai']['api_model']
        self.ollama_model = config['ai']['ollama_model']
        self.ai_routing = self.config.get('ai_routing', {
            'mode': 'api',  # 'api' or 'ollama'
            'ollama_url': 'http://localhost:11434',
            'api_key': os.environ.get('OPENAI_API_KEY')
        })

        # 基本モデル設定を取得
        self.mode = self.config.get('ai_routing', {}).get('mode', 'api')
        self.api_model = self.config.get('api_model', 'gpt-4o-mini')
        self.ollama_model = self.config.get('ollama_model', '')
        
        # quality_check（空なら model にフォールバック）
        quality_check_mode = config['ai'].get('quality_mode', self.mode)
        self.quality_check_model = (
            config['ai'].get('quality_check_ollama_model') if quality_check_mode == 'ollama'
            else config['ai'].get('quality_check_api_model')) or self.model

        # refiner（空なら model にフォールバック）
        refiner_mode = config['ai'].get('refiner_mode', self.mode)
        self.refiner_model = (
            config['ai'].get('refiner_ollama_model') if refiner_mode == 'ollama'
            else config['ai'].get('refiner_api_model')) or self.model

        # AIルーティング設定
        self.ai_routing = self.config.get('ai_routing', {
            'mode': 'api',  # 'api' or 'ollama'
            'ollama_url': 'http://localhost:11434',
            'api_key': os.environ.get('OPENAI_API_KEY')
        })
        
        self.logger.info(f"✅ AI Router initialized (mode: {self.mode})")
        self.logger.info(f"   TRIPLET model: {self.model}")
        self.logger.info(f"   Quality check model: {self.quality_check_model}")
        self.logger.info(f"   Refiner model: {self.refiner_model}")

    def call(self, task: Union[TaskType, str], prompt: str, system_prompt: str = "", **kwargs) -> str:
        """
        AI呼び出し（自動切り替え）
        Args:
            task: タスクタイプ
            prompt: ユーザープロンプト
            system_prompt: システムプロンプト
            **kwargs: その他のパラメータ（temperature, max_tokensなど）
        """
        if isinstance(task, str):
            task = TaskType(task)
        # ★ 品質チェック専用パス（最優先で専用モデルを使う）
                # タスクに応じて使うモデルを選択（これだけ！）
        if task == TaskType.QUALITY_CHECK:
            model = self.quality_check_model
            temperature = kwargs.get('temperature', 0.15)
            max_tokens = kwargs.get('max_tokens', 768)
        elif task == TaskType.SELF_RAG_REFINER:
            model = self.refiner_model
            temperature = kwargs.get('temperature', 0.7)
            max_tokens = kwargs.get('max_tokens', 2048)
        elif task == TaskType.TRIPLET:
            model = self.model
            temperature = kwargs.get('temperature', 0.3)
            max_tokens = kwargs.get('max_tokens', 1024)

        # モード決定（共通部分はそのまま）

        if self.mode == 'ollama':
            return self._call_ollama(task, prompt, system_prompt, model=model if 'model' in locals() else model, temperature=temperature, max_tokens=max_tokens, **kwargs)
        else:
            return self._call_api(task, prompt, system_prompt, model=model if 'model' in locals() else model, temperature=temperature, max_tokens=max_tokens, **kwargs)


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
        
        # 🔧 モデルが空の場合はエラー
        if not model:
            raise ValueError(
                f"Ollama model not configured. Please set 'ollama_model' in config."
            )
        
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
        def get_model_for_task(task_type: TaskType) -> str:
            key = 'ollama_model' if self.mode == 'ollama' else 'api_model'
            return self.task_defaults[task_type][key]
        return {
            'mode': self.mode,
            'ollama_available': self.ollama_available,
            'ollama_url': self.ollama_url,
            'api_configured': bool(self.api_key),
            'models': {
                'api': self.api_model,
                'ollama': self.ollama_model or '(not set)'
            },
            'quality_check_models': {
                'api': self.quality_check_api_model,
                'ollama': self.quality_check_ollama_model or '(not set)',
                'currently_using': (
                self.quality_check_ollama_model if self.mode == 'ollama'
                else self.quality_check_api_model)
            },
            'task_models': {
                'triplet': get_model_for_task(TaskType.TRIPLET),
                'quality_check': get_model_for_task(TaskType.QUALITY_CHECK),
                'self_rag_refiner': get_model_for_task(TaskType.SELF_RAG_REFINER),
            }
        }

def update_config(self, ai_config: Dict[str, Any]):
    """
    設定を動的に更新し、クライアントを再生成する
    
    Args:
        ai_config: 新しいAI設定（辞書形式）
    """
    self.logger.info("🔄 Updating AI Router configuration...")
    
    # 基本設定を更新
    self.mode = ai_config.get('mode', 'api')
    self.api_model = ai_config.get('api_model', 'gpt-4o-mini')
    self.ollama_model = ai_config.get('ollama_model', '')
    
    # 品質チェック専用モデル
    quality_mode = ai_config.get('quality_mode')
    if quality_mode == 'api':
        self.quality_check_api_model = ai_config.get('quality_check_api_model', 'gpt-4o-mini')
    elif quality_mode == 'ollama':
        self.quality_check_ollama_model = ai_config.get('quality_check_ollama_model', '')
    
    # Refiner専用モデル
    refiner_mode = ai_config.get('refiner_mode')
    if refiner_mode == 'api':
        self.refiner_api_model = ai_config.get('refiner_api_model', self.api_model)
    elif refiner_mode == 'ollama':
        self.refiner_ollama_model = ai_config.get('refiner_ollama_model', self.ollama_model)
    
    # APIキーを更新
    self.api_key = ai_config.get('api_key', '')
    self.ollama_url = ai_config.get('ollama_url', 'http://localhost:11434')
    
    # OpenAIクライアントを再生成
    if self.api_key:
        self.openai_client = OpenAI(api_key=self.api_key)
        self.logger.info(f"✅ OpenAI Client re-initialized (Mode: {self.mode})")
    else:
        self.openai_client = None
        if self.mode == 'api':
            self.logger.warning("⚠️ API mode but no API key provided")
    
    # Ollama接続を再確認
    self.ollama_available = self._check_ollama()
    
    # タスクデフォルトを更新
    self.task_defaults[TaskType.TRIPLET]['api_model'] = self.api_model
    self.task_defaults[TaskType.TRIPLET]['ollama_model'] = self.ollama_model
    
    self.task_defaults[TaskType.QUALITY_CHECK]['api_model'] = self.quality_check_api_model
    self.task_defaults[TaskType.QUALITY_CHECK]['ollama_model'] = self.quality_check_ollama_model
    
    if refiner_mode:
        self.task_defaults[TaskType.SELF_RAG_REFINER]['api_model'] = self.refiner_api_model if refiner_mode == 'api' else self.api_model
        self.task_defaults[TaskType.SELF_RAG_REFINER]['ollama_model'] = self.refiner_ollama_model if refiner_mode == 'ollama' else self.ollama_model

    self.logger.info("✅ AI Router configuration updated successfully")
    self.logger.info(f"   Mode: {self.mode}")
    self.logger.info(f"   Base model: {self.api_model if self.mode == 'api' else self.ollama_model}")
    self.logger.info(f"   Ollama available: {self.ollama_available}")    