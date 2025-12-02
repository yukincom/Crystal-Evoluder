"""
データ品質チェッカー（Qwen 8B）
"""
import logging
import json
import re
import csv
import requests
from pathlib import Path
from typing import List, Dict, Any
from llama_index.core import Document


class DataQualityChecker:
    """データ品質チェック（ローカルLLM: Qwen 8B）"""
    
    def __init__(self, ollama_url: str = 'http://localhost:11434', logger: logging.Logger = None):
        self.ollama_url = ollama_url
        self.logger = logger or logging.getLogger('DataQualityChecker')
        self.ollama_available = self._check_ollama()
        
        if self.ollama_available:
            self.logger.info("✅ Ollama (Qwen 8B) available")
        else:
            self.logger.warning("⚠️  Ollama not available (quality check disabled)")
    
    def _check_ollama(self) -> bool:
        """Ollamaが起動しているか確認"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    # ... 既存のDataQualityCheckerコードをそのままコピー
    # check_documents, _detect_issues, _ai_deep_check など