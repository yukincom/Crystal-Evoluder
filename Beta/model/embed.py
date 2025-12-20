# bge-m3を自動でロード

import subprocess
import importlib
import numpy as np
from pathlib import Path

def ensure_bge_m3():
    """
    bge-m3を自動でロード（なければインストール）
    
    Returns:
        HuggingFaceEmbedding インスタンス
    """
    # sentence-transformers チェック
    try:
        importlib.import_module("sentence_transformers")
    except ImportError:
        print("📦 Installing sentence-transformers...")
        subprocess.run(
            ["pip", "install", "sentence-transformers"],
            check=True
        )
    
    # llama-index-embeddings-huggingface チェック
    try:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    except ImportError:
        print("📦 Installing llama-index-embeddings-huggingface...")
        subprocess.run(
            ["pip", "install", "llama-index-embeddings-huggingface"],
            check=True
        )
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    
    model_name = "BAAI/bge-m3"
    
    try:
        embed_model = HuggingFaceEmbedding(
            model_name=model_name,
            device="mps",  # Mac用、他はcuda/cpu
            embed_batch_size=16
        )
        print(f"✅ Loaded embedding model: {model_name}")
        return embed_model
    
    except Exception as e:
        print(f"⚠️  Failed to load {model_name}: {e}")
        print("📦 Installing torch...")
        subprocess.run(["pip", "install", "torch"], check=True)
        
        embed_model = HuggingFaceEmbedding(
            model_name=model_name,
            device="mps",
            embed_batch_size=16
        )
class EmbeddingCache:
    """埋め込みキャッシュ管理クラス"""

    def __init__(self, embed_model=None, cache_size_limit: int = 10000):
        """
        Args:
            embed_model: 埋め込みモデルインスタンス
            cache_size_limit: キャッシュサイズ上限
        """
        self.embed_model = embed_model or ensure_bge_m3()
        self.cache_size_limit = cache_size_limit
        self.cache = {}

    def get_embedding(self, text: str) -> list:
        """
        キャッシュ付きで埋め込みを取得

        Args:
            text: 埋め込み対象テキスト

        Returns:
            埋め込みベクトル
        """
        if text in self.cache:
            return self.cache[text]

        # 新規計算
        embedding = np.array(self.embed_model.get_text_embedding(text))

        # キャッシュサイズチェック
        if len(self.cache) >= self.cache_size_limit:
            # LRU的に古いものを削除（簡易実装）
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]

        self.cache[text] = embedding
        return embedding

    def clear_cache(self):
        """キャッシュをクリア"""
        self.cache.clear()

    def get_cache_size(self) -> int:
        """キャッシュサイズを取得"""
        return len(self.cache)

    def get_cached_embedding(self, text: str) -> list:
        return self.get_embedding(text)
