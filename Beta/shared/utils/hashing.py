"""
ハッシュ計算ユーティリティ
"""
import hashlib
from pathlib import Path
from typing import Optional


def compute_file_hash(file_path: str, algorithm: str = 'sha256') -> str:
    """
    ファイルのハッシュを計算
    
    Args:
        file_path: ファイルパス
        algorithm: ハッシュアルゴリズム（sha256, md5など）
    
    Returns:
        ハッシュ値（16進数文字列）
    """
    hash_func = hashlib.new(algorithm)
    
    with open(file_path, 'rb') as f:
        # 大きなファイルでもメモリ効率的に処理
        for chunk in iter(lambda: f.read(8192), b''):
            hash_func.update(chunk)
    
    return hash_func.hexdigest()


def compute_text_hash(text: str, algorithm: str = 'sha256') -> str:
    """
    テキストのハッシュを計算
    
    Args:
        text: テキスト
        algorithm: ハッシュアルゴリズム
    
    Returns:
        ハッシュ値（16進数文字列）
    """
    hash_func = hashlib.new(algorithm)
    hash_func.update(text.encode('utf-8'))
    return hash_func.hexdigest()


def compute_fuzzy_hash(text: str, n_grams: int = 3) -> str:
    """
    Fuzzyハッシュ（類似検出用）

    Args:
        text: テキスト
        n_grams: n-gramのサイズ

    Returns:
        簡易的なfuzzyハッシュ
    """
    # 正規化: 小文字化、空白削除
    normalized = ''.join(text.lower().split())

    # n-gramを生成
    ngrams = [normalized[i:i+n_grams] for i in range(len(normalized) - n_grams + 1)]

    # n-gramのハッシュを結合
    ngram_hashes = [hashlib.md5(ng.encode()).hexdigest()[:8] for ng in ngrams[::10]]  # 10個おきにサンプリング

    return '-'.join(ngram_hashes[:5])  # 最初の5個を結合


# 互換性エイリアス
def hash_text(text: str, algorithm: str = 'sha256') -> str:
    """
    テキストのハッシュを計算（エイリアス）
    """
    return compute_text_hash(text, algorithm)


def generate_id(text: str, prefix: str = "", length: int = 8) -> str:
    """
    テキストから一意なIDを生成

    Args:
        text: テキスト
        prefix: IDのプレフィックス
        length: ハッシュの長さ

    Returns:
        生成されたID
    """
    hash_val = compute_text_hash(text)[:length]
    return f"{prefix}{hash_val}" if prefix else hash_val