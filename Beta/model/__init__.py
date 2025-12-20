"""
モデル関連モジュール
"""
from .embed import ensure_bge_m3
from .triplet import TripletExtractor

__all__ = ['ensure_bge_m3', 'TripletExtractor']