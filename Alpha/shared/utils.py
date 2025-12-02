"""
共通ユーティリティ
"""
import re
import pickle
from typing import Dict

from pathlib import Path
from typing import List, Dict, Any, Optional


def collect_files(target_dir: str, allowed_ext: List[str]) -> List[str]:
    """ファイル収集"""
    import os
    files = []
    for root, _, filenames in os.walk(target_dir):
        for fname in filenames:
            if fname.lower().endswith(tuple(allowed_ext)):
                files.append(os.path.join(root, fname))
    return files


def sanitize_filename(text: str, max_length: int = 50) -> str:
    """ファイル名用サニタイズ"""
    return re.sub(r'[<>:"/\\|?*]', '', text)[:max_length].strip()


def clean_text(text: str) -> str:
    """テキスト正規化"""
    try:
        import ftfy
        text = ftfy.fix_text(text)
    except ImportError:
        pass
    
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_path_dicts(pickle_path: str) -> List[Dict]:
    """
    Pickleファイルからパス情報を読み込む
    
    Args:
        pickle_path: Pickleファイルのパス
    
    Returns:
        パス情報の辞書リスト
        各要素は以下の形式:
        {
            'question': str,
            'translated_paths': List[List[str]],
            'reasoning_paths': List[List[str]],
            'path_distances': List[float]
        }
    
    Raises:
        FileNotFoundError: ファイルが存在しない
        pickle.UnpicklingError: Pickle形式が不正
    """
    path = Path(pickle_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Path dictionary file not found: {pickle_path}")
    
    with open(path, 'rb') as f:
        data = pickle.load(f)
    
    # データ形式はスクリプト依存(list of dict per sample)
    # Normalize to list of mapping
    normalized = []
    for item in data:
        # item might be {idx: {...}} so flatten
        if isinstance(item, dict) and len(item) == 1:
            # 単一キーの辞書をフラット化
            _, v = next(iter(item.items()))
            normalized.append(v)
        else:
            normalized.append(item)
    
    return normalized


def validate_path_dict(path_dict: Dict) -> bool:
    """
    パス辞書の形式が正しいか検証
    
    Args:
        path_dict: 検証対象の辞書
    
    Returns:
        形式が正しければ True
    """
    required_keys = {'translated_paths'}  # 最低限必要なキー
    optional_keys = {'question', 'reasoning_paths', 'path_distances'}
    
    # 必須キーの存在確認
    if not required_keys.issubset(path_dict.keys()):
        return False
    
    # translated_paths の型確認
    translated_paths = path_dict.get('translated_paths')
    if not isinstance(translated_paths, list):
        return False
    
    # 各パスがリストかどうか確認（最初の1件だけチェック）
    if translated_paths and not isinstance(translated_paths[0], (list, tuple, str)):
        return False
    
    return True


def load_and_validate_paths(pickle_path: str, logger=None) -> Optional[List[Dict]]:
    """
    パス辞書を読み込んで検証（エラーハンドリング付き）
    
    Args:
        pickle_path: Pickleファイルのパス
        logger: ロガー（オプション）
    
    Returns:
        検証済みパス辞書リスト、失敗時は None
    """
    try:
        path_dicts = load_path_dicts(pickle_path)
        
        # 検証
        valid_count = 0
        for pd in path_dicts:
            if validate_path_dict(pd):
                valid_count += 1
        
        if logger:
            logger.info(
                f"Loaded {len(path_dicts)} path dictionaries "
                f"({valid_count} valid, {len(path_dicts) - valid_count} invalid)"
            )
        
        # 全て無効なら警告
        if valid_count == 0:
            if logger:
                logger.warning("No valid path dictionaries found!")
            return None
        
        return path_dicts
    
    except FileNotFoundError as e:
        if logger:
            logger.error(f"Path file not found: {e}")
        return None
    
    except pickle.UnpicklingError as e:
        if logger:
            logger.error(f"Invalid pickle format: {e}")
        return None
    
    except Exception as e:
        if logger:
            logger.error(f"Failed to load paths: {type(e).__name__} - {str(e)[:100]}")
        return None


def safe_get_nested(data: Dict, *keys, default=None) -> Any:
    """
    ネストした辞書から安全に値を取得
    
    Example:
        >>> data = {'a': {'b': {'c': 123}}}
        >>> safe_get_nested(data, 'a', 'b', 'c')
        123
        >>> safe_get_nested(data, 'a', 'x', 'y', default=0)
        0
    """
    current = data
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key)
            if current is None:
                return default
        else:
            return default
    return current


def ensure_list(value: Any) -> List:
    """
    値をリストに変換（既にリストならそのまま）
    
    Args:
        value: 変換対象
    
    Returns:
        リスト形式
    """
    if value is None:
        return []
    elif isinstance(value, list):
        return value
    elif isinstance(value, tuple):
        return list(value)
    else:
        return [value]