"""
共通ユーティリティ
"""
import os
import re
import pickle
import ftfy
import chardet
from typing import Dict

from pathlib import Path
from typing import List, Dict, Any, Optional

def collect_files(
    target_dir: str,
    allowed_ext: List[str] = None,
    patterns: List[str] = None
) -> List[Path]:
    """
    ディレクトリから指定された拡張子/パターンのファイルを収集
    
    Args:
        target_dir: 対象ディレクトリ
        allowed_ext: 許可する拡張子のリスト（例: ['.pdf', '.md']）
        patterns: globパターンのリスト（例: ['*.pdf', '*.md']）
    
    Returns:
        ファイルパスのリスト
    
    Examples:
        # 拡張子で指定
        files = collect_files('data/', allowed_ext=['.pdf', '.md'])
        
        # パターンで指定
        files = collect_files('data/', patterns=['*.pdf', '*.md'])
    """
    target_path = Path(target_dir)
    
    if not target_path.exists():
        raise FileNotFoundError(f"Directory not found: {target_dir}")
    
    files = []
    
    # パターンベース（優先）
    if patterns:
        for pattern in patterns:
            files.extend(target_path.rglob(pattern))
        
        # 重複を除去してソート
        files = sorted(set(files))
    
    # 拡張子ベース
    elif allowed_ext:
        # 拡張子を正規化（先頭に.を追加）
        allowed_ext = [
            ext if ext.startswith('.') else f'.{ext}'
            for ext in allowed_ext
        ]
        
        for root, _, filenames in os.walk(target_dir):
            for fname in filenames:
                if any(fname.lower().endswith(ext.lower()) for ext in allowed_ext):
                    files.append(Path(root) / fname)
        
        files = sorted(files)
    
    else:
        # すべてのファイルを収集
        files = [
            f for f in target_path.rglob('*')
            if f.is_file()
        ]
        files = sorted(files)
    
    return files

def sanitize_filename(text: str, max_length: int = 50) -> str:
    """
    ファイル名として安全な文字列に変換
    
    Args:
        text: 元のテキスト
        max_length: 最大長
    
    Returns:
        サニタイズされた文字列
    
    Examples:
        >>> sanitize_filename("My Paper: A Study on <AI>")
        'My_Paper_A_Study_on_AI'
    """
    # 禁止文字を削除
    sanitized = re.sub(r'[<>:"/\\|?*]', '', text)
    
    # 空白をアンダースコアに
    sanitized = re.sub(r'\s+', '_', sanitized)
    
    # 連続するアンダースコアを1つに
    sanitized = re.sub(r'_+', '_', sanitized)
    
    # 先頭・末尾のアンダースコアを削除
    sanitized = sanitized.strip('_')
    
    # 長さ制限
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length].rstrip('_')
    
    # 空文字列の場合はデフォルト
    if not sanitized:
        sanitized = 'untitled'
    
    return sanitized

def clean_text(text: str, normalize_whitespace: bool = True) -> str:
    """
    テキストをクリーニング
    
    Args:
        text: 元のテキスト
        normalize_whitespace: 空白を正規化するか
    
    Returns:
        クリーニングされたテキスト
    
    Examples:
        >>> clean_text("Hello    World\\n\\n\\nTest")
        'Hello World Test'
    """
    # ftfyがあれば文字エンコーディングを修正
    try:
        text = ftfy.fix_text(text)
    except ImportError:
        pass
    
    if normalize_whitespace:
        # 複数の空白を1つに
        text = re.sub(r'\s+', ' ', text)
    
    # 先頭・末尾の空白を削除
    text = text.strip()
    
    return text

def chunk_by_paragraphs(
    content: str,
    chunk_size: int = 2000,
    overlap: int = 200
) -> List[str]:
    """
    段落ベースのチャンク分割
    
    Args:
        content: テキスト内容
        chunk_size: チャンクの最大サイズ（文字数）
        overlap: チャンク間のオーバーラップ
    
    Returns:
        チャンクのリスト
    """
    # 段落に分割
    paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for para in paragraphs:
        para_length = len(para)
        
        # チャンクサイズを超える場合
        if current_length + para_length > chunk_size and current_chunk:
            # 現在のチャンクを保存
            chunks.append('\n\n'.join(current_chunk))
            
            # オーバーラップ処理
            if overlap > 0 and current_chunk:
                # 最後のいくつかの段落を保持
                overlap_text = '\n\n'.join(current_chunk)
                if len(overlap_text) > overlap:
                    # オーバーラップサイズに収まるように調整
                    overlap_paras = []
                    overlap_len = 0
                    for p in reversed(current_chunk):
                        if overlap_len + len(p) <= overlap:
                            overlap_paras.insert(0, p)
                            overlap_len += len(p)
                        else:
                            break
                    current_chunk = overlap_paras
                    current_length = overlap_len
                else:
                    current_chunk = []
                    current_length = 0
            else:
                current_chunk = []
                current_length = 0
        
        current_chunk.append(para)
        current_length += para_length
    
    # 最後のチャンク
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))
    
    return chunks

def detect_encoding(file_path: str) -> str:
    """
    ファイルのエンコーディングを検出
    
    Args:
        file_path: ファイルパス
    
    Returns:
        エンコーディング名（例: 'utf-8', 'shift_jis'）
    """
    try:
        
        with open(file_path, 'rb') as f:
            raw_data = f.read(10000)  # 最初の10KBで判定
            result = chardet.detect(raw_data)
            return result['encoding'] or 'utf-8'
    
    except ImportError:
        # chardetがない場合はutf-8を仮定
        return 'utf-8'
    
    except Exception:
        return 'utf-8'


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

def ensure_dir(path: str) -> Path:
    """ディレクトリが存在することを保証"""
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj

def get_file_size_mb(file_path: str) -> float:
    """ファイルサイズをMB単位で取得"""
    return Path(file_path).stat().st_size / (1024 * 1024)

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