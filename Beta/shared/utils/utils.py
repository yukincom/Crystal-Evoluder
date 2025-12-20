"""
Shared Utilities
"""
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional


def load_and_validate_paths(path_pickle: str, logger: Optional[logging.Logger] = None) -> List[Dict[str, Any]]:
    """
    Path pickleファイルをロードして検証

    Args:
        path_pickle: pickleファイルのパス
        logger: ロガー

    Returns:
        path_dicts: パス情報のリスト

    Raises:
        FileNotFoundError: ファイルが存在しない場合
        Exception: ロード失敗時
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    path_file = Path(path_pickle)

    if not path_file.exists():
        raise FileNotFoundError(f"Path pickle file not found: {path_pickle}")

    try:
        with open(path_file, 'rb') as f:
            path_dicts = pickle.load(f)

        if not isinstance(path_dicts, list):
            logger.warning(f"Expected list, got {type(path_dicts)}")
            return []

        # 基本的な検証
        valid_paths = []
        for i, path_dict in enumerate(path_dicts):
            if not isinstance(path_dict, dict):
                logger.warning(f"Path dict {i} is not a dict, skipping")
                continue

            # translated_paths が存在するか確認
            if 'translated_paths' not in path_dict:
                logger.warning(f"Path dict {i} missing 'translated_paths', skipping")
                continue

            valid_paths.append(path_dict)

        logger.info(f"Loaded {len(valid_paths)} valid path dictionaries from {path_pickle}")
        return valid_paths

    except Exception as e:
        logger.error(f"Failed to load path pickle: {e}")
        raise