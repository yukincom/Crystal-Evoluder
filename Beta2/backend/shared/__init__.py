"""
共通モジュール
"""
from .logger import HierarchicalLogger, setup_logger

# 古いutils.pyの関数を直接インポート（名前衝突を避けるため別名でimport）
import importlib.util
spec = importlib.util.spec_from_file_location("old_utils", "shared/utils.py")
old_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old_utils)

load_and_validate_paths = old_utils.load_and_validate_paths
collect_files = old_utils.collect_files
sanitize_filename = old_utils.sanitize_filename
clean_text = old_utils.clean_text

from .error_handler import ErrorCollector, safe_execute
from .quality_checker import DataQualityChecker

# 新しい構造のインポート
from .duplicate_checker import ContentLevelDuplicateChecker
from .utils.hashing import hash_text, generate_id
from .utils.batching import batch_process, BatchProcessor
from .utils.neo4j_helpers import Neo4jConnection, execute_query

# 新しいユーティリティ
from .text_utils import clean_text, chunk_by_paragraphs
from .file_utils import collect_files, sanitize_filename, detect_encoding, detect_format

__all__ = [
    'HierarchicalLogger',
    'setup_logger',
    'load_and_validate_paths',
    'collect_files',
    'sanitize_filename',
    'clean_text',
    'chunk_by_paragraphs',
    'detect_encoding',
    'detect_format',
    'ErrorCollector',
    'safe_execute',
    'DataQualityChecker',
    'ContentLevelDuplicateChecker',
    'hash_text',
    'generate_id',
    'batch_process',
    'BatchProcessor',
    'Neo4jConnection',
    'execute_query',
]