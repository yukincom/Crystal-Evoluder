"""
Utility Functions
"""
from .. import old_utils
collect_files = old_utils.collect_files
sanitize_filename = old_utils.sanitize_filename
clean_text = old_utils.clean_text

from shared.utils.hashing import compute_file_hash, compute_text_hash, compute_fuzzy_hash, hash_text, generate_id
from shared.utils.batching import batch_process, BatchProcessor
from shared.utils.neo4j_helpers import Neo4jConnection, execute_query
from shared.utils.utils import load_and_validate_paths
from ..utils import collect_files, sanitize_filename, clean_text

__all__ = [
    'compute_file_hash',
    'compute_text_hash',
    'compute_fuzzy_hash',
    'hash_text',
    'generate_id',
    'batch_process',
    'BatchProcessor',
    'Neo4jConnection',
    'execute_query',
    'load_and_validate_paths',
    'collect_files',
    'sanitize_filename',
    'clean_text'
]