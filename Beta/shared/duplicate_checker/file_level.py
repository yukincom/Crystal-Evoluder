"""
File-Level Duplicate Checker
PDF→JSON化前の重複検知（ファイルハッシュベース）
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from shared.utils.hashing import compute_file_hash
from shared.duplicate_checker.provenance import ProvenanceManager


class FileLevelDuplicateChecker:
    """
    ファイルレベルの重複検知
    
    使い方:
        checker = FileLevelDuplicateChecker(cache_file='file_cache.json')
        is_duplicate, existing = checker.check_duplicate('document.pdf')
        if not is_duplicate:
            provenance = checker.add_file('document.pdf', source_type='pdf')
    """
    
    def __init__(
        self,
        cache_file: str = '.file_duplicate_cache.json',
        neo4j_store: Optional[Any] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Args:
            cache_file: ローカルキャッシュファイル（JSON）
            neo4j_store: Neo4jストア（オプション）
            logger: ロガー
        """
        self.cache_file = Path(cache_file)
        self.neo4j_store = neo4j_store
        self.logger = logger or logging.getLogger(__name__)
        self.provenance_mgr = ProvenanceManager(logger=self.logger)
        
        # キャッシュをロード
        self.cache = self._load_cache()
    
    def _load_cache(self) -> Dict[str, Dict]:
        """ローカルキャッシュをロード"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load cache: {e}")
        
        return {}
    
    def _save_cache(self):
        """ローカルキャッシュを保存"""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Failed to save cache: {e}")
    
    def check_duplicate(
        self,
        file_path: str,
        check_neo4j: bool = True
    ) -> Tuple[bool, Optional[Dict]]:
        """
        ファイルが重複しているかチェック
        
        Args:
            file_path: チェックするファイルパス
            check_neo4j: Neo4jもチェックするか
        
        Returns:
            (is_duplicate, existing_provenance)
        """
        # ファイルハッシュを計算
        try:
            file_hash = compute_file_hash(file_path)
        except Exception as e:
            self.logger.error(f"Failed to compute hash for {file_path}: {e}")
            return False, None
        
        # 1. ローカルキャッシュをチェック
        if file_hash in self.cache:
            existing = self.cache[file_hash]
            self.logger.info(
                f"✓ Duplicate found in cache: {file_path} "
                f"(original: {existing['source_path']})"
            )
            return True, existing
        
        # 2. Neo4jをチェック（オプション）
        if check_neo4j and self.neo4j_store:
            existing = self._check_neo4j_duplicate(file_hash)
            if existing:
                self.logger.info(
                    f"✓ Duplicate found in Neo4j: {file_path} "
                    f"(original: {existing['source_path']})"
                )
                # キャッシュにも追加
                self.cache[file_hash] = existing
                self._save_cache()
                return True, existing
        
        return False, None
    
    def _check_neo4j_duplicate(self, file_hash: str) -> Optional[Dict]:
        """Neo4jで重複をチェック"""
        query = """
        MATCH (d:Document {file_hash: $file_hash})
        RETURN d.source_path AS source_path,
               d.source_type AS source_type,
               d.file_hash AS file_hash,
               d.ingested_at AS ingested_at,
               d.version AS version
        LIMIT 1
        """
        
        try:
            result = self.neo4j_store.query(query, {'file_hash': file_hash})
            
            if result:
                return dict(result[0])
        
        except Exception as e:
            self.logger.debug(f"Neo4j duplicate check failed: {e}")
        
        return None
    
    def add_file(
        self,
        file_path: str,
        source_type: str,
        metadata: Optional[Dict] = None,
        save_to_neo4j: bool = True
    ) -> Dict:
        """
        ファイルを登録（重複チェック済み前提）
        
        Args:
            file_path: ファイルパス
            source_type: ファイルタイプ
            metadata: 追加メタデータ
            save_to_neo4j: Neo4jにも保存するか
        
        Returns:
            Provenance情報
        """
        # ファイルハッシュを計算
        file_hash = compute_file_hash(file_path)
        
        # Provenance作成
        provenance = self.provenance_mgr.create_provenance(
            source_path=file_path,
            source_type=source_type,
            file_hash=file_hash,
            metadata=metadata
        )
        
        # キャッシュに追加
        self.cache[file_hash] = provenance
        self._save_cache()
        
        # Neo4jに保存
        if save_to_neo4j and self.neo4j_store:
            self._save_to_neo4j(provenance)
        
        self.logger.info(f"✓ Registered file: {file_path} (hash: {file_hash[:8]}...)")
        
        return provenance
    
    def _save_to_neo4j(self, provenance: Dict):
        """ProvenanceをNeo4jに保存"""
        query = """
        MERGE (d:Document {file_hash: $file_hash})
        ON CREATE SET
            d.source_path = $source_path,
            d.source_type = $source_type,
            d.ingested_at = $ingested_at,
            d.version = $version,
            d.file_size = $file_size,
            d.metadata_json = $metadata_json
        ON MATCH SET
            d.last_checked = datetime()
        """
        
        try:
            props = self.provenance_mgr.to_neo4j_properties(provenance)
            self.neo4j_store.query(query, props)
        
        except Exception as e:
            self.logger.error(f"Failed to save to Neo4j: {e}")