"""
Content-Level Duplicate Checker
JSON化後の重複検知（テキスト内容ベース）
"""
import logging
from typing import Dict, List, Optional, Tuple, Any
from difflib import SequenceMatcher

from shared.utils.hashing import compute_text_hash, compute_fuzzy_hash
from shared.duplicate_checker.provenance import ProvenanceManager


class ContentLevelDuplicateChecker:
    """
    コンテンツレベルの重複検知
    
    使い方:
        checker = ContentLevelDuplicateChecker(
            similarity_threshold=0.85,
            neo4j_store=graph_store
        )
        
        is_duplicate, similar_docs = checker.check_duplicate(document.text)
        if not is_duplicate:
            checker.add_content(document.text, doc_id='doc_123', metadata={...})
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.85,
        neo4j_store: Optional[Any] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Args:
            similarity_threshold: 類似度閾値（0.0～1.0）
            neo4j_store: Neo4jストア（オプション）
            logger: ロガー
        """
        self.similarity_threshold = similarity_threshold
        self.neo4j_store = neo4j_store
        self.logger = logger or logging.getLogger(__name__)
        self.provenance_mgr = ProvenanceManager(logger=self.logger)
        
        # インメモリキャッシュ（起動時のみ有効）
        self.content_cache: Dict[str, Dict] = {}
    
    def check_duplicate(
        self,
        text: str,
        check_fuzzy: bool = True,
        check_neo4j: bool = True
    ) -> Tuple[bool, List[Dict]]:
        """
        テキストが重複しているかチェック
        
        Args:
            text: チェックするテキスト
            check_fuzzy: Fuzzy類似度チェックも行うか
            check_neo4j: Neo4jもチェックするか
        
        Returns:
            (is_duplicate, similar_documents)
        """
        # 完全一致チェック（ハッシュベース）
        text_hash = compute_text_hash(text)
        
        # 1. メモリキャッシュをチェック
        if text_hash in self.content_cache:
            existing = self.content_cache[text_hash]
            self.logger.info(
                f"✓ Exact duplicate found in cache "
                f"(doc_id: {existing.get('doc_id')})"
            )
            return True, [existing]
        
        # 2. Neo4jで完全一致をチェック
        if check_neo4j and self.neo4j_store:
            exact_match = self._check_neo4j_exact(text_hash)
            if exact_match:
                self.logger.info(
                    f"✓ Exact duplicate found in Neo4j "
                    f"(doc_id: {exact_match.get('doc_id')})"
                )
                # キャッシュに追加
                self.content_cache[text_hash] = exact_match
                return True, [exact_match]
        
        # 3. Fuzzy類似度チェック（オプション）
        similar_docs = []
        if check_fuzzy:
            similar_docs = self._check_fuzzy_similarity(text)
            
            if similar_docs:
                self.logger.info(
                    f"✓ Similar content found: {len(similar_docs)} documents "
                    f"(similarity > {self.similarity_threshold})"
                )
                return True, similar_docs
        
        return False, []
    
    def _check_neo4j_exact(self, text_hash: str) -> Optional[Dict]:
        """Neo4jで完全一致をチェック"""
        query = """
        MATCH (c:Content {text_hash: $text_hash})
        RETURN c.doc_id AS doc_id,
               c.text_hash AS text_hash,
               c.text_preview AS text_preview,
               c.ingested_at AS ingested_at,
               c.source_doc AS source_doc
        LIMIT 1
        """
        
        try:
            result = self.neo4j_store.query(query, {'text_hash': text_hash})
            
            if result:
                return dict(result[0])
        
        except Exception as e:
            self.logger.debug(f"Neo4j exact match check failed: {e}")
        
        return None
    
    def _check_fuzzy_similarity(self, text: str) -> List[Dict]:
        """
        Fuzzy類似度チェック
        
        戦略:
        1. Fuzzyハッシュで候補を絞る
        2. 候補に対してSequenceMatcherで詳細比較
        """
        fuzzy_hash = compute_fuzzy_hash(text)
        
        # Neo4jから候補を取得
        if not self.neo4j_store:
            return []
        
        # Fuzzyハッシュが部分一致する候補を取得
        query = """
        MATCH (c:Content)
        WHERE c.fuzzy_hash CONTAINS $fuzzy_hash_prefix
        RETURN c.doc_id AS doc_id,
               c.text_hash AS text_hash,
               c.text_preview AS text_preview,
               c.full_text AS full_text,
               c.ingested_at AS ingested_at
        LIMIT 50
        """
        
        try:
            fuzzy_hash_prefix = fuzzy_hash.split('-')[0]  # 最初のセグメントで検索
            candidates = self.neo4j_store.query(
                query,
                {'fuzzy_hash_prefix': fuzzy_hash_prefix}
            )
            
            # 詳細な類似度計算
            similar_docs = []
            for candidate in candidates:
                candidate_text = candidate.get('full_text', candidate.get('text_preview', ''))
                
                # SequenceMatcherで類似度計算
                similarity = SequenceMatcher(None, text, candidate_text).ratio()
                
                if similarity >= self.similarity_threshold:
                    similar_docs.append({
                        'doc_id': candidate['doc_id'],
                        'text_hash': candidate['text_hash'],
                        'similarity': similarity,
                        'text_preview': candidate['text_preview']
                    })
            
            # 類似度順にソート
            similar_docs.sort(key=lambda x: x['similarity'], reverse=True)
            
            return similar_docs
        
        except Exception as e:
            self.logger.debug(f"Fuzzy similarity check failed: {e}")
            return []
    
    def add_content(
        self,
        text: str,
        doc_id: str,
        metadata: Optional[Dict] = None,
        save_to_neo4j: bool = True,
        store_full_text: bool = False
    ) -> Dict:
        """
        コンテンツを登録（重複チェック済み前提）
        
        Args:
            text: テキスト内容
            doc_id: ドキュメントID
            metadata: 追加メタデータ
            save_to_neo4j: Neo4jにも保存するか
            store_full_text: 全文をNeo4jに保存するか（大きい場合は注意）
        
        Returns:
            コンテンツ情報
        """
        # ハッシュ計算
        text_hash = compute_text_hash(text)
        fuzzy_hash = compute_fuzzy_hash(text)
        
        # コンテンツ情報
        content_info = {
            'doc_id': doc_id,
            'text_hash': text_hash,
            'fuzzy_hash': fuzzy_hash,
            'text_preview': text[:200],  # 最初の200文字
            'text_length': len(text),
            'metadata': metadata or {}
        }
        
        # キャッシュに追加
        self.content_cache[text_hash] = content_info
        
        # Neo4jに保存
        if save_to_neo4j and self.neo4j_store:
            self._save_to_neo4j(content_info, text if store_full_text else None)
        
        self.logger.debug(
            f"✓ Registered content: doc_id={doc_id}, "
            f"hash={text_hash[:8]}..., length={len(text)}"
        )
        
        return content_info
    
    def _save_to_neo4j(self, content_info: Dict, full_text: Optional[str] = None):
        """コンテンツ情報をNeo4jに保存"""
        query = """
        MERGE (c:Content {text_hash: $text_hash})
        ON CREATE SET
            c.doc_id = $doc_id,
            c.fuzzy_hash = $fuzzy_hash,
            c.text_preview = $text_preview,
            c.text_length = $text_length,
            c.ingested_at = datetime(),
            c.metadata_json = $metadata_json
        """
        
        # 全文も保存する場合（オプション）
        if full_text:
            query += ", c.full_text = $full_text"
        
        try:
            import json
            params = {
                'text_hash': content_info['text_hash'],
                'doc_id': content_info['doc_id'],
                'fuzzy_hash': content_info['fuzzy_hash'],
                'text_preview': content_info['text_preview'],
                'text_length': content_info['text_length'],
                'metadata_json': json.dumps(content_info.get('metadata', {}))
            }
            
            if full_text:
                params['full_text'] = full_text
            
            self.neo4j_store.query(query, params)
        
        except Exception as e:
            self.logger.error(f"Failed to save content to Neo4j: {e}")