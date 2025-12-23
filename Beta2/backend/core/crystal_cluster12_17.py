"""
メイン統合クラス - Crystal Cluster
"""
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

from llama_index.core import Document
from llama_index.graph_stores.neo4j import Neo4jGraphStore

from config import load_config
from shared import setup_logger, HierarchicalLogger
from processors import DocumentProcessor, ChunkProcessor
from builders import GraphBuilder, RetrievalBuilder
from linkers import EntityLinker
from filters import TripletFilter
from rag import MultiHopExplorer


class CrystalCluster:
    """Crystal Cluster - Neo4j投入専用"""

    def __init__(self, log_level: int = 20, use_dual_chunk: bool = False, custom_config: dict = None):
        """
        Args:
            use_dual_chunk: Trueならデュアルチャンク機能を有効化
        """
        self.logger = setup_logger('CrystalCluster', log_level)
        self.hlogger = HierarchicalLogger(self.logger)
        self.use_dual_chunk = use_dual_chunk

        default_config = {
            'entity_linking_threshold': 0.88,
            'retrieval_chunk_size': 320,
            'retrieval_chunk_overlap': 120,
            'graph_chunk_size': 512,
            'graph_chunk_overlap': 50,
            'relation_compat_threshold': 0.11,
            'final_weight_cutoff': 0.035,
            'max_triplets_per_chunk': 15,
            'llm_model': 'gpt-4o-mini',
            'llm_timeout': 120.0,

            # Self-RAG設定
            'enable_self_rag': True,                    # Self-RAGを有効化
            'self_rag_confidence_threshold': 0.75,       # 再生成の閾値
            'self_rag_critic_model': 'gpt-4o-mini',     # 評価用LLM
            'self_rag_refiner_model': 'gpt-5-mini',         # 再生成用LLM（より高性能）
            'self_rag_max_retries': 1,                  # 最大再試行回数
            'self_rag_token_budget': 100000,            # トークン予算
            'self_rag_validation_checks': [             # 検証項目
                'entity_quality',
                'relation_clarity',
                'grammar',
                'redundancy'
            ],
            # Multi-hop設定（最適化版）
            'multihop_beam_width': 2,                   # ビーム幅を狭く
            'multihop_max_paths': 50,                   # パス数上限を追加
            # RAPL最適化設定
            'rapl_max_entities': 100,           # Inter計算で処理する最大エンティティ数
            'rapl_min_shared_triples': 3,       # 共有トリプル数の最小値（2→3）
            'neo4j_batch_size': 1000,           # Neo4jバッチサイズ
            
        }
        # カスタム設定があれば上書き
        if custom_config:
            default_config.update(custom_config)

        self.config = default_config
        self.config.setdefault('enable_triplet_filter', True)
        self.config.setdefault('triplet_quality_threshold', 0.3)


        # 関係タイプのブラックリスト
        self.relation_blacklist = {
            'is', 'has', 'are', 'was', 'were',
            'the', 'a', 'an',
            'of', 'in', 'on', 'at',
        }

        from model import ensure_bge_m3
        self.embed_model = ensure_bge_m3()

        # 各コンポーネントの初期化
        self.document_processor = DocumentProcessor(self.logger)
        self.chunk_processor = ChunkProcessor(self.config, self.logger)
        self.graph_builder = GraphBuilder(self.config, self.embed_model, self.logger)
        self.retrieval_builder = RetrievalBuilder(self.embed_model, self.logger)
        self.entity_linker = EntityLinker(self.config, self.logger)
        self.triplet_filter = TripletFilter(self.config, self.logger)
        self.multi_hop_explorer = MultiHopExplorer(self.config, self.logger)

        self.entity_emb_cache = {}          # エンティティ埋め込みキャッシュ
        self.relation_emb_cache = {}        # 関係埋め込みキャッシュ
        self.total_self_rag_tokens = 0      # Self-RAGトークンカウンタ
        self.visited_paths = set()          # Multi-hopパスキャッシュ

        self.logger.info(f"Crystal Cluster beta initialized")
        self.logger.info(f"Self-RAG: {'enabled' if self.config['enable_self_rag'] else 'disabled'}")
        self.logger.info(f"Config: {self.config}")

    def get_cached_embedding(
        self,
        text: str,
        cache_type: str = 'entity'
    ) -> np.ndarray:
        """
        キャッシュ付きで埋め込みを取得

        Args:
            text: テキスト
            cache_type: 'entity' または 'relation'

        Returns:
            正規化された埋め込みベクトル
        """
        # キャッシュ選択
        cache = self.entity_emb_cache if cache_type == 'entity' else self.relation_emb_cache

        # キャッシュヒット
        if text in cache:
            return cache[text]

        # キャッシュミス: 計算して保存
        try:
            emb = self.embed_model.get_text_embedding(text)
            emb = np.array(emb, dtype=np.float32)

            # 正規化
            norm = np.linalg.norm(emb)
            if norm > 1e-9:
                emb = emb / norm
            else:
                emb = np.zeros_like(emb)

            cache[text] = emb

            # キャッシュサイズが大きくなりすぎたら警告
            if len(cache) % 1000 == 0:
                self.logger.debug(f"  {cache_type} cache size: {len(cache)}")

            return emb

        except Exception as e:
            self.logger.debug(f"Embedding failed for '{text[:30]}': {type(e).__name__}")
            # フォールバック: ゼロベクトル
            return np.zeros(1024, dtype=np.float32)

    def load_documents(
        self,
        json_path: str,
        raw_docs: Optional[List[str]] = None,
        path_pickle: Optional[str] = None,
        kg=None,
        neo4j_store=None,
        enable_duplicate_check: bool = True
    ) -> List[Document]:
        """ドキュメントのロードと前処理"""
        self.document_processor.neo4j_store = neo4j_store
        return self.document_processor.load_documents(
            json_path=json_path,
            raw_docs=raw_docs,
            path_pickle=path_pickle,
            kg=kg,
            enable_duplicate_check=enable_duplicate_check
        )



    def create_dual_documents(
        self,
        documents: List[Document]
    ) -> Tuple[List[Document], List[Document]]:
        """デュアルドキュメント生成"""
        return self.chunk_processor.create_dual_documents(documents)

    def build_retrieval_store(
        self,
        retrieval_docs: List[Document]
    ) -> Dict[str, Any]:
        """検索ストア構築"""
        return self.retrieval_builder.build_retrieval_store(retrieval_docs)

    def retrieve(
        self,
        store: Dict,
        query: str,
        top_k: int = 5,
        chunk_mapping: Dict = None
    ) -> List[Tuple[float, Document, List[str]]]:
        """検索実行"""
        return self.retrieval_builder.retrieve(
            store=store,
            query=query,
            top_k=top_k,
            chunk_mapping=chunk_mapping
        )

    def commit_to_graph(self, documents: List[Document], graph_store: Neo4jGraphStore):
        """グラフ構築とNeo4j投入"""
        self.graph_builder.commit_to_graph(documents, graph_store)

    def commit_to_graph_with_retrieval(
        self,
        documents: List[Document],
        graph_store: Neo4jGraphStore
    ) -> Dict[str, Any]:
        """
        Graph index と Retrieval store を同時に構築（同期版）
        """
        with self.hlogger.section("Dual-Index Building (Synced)"):
            # 1. Dual-documents生成（同期マッピング付き）
            graph_docs, retrieval_docs = self.create_dual_documents(documents)

            # 簡易的なchunk_mapping作成
            chunk_mapping = {'graph_to_retrieval': {}, 'retrieval_to_graph': {}}

            self.logger.info(
                f"🔗 Chunk mapping: "
                f"{len(chunk_mapping['graph_to_retrieval'])} graph -> retrieval links"
            )

            # 2. Graph構築
            self.logger.info("📊 Building knowledge graph...")
            self.commit_to_graph(graph_docs, graph_store)

            # 3. Retrieval store構築
            self.logger.info("🔍 Building retrieval store...")
            retrieval_store = self.build_retrieval_store(retrieval_docs)

            # chunk_mappingをstoreに追加
            retrieval_store['chunk_mapping'] = chunk_mapping

        return {
            'retrieval_store': retrieval_store,
            'chunk_mapping': chunk_mapping,
            'stats': {
                'graph_docs': len(graph_docs),
                'retrieval_docs': len(retrieval_docs),
                'sync_links': len(chunk_mapping['retrieval_to_graph'])
            }
        }

    def query_with_multihop(
        self,
        query: str,
        kg,
        retrieval_store: Dict = None,
        max_steps: int = 5,
        top_k_retrieval: int = 5,
        top_k_paths: int = 10
    ) -> Dict[str, Any]:
        """Multi-hop探索を使ったクエリ実行"""
        return self.multi_hop_explorer.query_with_multihop(
            query=query,
            kg=kg,
            retrieval_store=retrieval_store,
            max_steps=max_steps,
            top_k_retrieval=top_k_retrieval,
            top_k_paths=top_k_paths
        )

    def filter_triplets(
        self,
        triplets: List[Tuple[str, str, str]],
        quality_threshold: float = 0.3
    ) -> Tuple[List[Tuple], List[Tuple], Dict]:
        """トリプレット品質フィルタリング"""
        return self.triplet_filter.filter_triplets(triplets, quality_threshold)

    def link_entities(
        self,
        kg,
        similarity_threshold: float = 0.88,
        use_embedding: bool = True
    ) -> Tuple[Any, Dict[str, str]]:
        """エンティティ統合"""
        return self.entity_linker.link_entities(
            kg=kg,
            similarity_threshold=similarity_threshold,
            use_embedding=use_embedding
        )