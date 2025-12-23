"""
Crystal Cluster beta
Knowledge Graph committer for Neo4j

"""

# ============================================================
# インポート
# ============================================================
import json
import logging
import pickle
import numpy as np
import networkx as nx
import traceback
import hashlib
import re
import argparse

from difflib import SequenceMatcher
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set

from llama_index.core import Settings
from llama_index.llms.openai import OpenAI  
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.core import Document, KnowledgeGraphIndex, StorageContext
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.core.node_parser import SentenceSplitter

# 共通モジュール
from shared.logger import setup_logger, HierarchicalLogger
from shared.utils import load_and_validate_paths
from shared.error_handler import ErrorCollector, safe_execute
from shared.duplicate_checker import ContentLevelDuplicateChecker


class CrystalCluster:
    """Crystal Cluster - Neo4j投入専用"""
    
    def __init__(self, log_level: int = logging.INFO, use_dual_chunk: bool = False, custom_config: dict = None):
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
            'self_rag_refiner_model': 'gpt-4o',         # 再生成用LLM（より高性能）
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
        
        self.embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-m3",
            device="mps",
            embed_batch_size=16,
        )
        
        Settings.embed_model = self.embed_model

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
        kg: Optional[nx.Graph] = None) -> List[Document]:
        enable_duplicate_check: bool = True  # ← 追加    
        """
        JSON と 生テキスト両方から Document を作る
        
        Args:
            json_path: JSONファイルのパス
            raw_docs: 生テキストのリスト（オプション）
            path_pickle: パス情報のPickleファイル（オプション）
            kg: ナレッジグラフ（パス情報統合時に必要）
        
        Returns:
            Documentのリスト（パス情報が統合されている場合もある）
        """

    
        if enable_duplicate_check:
        
            content_checker = ContentLevelDuplicateChecker(
                similarity_threshold=0.85,
                neo4j_store=getattr(self, 'graph_store', None),
                logger=self.logger
            )
        
            self.logger.info("🔍 Checking for content duplicates...")
            
        documents = []

        # --- JSON 側 ---
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # json_path の documents を追加
        for i, doc in enumerate(data.get('documents', [])):
            doc_text = doc['text']
            if enable_duplicate_check:
                is_duplicate, similar_docs = content_checker.check_duplicate(
                    doc_text,
                    check_fuzzy=True,
                    check_neo4j=True
                )
                if is_duplicate:
                    self.logger.info(
                        f"  ⊗ Skipping duplicate document {i} "
                        f"(similar to: {similar_docs[0].get('doc_id')})"
                    )
                    continue  # 重複ドキュメントはスキップ
                # 重複でない場合は登録
                content_checker.add_content(
                    text=doc_text,
                    doc_id=f"json_{i}",
                    metadata={
                        "source": "json",
                        "json_id": i,
                        **doc.get("metadata", {})
                    },
                    save_to_neo4j=True,
                    store_full_text=False  # 全文は保存しない（サイズ削減）
                )

            documents.append(
                Document(
                    text=doc['text'],
                    metadata={
                        "source": "json",
                        "json_id": i,
                        **doc.get("metadata", {})
                    }
                )
            )

        # --- 生テキスト側 ---
        if raw_docs:
            for i, text in enumerate(raw_docs):
                if enable_duplicate_check:
                    is_duplicate, similar_docs = content_checker.check_duplicate(
                        text,
                        check_fuzzy=True,
                        check_neo4j=True
                    )
                
                    if is_duplicate:
                        self.logger.info(
                            f"  ⊗ Skipping duplicate raw document {i}"
                        )
                        continue
                
                    content_checker.add_content(
                        text=text,
                        doc_id=f"raw_{i}",
                        metadata={"source": "raw", "raw_id": i},
                        save_to_neo4j=True,
                        store_full_text=False
                    )                

                documents.append(
                    Document(
                        text=text,
                        metadata={
                            "source": "raw",
                            "raw_id": i
                        }
                    )
                )

        json_count = len(data.get('documents', []))
        raw_count = len(raw_docs) if raw_docs else 0
        
        self.logger.info(
            f"📂 Loaded {len(documents)} documents "
            f"({json_count} from JSON, {raw_count} raw texts)"
        )

        # --- パス情報の統合（オプション）---
        if path_pickle and kg is not None:
            path_dicts = load_and_validate_paths(path_pickle, self.logger)
            if path_dicts:
                self.logger.info("Augmenting documents with path information...")
                documents = self.augment_documents_with_paths(
                    documents, 
                    path_dicts, 
                    kg,
                    entity_embeddings=getattr(self, 'entity_embeddings', None)
                )
                self.logger.info(f"✅ Path information added to {len(documents)} documents")
            else:
                self.logger.warning("Path information could not be loaded, continuing without it")

        return documents

    def augment_documents_with_paths(
        self,
        documents: List[Document], 
        path_dicts: List[Dict], 
        kg: nx.Graph,
        entity_embeddings: Dict[str, np.ndarray] = None,
        match_key='question') -> List[Document]:
        """
        documents に対応する path 情報を注入
        
        Args:
            documents: Documentのリスト
            path_dicts: load_path_dicts の戻り値
            kg: ナレッジグラフ
            entity_embeddings: エンティティの埋め込み（オプション）
            match_key: documents と path_dicts を突き合わせるキー
        
        Returns:
            パス情報が追加された documents

        """
        # defensive
        if entity_embeddings is None:
            entity_embeddings = {}
        
        # path_dictsが空なら元のドキュメントをそのまま返す
        if len(path_dicts) == 0:
            self.logger.info("  → No path information available, returning original documents")
            return documents

        # インデックス作成： path_dicts を match_key で引けるようにする
        pd_map = {}
        for p in path_dicts:
            key = p.get(match_key)
            if key is not None:
                pd_map[key] = p

        augmented = []
        matched_count = 0

        for doc in documents:
            meta = dict(getattr(doc, 'metadata', {}) or {})
            doc_key = meta.get(match_key)

            matched = None
            if doc_key is not None and doc_key in pd_map:
                matched = pd_map[doc_key]
                matched_count += 1
            else:
            # フォールバック：テキスト内に match_key の文字列が含まれる path_dict を探す
                text = getattr(doc, 'text', '') or ''
                for k, p in pd_map.items():
                    if isinstance(k, str) and k in text:
                        matched = p
                        matched_count += 1
                        break

            paths_meta = []
            if matched:
                for path in matched.get('translated_paths', []):
                # path: list of node names (entities)
                    path_len = len(path)
                    edge_weights = []
                    path_node_pairs = list(zip(path[:-1], path[1:])) if path_len >= 2 else []
                    for u, v in path_node_pairs:
                        if kg.has_edge(u, v):
                            edge_weights.append(kg[u][v].get('weight', 0.0))
                        elif kg.has_edge(v, u):
                            edge_weights.append(kg[v][u].get('weight', 0.0))
                        else:
                        # edge が存在しない場合は 0.0 を入れておく
                            edge_weights.append(0.0)

                    avg_edge_weight = float(np.mean(edge_weights)) if edge_weights else 0.0
                    sum_edge_weight = float(np.sum(edge_weights)) if edge_weights else 0.0

                # path 内ノードの埋め込みがあれば、ノード間類似度を計算（平均ペア類似度）
                    pair_sims = []
                    for i in range(len(path) - 1):
                        e1 = entity_embeddings.get(path[i])
                        e2 = entity_embeddings.get(path[i + 1])
                        if e1 is not None and e2 is not None:
                        # safe numpy dot / norms
                            denom = (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-9)
                            pair_sims.append(float(np.dot(e1, e2) / denom))
                    avg_pair_sim = float(np.mean(pair_sims)) if pair_sims else None

                # 最短距離（kg 上） — 存在しなければ None
                    shortest = None
                    try:
                        if path_len >= 2:
                        # path の端同士の最短長を計算（例）
                            s1, s2 = path[0], path[-1]
                            if kg.has_node(s1) and kg.has_node(s2):
                                shortest = int(nx.shortest_path_length(kg, s1, s2))
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        pass
                    except Exception:
                        pass

                    paths_meta.append({
                        'path': path,
                        'path_length_nodes': path_len,
                        'avg_edge_weight': avg_edge_weight,
                        'sum_edge_weight': sum_edge_weight,
                        'avg_adjacent_node_sim': avg_pair_sim, 
                        'kg_shortest_between_ends': shortest
                    })

        # attach (既存 metadata を壊さないようにコピー)
                new_meta = dict(meta)
                new_meta['paths'] = paths_meta
            # create a new Document preserving original text & adding metadata (or mutate in place if ok)
                new_doc = Document(text=getattr(doc, 'text', ''), metadata=new_meta)
                augmented.append(new_doc)
            else:
            # マッチしなかった場合も元のドキュメントを保持
                augmented.append(doc)
    
            self.logger.info(f"  → Matched {matched_count}/{len(documents)} documents with path information")
                  
            return augmented

        # naive match: by ordering if no explicit key available
        if len(path_dicts) == 0:
            return documents

        if len(path_dicts) == len(documents):
            for i, doc in enumerate(documents):
                doc.metadata['paths'] = path_dicts[i].get('translated_paths', [])
                doc.metadata['path_distances'] = path_dicts[i].get('path_distances', [])
        else:
            # fallback: attach top global paths to every doc (still useful)
            sample_paths = path_dicts[0].get('translated_paths', [])
            for doc in documents:
                doc.metadata.setdefault('paths', sample_paths)
                doc.metadata.setdefault('path_distances', path_dicts[0].get('path_distances', []))
        return documents
    
    def _generate_chunk_id(self, text: str, source_id: str, index: int) -> str:
        """
        チャンクの一意なIDを生成
        
        Args:
            text: チャンクのテキスト
            source_id: 元ドキュメントのID
            index: チャンク番号
        
        Returns:
            'doc123_chunk5_a7f3e9b2' のような一意ID
        """
        # テキストのハッシュ（最初の100文字から）
        text_hash = hashlib.md5(text[:100].encode()).hexdigest()[:8]
        return f"{source_id}_chunk{index}_{text_hash}"
    
    # ============================================================
    # Dual-documents 生成
    # ============================================================
    def create_dual_documents(
        self,
        documents: List[Document]) -> Tuple[List[Document], List[Document]]:
        """
        既存のDocumentから Graph用 と Retrieval用 の2種類を作る
        
        Args:
            documents: load_documents() で作成したDocumentリスト
        
        Returns:
            (graph_docs, retrieval_docs)
        """
        if not self.use_dual_chunk:
            # デュアルチャンク無効時は元のドキュメントをそのまま返す
            return documents, documents
        
        graph_splitter, retrieval_splitter = self.get_dual_splitters()
        graph_docs = []
        retrieval_docs = []
        
        for doc in documents:
            base_meta = dict(doc.metadata)
            
            # ------------------------------------------------------------
            # Graph用チャンク（小さめ）
            # ------------------------------------------------------------
            try:
                graph_nodes = graph_splitter.get_nodes_from_documents([doc])
                for j, node in enumerate(graph_nodes):
                    md = dict(base_meta)
                    md.update({
                        'chunk_type': 'structural',
                        'chunk_index': j,
                        'chunk_size': len(node.text)
                    })
                    graph_docs.append(Document(
                        text=node.text,
                        metadata=md
                    ))
            except Exception as e:
                self.logger.warning(f"Graph splitting failed: {e}")
                # フォールバック：元のドキュメントを使う
                md = dict(base_meta)
                md['chunk_type'] = 'structural'
                graph_docs.append(Document(text=doc.text, metadata=md))
            
            # ------------------------------------------------------------
            # Retrieval用チャンク（大きめ）
            # ------------------------------------------------------------
            try:
                retrieval_nodes = retrieval_splitter.get_nodes_from_documents([doc])
                for j, node in enumerate(retrieval_nodes):
                    md = dict(base_meta)
                    md.update({
                        'chunk_type': 'semantic',
                        'chunk_index': j,
                        'chunk_size': len(node.text)
                    })
                    retrieval_docs.append(Document(
                        text=node.text,
                        metadata=md
                    ))
            except Exception as e:
                self.logger.warning(f"Retrieval splitting failed: {e}")
                md = dict(base_meta)
                md['chunk_type'] = 'semantic'
                retrieval_docs.append(Document(text=doc.text, metadata=md))
        
        self.logger.info(
            f"📄 Created {len(graph_docs)} graph chunks, "
            f"{len(retrieval_docs)} retrieval chunks"
        )
        
        return graph_docs, retrieval_docs

    def _find_overlapping_chunks(
        self,
        start: int,
        end: int,
        graph_docs: List[Document]
    ) -> List[str]:
        """
        指定範囲と重なるGraphチャンクのIDを返す
        
        Args:
            start, end: 文字位置
            graph_docs: 同一ドキュメントのGraphチャンクリスト
        
        Returns:
            重なるチャンクのIDリスト
        """
        overlapping = []
        
        for doc in graph_docs:
            g_start = doc.metadata.get('start_char', 0)
            g_end = doc.metadata.get('end_char', 0)
            
            # 範囲の重なりチェック
            if not (end <= g_start or start >= g_end):
                overlapping.append(doc.metadata['chunk_id'])
        
        return overlapping
    

    # ============================================================
    # 修正1: get_dual_splitters（チャンクサイズ調整）
    # ============================================================
    
    def get_dual_splitters(self) -> Tuple[SentenceSplitter, SentenceSplitter]:
        """
        Graph用とRetrieval用の2系統を返す（チューニング版）
        """
        # Graph用：小さめチャンク
        graph_splitter = SentenceSplitter(
            chunk_size=self.config['graph_chunk_size'],
            chunk_overlap=self.config['graph_chunk_overlap'],
            paragraph_separator="\n\n",
            secondary_chunking_regex=r"[.!?。！?]\s+"
        )
        
        # Retrieval用：中サイズチャンク（512に変更）
        retrieval_splitter = SentenceSplitter(
            chunk_size=self.config['retrieval_chunk_size'],  # 512
            chunk_overlap=self.config['retrieval_chunk_overlap'],  # 100
            paragraph_separator="\n\n",
            secondary_chunking_regex=r"[.!?。！?]\s+"
        )
        
        self.logger.info(
            f"Splitters: graph={self.config['graph_chunk_size']}, "
            f"retrieval={self.config['retrieval_chunk_size']}"
        )
        
        return graph_splitter, retrieval_splitter

    # ============================================================
    # Retrieval Store 構築
    # ============================================================
    def build_retrieval_store(
        self,
        retrieval_docs: List[Document]
    ) -> Dict[str, Any]:
        """
        Retrieval用の埋め込みストアを構築
        
        Returns:
            {
                'docs': [Document, ...],
                'embeddings': np.array,
                'metadata': {...}
            }
        """
        self.logger.info("🔍 Building retrieval embeddings...")
        
        docs = []
        embeddings = []
        
        collector = ErrorCollector(self.logger)
        
        for doc in retrieval_docs:
            try:
                emb = self.embed_model.get_text_embedding(doc.text)
                emb = np.array(emb, dtype=np.float32)
                
                # 正規化
                norm = np.linalg.norm(emb)
                if norm > 1e-9:
                    emb = emb / norm
                else:
                    self.logger.debug("Zero-norm embedding, skipping")
                    continue
                
                docs.append(doc)
                embeddings.append(emb)
                collector.add_success()
            
            except Exception as e:
                collector.add_error(
                    context=f"doc_{doc.metadata.get('source_id', 'unknown')}",
                    error=e
                )
        
        collector.report("Retrieval embedding generation", threshold=0.3)
        
        embeddings = np.vstack(embeddings) if embeddings else np.zeros((0, 1024))
        
        self.logger.info(f"✅ Built retrieval store: {len(docs)} docs")
        
        return {
            'docs': docs,
            'embeddings': embeddings,
            'metadata': {
                'total_docs': len(docs),
                'embedding_dim': embeddings.shape[1] if len(embeddings) > 0 else 0
            }
        }
    
    def retrieve(
        self,
        store: Dict,
        query: str,
        top_k: int = 5,
        chunk_mapping: Dict = None
    ) -> List[Tuple[float, Document, List[str]]]:
        """
        クエリに対してコサイン類似度で検索
        
        Returns:
            [(score, Document, graph_chunk_ids), ...] のリスト
        """
        if len(store['docs']) == 0:
            self.logger.warning("⚠️  Retrieval store is empty")
            return []
        
        # クエリの埋め込み
        qemb = np.array(self.embed_model.get_text_embedding(query))
        qnorm = np.linalg.norm(qemb)
        if qnorm > 1e-9:
            qemb = qemb / qnorm
        
        # コサイン類似度計算
        sims = store['embeddings'] @ qemb
        top_indices = np.argsort(-sims)[:top_k]
        
        results = []
        for i in top_indices:
            if i >= len(store['docs']):
                continue
            
            doc = store['docs'][i]
            score = float(sims[i])
            
            # Graph chunk IDsを取得
            graph_chunk_ids = doc.metadata.get('graph_chunk_ids', [])
            
            # または、chunk_mappingから逆引き
            if not graph_chunk_ids and chunk_mapping:
                chunk_id = doc.metadata.get('chunk_id')
                graph_chunk_ids = chunk_mapping.get('retrieval_to_graph', {}).get(chunk_id, [])
            
            # 3. デュアルチャンク無効時のフォールバック
            if not graph_chunk_ids:
            # 同一ドキュメントIDを持つチャンクを推定
                source_id = doc.metadata.get('source_id') or doc.metadata.get('json_id') or doc.metadata.get('raw_id')
                if source_id is not None:
                    # 簡易的にドキュメント全体を指すIDを生成
                    graph_chunk_ids = [f"doc_{source_id}_all"]
                
                # 警告を1回だけ出す（初回のみ）
                    if not hasattr(self, '_warned_no_mapping'):
                        self.logger.warning(
                           "⚠️  chunk_mapping not available, using fallback document IDs. "
                            "Enable dual-chunk mode for better precision."
                        )
                        self._warned_no_mapping = True
        
                results.append((score, doc, graph_chunk_ids))
    
            return results
        
    def explore_multi_hop_paths(
        self,
        kg: nx.Graph,
        query: str,
        retrieval_chunks: List[str] = None,
        max_steps: int = 5,
        top_k_per_hop: int = 3,
        confidence_threshold: float = 0.7,
        extend_on_low_confidence: bool = True
    ) -> Dict[str, Any]:
        """
        Multi-hop探索を実行
    
        Args:
            kg: NetworkXグラフ
            query: 検索クエリ
            retrieval_chunks: 開始点となるチャンクID（オプション）
            max_steps: 最大ホップ数
            top_k_per_hop: 各ホップで探索する上位K個
            confidence_threshold: 信頼度の閾値
            extend_on_low_confidence: 信頼度が低い場合に探索を拡張するか
    
        Returns:
            {
                'paths': ランク付けされたパスのリスト,
                'entities': 訪問したエンティティ,
                'confidence': 信頼度スコア,
                'steps_used': 実際に使用したステップ数,
                'evidence': エビデンステキスト
            }
        """
        self.logger.info(f"🔍 Starting multi-hop exploration (max_steps={max_steps})")
    
    # ============================================================
    # 1. 開始エンティティの決定
    # ============================================================
        start_entities = set()
    
        if retrieval_chunks:
        # Retrievalで取得したチャンクから開始
            start_entities = self._resolve_entities_from_chunks(retrieval_chunks, kg)
    
        if not start_entities:
            # フォールバック: クエリに最も関連するエンティティを抽出
            start_entities = self._extract_query_entities(query, kg, top_k=5)
    
        if not start_entities:
            self.logger.warning("⚠️  No starting entities found")
            return {
                'paths': [],
                'entities': [],
                'confidence': 0.0,
                'steps_used': 0,
                'evidence': []
            }
    
        self.logger.info(f"  → Starting from {len(start_entities)} entities: {list(start_entities)[:3]}...")
    
    # ============================================================
    # 2. 各開始エンティティから探索
    # ============================================================
        all_paths = []
        visited_entities = set()
        evidence_texts = []
    
        for start_entity in list(start_entities)[:top_k_per_hop]:
            if start_entity not in kg.nodes():
                self.logger.debug(f"  Entity '{start_entity}' not in graph, skipping")
                continue
        
            path_result = self._explore_from_entity(
                kg,
                start_entity,
                query,
                max_steps=max_steps,
                visited=set()
            )
        
            all_paths.extend(path_result['paths'])
            visited_entities.update(path_result['visited'])
        
            # エビデンステキストを収集
            for path_info in path_result['paths'][:5]:  # Top 5のみ
                path = path_info['path']
                evidence_texts.append(' → '.join(path))
    
        # 全体の信頼度を計算
        if all_paths:
            confidence = np.mean([p['score'] for p in all_paths])
        else:
            confidence = 0.0
    
        current_step = max_steps
    
        self.logger.info(
            f"  → Found {len(all_paths)} paths with confidence {confidence:.2f}"
        )    

        # ============================================================
        # 3. 信頼度が低い場合は拡張
        # ============================================================
        if extend_on_low_confidence and confidence < confidence_threshold:
            extended_steps = max_steps + 2
            self.logger.info(
                f"  → Low confidence ({confidence:.2f} < {confidence_threshold}), "
                f"extending to {extended_steps} steps"
            )
            
            # 再探索
            extended_paths = []
            for start_entity in list(start_entities)[:top_k_per_hop]:
                if start_entity not in kg.nodes():
                    continue
                
                path_result = self._explore_from_entity(
                    kg,
                    start_entity,
                    query,
                    max_steps=extended_steps,
                    visited=set()  # リセット
                )
                
                extended_paths.extend(path_result['paths'])
                confidence = max(confidence, path_result['confidence'])
            
            if len(extended_paths) > len(paths):
                paths = extended_paths
                current_step = extended_steps
                self.logger.info(f"  → Extended search found {len(paths)} paths")
        
        # ============================================================
        # 4. パスのスコアリングとランキング
        # ============================================================
        ranked_paths = self._rank_paths(paths, query, kg)
        
        return {
            'paths': ranked_paths[:10],  # Top 10
            'entities': list(visited_entities),
            'confidence': confidence,
            'steps_used': current_step,
            'evidence': evidence_texts
        }
    
    def _precompute_representative_paths(
        self,
        kg: nx.Graph,
        documents: List[Document],
        num_sample_queries: int = 10
    ) -> None:
        """
        代表的なクエリでパスを事前計算し、グラフに保存
    
        Args:
            kg: NetworkXグラフ
            documents: ドキュメントリスト
            num_sample_queries: サンプルクエリ数
        """
        self.logger.info(f"Computing representative paths for {num_sample_queries} sample queries...")
    
    # ============================================================
    # 1. サンプルクエリの生成
    # ============================================================
        sample_queries = self._generate_sample_queries(documents, kg, num_sample_queries)
    
        if not sample_queries:
            self.logger.warning("  → No sample queries generated, skipping path pre-computation")
            return
    
        self.logger.info(f"  Generated {len(sample_queries)} sample queries")
    
    # ============================================================
    # 2. 各クエリでMulti-hop探索を実行
    # ============================================================
        all_paths = []
        path_count = 0
    
        for i, query in enumerate(sample_queries):
            try:
                result = self.explore_multi_hop_paths(
                    kg=kg,
                    query=query,
                    max_steps=5,
                    top_k_per_hop=3,
                    extend_on_low_confidence=False  # 事前計算では拡張しない
                )
            
            # 高品質なパスのみ保存（confidence > 0.5）
                for path_info in result['paths']:
                    if path_info.get('final_score', 0) > 0.5:
                        all_paths.append(path_info)
                        path_count += 1
            
                if (i + 1) % 5 == 0:
                    self.logger.info(f"  Processed {i+1}/{len(sample_queries)} queries...")
        
            except Exception as e:
                self.logger.debug(f"  Query '{query[:30]}...' failed: {type(e).__name__}")
                continue
    
        self.logger.info(f"  → Computed {path_count} high-quality paths")
    
    # ============================================================
    # 3. パス情報をグラフのノード/エッジに保存
    # ============================================================
        self._store_paths_in_graph(kg, all_paths)

    def _generate_sample_queries(
        self,
        documents: List[Document],
        kg: nx.Graph,
        num_queries: int = 10
    ) -> List[str]:
        """
        ドキュメントから代表的なクエリを生成
    
        Args:
            documents: ドキュメントリスト
            kg: NetworkXグラフ
            num_queries: 生成するクエリ数
    
        Returns:
            サンプルクエリのリスト
        """
        queries = []
    
    # ============================================================
    # 戦略1: 中心性の高いノードをクエリにする
    # ============================================================
        try:
        # 次数中心性を計算
            degree_centrality = nx.degree_centrality(kg)
        
        # 上位ノードを取得
            top_nodes = sorted(
                degree_centrality.items(),
                key=lambda x: x[1],
                reverse=True
            )[:num_queries // 2]
        
        # ノード名をクエリとして使用
            for node, _ in top_nodes:
                queries.append(f"What is {node}?")
                queries.append(f"How does {node} work?")
    
        except Exception as e:
            self.logger.debug(f"Centrality-based query generation failed: {e}")
    
    # ============================================================
    # 戦略2: ドキュメントのメタデータからクエリを生成
    # ============================================================
        for doc in documents[:num_queries // 2]:
        # メタデータに'question'があればそれを使用
            question = doc.metadata.get('question')
            if question:
                queries.append(question)
            else:
            # テキストの最初の文を使用
                text = doc.text.strip()
                if text:
                    first_sentence = text.split('.')[0][:100]
                    if len(first_sentence) > 10:
                        queries.append(first_sentence)
    
    # ============================================================
    # 戦略3: エンティティペアの関係を問うクエリ
    # ============================================================
        try:
        # 重みの高いエッジを取得
            high_weight_edges = sorted(
                kg.edges(data=True),
                key=lambda x: x[2].get('weight', 0),
                reverse=True
            )[:num_queries // 3]
        
            for u, v, data in high_weight_edges:
                relation = data.get('relation', 'related to')
                queries.append(f"How is {u} {relation} {v}?")
    
        except Exception as e:
            self.logger.debug(f"Edge-based query generation failed: {e}")
    
    # 重複を除去してシャッフル
        queries = list(set(queries))
        import random
        random.shuffle(queries)
    
        return queries[:num_queries]
    
    def _store_paths_in_graph(
        self,
        kg: nx.Graph,
        paths: List[Dict]
    ) -> None:
        """
        計算されたパスをグラフのノード/エッジ属性に保存
    
        Args:
            kg: NetworkXグラフ
            paths: パス情報のリスト
        """
        self.logger.info("  Storing path information in graph...")
    
    # ============================================================
    # 1. 各ノードが含まれるパス数をカウント
    # ============================================================
        node_path_counts = defaultdict(int)
        node_avg_scores = defaultdict(list)
    
        for path_info in paths:
            path = path_info.get('path', [])
            score = path_info.get('final_score', 0)
        
            for node in path:
                if kg.has_node(node):
                    node_path_counts[node] += 1
                    node_avg_scores[node].append(score)
    
    # ノードに属性を追加
        for node in kg.nodes():
            kg.nodes[node]['path_frequency'] = node_path_counts.get(node, 0)
        
            scores = node_avg_scores.get(node, [])
            if scores:
                kg.nodes[node]['avg_path_score'] = float(np.mean(scores))
            else:
                kg.nodes[node]['avg_path_score'] = 0.0
    
    # ============================================================
    # 2. 各エッジが含まれるパス数をカウント
    # ============================================================
        edge_path_counts = defaultdict(int)
        edge_avg_scores = defaultdict(list)
    
        for path_info in paths:
            path = path_info.get('path', [])
            score = path_info.get('final_score', 0)
        
        # パス内の連続するノードペアをエッジとして記録
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
            
            # 無向グラフとして扱う
                edge_key = tuple(sorted([u, v]))
                edge_path_counts[edge_key] += 1
                edge_avg_scores[edge_key].append(score)
    
    # エッジに属性を追加
        for u, v in kg.edges():
            edge_key = tuple(sorted([u, v]))
        
            kg[u][v]['path_frequency'] = edge_path_counts.get(edge_key, 0)
        
            scores = edge_avg_scores.get(edge_key, [])
            if scores:
                kg[u][v]['avg_path_score'] = float(np.mean(scores))
            else:
                kg[u][v]['avg_path_score'] = 0.0
    
    # 統計情報をログ出力
        total_nodes_with_paths = sum(1 for n in kg.nodes() if kg.nodes[n]['path_frequency'] > 0)
        total_edges_with_paths = sum(1 for u, v in kg.edges() if kg[u][v]['path_frequency'] > 0)
    
        self.logger.info(
            f"  → {total_nodes_with_paths}/{len(kg.nodes())} nodes and "
            f"{total_edges_with_paths}/{len(kg.edges())} edges have path information"
        )

    def _extract_query_entities(
        self,
        query: str,
        kg: nx.Graph,
        top_k: int = 5
    ) -> Set[str]:
        """
        クエリから関連エンティティを抽出
    
        Args:
            query: 検索クエリ
            kg: NetworkXグラフ
            top_k: 上位K個のエンティティを返す
    
        Returns:
            エンティティ名のセット
        """
    # クエリの埋め込み
        query_emb = self.get_cached_embedding(query, cache_type='entity')
    
    # 全エンティティとの類似度計算
        entity_scores = []
    
        for entity in kg.nodes():
            try:
                entity_emb = self.get_cached_embedding(entity, cache_type='entity')
            
                similarity = float(np.dot(query_emb, entity_emb))
                entity_scores.append((entity, similarity))
        
            except Exception:
                continue
    
        # スコア順にソート
        entity_scores.sort(key=lambda x: x[1], reverse=True)
    
        # Top K を返す
        top_entities = {entity for entity, _ in entity_scores[:top_k]}
    
        return top_entities

    def _resolve_entities_from_chunks(
        self,
        chunk_ids: Set[str],
        kg: nx.Graph
    ) -> Set[str]:
        """
        チャンクIDから実際のエンティティ名に変換
        
        Args:
            chunk_ids: チャンクIDのセット
            kg: NetworkXグラフ
        
        Returns:
            エンティティ名のセット
        """
        entities = set()
        
        for chunk_id in chunk_ids:
            # chunk_idがすでにエンティティ名の場合
            if chunk_id in kg.nodes():
                entities.add(chunk_id)
                continue

        # ============================================================
        # 2. チャンクIDからエンティティを推定
        # ============================================================
        
        # パターン1: "doc_X_chunkY_hash" 形式
        # → グラフのノード属性 'chunk_id' を持つノードを検索
            for node, data in kg.nodes(data=True):
                node_chunk_ids = data.get('chunk_ids', [])
            
                # chunk_ids が文字列の場合もあるので対応
                if isinstance(node_chunk_ids, str):
                    node_chunk_ids = [node_chunk_ids]
            
                if chunk_id in node_chunk_ids:
                    entities.add(node)
        
        # パターン2: チャンクID内にエンティティ名が含まれる
        # （例: chunk_id = "attention_mechanism_chunk3"）
        # → グラフ内のノード名がchunk_idに部分一致するか確認
            chunk_id_lower = chunk_id.lower()
            for node in kg.nodes():
                node_lower = node.lower()
            
            # 部分一致（少なくとも5文字以上）
                if len(node_lower) >= 5 and node_lower in chunk_id_lower:
                    entities.add(node)
                elif len(chunk_id_lower) >= 5 and chunk_id_lower in node_lower:
                    entities.add(node)
    
        if not entities:
            self.logger.debug(
                f"  Could not resolve entities from {len(chunk_ids)} chunk IDs"
            )
    
        return entities
    
    def _explore_from_entity(
        self,
        kg: nx.Graph,
        start_entity: str,
        query: str,
        max_steps: int,
        visited: Set[str]
    ) -> Dict[str, Any]:
        """
        特定エンティティから深さ優先探索
        
        Returns:
            {
                'paths': [パスのリスト],
                'visited': 訪問ノード,
                'steps': 最大ステップ数,
                'confidence': 信頼度
            }
        """
        paths = []
        visited.add(start_entity)
        
        # クエリの埋め込み
        query_emb = self.get_cached_embedding(query, cache_type='entity')
        
        # BFS
        queue = [(start_entity, [start_entity], 0)]  
        # パス数制限
        max_paths = self.config.get('multihop_max_paths', 50)
   
        while queue and len(paths) < max_paths: 
            current, path, depth = queue.pop(0)
            
            if depth >= max_steps:
                continue
            
            # 隣接ノードを探索
            neighbors = list(kg.neighbors(current))
            
            # 各隣接ノードのスコアを計算
            neighbor_scores = []
            for neighbor in neighbors:
                if neighbor in visited:
                    continue
                
                # エンティティ名の埋め込み
                entity_emb = self.get_cached_embedding(neighbor, cache_type='entity')
                    
                    # クエリとの類似度
                similarity = float(np.dot(query_emb, entity_emb))
                    
                    # エッジの重み
                edge_weight = kg[current][neighbor].get('weight', 0.5)
                    
                    # 総合スコア
                score = similarity * 0.6 + edge_weight * 0.4
                    
                neighbor_scores.append((neighbor, score))
                
            
            # スコア上位を選択
            neighbor_scores.sort(key=lambda x: x[1], reverse=True)
            beam_width = self.config.get('multihop_beam_width', 2)
            top_neighbors = neighbor_scores[:beam_width]  # 3 → 2
            
            for neighbor, score in top_neighbors:
                new_path = path + [neighbor]
                
            #パス重複チェック 
                path_tuple = tuple(new_path)
                if path_tuple in self.visited_paths:
                    continue
                self.visited_paths.add(path_tuple)
            
                visited.add(neighbor)

                # パスを保存
                paths.append({
                    'path': new_path,
                    'score': score,
                    'depth': depth + 1
                })
                
                # キューに追加
                queue.append((neighbor, new_path, depth + 1))
        
        # 信頼度計算（パスの平均スコア）
        confidence = np.mean([p['score'] for p in paths]) if paths else 0.0
        
        return {
            'paths': paths,
            'visited': visited,
            'steps': max_steps,
            'confidence': float(confidence)
        }
    
    def _rank_paths(
        self,
        paths: List[Dict],
        query: str,
        kg: nx.Graph
    ) -> List[Dict]:
        """
        パスをスコアでランキング
        """
        if not paths:
            return []
        
        query_emb = self.get_cached_embedding(query, cache_type='entity')

        # 各パスに最終スコアを計算
        for path_info in paths:
            path = path_info['path']
            
            # パスの長さペナルティ（長すぎると信頼度低下）
            length_penalty = 1.0 / (1.0 + 0.1 * len(path))
            
            # エッジ重みの平均
            edge_weights = []
            for i in range(len(path) - 1):
                if kg.has_edge(path[i], path[i+1]):
                    edge_weights.append(kg[path[i]][path[i+1]].get('weight', 0.5))
            
            avg_edge_weight = np.mean(edge_weights) if edge_weights else 0.5
            #  パス全体とクエリの関連性スコア
            path_query_relevance = 0.0     
            entity_similarities = []

            for entity in path:    
                entity_emb = self.get_cached_embedding(entity, cache_type='entity')    
                similarity = float(np.dot(query_emb, entity_emb))
                entity_similarities.append(similarity)
        
            if entity_similarities:
                # パス内の最大類似度を使用（最も関連するエンティティを重視）
                path_query_relevance = max(entity_similarities)
   
            # 最終スコア
            final_score = (
                path_info['score'] * 0.4 +
                avg_edge_weight * 0.25 +
                length_penalty * 0.15 +
                path_query_relevance * 0.2
            )
            
            path_info['final_score'] = final_score
            path_info['query_relevance'] = path_query_relevance  # デバッグ用に保存
        
        # スコア順にソート
        paths.sort(key=lambda x: x.get('final_score', 0), reverse=True)
        
        return paths

    # ============================================================
    # 統合ビルド関数
    # ============================================================
    
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
            graph_docs, retrieval_docs, chunk_mapping = self.create_dual_documents(documents)
            
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
        kg: nx.Graph,
        retrieval_store: Dict = None,
        max_steps: int = 5,
        top_k_retrieval: int = 5,
        top_k_paths: int = 10
    ) -> Dict[str, Any]:
        """
        Multi-hop探索を使ったクエリ実行
    
        Args:
            query: 検索クエリ
            kg: NetworkXグラフ
            retrieval_store: Retrievalストア（オプション）
            max_steps: 最大ホップ数
            top_k_retrieval: Retrieval結果の上位K件
            top_k_paths: 返すパスの上位K件
    
        Returns:
            {
                'paths': 発見されたパス,
                'retrieval_docs': Retrievalで取得したドキュメント,
                'confidence': 信頼度,
                'answer': 統合された回答（オプション）
            }
        """
        self.logger.info(f"🔍 Query: '{query}'")
    
        results = {
            'paths': [],
            'retrieval_docs': [],
            'confidence': 0.0,
            'answer': None
        }
    
    # ============================================================
    # 1. Retrieval（提供されている場合）
    # ============================================================
        retrieval_chunks = []
    
        if retrieval_store:
            try:
                retrieval_results = self.retrieve(
                    store=retrieval_store,
                    query=query,
                    top_k=top_k_retrieval
                )
            
                for score, doc, graph_chunk_ids in retrieval_results:
                    results['retrieval_docs'].append({
                        'text': doc.text,
                        'score': score,
                        'metadata': doc.metadata
                    })
                    retrieval_chunks.extend(graph_chunk_ids)
            
                self.logger.info(
                    f"  → Retrieval: {len(results['retrieval_docs'])} docs, "
                    f"{len(retrieval_chunks)} graph chunks"
                )
        
            except Exception as e:
                self.logger.warning(f"⚠️  Retrieval failed: {type(e).__name__}")
    
    # ============================================================
    # 2. Multi-hop探索
    # ============================================================
        try:
            path_result = self.explore_multi_hop_paths(
                kg=kg,
                query=query,
                retrieval_chunks=retrieval_chunks if retrieval_chunks else None,
                max_steps=max_steps,
                top_k_per_hop=3,
                confidence_threshold=0.7,
                extend_on_low_confidence=True
            )
        
            results['paths'] = path_result['paths'][:top_k_paths]
            results['confidence'] = path_result['confidence']
        
            self.logger.info(
                f"  → Multi-hop: {len(results['paths'])} paths, "
                f"confidence={results['confidence']:.2f}"
            )
    
        except Exception as e:
            self.logger.error(f"🚨 Multi-hop exploration failed: {type(e).__name__}")
            self.logger.error(f"   {str(e)[:200]}")
        
            if self.logger.level <= logging.DEBUG:
                self.logger.debug(traceback.format_exc())
    
    # ============================================================
    # 3. 結果の統合（オプション）
    # ============================================================
        if results['paths'] and results['retrieval_docs']:
            results['answer'] = self._synthesize_answer(
                query=query,
                paths=results['paths'],
                retrieval_docs=results['retrieval_docs']
            )
    
        return results

    def _synthesize_answer(
        self,
        query: str,
        paths: List[Dict],
        retrieval_docs: List[Dict]
    ) -> str:
        """
        パスとRetrievalドキュメントから回答を統合
    
        Args:
            query: クエリ
            paths: Multi-hopで発見されたパス
            retrieval_docs: Retrievalで取得したドキュメント
    
        Returns:
            統合された回答文字列
        """
    # 簡易実装（LLMを使った統合は別途実装可能）
    
        answer_parts = []
    
    # パスからのエビデンス
        answer_parts.append("**From Knowledge Graph:**")
        for i, path_info in enumerate(paths[:3], 1):
            path = path_info['path']
            score = path_info.get('final_score', 0)
            path_str = ' → '.join(path)
            answer_parts.append(f"{i}. {path_str} (score: {score:.2f})")
    
    # Retrievalドキュメントからのエビデンス
        answer_parts.append("\n**From Documents:**")
        for i, doc_info in enumerate(retrieval_docs[:3], 1):
            text_preview = doc_info['text'][:150] + "..."
            score = doc_info['score']
            answer_parts.append(f"{i}. {text_preview} (score: {score:.2f})")
    
        return '\n'.join(answer_parts)

    def link_entities(
        self,
        kg: nx.Graph,
        similarity_threshold: float = 0.88,
        use_embedding: bool = True
    ) -> Tuple[nx.Graph, Dict[str, str]]:
        """
        同一実体を統合してグラフをクリーンアップ
        
        Args:
            kg: NetworkXグラフ
            similarity_threshold: 統合する類似度の閾値（0.95推奨）
            use_embedding: True=埋め込み類似度、False=文字列類似度
        
        Returns:
            (統合後のグラフ, エンティティマッピング)
            
        例:
            mapping = {
                'Self-Attention': 'self_attention',
                'the attention mechanism': 'self_attention',
                'it': 'self_attention'  # coref解決が必要
            }
        """
        self.logger.info(f"🔗 Starting entity linking (threshold={similarity_threshold})")
        
        nodes = list(kg.nodes())
        entity_mapping = {}  # old_name -> canonical_name
        clusters = []  # [[類似エンティティのリスト], ...]
        
        # ============================================================
        # 1. エンティティのクラスタリング
        # ============================================================
        if use_embedding:
            clusters = self._cluster_entities_by_embedding(
                nodes, similarity_threshold
            )
        else:
            clusters = self._cluster_entities_by_string(nodes)
        
        # ============================================================
        # 2. 各クラスタの代表名を決定
        # ============================================================
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            
            # 代表名の選択戦略
            canonical = self._select_canonical_name(cluster, kg)
            
            for entity in cluster:
                if entity != canonical:
                    entity_mapping[entity] = canonical
        
        self.logger.info(f"  → {len(entity_mapping)} entities will be merged")
        
        # ============================================================
        # 3. グラフの統合
        # ============================================================
        merged_kg = self._merge_graph_entities(kg, entity_mapping)
        
        self.logger.info(
            f"✅ Entity linking complete: "
            f"{len(kg.nodes)} → {len(merged_kg.nodes)} nodes"
        )
        
        return merged_kg, entity_mapping
    
    def _cluster_entities_by_embedding(
        self,
        entities: List[str],
        threshold: float
    ) -> List[List[str]]:
        """
        埋め込みベースのクラスタリング
        
        Returns:
            [[類似エンティティ], [類似エンティティ], ...]
        """
        # エンティティの埋め込み計算
        embeddings = []
        valid_entities = []
        
        for entity in entities:
            try:
                emb = self.embed_model.get_text_embedding(entity)
                emb = np.array(emb, dtype=np.float32)
                norm = np.linalg.norm(emb)
                
                if norm > 1e-9:
                    emb = emb / norm
                    embeddings.append(emb)
                    valid_entities.append(entity)
            except Exception as e:
                self.logger.debug(f"Embedding failed for '{entity}': {e}")
        
        if len(embeddings) == 0:
            return []
        
        embeddings = np.vstack(embeddings)
        
        # 類似度マトリクス計算
        sim_matrix = embeddings @ embeddings.T
        
        # Union-Find でクラスタリング
        parent = {i: i for i in range(len(valid_entities))}
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # 類似度が閾値以上のペアを統合
        for i in range(len(valid_entities)):
            for j in range(i + 1, len(valid_entities)):
                if sim_matrix[i, j] >= threshold:
                    union(i, j)
        
        # クラスタを構築
        clusters_dict = defaultdict(list)
        for i, entity in enumerate(valid_entities):
            root = find(i)
            clusters_dict[root].append(entity)
        
        clusters = list(clusters_dict.values())
        
        self.logger.info(
            f"  → Found {len(clusters)} clusters from {len(valid_entities)} entities"
        )
        
        return clusters
    
    def _cluster_entities_by_string(
        self,
        entities: List[str]
    ) -> List[List[str]]:
        """
        文字列類似度ベースのクラスタリング（高速だが精度低い）
        
        使用ケース：
        - "Self-Attention" と "self-attention" を統合
        - "GPT-3" と "GPT3" を統合
        """
        
        clusters_dict = defaultdict(list)
        normalized = {}
        
        for entity in entities:
            # 正規化（小文字化、記号除去）
            norm = entity.lower().replace('-', '').replace('_', '').replace(' ', '')
            normalized[entity] = norm
            clusters_dict[norm].append(entity)
        
        # 2つ以上のエンティティがある正規化形のみ返す
        clusters = [v for v in clusters_dict.values() if len(v) > 1]
        
        return clusters
    
    def _select_canonical_name(
        self,
        cluster: List[str],
        kg: nx.Graph
    ) -> str:
        """
        クラスタの代表名を選択
        
        戦略：
        1. 最も次数が高い（多くの関係を持つ）
        2. 最も長い名前（情報量が多い）
        3. アルファベット順
        """
        # 次数でスコアリング
        scores = {}
        for entity in cluster:
            degree = kg.degree(entity) if kg.has_node(entity) else 0
            length = len(entity)
            
            # スコア = 次数 * 10 + 長さ
            scores[entity] = degree * 10 + length
        
        # スコアが最大のものを選択
        canonical = max(cluster, key=lambda e: scores[e])
        
        self.logger.debug(
            f"  Cluster: {cluster} → Canonical: '{canonical}'"
        )
        
        return canonical
    
    def _merge_graph_entities(
        self,
        kg: nx.Graph,
        entity_mapping: Dict[str, str]
    ) -> nx.Graph:
        """
        エンティティマッピングに従ってグラフを統合
        
        Args:
            kg: 元のグラフ
            entity_mapping: {old_name: canonical_name}
        
        Returns:
            統合後のグラフ
        """
        merged_kg = nx.Graph()
        
        # ノードをコピー（マッピング適用）
        for node, data in kg.nodes(data=True):
            canonical = entity_mapping.get(node, node)
            
            if merged_kg.has_node(canonical):
                # 既存ノードの属性をマージ
                for key, value in data.items():
                    if key not in merged_kg.nodes[canonical]:
                        merged_kg.nodes[canonical][key] = value
            else:
                merged_kg.add_node(canonical, **data)
        
        # エッジをコピー（マッピング適用 + 重み統合）
        edge_weights = defaultdict(lambda: {
            'weight': 0.0,
            'intra_raw': 0.0,
            'inter_raw': 0.0,
            'relations': []
        })
        
        for u, v, data in kg.edges(data=True):
            u_canonical = entity_mapping.get(u, u)
            v_canonical = entity_mapping.get(v, v)
            
            # 自己ループは除外
            if u_canonical == v_canonical:
                continue
            
            # 正規化されたエッジキー（方向なし）
            edge_key = tuple(sorted([u_canonical, v_canonical]))
            
            # 重みを累積
            edge_weights[edge_key]['weight'] += data.get('weight', 0.0)
            edge_weights[edge_key]['intra_raw'] += data.get('intra_raw', 0.0)
            edge_weights[edge_key]['inter_raw'] += data.get('inter_raw', 0.0)
            
            # 関係タイプを記録
            rel = data.get('relation', 'RELATED')
            if rel not in edge_weights[edge_key]['relations']:
                edge_weights[edge_key]['relations'].append(rel)
        
        # エッジを追加
        for (u, v), weights in edge_weights.items():
            merged_kg.add_edge(
                u, v,
                weight=weights['weight'],
                intra_raw=weights['intra_raw'],
                inter_raw=weights['inter_raw'],
                relation=weights['relations'][0] if weights['relations'] else 'RELATED',
                relation_types=weights['relations']
            )
        
        return merged_kg    

    def filter_triplets(
        self,
        triplets: List[Tuple[str, str, str]],
        quality_threshold: float = 0.3
    ) -> Tuple[List[Tuple], List[Tuple], Dict]:
        """
        トリプレットを品質でフィルタリング
        
        Args:
            triplets: [(subject, relation, object), ...] のリスト
            quality_threshold: 品質スコアの閾値（0.0~1.0）
        
        Returns:
            (filtered_triplets, rejected_triplets, stats)
        """
        self.logger.info(f"🔍 Filtering {len(triplets)} triplets...")
        
        filtered = []
        rejected = []
        quality_scores = []
        
        for s, r, o in triplets:
            # 品質スコア計算
            score = self._compute_triplet_quality(s, r, o)
            quality_scores.append(score)
            
            if score >= quality_threshold:
                filtered.append((s, r, o))
            else:
                rejected.append((s, r, o))
                self.logger.debug(
                    f"  Rejected: ({s}, {r}, {o}) [score={score:.2f}]"
                )
        
        # 統計情報
        stats = {
            'original': len(triplets),
            'filtered': len(filtered),
            'rejected': len(rejected),
            'avg_quality': sum(quality_scores) / len(quality_scores) if quality_scores else 0,
            'rejection_rate': len(rejected) / len(triplets) if triplets else 0
        }
        
        self.logger.info(
            f"  → Kept {len(filtered)} triplets, "
            f"rejected {len(rejected)} ({stats['rejection_rate']:.1%})"
        )
        self.logger.info(f"  → Avg quality: {stats['avg_quality']:.2f}")
        
        return filtered, rejected, stats
    
    def self_rag_triplets(
        self,
        triplets: List[Tuple[str, str, str]],
        chunk_text: str,
        llm: Any = None
    ) -> Tuple[List[Tuple], Dict]:
        """
        Self-RAG: トリプレットを評価し、低品質なものを再生成

        """
        if not self.config.get('enable_self_rag', False):
            return triplets, {'self_rag_applied': False}
    
        # トークン予算チェック
        token_budget = self.config.get('self_rag_token_budget', 100000)
    
        if self.total_self_rag_tokens >= token_budget:
            self.logger.warning(
                f"⚠️  Self-RAG token budget exhausted "
                f"({self.total_self_rag_tokens}/{token_budget}), skipping"
            )
            return triplets, {
                'self_rag_applied': False,
                'budget_exhausted': True
            }

        self.logger.info(f"🔄 Self-RAG: Evaluating {len(triplets)} triplets...")
    
    # Critic: トリプレットを評価
        evaluated_triplets = []
    
        for s, r, o in triplets:
            confidence = self._critic_evaluate_triplet(s, r, o, chunk_text)
            evaluated_triplets.append({
                'triplet': (s, r, o),
                'confidence': confidence,
                'needs_refinement': confidence < self.config['self_rag_confidence_threshold']
            })
    
    # 統計
        low_confidence_count = sum(1 for t in evaluated_triplets if t['needs_refinement'])
        avg_confidence = np.mean([t['confidence'] for t in evaluated_triplets])
    
        self.logger.info(
            f"  → Avg confidence: {avg_confidence:.2f}, "
            f"Low confidence: {low_confidence_count}/{len(triplets)}"
        )
    
    # Refiner: 低品質トリプレットを再生成
        refined_triplets = []
        refinement_stats = {
            'attempted': 0,
            'succeeded': 0,
            'failed': 0,
            'tokens_used': 0 
        }
    
        for triplet_info in evaluated_triplets:
            # 予算チェック
            if self.total_self_rag_tokens >= token_budget:
                self.logger.info("  → Budget limit reached, stopping refinement")
                # 残りは元のトリプレットを保持
                refined_triplets.append(triplet_info['triplet'])
                continue

            if triplet_info['needs_refinement']:
                # 再生成を試みる
                refined, tokens_used = self._refiner_regenerate_triplet(
                    triplet_info['triplet'],
                    chunk_text,
                    llm
                )
            
                refinement_stats['attempted'] += 1
                refinement_stats['tokens_used'] += tokens_used
                self.total_self_rag_tokens += tokens_used
            
                if refined:
                # 再評価
                    s, r, o = refined
                    new_confidence = self._critic_evaluate_triplet(s, r, o, chunk_text)
                
                    if new_confidence > triplet_info['confidence']:
                    # 改善された場合は置き換え
                        refined_triplets.append(refined)
                        refinement_stats['succeeded'] += 1
                    
                        self.logger.debug(
                            f"  ✓ Refined: {triplet_info['triplet']} → {refined} "
                            f"(confidence: {triplet_info['confidence']:.2f} → {new_confidence:.2f})"
                        )
                    else:
                    # 改善されなかった場合は元を保持
                        refined_triplets.append(triplet_info['triplet'])
                        refinement_stats['failed'] += 1
                else:
                # 再生成失敗時は元を保持
                    refined_triplets.append(triplet_info['triplet'])
                    refinement_stats['failed'] += 1
            else:
            # 高品質なものはそのまま
                refined_triplets.append(triplet_info['triplet'])
    
    # ============================================================
    # 3. Validator: 最終検証
    # ============================================================
        validated_triplets = self._validator_check_consistency(
            refined_triplets,
            chunk_text
        )
    
    # 統計情報
        stats = {
            'self_rag_applied': True,
            'original_count': len(triplets),
            'refined_count': len(validated_triplets),
            'avg_confidence': float(avg_confidence),
            'low_confidence_count': low_confidence_count,
            'refinement_stats': refinement_stats,
            'total_tokens_used': self.total_self_rag_tokens
        }
    
        self.logger.info(
            f"  → Self-RAG complete: "
            f"{refinement_stats['succeeded']} improved, "
            f"{refinement_stats['failed']} kept original"
            f"tokens: {refinement_stats['tokens_used']}"        
        )
    
        return validated_triplets, stats

    def _critic_evaluate_triplet(
        self,
        subject: str,
        relation: str,
        object_: str,
        context: str
    ) -> float:
        """
        トリプレットの品質を評価（信頼度スコア: 0.0～1.0）
    
        Args:
            subject: 主語
            relation: 関係
            object_: 目的語
            context: 元のテキスト
    
        Returns:
            信頼度スコア（高いほど高品質）
        """
        scores = []
    
    # ============================================================
    # 1. エンティティの品質（既存のメソッドを活用）
    # ============================================================
        entity_score = self._score_entities(subject, object_)
        scores.append(('entity', entity_score, 0.3))
    
    # ============================================================
    # 2. 関係の明確性（既存のメソッドを活用）
    # ============================================================
        relation_score = self._score_relation(relation)
        scores.append(('relation', relation_score, 0.3))
    
    # ============================================================
    # 3. 文法的正しさ（既存のメソッドを活用）
    # ============================================================
        grammar_score = self._score_grammar(subject, relation, object_)
        scores.append(('grammar', grammar_score, 0.2))
    
    # ============================================================
    # 4. コンテキストとの整合性（新規）
    # ============================================================
        context_score = self._score_context_alignment(
            subject, relation, object_, context
        )
        scores.append(('context', context_score, 0.2))
    
    # ============================================================
    # 5. 重み付き平均
    # ============================================================
        total_score = sum(score * weight for _, score, weight in scores)
    
    # デバッグログ（DEBUG時のみ）
        if self.logger.level <= logging.DEBUG:
            score_details = ', '.join(f"{name}={score:.2f}" for name, score, _ in scores)
            self.logger.debug(
                f"  Triplet: ({subject[:20]}, {relation}, {object_[:20]}) "
                f"→ {score_details} = {total_score:.2f}"
            )
    
        return total_score

    def _score_context_alignment(
        self,
        subject: str,
        relation: str,
        object_: str,
        context: str
    ) -> float:
        """
        トリプレットとコンテキストの整合性をスコアリング
    
        Returns:
            0.0（整合性なし）～ 1.0（完全に整合）
        """
        score = 0.0
        context_lower = context.lower()
    
    # ============================================================
    # 1. エンティティがコンテキストに存在するか
    # ============================================================
        subject_in_context = subject.lower() in context_lower
        object_in_context = object_.lower() in context_lower
    
        if subject_in_context and object_in_context:
            score += 0.5
        elif subject_in_context or object_in_context:
            score += 0.3
        else:
            # どちらもコンテキストにない場合は低スコア
            score += 0.1
    
    # ============================================================
    # 2. 関係がコンテキストの文脈と合致するか
    # ============================================================
        relation_lower = relation.lower().replace('_', ' ')
    
    # 関係の動詞形がコンテキストに存在するか
        if relation_lower in context_lower:
            score += 0.3
        else:
        # 類似表現をチェック（簡易実装）
            relation_synonyms = self._get_relation_synonyms(relation)
            if any(syn in context_lower for syn in relation_synonyms):
                score += 0.2
    
    # ============================================================
    # 3. トリプレット全体の近接性
    # ============================================================
    # 主語と目的語がコンテキスト内で近い位置にあるか
        if subject_in_context and object_in_context:
            try:
                subject_pos = context_lower.find(subject.lower())
                object_pos = context_lower.find(object_.lower())
            
                distance = abs(object_pos - subject_pos)
            
            # 距離に応じてスコアを調整（近いほど高スコア）
                if distance < 50:
                    score += 0.2
                elif distance < 100:
                    score += 0.1
            except Exception:
                pass
    
        return min(score, 1.0)

    def _get_relation_synonyms(self, relation: str) -> List[str]:
        """
        関係の同義語・類似表現を返す
    
        Args:
            relation: 関係名
    
        Returns:
            同義語のリスト
        """
    # 主要な関係の同義語マップ
        synonym_map = {
            'uses': ['use', 'utilizes', 'employs', 'applies'],
            'causes': ['cause', 'leads to', 'results in', 'triggers'],
            'part_of': ['part of', 'component of', 'belongs to'],
            'is_a': ['is a', 'type of', 'kind of', 'instance of'],
            'has': ['have', 'contains', 'includes', 'comprises'],
            'improves': ['improve', 'enhances', 'optimizes', 'boosts'],
            'based_on': ['based on', 'derived from', 'built on', 'relies on'],
            'enables': ['enable', 'allows', 'permits', 'facilitates'],
            'requires': ['require', 'needs', 'depends on', 'necessitates'],
        }
    
        relation_lower = relation.lower().replace('_', ' ')
    
    # 完全一致を探す
        for key, synonyms in synonym_map.items():
            if relation_lower == key.replace('_', ' ') or relation_lower in synonyms:
                return synonyms
    
    # マッチしない場合は元の関係のみ
        return [relation_lower]

    def _refiner_regenerate_triplet(
        self,
        original_triplet: Tuple[str, str, str],
        chunk_text: str,
        llm: Any = None
    ) -> Tuple[Optional[Tuple[str, str, str]], int]: 
        """
        低品質トリプレットを再生成
    
        Args:
            original_triplet: 元のトリプレット
            chunk_text: 元のテキスト
            llm: LLMインスタンス
    
        Returns:
            改善されたトリプレット（失敗時はNone）
        """
        s, r, o = original_triplet
    
    # LLMが提供されていない場合は初期化
        if llm is None:
            llm = OpenAI(
                model=self.config['self_rag_refiner_model'],
                timeout=self.config['llm_timeout']
            )
    
    # ============================================================
    # プロンプト構築
    # ============================================================
        prompt = f"""Given the following text, improve the quality of this knowledge triplet.

    Original triplet:
    - Subject: {s}
    - Relation: {r}
    - Object: {o}

    Text context:
    {chunk_text[:500]}

    Please provide an improved triplet that:
    1. Uses more specific and descriptive entities
    2. Uses a clear and meaningful relation
    3. Accurately reflects the text content
    4. Avoids vague terms like "it", "this", "that"

    Return ONLY the improved triplet in this exact format:
    Subject: [improved subject]
    Relation: [improved relation]
    Object: [improved object]

    If the original triplet cannot be improved, return "NO_IMPROVEMENT".
    """
    # 簡易実装: 文字数 / 4 ≈ トークン数（英語）
        prompt_tokens = len(prompt) // 4
        
    # ============================================================
    # LLMで再生成
    # ============================================================
        try:
            response = llm.complete(prompt)
            response_text = response.text.strip()
        
            response_tokens = len(response_text) // 4
            total_tokens = prompt_tokens + response_tokens

            # "NO_IMPROVEMENT"チェック
            if "NO_IMPROVEMENT" in response_text.upper():
                return None, total_tokens
        
        # レスポンスをパース
            refined = self._parse_triplet_response(response_text)
        
            if refined:
                return refined, total_tokens
            else:
                self.logger.debug(f"  Failed to parse refinement response")
                return None, total_tokens
    
        except Exception as e:
            self.logger.debug(f"  Refinement failed: {type(e).__name__}")
            return None, prompt_tokens
        
    def _parse_triplet_response(self, response: str) -> Optional[Tuple[str, str, str]]:
        """
        LLMレスポンスからトリプレットを抽出
    
        Args:
            response: LLMの出力テキスト
    
        Returns:
            (subject, relation, object) または None
        """
        try:
            lines = response.strip().split('\n')
        
            subject = None
            relation = None
            object_ = None
        
            for line in lines:
                line = line.strip()
            
                if line.startswith('Subject:'):
                    subject = line.replace('Subject:', '').strip()
                elif line.startswith('Relation:'):
                    relation = line.replace('Relation:', '').strip()
                elif line.startswith('Object:'):
                    object_ = line.replace('Object:', '').strip()
        
        # すべてが抽出できたか確認
            if subject and relation and object_:
            # 空白や記号のみでないか確認
                if (len(subject.strip()) > 1 and 
                    len(relation.strip()) > 1 and 
                    len(object_.strip()) > 1):
                    return (subject, relation, object_)
        
            return None
    
        except Exception as e:
            self.logger.debug(f"  Parse error: {e}")
            return None
        
    def _validator_check_consistency(
        self,
        triplets: List[Tuple[str, str, str]],
        context: str
    ) -> List[Tuple[str, str, str]]:
        """
        トリプレットの一貫性と矛盾をチェック
    
        Args:
            triplets: トリプレットのリスト
            context: 元のテキスト
    
        Returns:
            検証済みトリプレットのリスト（矛盾があるものは除外）
        """
        validated = []
        seen_triplets = set()  # 重複チェック用
    
        for s, r, o in triplets:
        # ============================================================
        # 1. 重複チェック
        # ============================================================
            triplet_key = (s.lower(), r.lower(), o.lower())
            if triplet_key in seen_triplets:
                self.logger.debug(f"  ⊗ Duplicate: ({s}, {r}, {o})")
                continue
        
        # ============================================================
        # 2. 自己参照チェック（主語と目的語が同じ）
        # ============================================================
            if s.lower().strip() == o.lower().strip():
                self.logger.debug(f"  ⊗ Self-reference: ({s}, {r}, {o})")
                continue
        
        # ============================================================
        # 3. 逆関係の矛盾チェック
        # ============================================================
            if self._has_contradictory_relation(s, r, o, validated):
                self.logger.debug(f"  ⊗ Contradictory: ({s}, {r}, {o})")
                continue
        
        # ============================================================
        # 4. コンテキスト妥当性の最終チェック
        # ============================================================
            if not self._is_contextually_valid(s, r, o, context):
                self.logger.debug(f"  ⊗ Context invalid: ({s}, {r}, {o})")
                continue
        
           # すべてのチェックをパス
            validated.append((s, r, o))
            seen_triplets.add(triplet_key)
    
        removed_count = len(triplets) - len(validated)
        if removed_count > 0:
            self.logger.info(f"  → Validator removed {removed_count} inconsistent triplets")
    
        return validated

    def _has_contradictory_relation(
        self,
        subject: str,
        relation: str,
        object_: str,
        existing_triplets: List[Tuple[str, str, str]]
    ) -> bool:
        """
        既存のトリプレットと矛盾する関係がないかチェック
    
        Args:
            subject: 主語
            relation: 関係
            object_: 目的語
            existing_triplets: 既に検証済みのトリプレット
    
        Returns:
            True: 矛盾あり, False: 矛盾なし
        """
    # 矛盾する関係のペア
        contradictory_pairs = [
        # 原因と結果の逆転
            ('causes', 'caused_by'),
            ('creates', 'created_by'),
            ('produces', 'produced_by'),
        
        # 包含関係の逆転
            ('part_of', 'contains'),
            ('component_of', 'has_component'),
            ('member_of', 'has_member'),
        
        # 肯定と否定
            ('is', 'is_not'),
            ('has', 'lacks'),
            ('enables', 'prevents'),
        
        # 時間的矛盾
            ('before', 'after'),
            ('precedes', 'follows'),
        ]
    
        subject_lower = subject.lower()
        object_lower = object_.lower()
        relation_lower = relation.lower().replace('_', ' ').replace('-', ' ')
    
        for s_exist, r_exist, o_exist in existing_triplets:
            s_exist_lower = s_exist.lower()
            o_exist_lower = o_exist.lower()
            r_exist_lower = r_exist.lower().replace('_', ' ').replace('-', ' ')
        
        # 同じエンティティペアで異なる関係
            if ((subject_lower == s_exist_lower and object_lower == o_exist_lower) or
                (subject_lower == o_exist_lower and object_lower == s_exist_lower)):
            
            # 矛盾する関係のペアをチェック
                for rel1, rel2 in contradictory_pairs:
                    if ((relation_lower == rel1 and r_exist_lower == rel2) or
                        (relation_lower == rel2 and r_exist_lower == rel1)):
                        self.logger.debug(
                            f"  Found contradiction: "
                            f"({subject}, {relation}, {object_}) vs "
                            f"({s_exist}, {r_exist}, {o_exist})"
                        )
                        return True
    
        return False

    def _is_contextually_valid(
        self,
        subject: str,
        relation: str,
        object_: str,
        context: str,
        min_score: float = 0.3
    ) -> bool:
        """
        トリプレットがコンテキストに対して妥当かチェック
    
        Args:
            subject: 主語
            relation: 関係
            object_: 目的語
            context: 元のテキスト
            min_score: 最小スコア閾値
    
        Returns:
            True: 妥当, False: 不適切
        """
    # コンテキストアライメントスコアを使用
        score = self._score_context_alignment(subject, relation, object_, context)
    
        return score >= min_score

    def _compute_triplet_quality(
        self,
        subject: str,
        relation: str,
        object_: str
    ) -> float:
        """
        トリプレットの品質スコアを計算
        
        スコアリング基準：
        - 関係の明確性（0.4）
        - エンティティの具体性（0.3）
        - 文法的正しさ（0.3）
        
        Returns:
            0.0~1.0 のスコア
        """
        score = 0.0
        
        # ============================================================
        # 1. 関係の明確性（0.4）
        # ============================================================
        relation_score = self._score_relation(relation)
        score += relation_score * 0.4
        
        # ============================================================
        # 2. エンティティの具体性（0.3）
        # ============================================================
        entity_score = self._score_entities(subject, object_)
        score += entity_score * 0.3
        
        # ============================================================
        # 3. 文法的正しさ（0.3）
        # ============================================================
        grammar_score = self._score_grammar(subject, relation, object_)
        score += grammar_score * 0.3
        
        return min(max(score, 0.0), 1.0)
    
    def _map_triplets_to_documents(
        self,
        triplets: List[Tuple[str, str, str]],
        documents: List[Document]
    ) -> Dict[Document, List[Tuple[str, str, str]]]:
        """
        トリプレットをドキュメントにマッピング
    
        Args:
            triplets: トリプレットのリスト
            documents: ドキュメントのリスト
    
        Returns:
            {Document: [triplets]} の辞書
        """
        mapping = {doc: [] for doc in documents}
    
    # 各トリプレットがどのドキュメントに属するか判定
        for s, r, o in triplets:
        # エンティティがドキュメント内に存在するか確認
            for doc in documents:
                doc_text_lower = doc.text.lower()
            
            # 主語または目的語がドキュメントに含まれる
                if (s.lower() in doc_text_lower or o.lower() in doc_text_lower):
                    mapping[doc].append((s, r, o))
                    break  # 最初にマッチしたドキュメントに割り当て
            else:
            # どのドキュメントにもマッチしない場合は最初のドキュメントに割り当て
                if documents:
                    mapping[documents[0]].append((s, r, o))
    
    # 空のエントリを削除
        mapping = {doc: trips for doc, trips in mapping.items() if trips}
    
        self.logger.info(f"  Mapped {len(triplets)} triplets to {len(mapping)} documents")
    
        return mapping
    
    def _score_relation(self, relation: str) -> float:
        """
        関係の明確性をスコアリング
        
        Returns:
            0.0（最悪）～ 1.0（最良）
        """
        relation_lower = relation.lower().strip()
        
        # ブラックリスト（即座に0.0）
        if relation_lower in self.relation_blacklist:
            return 0.0
        
        # 空または短すぎる
        if len(relation_lower) < 2:
            return 0.0
        
        # 高品質な関係（専門的・具体的）
        high_quality_relations = {
            # 因果関係
            'causes', 'results_in', 'leads_to', 'enables', 'triggers',
            'produces', 'generates', 'influences', 'affects',
            
            # 構成関係
            'part_of', 'component_of', 'consists_of', 'comprises',
            'contains', 'includes',
            
            # 使用関係
            'uses', 'utilizes', 'employs', 'applies', 'leverages',
            'implements', 'adopts',
            
            # 派生関係
            'based_on', 'derived_from', 'inspired_by', 'extends',
            'improves_upon', 'builds_on',
            
            # 専門関係
            'optimizes', 'parameterizes', 'regularizes', 'approximates',
            'encodes', 'decodes', 'transforms', 'projects',
            
            # 比較関係
            'outperforms', 'surpasses', 'exceeds', 'improves',
        }
        
        if relation_lower in high_quality_relations:
            return 1.0
        
        # 中品質な関係（一般的だが有用）
        medium_quality_relations = {
            'is_a', 'type_of', 'instance_of', 'subclass_of',
            'related_to', 'associated_with', 'connected_to',
            'depends_on', 'requires', 'needs',
        }
        
        if relation_lower in medium_quality_relations:
            return 0.7
        
        # 動詞形式（-s, -ed, -ing）なら中程度
        if re.match(r'\w+(s|ed|ing)$', relation_lower):
            return 0.6
        
        # それ以外は低品質
        return 0.3
    
    def _score_entities(self, subject: str, object_: str) -> float:
        """
        エンティティの具体性をスコアリング
        
        Returns:
            0.0（抽象的・曖昧）～ 1.0（具体的）
        """
        score = 0.0
        
        # 両方のエンティティをチェック
        for entity in [subject, object_]:
            entity_lower = entity.lower().strip()
            
            # 空または短すぎる
            if len(entity_lower) < 2:
                continue
            
            # 代名詞（低品質）
            pronouns = {'it', 'this', 'that', 'these', 'those', 'they', 'them'}
            if entity_lower in pronouns:
                score += 0.0
                continue
            
            # 単語数（複数単語 = より具体的）
            word_count = len(entity_lower.split())
            if word_count >= 3:
                score += 1.0
            elif word_count == 2:
                score += 0.8
            else:
                score += 0.5
        
        # 2つのエンティティの平均
        return score / 2.0
    
    def _score_grammar(
        self,
        subject: str,
        relation: str,
        object_: str
    ) -> float:
        """
        文法的正しさをスコアリング
        
        Returns:
            0.0（文法的におかしい）～ 1.0（正しい）
        """
        score = 1.0
        
        # 全て小文字（抽出ミスの可能性）
        if subject.islower() and object_.islower():
            score -= 0.2
        
        # 数字だけのエンティティ（低品質）
        if subject.isdigit() or object_.isdigit():
            score -= 0.3
        
        # 記号のみ
        if not re.search(r'[a-zA-Z]', subject) or not re.search(r'[a-zA-Z]', object_):
            score -= 0.5
        
        # 主語と目的語が同じ（自己参照）
        if subject.lower() == object_.lower():
            score -= 0.5
        

    # ------------------------------------------------------------
    # 2. 関係の品質チェック（新規追加）
    # ------------------------------------------------------------
    
        relation_lower = relation.lower().strip()
    
        # 関係が空または短すぎる
        if len(relation_lower) < 2:
            score -= 0.4
    
        # 関係がブラックリストに含まれる（低品質）
        if relation_lower in self.relation_blacklist:
            score -= 0.3
    
        # 関係が記号のみ
        if not re.search(r'[a-zA-Z]', relation):
            score -= 0.4
    
    # ------------------------------------------------------------
    # 3. トリプレット全体の整合性チェック
    # ------------------------------------------------------------
    
        # 主語と関係が同じ（例: "uses uses object"）
        if subject.lower() == relation_lower:
            score -= 0.3
    
    # 目的語と関係が同じ（例: "subject uses uses"）
        if object_.lower() == relation_lower:
            score -= 0.3
    
    # 3つとも同じ（最悪）
        if subject.lower() == relation_lower == object_.lower():
            score -= 0.5

        return max(score, 0.0)

    def commit_to_graph(self, documents: List[Document], graph_store: Neo4jGraphStore):
        """Neo4jにグラフを投入"""
        #　接続確認　===========================================
        try:
            graph_store.query("RETURN 1")
            self.logger.info("✅ Neo4j connection verified")
        except Exception as e:
            self.logger.error(f"🚨 Neo4j connection failed: {type(e).__name__}")
            raise  # 接続できないなら処理を中断
        # 2. グラフ生成 ===========================================
        try:
            with self.hlogger.section("Graph Generation"):
                llm = OpenAI(
                    model=self.config['llm_model'],
                    timeout=self.config['llm_timeout']
                )
            #    storage_context = StorageContext.from_defaults(graph_store=graph_store)
            
                local_graph_store = SimpleGraphStore()
                local_storage = StorageContext.from_defaults(graph_store=local_graph_store)

                self.logger.info("Building local knowledge graph...")
                index = KnowledgeGraphIndex.from_documents(
                    documents,
                    storage_context=local_storage, 
                    llm=llm,
                    transformations=[SimpleNodeParser.from_defaults(chunk_size=512)],
                    embed_model=self.embed_model,
                    show_progress=True,
                    max_triplets_per_chunk=self.config['max_triplets_per_chunk']    # 15
                )
            
                kg = index.get_networkx_graph()
                self.logger.info(f"✅ Initial graph: {len(kg.nodes)} nodes, {len(kg.edges)} edges")

                # トリプレットをメタデータに保存
                all_triples = []

                for subj, obj, data in kg.edges(data=True):
                    rel = data.get('relation', 'RELATED')
                    all_triples.append((subj, rel, obj))
                # rel_map処理
                if hasattr(local_graph_store, 'get_rel_map'):
                    try:
                        rel_map = local_graph_store.get_rel_map()
                        self.logger.debug(f"rel_map structure: {type(rel_map)}")
        
                        for subj, relations in rel_map.items():
                        # relations が辞書か、リストか確認
                            if isinstance(relations, dict):
                                # 辞書の場合
                                for rel, objs in relations.items():
                                    if isinstance(objs, list):
                                        for obj in objs:
                                            if (subj, rel, obj) not in all_triples:
                                                all_triples.append((subj, rel, obj))
                                    else:
                                        if (subj, rel, objs) not in all_triples:
                                            all_triples.append((subj, rel, objs))
                            elif isinstance(relations, list):
                                # リストの場合
                                for item in relations:
                                    if isinstance(item, tuple) and len(item) == 2:
                                        rel, obj = item
                                        if (subj, rel, obj) not in all_triples:
                                            all_triples.append((subj, rel, obj))
                    except Exception as e:
                        self.logger.warning(f"Could not parse rel_map: {e}")

                self.logger.info(f"Extracted {len(all_triples)} triples (before filtering)")

# Self-RAG統合 
# 品質フィルタを適用
                if self.config.get('enable_triplet_filter', True):
                    filtered_triples, rejected_triples, filter_stats = self.filter_triplets(
                        all_triples,
                        quality_threshold=self.config.get('triplet_quality_threshold', 0.3)
                    )
                    all_triples = filtered_triples
    
                    self.logger.info(
                        f"After filtering: {len(all_triples)} triples "
                        f"(rejection rate: {filter_stats['rejection_rate']:.1%})"
                    )

# Self-RAGを適用（チャンクごとに処理）
                if self.config.get('enable_self_rag', False):
                    with self.hlogger.section("Self-RAG Refinement"):
        # ドキュメントごとにトリプレットを再生成
                        refined_all_triples = []
                        total_self_rag_stats = {
                            'attempted': 0,
                            'succeeded': 0,
                            'failed': 0
                        }
        
        # ドキュメントとそのトリプレットをマッピング
                        doc_triplet_map = self._map_triplets_to_documents(all_triples, documents)
        
                        for doc_idx, (doc, doc_triplets) in enumerate(doc_triplet_map.items()):
                            if not doc_triplets:
                                continue
            
                            try:
                                refined_triplets, stats = self.self_rag_triplets(
                                    doc_triplets,
                                    doc.text,
                                    llm=llm  # 既存のLLMインスタンスを使用
                                )

                                refined_all_triples.extend(refined_triplets)
                
                # 統計を集計
                                if stats.get('self_rag_applied'):
                                    ref_stats = stats['refinement_stats']
                                    total_self_rag_stats['attempted'] += ref_stats['attempted']
                                    total_self_rag_stats['succeeded'] += ref_stats['succeeded']
                                    total_self_rag_stats['failed'] += ref_stats['failed']
                
                                if (doc_idx + 1) % 10 == 0:
                                    self.logger.info(f"  Processed {doc_idx + 1}/{len(doc_triplet_map)} documents...")
            
                            except Exception as e:
                                self.logger.warning(f"  Self-RAG failed for doc {doc_idx}: {type(e).__name__}")
                        # 失敗時は元のトリプレットを保持
                                refined_all_triples.extend(doc_triplets)
        
        # トリプレットを更新
                        all_triples = refined_all_triples
        
                        self.logger.info(
                            f"✅ Self-RAG complete: "
                            f"{total_self_rag_stats['succeeded']} improved, "
                            f"{total_self_rag_stats['attempted']} attempted, "
                            f"final count: {len(all_triples)}"
                        )
                # 再度品質フィルタを適用
                if self.config.get('enable_triplet_filter', True):
                    filtered_triples,rejected_triples, filter_stats = self.filter_triplets(
                        all_triples,
                        quality_threshold=self.config.get('triplet_quality_threshold', 0.3)
                    )
                    all_triples = filtered_triples
                
                    # 統計情報を活用
                    self.logger.info(
                        f"After filtering: {len(all_triples)} triples "
                        f"(rejection rate: {filter_stats['rejection_rate']:.1%})"
                    )

                    # 品質が低い場合は警告
                    if filter_stats['avg_quality'] < 0.5:
                        self.logger.warning("⚠️  Low average triplet quality!")

                    # デバッグモードならリジェクト例を表示
                    if rejected_triples and self.logger.level <= logging.DEBUG:
                        self.logger.debug("Sample rejected triplets:")
                        for s, r, o in rejected_triples[:3]:
                            self.logger.debug(f"  ({s}, {r}, {o})")

                # すべてのドキュメントに全トリプルを共有
                for doc in documents:
                    doc.metadata['triples'] = all_triples

        except Exception as e:
            self.logger.error(
                f"🚨 Graph generation failed: {type(e).__name__}"
            )
            raise

        # Entity Linking
        try:
            with self.hlogger.section("Entity Linking"):
                kg, entity_mapping = self.link_entities(
                    kg,
                    similarity_threshold=self.config['entity_linking_threshold'],
                    use_embedding=True
                )
                
                # トリプレット更新
                updated_triples = []
                for s, r, o in all_triples:
                    s_new = entity_mapping.get(s, s)
                    o_new = entity_mapping.get(o, o)
                    if s_new != o_new:  # 自己ループ除外
                        updated_triples.append((s_new, r, o_new))
                
                # ドキュメントのメタデータを更新
                for doc in documents:
                    doc.metadata['triples'] = updated_triples
                
                self.logger.info(f"Updated triples: {len(all_triples)} → {len(updated_triples)}")
        
        except Exception as e:
            self.logger.warning(f"⚠️  Entity linking failed: {e}")
            # Entity Linking失敗でも処理は継続        

        # パス情報をグラフに統合　================================
        try:
            with self.hlogger.section("Merging Path Information"):
                self.merge_paths_into_kg(kg, documents)
                self.logger.info(f"✅ Path info merged: {len(kg.nodes)} nodes, {len(kg.edges)} edges")

        except Exception as e:
            self.logger.warning(f"⚠️  Path merging failed: {type(e).__name__} - {str(e)[:100]}")
    
        self.logger.info("  → Continuing without path information")
    
    # デバッグ情報を記録
        if self.logger.level <= logging.DEBUG:
            self.logger.debug(f"Path merge traceback:\n{traceback.format_exc()}")
    
    # documentsからpaths情報を削除（中途半端なデータを残さない）
        for doc in documents:
            doc.metadata.pop('paths', None)
            doc.metadata.pop('path_distances', None)

        # RAPL最適化
        try:
            with self.hlogger.section("Graph Optimization (RAPL)"):
                optimized_kg = self._optimize_graph_rapl(kg, documents)
                self.logger.info(
                    f"✅ Optimized graph: {len(optimized_kg.nodes)} nodes, "
                    f"{len(optimized_kg.edges)} edges"
                )
        except Exception as e:
            self.logger.error(  
                f"🚨 Graph optimization failed: {e}")
            optimized_kg = kg 

    # Multi-hop パス探索（サンプルクエリで代表的なパスを計算）
        try:
            with self.hlogger.section("Multi-hop Path Pre-computation"):
                self._precompute_representative_paths(optimized_kg, documents)
                self.logger.info("✅ Representative paths computed and stored")
    
        except Exception as e:
            self.logger.warning(f"⚠️  Path pre-computation failed: {type(e).__name__} - {str(e)[:100]}")
            self.logger.info("  → Continuing without pre-computed paths")
        
            if self.logger.level <= logging.DEBUG:
                self.logger.debug(f"Path pre-computation traceback:\n{traceback.format_exc()}")
    
            # 最適化されたグラフをNeo4jに反映
        try:
            with self.hlogger.section("Updating Neo4j"):
                result = self._update_neo4j_structure(optimized_kg, graph_store)
            
            # result が None の場合のフォールバック
                if result is None:
                    result = {'updated': 0, 'skipped': 0, 'failed': 0, 'error_details': []}
                    
                    self.logger.warning("⚠️  _update_neo4j_structure returned None")

            # 結果サマリー
                self.logger.info(
                    f"✅ Neo4j update complete:\n"
                    f"   - Updated: {result.get('updated', 0)} edges\n"
                    f"   - Skipped: {result.get('skipped', 0)} edges\n"
                    f"   - Failed: {result.get('failed', 0)} edges"
                )
            
            # 失敗率が高い場合は警告
                total = result.get('updated', 0) + result.get('failed', 0)
                if total > 0 and result.get('failed', 0) / total > 0.3:
                    self.logger.warning(
                        f"⚠️  High failure rate ({result.get('failed', 0)/total:.1%}). "
                        f"Check Neo4j constraints and data format."
                    )
    
        except Exception as e:
            self.logger.error(f"🚨 Neo4j update failed: {e}")
            raise

    def merge_paths_into_kg(self, kg, documents: List[Document]):
        """
        kg: networkx.Graph (triples turned into nodes/edges)
        documents: the same documents that have metadata['paths'] etc.
        This will:
          - count how many times each entity appears in top-k paths
          - add edge/node attributes: top_path_count, avg_path_length
        """
        from collections import Counter, defaultdict
        path_entity_counts = Counter()
        entity_path_lengths = defaultdict(list)

        for doc in documents:
            paths = doc.metadata.get('paths', [])  # each path is a str like "A -> B -> C" OR list; adapt if needed
            distances = doc.metadata.get('path_distances', [])
            for i, p in enumerate(paths):
                # normalize path representation
                if isinstance(p, str):
                    nodes = [n.strip() for n in p.split('->') if n.strip()]
                elif isinstance(p, (list, tuple)):
                   nodes = list(p)
                else:
                    continue

                dist = distances[i] if i < len(distances) else len(nodes)-1
                for n in nodes:
                    path_entity_counts[n] += 1
                    entity_path_lengths[n].append(dist)

                # if the path describes relations, you could also add edges for consecutive nodes
                for a, b in zip(nodes, nodes[1:]):
                    if kg.has_edge(a, b):
                        # add a path_support counter on existing edge
                        kg[a][b].setdefault('path_support', 0)
                        kg[a][b]['path_support'] += 1
                    else:
                        kg.add_edge(a, b, relation='path_inferred', path_support=1)

        # inject aggregated attrs to nodes
        for n in kg.nodes():
            cnt = path_entity_counts.get(n, 0)
            lens = entity_path_lengths.get(n, [])
            avg_len = sum(lens)/len(lens) if lens else None
            kg.nodes[n]['path_top_count'] = cnt
            if avg_len is not None:
                kg.nodes[n]['path_avg_length'] = avg_len

    def _optimize_graph_rapl(self, kg, documents):
        """
        RAPL 最適化
        """
    
    # 1. Triples 抽出
        doc_triples = {}
        for idx, doc in enumerate(documents):
            triples = doc.metadata.get("triples", [])
            if triples:  # 空リストは除外
                doc_triples[idx] = triples
        
        all_triples = [t for lst in doc_triples.values() for t in lst]
    
        self.logger.info(f"Total triples: {len(all_triples)}")
    
    # Weight 格納領域の初期化
        for u, v in kg.edges():
            kg[u][v]["intra_raw"] = 0.0
            kg[u][v]["inter_raw"] = 0.0
    
    # 2. Intra: 文書内 triple 間相互作用
        self.logger.info("Computing intra-interactions...")
        intra_collector = ErrorCollector(self.logger)
        intra_edges = 0
    
        for doc_id, triples in doc_triples.items():
            try:
                entities = set()
                for s, _, o in triples:
                    entities.add(s)
                    entities.add(o)
        
        # Triple 間の相互作用（関係の相性を考慮）
                for i in range(len(triples)):
                    s1, r1, o1 = triples[i]
                    for j in range(i + 1, len(triples)):
                        s2, r2, o2 = triples[j]
                
                # 関係の相性
                        try:
                            rel_compat = self._compute_relation_compatibility(r1, r2)
                
                # エンティティの共有度
                            shared = len({s1, o1} & {s2, o2})
                            shared_score = shared * 0.5
                
                # 統合重み
                            w = rel_compat * 0.6 + shared_score * 0.4
                
                            if w > 0.3:
                                if kg.has_edge(s1, o1):
                                    kg[s1][o1]["intra_raw"] += w
                                if kg.has_edge(s2, o2):
                                    kg[s2][o2]["intra_raw"] += w
                            intra_collector.add_success()
                        except Exception as e:
                            intra_collector.add_error(
                                context=f"doc_{doc_id}_triple_{i}_{j}",
                                error=e,
                                triple1=(s1, r1, o1),
                                triple2=(s2, r2, o2)
                            )
        
        # エンティティペア間のエッジ追加
                for e1 in entities:
                    for e2 in entities:
                        if e1 != e2:
                            try:
                                w = self._compute_intra_weight(e1, e2, triples, kg)
                                if w > 0.5:
                                    if kg.has_edge(e1, e2):
                                        kg[e1][e2]["weight"] = kg[e1][e2].get("weight", 0) + w
                                    else:
                                        kg.add_edge(e1, e2, relation="intra_doc", weight=w)
                                        intra_edges += 1
                            except Exception as e:
                                intra_collector.add_error(
                                    context=f"entity_pair_{e1}_{e2}",
                                    error=e
                                )
            except Exception as e:
                self.logger.error(f"Failed to process document {doc_id}: {type(e).__name__}")
                continue

        intra_collector.report("Intra-document processing", threshold=0.3)
        self.logger.info(f"Added {intra_edges} intra-document edges")
    
    # 3. Inter: 共有エンティティベースの高速化
        self.logger.info("Computing inter-interactions (optimized & sampled)...")
        inter_collector = ErrorCollector(self.logger)
    
    # 3-1. エンティティ→Triple インデックス構築
        entity_to_triples = defaultdict(set)
        for idx, (s, r, o) in enumerate(all_triples):
            entity_to_triples[s].add(idx)
            entity_to_triples[o].add(idx)
    
        # 3-2. エンティティを出現頻度でソート（上位のみ処理）
        entity_freq = [(entity, len(triple_indices)) 
                       for entity, triple_indices in entity_to_triples.items()]
        entity_freq.sort(key=lambda x: x[1], reverse=True)
    
    # 上位100エンティティのみ処理（調整可能）
        max_entities = min(100, len(entity_freq))
        top_entities = set(entity for entity, _ in entity_freq[:max_entities])
    
        self.logger.info(
            f"  Sampled {max_entities}/{len(entity_to_triples)} entities "
            f"(covering {sum(freq for _, freq in entity_freq[:max_entities])} triples)"
        )
    
    # 3-2. 共有エンティティがある Triple ペアのみ計算
        seen_pairs = set()
        inter_count = 0
    
        for _entity, triple_indices in entity_to_triples.items():
            if _entity not in top_entities:
                continue  # 上位エンティティ以外はスキップ
            if len(triple_indices) < 3:
                continue  
        
            indices = list(triple_indices)
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    idx1, idx2 = indices[i], indices[j]
                    pair = (min(idx1, idx2), max(idx1, idx2))
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                
                # 重み計算
                    try:
                        t1 = all_triples[idx1]
                        t2 = all_triples[idx2]
                        w = self._compute_inter_weight(t1, t2, kg=kg)
                
                        if w > self.config['relation_compat_threshold']: 
                            s1, _, o1 = t1
                            s2, _, o2 = t2
                    
                    # 双方向に重みを加算
                            if kg.has_edge(s1, o1):
                                kg[s1][o1]["inter_raw"] = kg[s1][o1].get("inter_raw", 0.0) + w
                            if kg.has_edge(s2, o2):
                                kg[s2][o2]["inter_raw"] = kg[s2][o2].get("inter_raw", 0.0) + w
                    
                            inter_count += 1
                        inter_collector.add_success()

                    except Exception as e:
                        inter_collector.add_error(
                            context=f"triple_pair_{idx1}_{idx2}",
                            error=e
                        )

        inter_collector.report("Inter-document processing", threshold=0.3)
        self.logger.info(f"Added {inter_count} meaningful inter-interactions")
    
    # 4. Document-level linking
        self.logger.info("Computing document-level connections...")
    
        try:
            entity_docs = {}
            for doc_id, triples in doc_triples.items():
                for s, _, o in triples:
                    entity_docs.setdefault(s, set()).add(doc_id)
                    entity_docs.setdefault(o, set()).add(doc_id)
    
            doc_pairs = {}
            bridge_entities = []
    
            for entity_name, doc_set in entity_docs.items():
                if len(doc_set) > 1:
                    docs = list(doc_set)
                    for i, d1 in enumerate(docs):
                        for d2 in docs[i+1:]:
                            pair = (d1, d2)
                            doc_pairs[pair] = doc_pairs.get(pair, 0) + 1
            
                    if len(doc_set) > 2:
                        bridge_entities.append((entity_name, len(doc_set)))

                # ブリッジエンティティのログ
            if bridge_entities:
                bridge_entities.sort(key=lambda x: x[1], reverse=True)
                self.logger.info("Top bridge entities:")
                for entity_name, count in bridge_entities[:5]:
                    self.logger.info(f"  '{entity_name}': {count} documents")

            inter_doc_count = 0
            for (d1, d2), ct in doc_pairs.items():
                if ct > 2:
                    n1 = f"doc_{d1}"
                    n2 = f"doc_{d2}"
                
                    if not kg.has_node(n1):
                        kg.add_node(n1, type="document")
                    if not kg.has_node(n2):
                        kg.add_node(n2, type="document")
                
                    kg.add_edge(n1, n2, relation="inter_doc", weight=ct)
                    inter_doc_count += 1
            self.logger.info(f"Added {inter_doc_count} inter-document links")

        except Exception as e:
            self.logger.error(f"Document linking failed: {type(e).__name__} - {str(e)[:100]}")

            kg = self._normalize_edge_weights(kg, doc_triples, method='minmax')
            self.logger.info("Finalizing edge weights with normalization...")
    
            for u, v, d in kg.edges(data=True):
                intra = d.get("intra_normalized", d.get("intra_raw", 0.0))
                inter = d.get("inter_normalized", d.get("inter_raw", 0.0))
        
        # RAPL論文: intra重視 + inter補完
                d["weight"] = min(0.7 * intra + 0.3 * inter, 1.0)    
    
            self.logger.info(f"Weight calculation complete: {len(kg.edges())} edges")
            return kg

    def _normalize_edge_weights(
        self,
        kg: nx.Graph,
        doc_triples: Dict[int, List[Tuple]],
        method: str = 'minmax'
    ) -> nx.Graph:
        """
        エッジ重みをドキュメントごとに正規化
        
        Args:
            kg: NetworkXグラフ
            doc_triples: {doc_id: [(s, r, o), ...]} の辞書
            method: 'minmax' または 'zscore'
        
        Returns:
            正規化されたグラフ
        """
        self.logger.info(f"Normalizing edge weights (method={method})...")
        
        # ============================================================
        # 1. ドキュメントごとに重みを収集
        # ============================================================
        doc_edge_weights = defaultdict(lambda: {'intra': [], 'inter': []})
        edge_to_docs = defaultdict(set)  # エッジがどのドキュメントに属するか
        
        for doc_id, triples in doc_triples.items():
            doc_entities = set()
            for s, _, o in triples:
                doc_entities.add(s)
                doc_entities.add(o)
            
            # このドキュメントに関連するエッジを探す
            for u, v, data in kg.edges(data=True):
                if u in doc_entities or v in doc_entities:
                    edge_key = (u, v)
                    edge_to_docs[edge_key].add(doc_id)
                    
                    intra_raw = data.get('intra_raw', 0.0)
                    inter_raw = data.get('inter_raw', 0.0)
                    
                    if intra_raw > 0:
                        doc_edge_weights[doc_id]['intra'].append(intra_raw)
                    if inter_raw > 0:
                        doc_edge_weights[doc_id]['inter'].append(inter_raw)
        
        # ============================================================
        # 2. ドキュメントごとに正規化パラメータを計算
        # ============================================================
        norm_params = {}
        
        for doc_id, weights in doc_edge_weights.items():
            params = {}
            
            for weight_type in ['intra', 'inter']:
                values = weights[weight_type]
                
                if not values:
                    params[weight_type] = None
                    continue
                
                if method == 'minmax':
                    min_val = min(values)
                    max_val = max(values)
                    params[weight_type] = {
                        'min': min_val,
                        'max': max_val,
                        'range': max_val - min_val
                    }
                
                elif method == 'zscore':
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    params[weight_type] = {
                        'mean': mean_val,
                        'std': std_val if std_val > 0 else 1.0
                    }
            
            norm_params[doc_id] = params
        
        # 統計情報をログ出力
        self._log_normalization_stats(doc_edge_weights, norm_params)
        
        # ============================================================
        # 3. エッジごとに正規化を適用
        # ============================================================
        normalized_count = 0
        
        for u, v, data in kg.edges(data=True):
            edge_key = (u, v)
            related_docs = edge_to_docs.get(edge_key, set())
            
            if not related_docs:
                continue
            
            # このエッジに関連する全ドキュメントの正規化値を平均
            intra_normalized = []
            inter_normalized = []
            
            for doc_id in related_docs:
                if doc_id not in norm_params:
                    continue
                
                params = norm_params[doc_id]
                intra_raw = data.get('intra_raw', 0.0)
                inter_raw = data.get('inter_raw', 0.0)
                
                # Intra正規化
                if params['intra'] and intra_raw > 0:
                    norm_val = self._normalize_value(
                        intra_raw,
                        params['intra'],
                        method
                    )
                    intra_normalized.append(norm_val)
                
                # Inter正規化
                if params['inter'] and inter_raw > 0:
                    norm_val = self._normalize_value(
                        inter_raw,
                        params['inter'],
                        method
                    )
                    inter_normalized.append(norm_val)
            
            # 正規化後の値を平均
            if intra_normalized:
                data['intra_normalized'] = np.mean(intra_normalized)
                normalized_count += 1
            else:
                data['intra_normalized'] = data.get('intra_raw', 0.0)
            
            if inter_normalized:
                data['inter_normalized'] = np.mean(inter_normalized)
            else:
                data['inter_normalized'] = data.get('inter_raw', 0.0)
        
        self.logger.info(f"  → Normalized {normalized_count} edges")
        
        return kg
    
    def _normalize_value(
        self,
        value: float,
        params: dict,
        method: str
    ) -> float:
        """
        単一の値を正規化
        
        Args:
            value: 正規化する値
            params: 正規化パラメータ
            method: 'minmax' または 'zscore'
        
        Returns:
            正規化された値
        """
        if method == 'minmax':
            min_val = params['min']
            max_val = params['max']
            range_val = params['range']
            
            if range_val < 1e-9:
                return 0.5  # 全て同じ値の場合は中間値
            
            # [0, 1] に正規化
            normalized = (value - min_val) / range_val
            return max(0.0, min(1.0, normalized))
        
        elif method == 'zscore':
            mean_val = params['mean']
            std_val = params['std']
            
            # z-scoreを計算後、sigmoidで [0, 1] に変換
            z = (value - mean_val) / std_val
            sigmoid = 1 / (1 + np.exp(-z))
            return sigmoid
        
        return value
    
    def _log_normalization_stats(
        self,
        doc_edge_weights: dict,
        norm_params: dict
    ):
        """正規化統計をログ出力"""
        self.logger.info("  Normalization statistics:")
        
        for doc_id in list(norm_params.keys())[:3]:  # 最初の3ドキュメント
            params = norm_params[doc_id]
            
            intra_weights = doc_edge_weights[doc_id]['intra']
            inter_weights = doc_edge_weights[doc_id]['inter']
            
            if intra_weights:
                self.logger.info(
                    f"    Doc {doc_id} intra: "
                    f"min={min(intra_weights):.3f}, "
                    f"max={max(intra_weights):.3f}, "
                    f"mean={np.mean(intra_weights):.3f}"
                )
            
            if inter_weights:
                self.logger.info(
                    f"    Doc {doc_id} inter: "
                    f"min={min(inter_weights):.3f}, "
                    f"max={max(inter_weights):.3f}, "
                    f"mean={np.mean(inter_weights):.3f}"
                )

    # 5. 統合重み
        kg = self._normalize_edge_weights(kg, doc_triples, method='minmax')
        self.logger.info("Finalizing edge weights with normalization...")
    
        for u, v, d in kg.edges(data=True):
            intra = d.get("intra_normalized", d.get("intra_raw", 0.0))
            inter = d.get("inter_normalized", d.get("inter_raw", 0.0))
        
        # RAPL論文: intra重視 + inter補完
            d["weight"] = min(0.7 * intra + 0.3 * inter, 1.0)    
        self.logger.info(f"Weight calculation complete: {len(kg.edges())} edges")
        return kg
    
    def _group_triples_by_document(self, kg, documents):
        """トリプレットをDocument別にグループ化"""
        # 簡易実装: メタデータから推定
        doc_triples = {}
        
        for i, doc in enumerate(documents):
            doc_id = f"doc_{i}"
            doc_triples[doc_id] = []
            
            # エンティティがDocument内に出現するトリプレットを抽出
            for s, o, data in kg.edges(data=True):
                if s in doc.text or o in doc.text:
                    doc_triples[doc_id].append((s, data.get('relation', ''), o))
        return doc_triples
    
    def _compute_intra_weight(self, e1: str, e2: str, triples: List, kg=None) -> float:
        """
        同一Document内のエンティティ間重み計算
        
        Args:
            e1, e2: エンティティ名
            triples: (s, r, o) のリスト
            kg: NetworkXグラフ（オプション）
        """
    # ------------------------------------------------------------
    # 1) 共起頻度（基本）
    # ------------------------------------------------------------
        cooccur = sum(
            1 for s, _, o in triples
            if (s == e1 and o == e2) or (s == e2 and o == e1)
        )
        co_norm = min(cooccur / 5.0, 1.0)   # 正規化

    # ------------------------------------------------------------
    # 2) 関係タイプの多様性
    # ------------------------------------------------------------
        rel_pairs = [
            (r, True) for s, r, o in triples
            if (s == e1 and o == e2)
        ] + [
            (r, False) for s, r, o in triples
            if (s == e2 and o == e1)  # 逆向き
        ]
    
        if not rel_pairs:
            rel_bonus = 0.0
        else:
            # 関係の多様性
            unique_rels = set(r for r, _ in rel_pairs)
            diversity_bonus = min(len(unique_rels) * 0.2, 0.6)

        # 方向の一貫性（同じ向きが多いほど強い関係）
            same_direction_count = sum(1 for _, is_forward in rel_pairs if is_forward)
            opposite_direction_count = len(rel_pairs) - same_direction_count

        # 関係の質（同じ向きか逆向きかで評価）
            if same_direction_count > opposite_direction_count:
                direction_score = same_direction_count / len(rel_pairs)
            else:
            # 逆方向が多い = 双方向の関係（これも有用）
                direction_score = 0.7  # やや高めに評価

            rel_bonus = diversity_bonus * 0.5 + direction_score * 0.5

    # ------------------------------------------------------------
    # 3) パスサポート（kgに path_support があれば）
    # ------------------------------------------------------------
        path_bonus = 0.0
        if kg is not None and kg.has_edge(e1, e2):
            path_bonus = min(kg[e1][e2].get("path_support", 0) * 0.1, 0.5)

    # ------------------------------------------------------------
    # 4) 合成
    # ------------------------------------------------------------
        weight = co_norm * 0.5 + rel_bonus * 0.4 + path_bonus * 0.1
        return min(weight, 1.0)


    def _compute_inter_weight(self, t1: tuple, t2: tuple, kg=None): 
        """inter-triple interaction weight計算"""

        s1, r1, o1 = t1
        s2, r2, o2 = t2

        # 共有エンティティ（最重要）
        shared = len({s1, o1} & {s2, o2})
        shared_bonus = min(shared * 0.5, 1.0)
        # 関係の相性計算
        rel_compatibility = safe_execute(
            self._compute_relation_compatibility,
            args=(r1, r2),
            default=0.3,  
            logger=self.logger,
            context=f"relation_compatibility({r1}, {r2})"
        )
        # エンティティ類似度
        sim_bonus = 0.0
        try:        
            e1 = self.get_cached_embedding(s1, cache_type='entity')
            e2 = self.get_cached_embedding(s2, cache_type='entity')

            # 正規化済みなので直接内積を計算
            sim = float(np.dot(e1, e2))
            sim_bonus = max(sim, 0) * 0.3

        except Exception as e:
            if not hasattr(self, '_embedding_error_warned'):
                    self.logger.warning(f"⚠️  Embedding similarity errors detected")
                    self._embedding_error_warned = True
            
        # 3) graph path-based support（kgが与えられた場合）
        path_bonus = 0.0
        if kg is not None:
            try:
            # 2-hop以内でつながってたら評価        
                if kg.has_node(s1) and kg.has_node(s2):
                    length = nx.shortest_path_length(kg, s1, s2)
                    if length <= 2:
                        path_bonus = 0.3 * (1.0 - length / 3.0)  # 近いほど高スコア
            except nx.NetworkXNoPath:
                pass
            except nx.NodeNotFound:
                if self.logger.level <= logging.DEBUG:
                    self.logger.debug(f"Node not found in graph: {s1} or {s2}")
            except Exception as e:
                if self.logger.level <= logging.DEBUG:
                    self.logger.debug(f"Path calc failed ({s1}->{s2}): {type(e).__name__}")

        # 4) 総合
        w = (
            shared_bonus * 0.4 +       # 共有エンティティ
            rel_compatibility * 0.3 +   # 関係の相性（ここに統合済み）
            sim_bonus * 0.2 +           # エンティティ類似度
            path_bonus * 0.1            # パス距離
        )

        return min(w, 1.0)
    
    def _compute_relation_compatibility(self, r1: str, r2: str) -> float:
        """
        関係の相性スコア
        """
        # 正規化（小文字化、アンダースコア統一）
        r1 = r1.lower().replace('-', '_')
        r2 = r2.lower().replace('-', '_')
    # 1. 完全一致
        if r1 == r2:
            return 1.0
    
    # 2. 逆関係のペア（高スコア）
        inverse_pairs = {
            ("cause_of", "caused_by"),
            ("cause_of", "effect_of"), 
            ("part_of", "has_part"),
            ("component_of", "has_component"),
            ("parent_of", "child_of"),
            ("author_of", "written_by"),
            ("owns", "owned_by"),
            ("manages", "managed_by"),
            ("teaches", "taught_by"),
            ("supervises", "supervised_by"),
        }
    
        if (r1, r2) in inverse_pairs or (r2, r1) in inverse_pairs:
            return 0.9
    
    # 3. 関連する関係グループ（中スコア）
        related_groups = [
        # 因果関係グループ
            {
                "cause_of", "caused_by", "leads_to", "results_in", 
                "triggers", "produces", "generates", "effect_of"
            },
        
        # 構成要素グループ
            {
                "part_of", "has_part", "component_of", "has_component",
                "contains", "includes", "consists_of", "comprises"
            },
        
        # 所属グループ
            {
                "member_of", "has_member", "belongs_to", "works_at", 
                "employed_by", "affiliated_with"
            },
        
        # 時間関係グループ
            {
                "before", "after", "during", "precedes", "follows",
                "happens_before", "happens_after"
            },
        
        # 空間関係グループ
            {
                "located_in", "location_of", "near", "adjacent_to",
                "contains", "inside", "outside"
            },
        
        # 属性・性質グループ
            {
                "is_a", "type_of", "instance_of", "has_property",
                "characterized_by", "defined_by"
            },
        
        # 相互作用グループ
            {
                "interacts_with", "collaborates_with", "competes_with",
                "influences", "affected_by"
            },
        ]
    
        for group in related_groups:
            if r1 in group and r2 in group:
                return 0.7
    
    # 4. 同じカテゴリ（動詞の性質で判定）
    # 例: action 系、state 系など
        action_verbs = {
            "creates", "builds", "develops", "produces", "makes",
            "constructs", "designs", "implements", "generates",
            "enables", "powers", "leverages", "accelerates"
            # （ML/AI専門）
            "utilizes", "parameterizes", "fine_tunes", "approximates",
            "encodes", "regularizes", "iterates", "optimizes",
            "traverses", "samples", "augments", "normalizes",
            "quantizes", "distills", "ensembles", "prunes",
            "compresses", "aggregates", "fuses", "aligns",
            "projects", "embeds", "transforms", "adapts",
            
            # CV系
            "detects", "segments", "classifies", "recognizes",
            "extracts", "filters", "convolves", "pools",
            
            # NLP系
            "tokenizes", "parses", "generates_text", "translates",
            "attends_to", "masks", "predicts",
            
            # Graph系
            "propagates", "aggregates_neighbors", "diffuses",
            "clusters", "partitions", "samples_neighbors"
        }
    
        state_verbs = {
            "is", "has", "contains", "includes", "comprises",
            "exists", "represents", "defines", "consists_of",
            "maintains", "preserves", "exhibits", "displays"
        }
    
        relation_verbs = {
            "relates_to", "associated_with", "connected_to",
            "linked_to", "corresponds_to", "depends_on",
            "derived_from", "based_on", "inspired_by"
        }

        # --- 3-4. 計算動詞 ---
        computational_verbs = {
            "computes", "calculates", "evaluates", "measures",
            "estimates", "infers", "learns", "trains",
            "updates", "backpropagates", "forward_passes"
        }
        
        # --- 3-5. 比較動詞 ---
        comparison_verbs = {
            "outperforms", "surpasses", "exceeds", "improves_upon",
            "compares_to", "contrasts_with", "benchmarks_against"
        }

            # カテゴリマッチング
        verb_categories = [
            action_verbs,
            state_verbs,
            relation_verbs,
            computational_verbs,
            comparison_verbs
        ]
                
        for category in verb_categories:
            if r1 in category and r2 in category:
                return 0.5
    
    # 5. 埋め込みフォールバック（低スコア）
        try:
            emb1 = self.relation_embedder.get_text_embedding(r1)
            emb2 = self.relation_embedder.get_text_embedding(r2)
            sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-9)
            return max(0.3, float(sim))
        except Exception:
            return 0.3

    def _update_neo4j_structure(self, kg, graph_store):
        """
        Neo4j更新
        """        
        batch_query  = """
        UNWIND $batch AS row
        MERGE (a:Concept {name: row.source})
        MERGE (b:Concept {name: row.target})
        MERGE (a)-[r:RELATED]->(b)
        ON CREATE SET r.weight = row.weight
        ON MATCH SET r.weight = row.weight
        """
        collector = ErrorCollector(self.logger)

        batch = []
        batch_size = 1000  # 1000件ごとに送信
        
        for s, o, data in kg.edges(data=True):
            weight = data.get('weight', 0.0)

            if weight <= self.config['final_weight_cutoff']: 
                collector.add_skip()
                continue 
            
            # バッチに追加
            batch.append({
                'source': s,
                'target': o,
                'weight': float(weight)
            })
        
            # バッチサイズに達したら送信
            if len(batch) >= batch_size:

                try:
                    graph_store.query(batch_query, {'batch': batch})
                    collector.add_success(count=len(batch))
                
                    self.logger.debug(f"  Sent batch of {len(batch)} edges")
                    batch = []  # バッチをクリア
            
                except Exception as e:
                    collector.add_error(
                        context=f"batch_{len(batch)}_edges",
                        error=e
                    )
                # 失敗したバッチは破棄（または個別処理）
                    batch = []
    
    # 残りのバッチを送信
        if batch:
            try:
                graph_store.query(batch_query, {'batch': batch})
                collector.add_success(count=len(batch))
                self.logger.debug(f"  Sent final batch of {len(batch)} edges")
        
            except Exception as e:
                collector.add_error(
                    context=f"final_batch_{len(batch)}_edges",
                    error=e
                )
        # レポート生成（自動でログ出力）
        collector.report("Neo4j edge update", threshold=0.3)
    # 戻り値も取得可能
        return collector.get_summary()        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Crystal Cluster beta')
    parser.add_argument('json_file', help='Clean documents JSON file')
    parser.add_argument('--neo4j-uri', default='bolt://localhost:7687')
    parser.add_argument('--neo4j-user', default='neo4j')
    parser.add_argument('--neo4j-pass', required=True)
    parser.add_argument('--dual-chunk', action='store_true', help='Enable dual-chunk mode')
    parser.add_argument('--test-query', help='Test retrieval with a query')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument(
        '--llm',
        choices=['gpt-4o-mini', 'claude-3.5-sonnet'],
        default='gpt-4o-mini',
        help='LLM model for triplet extraction'
    )
 # Self-RAG引数
    parser.add_argument(
        '--enable-self-rag',
        action='store_true',
        help='Enable Self-RAG for triplet refinement'
    )
    parser.add_argument(
        '--self-rag-threshold',
        type=float,
        default=0.5,
        help='Self-RAG confidence threshold (default: 0.5)'
    )
    parser.add_argument(
        '--self-rag-refiner',
        choices=['gpt-4o-mini', 'gpt-4o', 'claude-3.5-sonnet'],
        default='gpt-4o',
        help='LLM model for Self-RAG refinement (default: gpt-4o)'
    )
    parser.add_argument(
        '--enable-duplicate-check',
        action='store_true',
        default=True,
        help='Enable duplicate detection (default: enabled)'
    )
    parser.add_argument(
        '--no-duplicate-check',
        dest='enable_duplicate_check',
        action='store_false',
        help='Disable duplicate detection'
    )
    parser.add_argument(
        '--duplicate-similarity',
        type=float,
        default=0.85,
        help='Similarity threshold for fuzzy duplicate detection (default: 0.85)'
    )

    args = parser.parse_args()
    
    print("❄️ Crystal Cluster beta")
    print(f"🤖 LLM: {args.llm}")
    if args.dual_chunk:
        print("🔀 Dual-chunk mode enabled")   
    if args.enable_self_rag:
        print(f"🔄 Self-RAG enabled (refiner: {args.self_rag_refiner})")         
    print("━" * 42)

    # カスタム設定の構築
    custom_config = {}
    
    if args.llm != 'gpt-4o-mini':
        custom_config['llm_model'] = args.llm

    # Self-RAG設定
    if args.enable_self_rag:
        custom_config['enable_self_rag'] = True
        custom_config['self_rag_confidence_threshold'] = args.self_rag_threshold
        custom_config['self_rag_refiner_model'] = args.self_rag_refiner
    else:
        custom_config['enable_self_rag'] = False

    cluster = CrystalCluster(
        log_level=logging.DEBUG if args.debug else logging.INFO,
        use_dual_chunk=args.dual_chunk,
        custom_config=custom_config if custom_config else None
    )
    documents = cluster.load_documents(args.json_file)
    
    graph_store = Neo4jGraphStore(
        username=args.neo4j_user,
        password=args.neo4j_pass,
        url=args.neo4j_uri
    )

    if args.dual_chunk:
        # デュアルチャンクモード
        result = cluster.commit_to_graph_with_retrieval(documents, graph_store)
        
        # テストクエリがあれば検索
        if args.test_query:
            print(f"\n🔍 Testing retrieval: '{args.test_query}'")

            hits = cluster.retrieve(
                result['retrieval_store'], 
                args.test_query, 
                top_k=3,
                chunk_mapping=result.get('chunk_mapping')
            )
            if not hits:
                print("  ⚠️  No results found")
            else:        
            
                for i, (score, doc, graph_chunk_ids) in enumerate(hits, 1):
                    print(f"\n{i}. Score: {score:.3f}")
                    print(f"   Text: {doc.text[:150]}...")
                    
                                # 追加情報も表示
                    if graph_chunk_ids:
                        print(f"   ({len(graph_chunk_ids)} graph chunks linked)")
                        if len(graph_chunk_ids) > 3:
                            print(f"   ... and {len(graph_chunk_ids) - 3} more")
                    else:
                        print(f"   Graph chunks: (none)")

    else:

        cluster.commit_to_graph(documents, graph_store)
    
    print("\n✨ Complete!")