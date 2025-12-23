"""
メイン統合クラス - Crystal Cluster
"""
import numpy as np
import networkx as nx
from typing import List, Dict, Any, Optional, Tuple, Iterable
from collections import defaultdict
import itertools

from llama_index.core import Document
from llama_index.graph_stores.neo4j import Neo4jGraphStore


from config.config_manager import _load_config
from shared import setup_logger, HierarchicalLogger
from shared.ai_router import AIRouter
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

            # AIルーティング設定
            'ai_routing': {
                'mode': 'api',  # 'api' or 'ollama'
                'ollama_url': 'http://localhost:11434',
                'api_key': None  # 環境変数から取得も可
            }
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

        from ..model import ensure_bge_m3
        self.embed_model = ensure_bge_m3()

        # AI Router初期化
        self.ai_router = AIRouter(config=self.config, logger=self.logger)

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
    
    def _extract_entity_contexts(
        self, 
        documents: List[Document], 
        window_sentences: int = 1
    ) -> Dict[str, List[str]]:
        """
        documentsからエンティティとそのコンテキスト文を抽出
        
        Args:
            documents: Documentリスト
            window_sentences: entity が登場する文の前後に何文取るか
        
        Returns:
            {entity_name: [context_str1, context_str2, ...]}
        """
        import re
        
        entity_contexts = defaultdict(list)
        
        for doc in documents:
            text = getattr(doc, "text", "") or ""
            # 文分割（日本語・英語対応）
            pieces = [s.strip() for s in re.split(r'(?<=[。．.!?])\s+', text) if s.strip()]

            # triples が metadata にあればエンティティソースとして使う
            triples = doc.metadata.get("triples", [])
            candidate_entities = set()
            for s, r, o in triples:
                candidate_entities.add(str(s))
                candidate_entities.add(str(o))

            # metadata の 'keywords' も候補に追加
            for kw in doc.metadata.get("keywords", []) if doc.metadata.get("keywords") else []:
                candidate_entities.add(str(kw))

            if not candidate_entities:
                continue

            # 各 entity の登場文を探して前後 window_sentences 文を取る
            for ent in candidate_entities:
                ent_norm = str(ent).strip()
                if not ent_norm:
                    continue
                
                for i, sent in enumerate(pieces):
                    if ent_norm.lower() in sent.lower():
                        start = max(0, i - window_sentences)
                        end = min(len(pieces), i + window_sentences + 1)
                        ctx = " ".join(pieces[start:end])
                        entity_contexts[ent_norm].append(ctx)
                
                # 出現がなければ文書冒頭をフォールバック
                if len(entity_contexts[ent_norm]) == 0:
                    entity_contexts[ent_norm].append(pieces[0] if pieces else text[:200])

        return entity_contexts

    def _batch_embed_texts(
        self, 
        texts: Iterable[str], 
        batch_size: int = 32
    ) -> List[np.ndarray]:
        """
        self.embed_modelを用いてテキストをバッチ埋め込み
        
        Args:
            texts: テキストのイテレータ
            batch_size: バッチサイズ
        
        Returns:
            埋め込みベクトルのリスト
        """
        embeddings = []
        batch = []
        
        for t in texts:
            batch.append(t)
            if len(batch) >= batch_size:
                # llama_indexの埋め込みモデルに対応
                if hasattr(self.embed_model, 'get_text_embedding_batch'):
                    embs = self.embed_model.get_text_embedding_batch(batch)
                elif hasattr(self.embed_model, 'get_text_embedding'):
                    embs = [self.embed_model.get_text_embedding(t) for t in batch]
                else:
                    embs = self.embed_model.embed_batch(batch)
                
                for e in embs:
                    embeddings.append(np.array(e, dtype=np.float32))
                batch = []
        
        # 残りのバッチ処理
        if batch:
            if hasattr(self.embed_model, 'get_text_embedding_batch'):
                embs = self.embed_model.get_text_embedding_batch(batch)
            elif hasattr(self.embed_model, 'get_text_embedding'):
                embs = [self.embed_model.get_text_embedding(t) for t in batch]
            else:
                embs = self.embed_model.embed_batch(batch)
            
            for e in embs:
                embeddings.append(np.array(e, dtype=np.float32))
        
        return embeddings

    def _detect_language_simple(self, text: str) -> str:
        """
        簡易言語判定（日本語・中国語・韓国語・英語対応）
        
        フォールバック用。langdetect がインストールされていれば
        _detect_language_accurate() が優先される。
        
        Args:
            text: 判定するテキスト
        
        Returns:
            言語コード ('en', 'ja', 'zh', 'ko', 'other', 'unknown')
        """
        if not text:
            return "unknown"
        
        sample = text[:300]
        
        # 文字種別カウント
        hiragana = sum(1 for c in sample if '\u3040' <= c <= '\u309f')
        katakana = sum(1 for c in sample if '\u30a0' <= c <= '\u30ff')
        kanji = sum(1 for c in sample if '\u4e00' <= c <= '\u9faf')
        hangul = sum(1 for c in sample if '\uac00' <= c <= '\ud7af')
        ascii_chars = sum(1 for c in sample if ord(c) < 128)
        
        total = max(len(sample), 1)
        
        # 日本語判定（ひらがな・カタカナが多い）
        if (hiragana + katakana) / total > 0.15:
            return "ja"
        
        # 韓国語判定
        if hangul / total > 0.3:
            return "ko"
        
        # 中国語判定（漢字のみで日本語的な文字がない）
        if kanji / total > 0.3 and (hiragana + katakana) / total < 0.05:
            return "zh"
        
        # 英語判定
        if ascii_chars / total > 0.7:
            return "en"
        
        return "other"

    def _detect_language_accurate(self, text: str) -> str:
        """
        高精度言語判定（langdetect使用）
        
        インストール: pip install langdetect
        
        Args:
            text: 判定するテキスト
        
        Returns:
            言語コード
        """
        try:
            from langdetect import detect
            return detect(text[:500])
        except ImportError:
            self.logger.debug("langdetect not installed, using simple detection")
            return self._detect_language_simple(text)
        except Exception as e:
            self.logger.debug(f"Language detection failed: {e}, falling back to simple")
            return self._detect_language_simple(text)

    def multilingual_entity_linking(
        self,
        kg: nx.Graph,
        documents: List[Document],
        *,
        window_sentences: int = 1,
        batch_size: int = 32,
        same_lang_threshold: float = 0.90,
        cross_lang_threshold: float = 0.85,
        preserve_original_triples: bool = True,
        use_accurate_detection: bool = True
    ) -> Dict[str, Any]:
        """
        多言語対応のEntity Linkingを実行
        
        Args:
            kg: NetworkXグラフ
            documents: Documentのリスト
            window_sentences: コンテキスト抽出の前後文数
            batch_size: 埋め込みのバッチサイズ
            same_lang_threshold: 同言語比較の閾値
            cross_lang_threshold: 異言語比較の閾値
            preserve_original_triples: 元のトリプルを保存するか
            use_accurate_detection: langdetectを使うか
        
        Returns:
            サマリー辞書
        """
        self.logger.info("🌐 Starting multilingual entity linking...")
        
        # 1) entity -> contexts を集める
        entity_contexts = self._extract_entity_contexts(documents, window_sentences)
        
        # 2) 各 entity に代表文脈を作る
        entity_representations = {}
        for ent, ctxs in entity_contexts.items():
            rep = " ".join(ctxs[:2])  # 最大2文をつなげる
            entity_representations[ent] = rep
        
        # 3) 埋め込みを作る（バッチ）
        ents = list(entity_representations.keys())
        reps = [entity_representations[e] for e in ents]
        
        self.logger.info(f"  Embedding {len(reps)} entity contexts (batch_size={batch_size})")
        emb_list = self._batch_embed_texts(reps, batch_size=batch_size)
        
        # 4) 言語判定
        detect_func = self._detect_language_accurate if use_accurate_detection else self._detect_language_simple
        lang_map = {ent: detect_func(entity_representations[ent]) for ent in ents}
        
        # 5) コサイン類似度計算とマージ
        def cosine(a: np.ndarray, b: np.ndarray) -> float:
            na = np.linalg.norm(a)
            nb = np.linalg.norm(b)
            if na < 1e-9 or nb < 1e-9:
                return 0.0
            return float(np.dot(a, b) / (na * nb))
        
        n = len(ents)
        merged_groups = []
        visited = set()
        
        for i in range(n):
            if ents[i] in visited:
                continue
            group = [ents[i]]
            visited.add(ents[i])
            
            for j in range(i + 1, n):
                if ents[j] in visited:
                    continue
                
                lang_i = lang_map[ents[i]]
                lang_j = lang_map[ents[j]]
                score = cosine(emb_list[i], emb_list[j])
                thresh = same_lang_threshold if lang_i == lang_j else cross_lang_threshold
                
                if score >= thresh:
                    group.append(ents[j])
                    visited.add(ents[j])
            
            merged_groups.append(group)
        
        # 6) canonical name を決定
        name_to_canonical = {}
        canonical_stats = []
        
        for group in merged_groups:
            if len(group) == 1:
                name_to_canonical[group[0]] = group[0]
                canonical_stats.append((group[0], 1))
                continue
            
            # 出現頻度ベースで代表を選ぶ
            counts = {g: 0 for g in group}
            for doc in documents:
                txt = (getattr(doc, "text", "") or "").lower()
                for g in group:
                    if g.lower() in txt:
                        counts[g] += 1
            
            canonical = max(group, key=lambda x: (counts.get(x, 0), -len(x)))
            for g in group:
                name_to_canonical[g] = canonical
            canonical_stats.append((canonical, len(group)))
        
        # 7) NetworkX ノードのマージ
        for old, canon in name_to_canonical.items():
            if old == canon:
                continue
            
            if not kg.has_node(canon):
                if kg.has_node(old):
                    kg.add_node(canon, **kg.nodes[old])
                else:
                    kg.add_node(canon)
            
            if kg.has_node(old):
                # エッジのリダイレクト
                for u, v, data in list(kg.in_edges(old, data=True)):
                    if not kg.has_edge(u, canon):
                        kg.add_edge(u, canon, **data)
                    else:
                        kg[u][canon]['weight'] = max(
                            kg[u][canon].get('weight', 0.0), 
                            data.get('weight', 0.0)
                        )
                
                for u, v, data in list(kg.out_edges(old, data=True)):
                    if not kg.has_edge(canon, v):
                        kg.add_edge(canon, v, **data)
                    else:
                        kg[canon][v]['weight'] = max(
                            kg[canon][v].get('weight', 0.0), 
                            data.get('weight', 0.0)
                        )
                
                try:
                    kg.remove_node(old)
                except Exception:
                    pass
        
        # 8) Document metadata (triples) の更新
        for doc in documents:
            triples = doc.metadata.get("triples", [])
            new_triples = []
            for s, r, o in triples:
                s2 = name_to_canonical.get(s, s)
                o2 = name_to_canonical.get(o, o)
                new_triples.append((s2, r, o2))
            
            if preserve_original_triples:
                doc.metadata.setdefault("_original_triples", doc.metadata.get("triples", []).copy())
            doc.metadata["triples"] = new_triples
        
        summary = {
            "num_entities_before": n,
            "num_groups": len(merged_groups),
            "merged_count": sum(1 for g in merged_groups if len(g) > 1),
            "canonical_stats": canonical_stats[:10],  # 最初の10個だけ
        }
        
        self.logger.info(
            f"✅ Multilingual EL completed: "
            f"{summary['num_groups']} groups, "
            f"{summary['merged_count']} merges"
        )
        
        return summary