"""
ドキュメント処理クラス
"""
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

from llama_index.core import Document

from shared import ContentLevelDuplicateChecker, load_and_validate_paths

class DocumentProcessor:
    """ドキュメントのロードと前処理を担当"""

    def __init__(self, logger):
        self.logger = logger

        LANGUAGE_CHUNK_CONFIG = {
            'en': {
                'retrieval_chunk_size': 320,
                'retrieval_chunk_overlap': 120,
                'graph_chunk_size': 512,
                'graph_chunk_overlap': 50,
            },
            'ja': {
                # 日本語は1文字あたりの情報密度が高いため、サイズを縮小
                'retrieval_chunk_size': 200,    # 英語の約60%
                'retrieval_chunk_overlap': 80,
                'graph_chunk_size': 350,        # 英語の約70%
                'graph_chunk_overlap': 35,

            },
        }
    
    def _detect_document_language(self, text: str) -> str:
        """
        ドキュメントの主要言語を検出
    
        Args:
            text: ドキュメントテキスト
    
        Returns:
            言語コード ('en', 'ja', 'zh', etc.)
        """
    # 簡易判定（本格的にはlangdetectを使う）
        sample = text[:500]  # 最初の500文字で判定
    
    # 日本語文字（ひらがな・カタカナ・漢字）の割合
        ja_chars = sum(1 for c in sample if '\u3040' <= c <= '\u30ff' or '\u4e00' <= c <= '\u9faf')
        ja_ratio = ja_chars / max(len(sample), 1)
    
        if ja_ratio > 0.3:
            return 'ja'
    
    # 中国語簡体字の判定（必要なら）
    # zh_chars = sum(1 for c in sample if '\u4e00' <= c <= '\u9faf')
    # if zh_chars / max(len(sample), 1) > 0.3:
    #     return 'zh'
    
        return 'en'  # デフォルトは英語


    def _get_language_aware_config(self, documents: List[Document]) -> Dict[str, int]:
        """
        ドキュメント群の言語を検出し、適切なチャンク設定を返す
    
        Args:
            documents: ドキュメントリスト
    
        Returns:
            言語別チャンク設定
        """
    # 最初の数ドキュメントで言語を判定
        sample_size = min(5, len(documents))
        lang_counts = {}
    
        for doc in documents[:sample_size]:
            lang = self._detect_document_language(doc.text)
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
    
    # 最も多い言語を採用
        primary_lang = max(lang_counts.items(), key=lambda x: x[1])[0]
    
        config = LANGUAGE_CHUNK_CONFIG.get(primary_lang, LANGUAGE_CHUNK_CONFIG['en'])
    
        self.logger.info(
            f"🌐 Detected primary language: {primary_lang} "
            f"(chunk_size: graph={config['graph_chunk_size']}, "
            f"retrieval={config['retrieval_chunk_size']})"
        )

        return config
    
# ChunkProcessor.create_dual_documents を更新
    def create_dual_documents(
        self,
        documents: List[Document],
        auto_detect_language: bool = True
    ) -> Tuple[List[Document], List[Document]]:
        """
        デュアルドキュメント生成（言語対応版）
        """
    # 言語検出してチャンクサイズを調整
        if auto_detect_language:
            lang_config = self._get_language_aware_config(documents)
        
        # 一時的に設定を上書き
            original_config = {
                'graph_chunk_size': self.config['graph_chunk_size'],
                'graph_chunk_overlap': self.config['graph_chunk_overlap'],
                'retrieval_chunk_size': self.config['retrieval_chunk_size'],
                'retrieval_chunk_overlap': self.config['retrieval_chunk_overlap'],
            }
        
            self.config.update(lang_config)
    
    # 既存のチャンク処理
        graph_docs = self._create_graph_chunks(documents)
        retrieval_docs = self._create_retrieval_chunks(documents)
    
    # 設定を元に戻す
        if auto_detect_language:
            self.config.update(original_config)

        return graph_docs, retrieval_docs
    
    def load_documents(
        self,
        json_path: str,
        raw_docs: Optional[List[str]] = None,
        path_pickle: Optional[str] = None,
        kg=None,
        enable_duplicate_check: bool = True
    ) -> List[Document]:
        """
        JSON と 生テキスト両方から Document を作る

        Args:
            json_path: JSONファイルのパス
            raw_docs: 生テキストのリスト（オプション）
            path_pickle: パス情報のPickleファイル（オプション）
            kg: ナレッジグラフ（パス情報統合時に必要）
            enable_duplicate_check: 重複チェックを有効化

        Returns:
            Documentのリスト（パス情報が統合されている場合もある）
        """

        if enable_duplicate_check:
            content_checker = ContentLevelDuplicateChecker(
                similarity_threshold=0.85,
                neo4j_store=self.neo4j_store,  
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
                    entity_embeddings=None  # 必要に応じて渡す
                )
                self.logger.info(f"✅ Path information added to {len(documents)} documents")
            else:
                self.logger.warning("Path information could not be loaded, continuing without it")

        return documents

    def augment_documents_with_paths(
        self,
        documents: List[Document],
        path_dicts: List[Dict],
        kg,
        entity_embeddings: Dict[str, Any] = None,
        match_key='question'
    ) -> List[Document]:
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
                                shortest = int(kg.shortest_path_length(s1, s2))
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
            # マッチしなかった場合も元のドキュメントを保持
            augmented.append(doc)

        self.logger.info(
            f"  → Matched {matched_count}/{len(documents)} documents with path information"
        )
        return augmented
    
class DocumentLoader:
    """ドキュメントローダー（簡易版）"""

    def __init__(self, logger=None):
        self.logger = logger
        self.processor = DocumentProcessor(logger)

    def load_from_json(self, json_path: str) -> List[Document]:
        """
        JSONファイルからドキュメントをロード

        Args:
            json_path: JSONファイルのパス

        Returns:
            Documentのリスト
        """
        return self.processor.load_documents(json_path)

    def load_from_text(self, texts: List[str]) -> List[Document]:
        """
        テキストリストからドキュメントをロード

        Args:
            texts: テキストのリスト

        Returns:
            Documentのリスト
        """
        return self.processor.load_documents(
            json_path="",  # 空のJSONパス
            raw_docs=texts
        )
