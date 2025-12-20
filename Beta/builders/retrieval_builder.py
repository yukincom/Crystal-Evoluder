"""
検索ストア構築クラス
"""
import numpy as np
from typing import List, Dict, Any, Tuple

from llama_index.core import Document

from shared import ErrorCollector


class RetrievalBuilder:
    """検索ストアの構築と検索を担当"""

    def __init__(self, embed_model, logger):
        self.embed_model = embed_model
        self.logger = logger

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