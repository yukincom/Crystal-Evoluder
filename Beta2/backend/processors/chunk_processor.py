"""
チャンク処理クラス
"""
import hashlib
from typing import List, Tuple

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter


class ChunkProcessor:
    """ドキュメントのチャンク分割を担当"""

    def __init__(self, config: dict, logger):
        self.config = config
        self.logger = logger

    def create_dual_documents(
        self,
        documents: List[Document]
    ) -> Tuple[List[Document], List[Document]]:
        """
        既存のDocumentから Graph用 と Retrieval用 の2種類を作る

        Args:
            documents: load_documents() で作成したDocumentリスト

        Returns:
            (graph_docs, retrieval_docs)
        """
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