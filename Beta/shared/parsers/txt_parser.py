"""
Text Parser
プレーンテキストのパース
"""
import os
from pathlib import Path
from typing import List
from llama_index.core import Document

from ..text_utils import clean_text, chunk_by_paragraphs

try:
    from langchain_experimental.text_splitter import SemanticChunker
    from langchain_openai import OpenAIEmbeddings
    HAS_SEMANTIC_CHUNKER = True
except ImportError:
    HAS_SEMANTIC_CHUNKER = False

def parse_txt(txt_path: Path, config=None, logger=None) -> List[Document]:
    """プレーンテキストをパース"""
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            content = f.read()

        content = clean_text(content)

        # APIキーの取得（優先順位）
        api_key = (
            (config or {}).get('openai_api_key') or  # 1. 設定ファイル/引数
            os.environ.get('OPENAI_API_KEY')      # 2. 環境変数（.zshrc）
        )

        # セマンティックチャンキング
        if HAS_SEMANTIC_CHUNKER and api_key:
            try:
                os.environ['OPENAI_API_KEY'] = api_key

                if logger:
                    logger.info("🤖 Using SemanticChunker")
                embeddings = OpenAIEmbeddings()
                splitter = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")
                chunks = splitter.split_text(content)
                if logger:
                    logger.info(f"✅ {len(chunks)} semantic chunks created")
            except Exception as e:
                if logger:
                    logger.warning(f"⚠️  SemanticChunker failed: {e}")
                chunks = chunk_by_paragraphs(content)
        else:
            if HAS_SEMANTIC_CHUNKER and not api_key:
                if logger:
                    logger.info("ℹ️  OpenAI API key not provided, using basic chunking")
            chunks = chunk_by_paragraphs(content)

        documents = []
        for i, chunk in enumerate(chunks):
            documents.append(Document(
                text=chunk,
                metadata={
                    'title': txt_path.stem,
                    'authors': 'Unknown',
                    'section': f"Chunk {i+1}",
                    'section_index': i,
                    'source_format': 'txt'
                }
            ))

        metadata = {'title': txt_path.stem, 'authors': ['Unknown']}
        if logger:
            logger.info(f"Text parsed: {len(documents)} chunks")
        return documents, metadata

    except Exception as e:
        if logger:
            logger.error(f"Text parse failed: {e}")
        return [], {'title': txt_path.stem, 'authors': ['Unknown']}