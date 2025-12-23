# crystal_cluster/process_pdf_with_figures.py

from pipelines.figure_pipeline import FigurePipeline
from llama_index.graph_stores.neo4j import Neo4jGraphStore
import logging

def process_pdf_with_figures(pdf_path: str, neo4j_store: Neo4jGraphStore):
    """
    PDFを処理して、テキストチャンク + 図表解析を実行
    """
    logger = logging.getLogger(__name__)

    # 1. 既存のテキストチャンク処理（既存コード）
    chunks = your_existing_chunker.process(pdf_path)

    # 2. 図表パイプライン初期化
    fig_pipeline = FigurePipeline(
        ollama_url="http://localhost:11434",
        vision_model="granite3.2-vision",
        dpi=200,  # Mac Siliconでも高速
        use_cache=True,
        logger=logger
    )

    # 3. 図表処理
    fig_results = fig_pipeline.process_pdf(
        pdf_path=pdf_path,
        chunks=chunks,
        graph_store=neo4j_store
    )

    logger.info(f"Figures: {fig_results}")

    return {
        'chunks': len(chunks),
        'figures': fig_results
    }