"""
PDF Parser
PDF文書のパース（GROBID使用）
"""
import time
import requests
from pathlib import Path
from typing import List
from llama_index.core import Document

def parse_pdf(pdf_path: Path, grobid_client=None, logger=None) -> List[Document]:
    """PDF → GROBID → TEI → Document"""
    if not grobid_client:
        if logger:
            logger.error("Grobid client not available")
        raise RuntimeError("Grobid client not available")

    try:
        # TEI出力先
        tei_path = pdf_path.with_suffix('.tei.xml')

        if tei_path.exists():
            if logger:
                logger.info(f"Using cached TEI: {tei_path.name}")
            # TEIをパース（tei_parserを使用）
            from .tei_parser import parse_tei
            return parse_tei(tei_path, logger)[0]  # documentsのみ返す

        # Grobid処理
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if logger:
                    logger.info(
                        f"Processing PDF with Grobid: {pdf_path.name}"
                        f"(attempt {attempt+1}/{max_retries})"
                    )

                grobid_client.process(
                'processFulltextDocument',
                str(pdf_path),
                output=str(tei_path.parent),
                consolidate_citations=True,
                consolidate_header=True,
                include_raw_citations=False,
                include_raw_affiliations=False,
                tei_coordinates=False,
                segment_sentences=False
                )
                # 既存のTEIパーサーで処理
                if tei_path.exists():
                    from .tei_parser import parse_tei
                    return parse_tei(tei_path, logger)[0]
                else:
                    raise FileNotFoundError(f"Grobid output not found: {tei_path}")

            except requests.exceptions.Timeout:
                if logger:
                    logger.warning(f"⏱️  Timeout on attempt {attempt+1}")
                if attempt == max_retries - 1:
                    raise RuntimeError("Grobid processing timeout")
                time.sleep(2 ** attempt)  # exponential backoff

            except requests.exceptions.ConnectionError:
                if logger:
                    logger.error("❌ Grobid server unreachable")
                raise RuntimeError(
                    "Grobid server not responding. Check:\n"
                    "1. Server is running: docker ps\n"
                    "2. URL is correct: http://localhost:8070\n"
                    "3. Restart: docker restart <container_id>"
                )

            except Exception as e:
                if logger:
                    logger.error(f"❌ Grobid processing failed: {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)

    except Exception as e:
        if logger:
            logger.error(f"PDF parse failed: {e}")
        return []