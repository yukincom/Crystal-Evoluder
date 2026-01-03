"""
Crystal Geode
Knowledge Crystallization System - 

パース → Quality Check → JSON出力
"""

# ============================================================
# インポート
# ============================================================
import os
import re
import logging
import concurrent.futures
import time
import argparse
import json
import requests

from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any, Set
from sentence_transformers import SentenceTransformer

from llama_index.core import Document
from llama_index.graph_stores.neo4j import Neo4jGraphStore

from grobid_client.grobid_client import GrobidClient

# 共通モジュール
from shared.logger import (
    HierarchicalLogger,
    setup_logger,
)
from shared.quality_checker import DataQualityChecker
from shared.duplicate_checker import ProvenanceManager
from shared.utils.hashing import compute_file_hash

from shared.text_utils import (
    clean_text,
    chunk_by_paragraphs,
)
from shared.file_utils import (
    collect_files,
#    sanitize_filename,
#    detect_encoding,
    detect_format
)
from shared.parsers import parse_tei, parse_markdown, parse_txt, parse_docx, parse_html, parse_pdf

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ============================================================
# メインクラス
# ============================================================
class CrystalGeode:
    """Crystal Geode - 文書パース専用"""
    
    def __init__(self, config: Optional[Dict] = None, log_level: int = logging.INFO):
        self.config = config or {}
        self.crystal = None
        self.metadata = {}
        self.logger = setup_logger('CrystalGeode', log_level)
        self.hlogger = HierarchicalLogger(self.logger)
       
        # BGE-M3共有キャッシュをここで初期化（1回だけロード）
        from .model.embed import ensure_bge_m3, EmbeddingCache 
        try:
            self.embed_model = ensure_bge_m3()  # 自動ロード
            self.embedding_cache = EmbeddingCache(embed_model=self.embed_model)
            self.logger.info("✅ BGE-M3 shared embedding cache initialized")
        except Exception as e:
            self.logger.warning(f"⚠️ BGE-M3 initialization failed: {e}. Embedding features disabled.")
            self.embedding_cache = None

        # Grobid設定
        self.grobid_url = self.config.get('grobid_url', 'http://localhost:8070')
        self.grobid_available = self._check_grobid()
        self.grobid_client = None
        
        if self.grobid_available:
            self._init_grobid()
            self.logger.info(f"✅ Grobid server available at {self.grobid_url}")
        else:
            self.logger.warning("⚠️  Grobid server not available (PDF support disabled)")
        
        # 重複チェック設定
        self.enable_duplicate_check = config.get('enable_duplicate_check', True)
        self.enable_provenance = config.get('enable_provenance', True)
        
        # キャッシュ
        self.file_hash_cache: Set[str] = set()  # ファイルハッシュキャッシュ
        self.doc_hash_cache: Set[str] = set()   # 後でCrystalClusterに渡す
        
        # Provenance Manager
        if self.enable_provenance:
            
            self.provenance_mgr = ProvenanceManager(logger=self.logger)
        else:
            self.provenance_mgr = None
        
        # Neo4j接続（オプション）
        self.neo4j_store = self._init_neo4j_connection()
        
        # 起動時に既存ハッシュをロード
        if self.enable_duplicate_check and self.neo4j_store:
            self.load_file_hashes_from_neo4j()
        
        self.logger.info("Crystal Geode bata")
    
    def _check_grobid(self) -> bool:
        """Grobidサーバーが起動しているか確認"""
        try:
            
            response = requests.get(f"{self.grobid_url}/api/isalive", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def _init_grobid(self):
        """Grobidクライアント初期化"""
        
        self.grobid_client = GrobidClient(
            grobid_server=self.config.get('grobid_url', 'http://localhost:8070'),
            timeout=120
        )
        self.logger.info("✅ Grobid client initialized")

    def _init_neo4j_connection(self) -> Optional[Any]:
        """Neo4j接続を初期化（オプション）"""
        neo4j_config = self.config.get('neo4j', {})
        
        if not neo4j_config.get('enabled', False):
            self.logger.info("Neo4j integration disabled")
            return None
        
        try:           
            store = Neo4jGraphStore(
                username=neo4j_config.get('username', 'neo4j'),
                password=neo4j_config.get('password'),
                url=neo4j_config.get('url', 'bolt://localhost:7687')
            )
            
            store.query("RETURN 1")
            self.logger.info("✅ Neo4j connection established")
            
            return store
        
        except Exception as e:
            self.logger.warning(f"⚠️  Neo4j connection failed: {type(e).__name__}")
            return None

    def load_file_hashes_from_neo4j(self):
        """Neo4jから既存のファイルハッシュを一括取得"""
        if not self.neo4j_store:
            return
        
        query = """
        MATCH (d:Document)
        WHERE d.file_hash IS NOT NULL
        RETURN d.file_hash AS file_hash
        """
        
        try:
            results = self.neo4j_store.query(query)
            
            for record in results:
                self.file_hash_cache.add(record['file_hash'])
            
            self.logger.info(
                f"📥 Loaded {len(self.file_hash_cache)} existing file hashes from Neo4j"
            )
        
        except Exception as e:
            self.logger.warning(f"Failed to load file hashes from Neo4j: {e}")

    def check_file_duplicate_and_provenance(
        self,
        file_path: str
    ) -> Optional[Dict[str, Any]]:
        """
        ファイルレベルの重複チェック + Provenance生成
        
        Args:
            file_path: ファイルパス
        
        Returns:
            Provenance辞書（重複の場合はNone）
        """
        
        # ファイルハッシュを計算
        try:
            file_hash = compute_file_hash(file_path, algorithm='sha256')
        except Exception as e:
            self.logger.error(f"Failed to compute hash for {file_path}: {e}")
            return None
        
        # 重複チェック
        if self.enable_duplicate_check and file_hash in self.file_hash_cache:
            self.logger.warning(
                f"⊗ Duplicate file skipped: {file_path} "
                f"(hash: {file_hash[:8]}...)"
            )
            return None
        
        # Provenance生成
        if self.enable_provenance and self.provenance_mgr:
            # ファイルタイプを判定
            source_type = Path(file_path).suffix.lstrip('.')
            
            provenance = self.provenance_mgr.create_provenance(
                source_path=file_path,
                source_type=source_type,
                file_hash=file_hash,
                metadata={
                    'parsed_by': 'crystal_geode',
                    'version': 'beta'
                }
            )
            
            # キャッシュに追加
            self.file_hash_cache.add(file_hash)
            
            self.logger.info(
                f"✓ New file registered: {Path(file_path).name} "
                f"(hash: {file_hash[:8]}...)"
            )
            
            return provenance
        
        else:
            # Provenance無効時は簡易辞書を返す
            self.file_hash_cache.add(file_hash)
            return {
                'file_hash': file_hash,
                'source_path': file_path,
                'source_type': Path(file_path).suffix.lstrip('.')
            }

    def crystallize(self, input_path: str, format: str = 'auto') -> List[Document]:
        """結晶化: 入力ファイルをパース"""
        self.logger.info("Crystallizing knowledge structure...")

        input_path = Path(input_path).expanduser()

        if not input_path.exists():
            raise FileNotFoundError(f"File not found: {input_path}")
        
        # 1. ファイルレベル重複チェック + Provenance生成
        provenance = self.check_file_duplicate_and_provenance(str(input_path))
    
        if provenance is None:
            # 重複ファイル → スキップ
            self.logger.warning(f"⊗ Skipping duplicate file: {input_path.name}")
            return []

        if format == 'auto':
            format = detect_format(str(input_path))

        parsers = {
            'tei': parse_tei,
            'markdown': parse_markdown,
            'txt': parse_txt,
            'docx': parse_docx,
            'html': parse_html,
            'pdf': parse_pdf
        }
        try:
            if format not in parsers:
                raise ValueError(f"Unsupported format: {format}")

        # パーサー呼び出し
            if format == 'pdf':
                documents, metadata = parsers[format](input_path, grobid_client=self.grobid_client, logger=self.logger)
            elif format == 'txt':
                documents, metadata = parsers[format](input_path, config=self.config, logger=self.logger)
            else:
                documents, metadata = parsers[format](input_path, logger=self.logger)

        except Exception as e:
            self.logger.error(f"❌ Failed to parse {input_path.name}: {e}")
            raise
    
       # 4. Provenance情報を各Documentに注入
        for doc in documents:
        # 既存のmetadataを保持しつつProvenanceを追加
            doc.metadata.update({
               # Provenance情報
               'file_hash': provenance['file_hash'],
               'source_path': provenance['source_path'],
                'source_name': provenance['source_name'],
                'source_type': provenance['source_type'],
                'ingested_at': provenance['ingested_at'],
                'version': provenance['version'],
                'pipeline_stage': 'geode_parse',
            
            # パース情報
                'parsed_at': datetime.now().isoformat(),
                'parsed_by': 'crystal_geode',
                'format': format,
            })

        self.crystal = documents
        self.metadata = metadata
        self.logger.info(
            f"✨ Crystal structure stabilized: {len(documents)} nodes "
            f"(hash: {provenance['file_hash'][:8]}...)"
        )

        return documents

    def batch_crystallize(
        self,
        input_dir: str,
        patterns: List[str] = None,
        max_workers: int = 4,
        fail_fast: bool = False,
        output_json: str = None
    ) -> Dict[str, Any]:
        """
        ディレクトリ内のファイルを一括処理
    
        Args:
            input_dir: 入力ディレクトリ
            patterns: ファイルパターン（例: ['*.md', '*.pdf']）
           max_workers: 並列処理数
            fail_fast: True=最初のエラーで停止, False=全部試す
           output_json: JSON出力パス（オプション）
    
        Returns:
            {
                'success': {filepath: [Document, ...], ...},
                'failed': [(filepath, error_msg), ...],
                'skipped': [filepath, ...],  # 重複スキップ
                'stats': {...}
            }
        """
        self.logger.info(f"Starting batch crystallization: {input_dir}")
    
        # ========================================
            # 1. ファイル収集
        # ========================================
        if patterns is None:
            patterns = ['*.md', '*.docx', '*.html', '*.txt', '*.tei.xml']
        
            # Grobid有効時のみPDFを追加
            if self.grobid_available:
                patterns.append('*.pdf')
                self.logger.info("✅ PDF processing enabled")
            else:
                self.logger.warning("⚠️  PDF skipped (Grobid server not available)")
    
        files = collect_files(input_dir, patterns=patterns)
        self.logger.info(f"Found {len(files)} files")
    
        # ========================================
        # 2. 結果格納
        # ========================================
        results = {
            'success': {},
            'failed': [],
            'skipped': [],  # 重複でスキップされたファイル
            'stats': {}
        }
    
    # ========================================
    # 3. 並列処理
    # ========================================
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {}
        
            for f in files:
                future = executor.submit(self._crystallize_with_retry, str(f))
                future_to_file[future] = f
        
        # プログレスバー
            if HAS_TQDM:
                iterator = tqdm(
                    concurrent.futures.as_completed(future_to_file),
                    total=len(future_to_file),
                    desc="Crystallizing"
                )
            else:
                iterator = concurrent.futures.as_completed(future_to_file)
        
        # 結果収集
            for future in iterator:
                file_path = future_to_file[future]
            
                try:
                    docs = future.result(timeout=300)
                
                # 空リスト = 重複スキップ
                    if not docs:
                        results['skipped'].append(str(file_path))
                    else:
                        results['success'][str(file_path)] = docs
            
                except concurrent.futures.TimeoutError:
                    error_msg = "Processing timeout (>5min)"
                    results['failed'].append((str(file_path), error_msg))
                    self.logger.error(f"❌ Failed: {file_path.name} - {error_msg}")
                
                    if fail_fast:
                        executor.shutdown(wait=False)
                        break
            
                except Exception as e:
                    error_msg = str(e)
                    results['failed'].append((str(file_path), error_msg))
                    self.logger.error(f"❌ Failed: {file_path.name} - {error_msg}")
                
                    if fail_fast:
                        executor.shutdown(wait=False)
                        break
    
    # ========================================
    # 4. 統計情報
    # ========================================
        results['stats'] = {
            'total': len(files),
            'success': len(results['success']),
            'failed': len(results['failed']),
            'skipped': len(results['skipped']),
            'total_documents': sum(len(docs) for docs in results['success'].values())
        }
    
    # ========================================
    # 5. サマリー
    # ========================================
        self.logger.info(
            f"\n{'='*60}\n"
            f"✅ Batch Complete!\n"
            f"   Success: {results['stats']['success']}\n"
            f"   Failed:  {results['stats']['failed']}\n"
            f"   Skipped: {results['stats']['skipped']} (duplicates)\n"
            f"   Total documents: {results['stats']['total_documents']}\n"
            f"{'='*60}"
        )
    
    # ========================================
    # 6. 失敗レポート保存
    # ========================================
        if results['failed']:
            self._save_error_report(results['failed'], input_dir)
    
    # ========================================
    # 7. JSON出力（オプション）
    # ========================================
        if output_json:
            self._save_batch_results(results, output_json)
    
        return results

    def _crystallize_with_retry(
        self,
        file_path: str,
        max_retries: int = 3
    ) -> List[Document]:
        """
        リトライ機構付き crystallize
    
        Args:
            file_path: ファイルパス
            max_retries: 最大リトライ回数
    
        Returns:
            Documentのリスト（重複の場合は空リスト）
        """
        for attempt in range(max_retries):
            try:
                return self.crystallize(file_path)
        
            except Exception as e:
                if attempt == max_retries - 1:
                # 最後のリトライで失敗 → 例外を上げる
                    raise
            
                self.logger.warning(
                    f"⚠️  Retry {attempt+1}/{max_retries}: {Path(file_path).name}\n"
                    f"   Error: {e}"
                )
                time.sleep(2 ** attempt)  # exponential backoff

    def parse_and_check(self, input_file: str, review_dir: str = './review') -> Dict[str, Any]:
        """パース → Quality Check → 停止"""

        # 1. Crystallize
        with self.hlogger.section("Parsing"):
            self.crystallize(input_file)

        # 2. Quality Check
        with self.hlogger.section("Quality Check"):
            checker = DataQualityChecker(logger=self.logger)
            result = checker.check_documents(self.crystal, output_dir=review_dir)

        # 3. 結果保存
        with self.hlogger.section("Saving Results"):
            # Clean データ
            clean_path = Path(review_dir) / 'clean_documents.json'
            self._save_documents(result['clean'], clean_path)

            # 統計情報
            stats_path = Path(review_dir) / 'stats.json'
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(result['stats'], f, indent=2)

        self.logger.info(
            f"\n{'='*60}\n"
            f"✅ Parsing Complete!\n"
            f"   Clean documents: {result['stats']['clean']}\n"
            f"   Flagged documents: {result['stats']['flagged']}\n"
            f"\n"
            f"📁 Output:\n"
            f"   Clean data: {clean_path}\n"
            f"   Review queue: {review_dir}/review_queue.csv\n"
            f"\n"
            f"▶️  Next Step:\n"
            f"   1. Review: {review_dir}/review_queue.csv\n"
            f"   2. Run: crystal_committer.py {clean_path}\n"
            f"{'='*60}"
        )

        return result

    def save_parsed_data(self, output_path: str):
        """パース結果をJSONで保存"""
        if not self.crystal:
            raise ValueError("No data to save. Run crystallize() first")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'metadata': self.metadata,
            'documents': [
                {
                    'text': doc.text,
                    'metadata': doc.metadata
                }
                for doc in self.crystal
            ],
            'created_at': datetime.now().isoformat()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"💾 Saved parsed data: {output_path}")
        

    def _save_error_report(
        self,
        failed: List[Tuple[str, str]],
        base_dir: str
    ):
        """エラーレポート保存"""
        report_path = Path(base_dir) / 'crystal_geode_errors.json'
    
        report = {
            'timestamp': datetime.now().isoformat(),
            'failed_count': len(failed),
            'failed_files': [
                {'file': filepath, 'error': error}
                for filepath, error in failed
            ]
        }
    
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
    
        self.logger.info(f"📋 Error report saved: {report_path}")

    def _save_batch_results(
        self,
        results: Dict[str, Any],
        output_path: str
    ):
        """バッチ処理結果をJSON保存"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Documentをシリアライズ可能な形式に変換
        serializable_results = {
            'metadata': {
                'batch_timestamp': datetime.now().isoformat(),
                'stats': results['stats']
            },
            'documents': {}
        }
    
        for filepath, docs in results['success'].items():
            serializable_results['documents'][filepath] = [
                {
                    'text': doc.text,
                    'metadata': doc.metadata
                }
                for doc in docs
            ]
    
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
    
        self.logger.info(f"💾 Batch results saved: {output_path}")

# ============================================================
# メイン処理
# ============================================================
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Crystal Geode Beta')
    
    parser.add_argument('command', choices=['parse', 'batch'])
    parser.add_argument('input_file', help='Input file or directory')
    parser.add_argument('--format', default='auto')
    parser.add_argument('--review-dir', default='./review')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--max-workers', type=int, default=4)
    
    args = parser.parse_args()
    
    print("🌋 Crystal Geode Beta")
    print("━" * 42)
    
    app = CrystalGeode(log_level=logging.DEBUG if args.debug else logging.INFO)
    
    if args.command == 'parse':
        app.parse_and_check(args.input_file, review_dir=args.review_dir)
    
    elif args.command == 'batch':
        # バッチ処理実装
        app.batch_crystallize(args.input_file, max_workers=args.max_workers)
    
    print("✨ Complete!")