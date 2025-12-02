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
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any
from contextlib import contextmanager

import requests
from bs4 import BeautifulSoup
from llama_index.core import Document

# 共通モジュール
from shared import (
    HierarchicalLogger,
    setup_logger,
    DataQualityChecker,
    collect_files,
    sanitize_filename,
    clean_text
)

# オプションパッケージ
try:
    import frontmatter
    HAS_FRONTMATTER = True
except ImportError:
    HAS_FRONTMATTER = False
    print("⚠️  python-frontmatter not installed")

# ... 他のオプションパッケージ

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
        
        # Grobid設定
        self.grobid_url = self.config.get('grobid_url', 'http://localhost:8070')
        self.grobid_available = self._check_grobid()
        
        if self.grobid_available:
            self.logger.info(f"✅ Grobid server available at {self.grobid_url}")
        else:
            self.logger.warning("⚠️  Grobid server not available (PDF support disabled)")
        
        self.logger.info("Crystal Geode v1.1 initialized")
    
    # ... crystallize, _parse_* メソッドをコピー
    # ... batch_crystallize メソッドをコピー
    
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
    
    def _save_documents(self, documents: List[Document], output_path: Path):
        """Documents をJSON保存"""
        data = {
            'documents': [
                {'text': doc.text, 'metadata': doc.metadata}
                for doc in documents
            ],
            'count': len(documents),
            'created_at': datetime.now().isoformat()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


# ============================================================
# メイン処理
# ============================================================
if __name__ == "__main__":
    import argparse
    
    geode = argparse.ArgumentGeode(description='Crystal Geode v1.1')
    geode.add_argument('command', choices=['parse', 'batch'])
    geode.add_argument('input_file', help='Input file or directory')
    geode.add_argument('--format', default='auto')
    geode.add_argument('--review-dir', default='./review')
    geode.add_argument('--debug', action='store_true')
    geode.add_argument('--max-workers', type=int, default=4)
    
    args = geode.parse_args()
    
    print("🔮 Crystal Geode v1.1")
    print("━" * 42)
    
    geode_app = CrystalGeode(log_level=logging.DEBUG if args.debug else logging.INFO)
    
    if args.command == 'parse':
        geode_app.parse_and_check(args.input_file, review_dir=args.review_dir)
    
    elif args.command == 'batch':
        # バッチ処理実装
        pass
    
    print("✨ Complete!")