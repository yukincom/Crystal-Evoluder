"""
Data Provenance（データ出所・履歴管理）
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging


class ProvenanceManager:
    """データの出所と履歴を管理"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def create_provenance(
        self,
        source_path: str,
        source_type: str,
        file_hash: str,
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Provenance情報を作成
        
        Args:
            source_path: 元ファイルのパス
            source_type: ファイルタイプ（pdf, json, md, etc.）
            file_hash: ファイルのハッシュ値
            metadata: 追加メタデータ
        
        Returns:
            Provenance辞書
        """
        provenance = {
            'source_path': str(source_path),
            'source_type': source_type,
            'file_hash': file_hash,
            'ingested_at': datetime.now().isoformat(),
            'version': self._extract_version(source_path),
            'file_size': Path(source_path).stat().st_size if Path(source_path).exists() else None,
            'metadata': metadata or {}
        }
        
        return provenance
    
    def _extract_version(self, source_path: str) -> Optional[str]:
        """
        ファイル名からバージョンを抽出
        
        例:
            document_v1.2.3.pdf → "1.2.3"
            paper_2024-01-15.pdf → "2024-01-15"
        """
        path = Path(source_path)
        name = path.stem
        
        # パターン1: v1.2.3 形式
        import re
        version_match = re.search(r'v?(\d+\.\d+(?:\.\d+)?)', name)
        if version_match:
            return version_match.group(1)
        
        # パターン2: 日付形式 (YYYY-MM-DD)
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', name)
        if date_match:
            return date_match.group(1)
        
        # パターン3: 修正タイムスタンプ
        if path.exists():
            mtime = datetime.fromtimestamp(path.stat().st_mtime)
            return mtime.strftime('%Y%m%d')
        
        return None
    
    def compare_versions(self, version1: str, version2: str) -> int:
        """
        バージョンを比較
        
        Returns:
            1: version1 > version2
            0: version1 == version2
            -1: version1 < version2
        """
        # セマンティックバージョニング形式
        try:
            v1_parts = [int(x) for x in version1.split('.')]
            v2_parts = [int(x) for x in version2.split('.')]
            
            # パディング（長さを揃える）
            max_len = max(len(v1_parts), len(v2_parts))
            v1_parts += [0] * (max_len - len(v1_parts))
            v2_parts += [0] * (max_len - len(v2_parts))
            
            for a, b in zip(v1_parts, v2_parts):
                if a > b:
                    return 1
                elif a < b:
                    return -1
            
            return 0
        
        except ValueError:
            # 数値変換失敗時は文字列比較
            if version1 > version2:
                return 1
            elif version1 < version2:
                return -1
            return 0
    
    def to_neo4j_properties(self, provenance: Dict) -> Dict:
        """
        Neo4jプロパティ形式に変換
        
        Returns:
            Neo4jに保存可能な辞書
        """
        return {
            'source_path': provenance['source_path'],
            'source_type': provenance['source_type'],
            'file_hash': provenance['file_hash'],
            'ingested_at': provenance['ingested_at'],
            'version': provenance.get('version', 'unknown'),
            'file_size': provenance.get('file_size', 0),
            'metadata_json': json.dumps(provenance.get('metadata', {}))
        }
    
    def from_neo4j_properties(self, properties: Dict) -> Dict:
        """
        Neo4jプロパティからProvenance辞書に復元
        """
        provenance = {
            'source_path': properties.get('source_path'),
            'source_type': properties.get('source_type'),
            'file_hash': properties.get('file_hash'),
            'ingested_at': properties.get('ingested_at'),
            'version': properties.get('version'),
            'file_size': properties.get('file_size'),
            'metadata': json.loads(properties.get('metadata_json', '{}'))
        }
        
        return provenance