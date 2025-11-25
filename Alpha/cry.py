"""
Crystal Evoluder v1.0
Knowledge Crystallization System

"""

from bs4 import BeautifulSoup
from llama_index.core import Document, KnowledgeGraphIndex, StorageContext
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.llms.openai import OpenAI
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os
import re
import logging
import concurrent.futures
import time
from pathlib import Path
from typing import Optional, List, Dict


class CrystalEvoluder:
    """
    Crystal Evoluder - Knowledge Crystallization System
    
    デジタル技術の基盤となる秩序化された構造
    - 水晶振動子: デジタル回路の心臓
    - 液晶: 情報の表示面
    - 結晶成長: 知識が秩序を持って増殖
    """
    
    def __init__(self, config: Optional[Dict] = None, log_level: int = logging.INFO):
        self.config = config or {}
        self.crystal = None
        self.metadata = {}
        
        # ロガー設定
        self.logger = self._setup_logger(log_level)
        self.logger.info("Crystal Evoluder initialized")
    
    def _setup_logger(self, level: int) -> logging.Logger:
        """階層化されたロガー設定"""
        logger = logging.getLogger('CrystalEvoluder')
        logger.setLevel(level)
        
        # 既存ハンドラをクリア
        logger.handlers.clear()
        
        # コンソールハンドラ（厨二演出用）
        console = logging.StreamHandler()
        
        class IconFormatter(logging.Formatter):
            ICONS = {
                'DEBUG': '🔍',
                'INFO': '✨',
                'WARNING': '⚠️',
                'ERROR': '❌',
                'CRITICAL': '💥'
            }
            
            def format(self, record):
                icon = self.ICONS.get(record.levelname, 'ℹ️')
                record.icon = icon
                return super().format(record)
        
        console.setFormatter(IconFormatter('%(icon)s %(message)s'))
        logger.addHandler(console)
        
        # ファイルハンドラ（研究者用）
        log_file = Path('crystal_evoluder.log')
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(file_handler)
        
        return logger
    
    def crystallize(self, tei_path: str, strict_mode: bool = False) -> List[Document]:
        """
        🔮 結晶化: TEIを秩序化された構造に変換
        
        Args:
            tei_path: GROBIDが出力したTEIファイル
            strict_mode: Trueなら壊れたTEIで停止、Falseなら続行
        
        Returns:
            crystallized_documents: 結晶化されたドキュメント
        """
        self.logger.info("Crystallizing knowledge structure...")
        
        tei_path = Path(tei_path).expanduser()
        
        if not tei_path.exists():
            self.logger.error(f"TEI file not found: {tei_path}")
            raise FileNotFoundError(f"TEI file not found: {tei_path}")
        
        # TEIをパース
        try:
            with open(tei_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f, 'xml')
            self.logger.debug(f"TEI parsed successfully: {tei_path}")
        except Exception as e:
            self.logger.error(f"Failed to parse TEI: {e}", exc_info=True)
            if strict_mode:
                raise
            return []
        
        # メタデータ抽出（フォールバック付き）
        title = self._extract_title_safe(soup, tei_path)
        authors = self._extract_authors_safe(soup)
        
        self.logger.info(f"  ├─ Title: \"{title}\"")
        self.logger.info(f"  ├─ Authors: {', '.join(authors[:3])}")
        
        # セクション抽出（壊れてても続行）
        documents = []
        divs = soup.find_all('div')
        self.logger.debug(f"Found {len(divs)} div elements")
        
        for i, div in enumerate(divs):
            try:
                doc = self._extract_section_safe(div, i, title, authors)
                if doc:
                    documents.append(doc)
                    self.logger.debug(f"Section {i} extracted: {doc.metadata['section']}")
            except Exception as e:
                self.logger.warning(f"Skipping broken section {i}: {e}")
                if strict_mode:
                    raise
                continue
        
        # 検証
        if len(documents) == 0:
            self.logger.error("No valid sections found in TEI")
            if strict_mode:
                raise ValueError("TEI completely broken - no sections extracted")
        
        self.logger.info(f"  ├─ Sections: {len(documents)} fragments detected")
        self.logger.info(f"  └─ Crystal formed: {len(documents)} nodes")
        self.logger.info("✨ Crystal structure stabilized")
        
        self.crystal = documents
        self.metadata = {'title': title, 'authors': authors}
        
        return documents
    
    def _extract_title_safe(self, soup: BeautifulSoup, filepath: Path) -> str:
        """タイトル抽出（フォールバック付き）"""
        try:
            title_tag = soup.find('titleStmt')
            if title_tag:
                title_tag = title_tag.find('title', level='a', type='main')
            title = title_tag.text.strip() if title_tag else None
            
            if title and len(title) > 10:
                return title
        except Exception as e:
            self.logger.debug(f"Title extraction failed: {e}")
        
        # フォールバック1: ファイル名から推測
        self.logger.warning("Title not found in TEI, using filename")
        return filepath.stem.replace('_', ' ').replace('.tei', '').title()
    
    def _extract_authors_safe(self, soup: BeautifulSoup) -> List[str]:
        """著者抽出（エラー耐性）"""
        authors = []
        try:
            for persName in soup.find_all('persName'):
                try:
                    forenames = [f.text for f in persName.find_all('forename') if f.text]
                    surname = persName.find('surname')
                    author_name = f"{' '.join(forenames)} {surname.text if surname else ''}".strip()
                    if author_name and len(author_name) > 1:
                        authors.append(author_name)
                except Exception as e:
                    self.logger.debug(f"Skipping malformed author entry: {e}")
                    continue
        except Exception as e:
            self.logger.warning(f"Author extraction failed: {e}")
        
        if not authors:
            self.logger.warning("No authors found, using placeholder")
            authors = ["Unknown Author"]
        
        return authors
    
    def _extract_section_safe(self, div, index: int, title: str, authors: List[str]) -> Optional[Document]:
        """セクション抽出（エラー耐性）"""
        try:
            # セクションタイトル
            head = div.find('head')
            section_title = head.text.strip() if head and head.text else f"Section {index}"
            
            # パラグラフ抽出
            paragraphs = []
            for p in div.find_all('p'):
                text = p.get_text(strip=True)
                if text:
                    paragraphs.append(text)
            
            text = '\n\n'.join(paragraphs)
            
            # 空セクションをスキップ
            if not text or len(text) < 50:
                self.logger.debug(f"Skipping empty/short section: {section_title}")
                return None
            
            return Document(
                text=text,
                metadata={
                    'title': title[:200],
                    'authors': ', '.join(authors[:5]),
                    'section': section_title[:100],
                    'section_index': index,
                    'char_count': len(text),
                    'paragraph_count': len(paragraphs)
                }
            )
        except Exception as e:
            self.logger.warning(f"Failed to extract section {index}: {e}")
            return None
    
    def evolve_to_notes(self, output_dir: str, granularity: str = 'section'):
        """
        📝 ノートに進化: 液晶のように情報を可視化
        
        Args:
            output_dir: Obsidian vault のパス
            granularity: 'section' | 'paragraph' (v2.0) | 'sentence' (v3.0)
        """
        if not self.crystal:
            raise ValueError("❌ No crystal found. Run crystallize() first.")
        
        # 粒度の警告（将来の拡張用）
        if granularity != 'section':
            self.logger.warning(f"Granularity '{granularity}' not yet implemented, using 'section'")
            # TODO: v2.0で実装
        
        self.logger.info("Evolving to observable notes...")
        self.logger.info("  ├─ Generating markdown lattice")
        
        output_dir = Path(output_dir).expanduser()
        paper_title = self._sanitize(self.metadata['title'])
        paper_dir = output_dir / "Papers" / paper_title
        paper_dir.mkdir(parents=True, exist_ok=True)
        
        # 各セクションをMarkdown化
        for i, doc in enumerate(self.crystal):
            section = doc.metadata.get('section', 'Untitled')
            
            md_content = f"""---
title: {self.metadata['title']}
authors: {', '.join(self.metadata['authors'][:3])}
section: {section}
index: {i}
total: {len(self.crystal)}
type: paper-section
created: {time.strftime('%Y-%m-%d %H:%M:%S')}
---

# {section}

{doc.text}

---
**Navigation**
- [[{paper_title}_index|📑 Back to Index]]
{"- [[" + paper_title + f"_{i-1:03d}|← Previous]]" if i > 0 else ""}
{"- [[" + paper_title + f"_{i+1:03d}|Next →]]" if i < len(self.crystal)-1 else ""}

**Metadata**
- Paragraphs: {doc.metadata.get('paragraph_count', 'N/A')}
- Characters: {doc.metadata.get('char_count', 'N/A')}
"""
            
            filename = f"{paper_title}_{i:03d}_{self._sanitize(section)}.md"
            filepath = paper_dir / filename
            
            with open(filepath, "w", encoding='utf-8') as f:
                f.write(md_content)
            
            self.logger.debug(f"  ├─ {filename} ✓")
        
        # インデックスページ
        index_content = f"""---
title: {self.metadata['title']}
type: paper-index
authors: {', '.join(self.metadata['authors'])}
created: {time.strftime('%Y-%m-%d %H:%M:%S')}
---

# {self.metadata['title']}

**Authors:** {', '.join(self.metadata['authors'])}

## Sections

"""
        for i, doc in enumerate(self.crystal):
            section = doc.metadata.get('section', 'Untitled')
            index_content += f"{i+1}. [[{paper_title}_{i:03d}_{self._sanitize(section)}|{section}]]\n"
        
        index_path = paper_dir / f"{paper_title}_index.md"
        with open(index_path, "w", encoding='utf-8') as f:
            f.write(index_content)
        
        self.logger.info("  └─ Index matrix created")
        self.logger.info("✅ Notes evolution complete")
        self.logger.info(f"   Location: {paper_dir}")
    
    def evolve_to_graph(self, graph_store: Neo4jGraphStore, normalize_labels: bool = False):
        """
        🕸️ グラフに進化: 水晶格子のように概念を配置
        
        Args:
            graph_store: Neo4jGraphStore インスタンス
            normalize_labels: ⚠️ 実験的機能（v2.0で実装予定）
        """
        if not self.crystal:
            raise ValueError("❌ No crystal found. Run crystallize() first.")
        
        if normalize_labels:
            self.logger.warning("⚠️ Label normalization is not yet implemented (coming in v2.0)")
            # TODO: v2.0で実装
        
        self.logger.info("Evolving to crystal lattice structure...")
        
        # Neo4jスキーマ初期化
        self._setup_neo4j_schema(graph_store)
        
        self.logger.info("  ├─ Resonating with GPT-4o-mini")
        
        llm = OpenAI(model="gpt-4o-mini", timeout=120.0, max_retries=3)
        node_parser = SimpleNodeParser.from_defaults(chunk_size=512, chunk_overlap=50)
        storage_context = StorageContext.from_defaults(graph_store=graph_store)
        
        self.logger.info("  ├─ Extracting concept nodes")
        self.logger.info("  ├─ Forming relationship bonds")
        
        try:
            index = KnowledgeGraphIndex.from_documents(
                self.crystal,
                storage_context=storage_context,
                llm=llm,
                transformations=[node_parser],
                embed_model=HuggingFaceEmbedding(model_name="BAAI/bge-m3"),
                show_progress=False,
                max_triplets_per_chunk=10,
                include_embeddings=True,
            )
            
            # Data Provenance（出典情報）を追加
            self._add_provenance(graph_store)
            
            kg = index.get_networkx_graph()
            
            self.logger.info("  ├─ Injecting into Neo4j lattice")
            self.logger.info("  └─ Structure crystallized")
            self.logger.info("✅ Graph evolution complete")
            self.logger.info(f"   Nodes: {len(kg.nodes)} | Edges: {len(kg.edges)}")
            
        except Exception as e:
            self.logger.error(f"Graph evolution failed: {e}", exc_info=True)
            raise
    
    def _setup_neo4j_schema(self, graph_store: Neo4jGraphStore):
        """Neo4jスキーマ初期化（制約・インデックス）"""
        self.logger.debug("Setting up Neo4j schema...")
        
        try:
            with graph_store.client.session() as session:
                # UNIQUE制約
                session.run("""
                    CREATE CONSTRAINT entity_id IF NOT EXISTS
                    FOR (n:Entity) REQUIRE n.id IS UNIQUE
                """)
                
                # インデックス（検索高速化）
                session.run("""
                    CREATE INDEX entity_name IF NOT EXISTS
                    FOR (n:Entity) ON (n.name)
                """)
                
                self.logger.debug("Neo4j schema initialized")
        except Exception as e:
            self.logger.warning(f"Schema setup failed (may already exist): {e}")
    
    def _add_provenance(self, graph_store: Neo4jGraphStore):
        """Data Provenance（出典情報）を追加"""
        self.logger.debug("Adding provenance metadata...")
        
        try:
            with graph_store.client.session() as session:
                session.run("""
                    MATCH (n:Entity)
                    WHERE NOT EXISTS(n.source_paper)
                    SET n.source_paper = $title,
                        n.source_authors = $authors,
                        n.extracted_at = datetime(),
                        n.extractor_model = 'gpt-4o-mini',
                        n.extractor_version = '1.0'
                """, 
                title=self.metadata['title'], 
                authors=', '.join(self.metadata['authors']))
                
                self.logger.debug("Provenance metadata added")
        except Exception as e:
            self.logger.warning(f"Provenance addition failed: {e}")
    
    def evolve_all(self, markdown_dir: str, graph_store: Neo4jGraphStore):
        """
        🌟 完全進化: 全形態に共振
        
        Args:
            markdown_dir: Markdown出力先
            graph_store: Neo4jGraphStore
        """
        if not self.crystal:
            raise ValueError("❌ No crystal found. Run crystallize() first.")
        
        self.logger.info("Resonating across all forms...")
        
        start_time = time.time()
        
        try:
            # 並行処理
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                notes_future = executor.submit(self.evolve_to_notes, markdown_dir)
                graph_future = executor.submit(self.evolve_to_graph, graph_store)
                
                notes_future.result()
                graph_future.result()
            
            elapsed = time.time() - start_time
            self.logger.info(f"All forms resonating in harmony (took {elapsed:.1f}s)")
            
        except Exception as e:
            self.logger.error(f"Evolution failed: {e}", exc_info=True)
            raise
    
    def export_graph(self, format: str = 'neo4j', output_path: Optional[str] = None):
        """
        グラフをエクスポート（拡張用スロット）
        
        Args:
            format: 'neo4j' | 'json-ld' (v2.0) | 'rdf' (v2.0) | 'custom' (v3.0)
            output_path: 出力先（formatによる）
        """
        exporters = {
            'neo4j': self._export_neo4j,
            'json-ld': self._export_jsonld,
            'rdf': self._export_rdf,
            'custom': self._export_custom
        }
        
        if format not in exporters:
            raise ValueError(f"Unknown format: {format}. Supported: {list(exporters.keys())}")
        
        self.logger.info(f"Exporting to {format}...")
        return exporters[format](output_path)
    
    def _export_neo4j(self, output_path: Optional[str] = None):
        """Neo4j（デフォルト）"""
        self.logger.info("Neo4j is the default storage, no export needed")
        return None
    
    def _export_jsonld(self, output_path: str):
        """JSON-LD（汎用）- v2.0実装予定"""
        raise NotImplementedError("JSON-LD export coming in v2.0")
    
    def _export_rdf(self, output_path: str):
        """RDF/Turtle（Blazegraph用）- v2.0実装予定"""
        raise NotImplementedError("RDF export coming in v2.0")
    
    def _export_custom(self, output_path: str):
        """カスタムプラグイン - v3.0実装予定"""
        raise NotImplementedError("Custom plugins coming in v3.0")
    
    def _sanitize(self, text: str) -> str:
        """ファイル名用のサニタイズ"""
        sanitized = re.sub(r'[<>:"/\\|?*]', '', text)
        return sanitized[:50].strip()


# CLI
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Crystal Evoluder - Knowledge Crystallization System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Crystallize only
  python crystal_evoluder.py crystallize paper.tei.xml
  
  # Generate notes
  python crystal_evoluder.py evolve-notes paper.tei.xml --markdown-dir ~/Notes
  
  # Build knowledge graph
  python crystal_evoluder.py evolve-graph paper.tei.xml --neo4j-pass mypass
  
  # Do everything
  python crystal_evoluder.py evolve-all paper.tei.xml --neo4j-pass mypass
        """
    )
    
    parser.add_argument('command', choices=['crystallize', 'evolve-notes', 'evolve-graph', 'evolve-all'])
    parser.add_argument('tei_file', help='TEI XML file path')
    parser.add_argument('--markdown-dir', default='~/CrystalEvoluder/Library', help='Markdown output directory')
    parser.add_argument('--neo4j-uri', default='bolt://localhost:7687', help='Neo4j connection URI')
    parser.add_argument('--neo4j-user', default='neo4j', help='Neo4j username')
    parser.add_argument('--neo4j-pass', help='Neo4j password (required for graph operations)')
    parser.add_argument('--api-key', help='OpenAI API key (or set OPENAI_API_KEY env var)')
    parser.add_argument('--strict', action='store_true', help='Strict mode: fail on any error')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    # ロギングレベル
    log_level = logging.DEBUG if args.debug else logging.INFO
    
    # APIキー設定
    if args.api_key:
        os.environ['OPENAI_API_KEY'] = args.api_key
    
    # ヘッダー
    print("🔮 Crystal Evoluder v1.0.0")
    print("━" * 42)
    
    start_time = time.time()
    
    try:
        evoluder = CrystalEvoluder(log_level=log_level)
        evoluder.crystallize(args.tei_file, strict_mode=args.strict)
        
        if args.command in ['evolve-notes', 'evolve-all']:
            evoluder.evolve_to_notes(args.markdown_dir)
        
        if args.command in ['evolve-graph', 'evolve-all']:
            if not args.neo4j_pass:
                raise ValueError("--neo4j-pass required for graph operations")
            
            graph_store = Neo4jGraphStore(
                username=args.neo4j_user,
                password=args.neo4j_pass,
                url=args.neo4j_uri,
            )
            evoluder.evolve_to_graph(graph_store)
        
        elapsed = time.time() - start_time
        print("\n" + "━" * 42)
        print(f"✨ Process completed in {int(elapsed//60)}m {int(elapsed%60)}s")
        print("💎 Knowledge crystallization successful")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        exit(1)