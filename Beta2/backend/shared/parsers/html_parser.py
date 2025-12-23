"""
HTML Parser
HTML文書のパース
"""
from pathlib import Path
from typing import List
from bs4 import BeautifulSoup
from llama_index.core import Document

from ..text_utils import clean_text

try:
    import trafilatura
    HAS_TRAFILATURA = True
except ImportError:
    HAS_TRAFILATURA = False

def parse_html(html_path: Path, logger=None) -> List[Document]:
    """HTMLをパース"""
    try:
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()

        # 1) trafilatura を試す（整形済みテキストが欲しい時）
        if HAS_TRAFILATURA:
            extracted = trafilatura.extract(html_content, include_formatting=True)
            if extracted:
                text = clean_text(extracted)
                title = html_path.stem

                soup = BeautifulSoup(html_content, 'html.parser')
                authors = _extract_html_authors(soup)

                metadata = {'title': title, 'authors': authors}
                if logger:
                    logger.info("HTML parsed with trafilatura")
                return [Document(
                    text=text,
                    metadata={'title': title, 'authors': ', '.join(authors),
                    'section': 'Main',
                    'source_format': 'html'}
                )], metadata

        # 2) fallback BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
        title_tag = soup.find('title')
        title = title_tag.get_text(strip=True) if title_tag else html_path.stem

        # 著者メタ情報取得
        authors = _extract_html_authors(soup)

        # 見出しと段落でセクション分割
        sections = []
        current_heading = "Introduction"
        current_text = []

        for element in soup.find_all(['h1', 'h2', 'h3', 'p']):
            if element.name in ['h1', 'h2', 'h3']:
                if current_text:
                    sections.append((current_heading, clean_text('\n\n'.join(current_text))))
                current_heading = element.get_text(strip=True)
                current_text = []
            elif element.name == 'p':
                text = element.get_text(strip=True)
                if text:
                    current_text.append(text)

        if current_text:
            sections.append((current_heading, clean_text('\n\n'.join(current_text))))

        documents = []
        for i, (heading, text) in enumerate(sections):
            if len(text) >= 50:
                documents.append(Document(
                    text=text,
                    metadata={
                        'title': title,
                        'authors': ', '.join(authors),
                        'section': heading,
                        'section_index': i,
                        'source_format': 'html'
                    }
                ))

        metadata = {'title': title, 'authors': authors}
        if logger:
            logger.info(f"HTML parsed: {len(documents)} sections")
        return documents, metadata

    except Exception as e:
        if logger:
            logger.error(f"HTML parse failed: {e}")
        return [], {'title': html_path.stem, 'authors': ['Unknown']}

def _extract_html_authors(soup: BeautifulSoup) -> List[str]:
    """HTML から著者情報を抽出（共通メソッド）"""
     # 複数のメタタグパターンを試す
    for meta_name in ('author', 'article:author', 'og:author', 'byline'):
        meta = soup.find('meta', attrs={'name': meta_name})
        if not meta:
            meta = soup.find('meta', attrs={'property': meta_name})

        if meta and meta.get('content'):
            author = meta.get('content').strip()
            if author:
                return [author]

    # rel="author" を試す
    author_link = soup.find('a', rel='author')
    if author_link:
        author = author_link.get_text(strip=True)
        if author:
            return [author]

    # 見つからなければ Unknown
    return ['Unknown']