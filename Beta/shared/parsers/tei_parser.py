"""
TEI Parser
TEI XML文書のパース
"""
from pathlib import Path
from typing import List, Optional
from bs4 import BeautifulSoup
from llama_index.core import Document

from ..text_utils import clean_text

def parse_tei(tei_path: Path, logger=None) -> List[Document]:
    """TEIをパース"""
    try:
        with open(tei_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'xml')

        title = _extract_title_safe(soup, tei_path)
        authors = _extract_authors_safe(soup)

        documents = []
        for i, div in enumerate(soup.find_all('div')):
            doc = _extract_section_safe(div, i, title, authors)
            if doc:
                documents.append(doc)

        metadata = {'title': title, 'authors': authors}
        if logger:
            logger.info(f"TEI parsed: {len(documents)} sections")
        return documents, metadata

    except Exception as e:
        if logger:
            logger.error(f"TEI parse failed: {e}")
        return [], {'title': tei_path.stem, 'authors': ['Unknown']}

def _extract_title_safe(soup: BeautifulSoup, filepath: Path) -> str:
    """タイトル抽出"""
    try:
        title_tag = soup.find('titleStmt')
        if title_tag:
            title_tag = title_tag.find('title', level='a', type='main')
        title = title_tag.text.strip() if title_tag else None
        if title and len(title) > 10:
            return title
    except:
        pass
    return filepath.stem.replace('_', ' ').title()

def _extract_authors_safe(soup: BeautifulSoup) -> List[str]:
    """著者抽出"""
    authors = []
    try:
        for persName in soup.find_all('persName'):
            try:
                forenames = [f.text for f in persName.find_all('forename') if f.text]
                surname = persName.find('surname')
                author_name = f"{' '.join(forenames)} {surname.text if surname else ''}".strip()
                if author_name:
                    authors.append(author_name)
            except:
                continue
    except:
        pass
    return authors if authors else ["Unknown Author"]

def _extract_section_safe(div, index: int, title: str, authors: List[str]) -> Optional[Document]:
    """セクション抽出"""
    try:
        head = div.find('head')
        section_title = head.text.strip() if head and head.text else f"Section {index}"

        paragraphs = [p.get_text(strip=True) for p in div.find_all('p') if p.get_text(strip=True)]
        text = '\n\n'.join(paragraphs)

        if len(text) < 50:
            return None

        return Document(
            text=clean_text(text),
            metadata={
                'title': title[:200],
                'authors': ', '.join(authors[:5]),
                'section': section_title[:100],
                'section_index': index,
                'source_format': 'tei'
            }
        )
    except:
        return None