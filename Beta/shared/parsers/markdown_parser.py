"""
Markdown Parser
Markdown文書のパース
"""
import re
from pathlib import Path
from typing import List, Tuple
from llama_index.core import Document

from ..text_utils import clean_text

try:
    import frontmatter
    HAS_FRONTMATTER = True
except ImportError:
    HAS_FRONTMATTER = False

def parse_markdown(md_path: Path, logger=None) -> List[Document]:
    """Markdownをパース"""
    try:
        with open(md_path, 'r', encoding='utf-8') as f:
            content = f.read()

        if HAS_FRONTMATTER:
            post = frontmatter.load(md_path)
            content = post.content
            title = post.metadata.get('title', md_path.stem)
            authors = post.metadata.get('authors', ['Unknown'])
        else:
            title = md_path.stem
            authors = ['Unknown']

        sections = _split_by_headers_safe(content)

        documents = []
        for i, (heading, text) in enumerate(sections):
            clean_text_content = clean_text(text)
            if len(clean_text_content) >= 50:
                documents.append(Document(
                    text=clean_text_content,
                    metadata={
                        'title': title,
                        'authors': ', '.join(authors) if isinstance(authors, list) else authors,
                        'section': heading,
                        'section_index': i,
                        'source_format': 'markdown'
                    }
                ))

        metadata = {'title': title, 'authors': authors if isinstance(authors, list) else [authors]}
        if logger:
            logger.info(f"Markdown parsed: {len(documents)} sections")
        return documents, metadata

    except Exception as e:
        if logger:
            logger.error(f"Markdown parse failed: {e}")
        return [], {'title': md_path.stem, 'authors': ['Unknown']}

def _split_by_headers_safe(content: str) -> List[Tuple[str, str]]:
    """Markdownを見出しで分割（コードブロック対応）"""
    code_blocks = []
    def replace_code(match):
        code_blocks.append(match.group(0))
        return f"__CODE_BLOCK_{len(code_blocks)-1}__"

    content_no_code = re.sub(r'```.*?```', replace_code, content, flags=re.DOTALL)

    pattern = r'^(#{1,6})\s+(.+)$'
    lines = content_no_code.split('\n')

    sections = []
    current_heading = "Introduction"
    current_text = []

    for line in lines:
        match = re.match(pattern, line)
        if match:
            if current_text:
                text = '\n'.join(current_text)
                for i, block in enumerate(code_blocks):
                    text = text.replace(f"__CODE_BLOCK_{i}__", block)
                sections.append((current_heading, text))
            current_heading = match.group(2)
            current_text = []
        else:
            current_text.append(line)

    if current_text:
        text = '\n'.join(current_text)
        for i, block in enumerate(code_blocks):
            text = text.replace(f"__CODE_BLOCK_{i}__", block)
        sections.append((current_heading, text))

    return sections