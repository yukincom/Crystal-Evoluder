"""
Parsers
文書パーサーモジュール
"""
from .tei_parser import parse_tei
from .markdown_parser import parse_markdown
from .txt_parser import parse_txt
from .docx_parser import parse_docx
from .html_parser import parse_html
from .pdf_parser import parse_pdf

__all__ = [
    'parse_tei',
    'parse_markdown',
    'parse_txt',
    'parse_docx',
    'parse_html',
    'parse_pdf'
]