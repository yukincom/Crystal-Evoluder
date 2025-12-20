"""
Text Utilities
テキスト処理関連のユーティリティ関数
"""
import re
from typing import List

try:
    import ftfy
    HAS_FTFY = True
except ImportError:
    HAS_FTFY = False

def clean_text(text: str, normalize_whitespace: bool = True) -> str:
    """
    テキストをクリーニング

    Args:
        text: 元のテキスト
        normalize_whitespace: 空白を正規化するか

    Returns:
        クリーニングされたテキスト

    Examples:
        >>> clean_text("Hello    World\\n\\n\\nTest")
        'Hello World Test'
    """
    # ftfyがあれば文字エンコーディングを修正
    if HAS_FTFY:
        text = ftfy.fix_text(text)

    if normalize_whitespace:
        # 複数の空白を1つに
        text = re.sub(r'\s+', ' ', text)

    # 先頭・末尾の空白を削除
    text = text.strip()

    return text

def detect_text_language(text: str) -> str:
    """
    テキストの言語を検出
    
    Args:
        text: 検出するテキスト
    
    Returns:
        言語コード ('en', 'ja', 'zh', 'ko', 'other')
    """
    if not text:
        return "unknown"
    
    sample = text[:300]
    
    hiragana = sum(1 for c in sample if '\u3040' <= c <= '\u309f')
    katakana = sum(1 for c in sample if '\u30a0' <= c <= '\u30ff')
    kanji = sum(1 for c in sample if '\u4e00' <= c <= '\u9faf')
    ascii_chars = sum(1 for c in sample if ord(c) < 128)
    
    total = max(len(sample), 1)
    
    if (hiragana + katakana) / total > 0.15:
        return "ja"
    
    if kanji / total > 0.3 and (hiragana + katakana) / total < 0.05:
        return "zh"
    
    if ascii_chars / total > 0.7:
        return "en"
    
    return "other"

def split_japanese_sentences(text: str) -> List[str]:
    """
    日本語テキストを文単位で分割
    
    Args:
        text: 分割するテキスト
    
    Returns:
        文のリスト
    """
    # 日本語の文末パターン
    sentence_endings = ['。', '！', '？', '.\n', '!\n', '?\n']
    
    sentences = []
    temp_sentence = ""
    
    for char in text:
        temp_sentence += char
        if any(temp_sentence.endswith(end) for end in sentence_endings):
            sentences.append(temp_sentence.strip())
            temp_sentence = ""
    
    if temp_sentence.strip():
        sentences.append(temp_sentence.strip())
    
    return sentences

def chunk_by_paragraphs(
    content: str,
    chunk_size: int = 2000,
    overlap: int = 200,
    language: str = None  # 👈 言語パラメータ追加
) -> List[str]:
    """
    段落ベースのチャンク分割（言語対応版）
    
    Args:
        content: テキスト内容
        chunk_size: チャンクの最大サイズ（文字数）
        overlap: チャンク間のオーバーラップ
        language: 言語コード（None=自動検出）
    
    Returns:
        チャンクのリスト
    """
    # 言語自動検出
    if language is None:
        language = detect_text_language(content)
    
    # 日本語の場合は専用処理
    if language == 'ja':
        return _chunk_japanese_text(content, chunk_size, overlap)
    
    # 英語など（既存処理）
    paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for para in paragraphs:
        para_length = len(para)
        
        if current_length + para_length > chunk_size and current_chunk:
            chunks.append('\n\n'.join(current_chunk))
            
            if overlap > 0 and current_chunk:
                overlap_text = '\n\n'.join(current_chunk)
                if len(overlap_text) > overlap:
                    overlap_paras = []
                    overlap_len = 0
                    for p in reversed(current_chunk):
                        if overlap_len + len(p) <= overlap:
                            overlap_paras.insert(0, p)
                            overlap_len += len(p)
                        else:
                            break
                    current_chunk = overlap_paras
                    current_length = overlap_len
                else:
                    current_chunk = []
                    current_length = 0
            else:
                current_chunk = []
                current_length = 0
        
        current_chunk.append(para)
        current_length += para_length
    
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))
    
    return chunks

def _chunk_by_sentences(sentences: List[str], chunk_size: int, overlap: int) -> List[str]:
    """
    文単位でチャンク分割（内部ヘルパー）
    """
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sent in sentences:
        sent_length = len(sent)
        
        if current_length + sent_length > chunk_size and current_chunk:
            chunks.append(''.join(current_chunk))
            
            if overlap > 0:
                overlap_sents = []
                overlap_len = 0
                for s in reversed(current_chunk):
                    if overlap_len + len(s) <= overlap:
                        overlap_sents.insert(0, s)
                        overlap_len += len(s)
                    else:
                        break
                current_chunk = overlap_sents
                current_length = overlap_len
            else:
                current_chunk = []
                current_length = 0
        
        current_chunk.append(sent)
        current_length += sent_length
    
    if current_chunk:
        chunks.append(''.join(current_chunk))
    
    return chunks

def chunk_by_paragraphs(
    content: str,
    chunk_size: int = 2000,
    overlap: int = 200,
    language: str = 'en'
) -> List[str]:
    """
    言語対応段落チャンク分割
    
    Args:
        content: テキスト
        chunk_size: チャンクサイズ
        overlap: オーバーラップ
        language: 言語コード ('en', 'ja', etc.)
    Returns:
        チャンクのリスト    
    """
    if language == 'ja':
        language = detect_text_language(content)
    
    # 日本語の場合は専用処理
    if language == 'ja':
        return _chunk_japanese_text(content, chunk_size, overlap)
    
    # 英語など（既存処理）
    return _chunk_english_text(content, chunk_size, overlap)
        
def _chunk_japanese_text(content: str, chunk_size: int, overlap: int) -> List[str]:
    """
    日本語テキストのチャンク分割
    
    Args:
        content: テキスト
        chunk_size: チャンクサイズ
        overlap: オーバーラップ
    
    Returns:
        チャンクのリスト
    """
    # 段落分割
    paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
    
    if not paragraphs:
        # 段落がない場合は文単位で分割
        sentences = split_japanese_sentences(content)
        return _chunk_by_sentences(sentences, chunk_size, overlap)
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for para in paragraphs:
        para_length = len(para)
        
        if current_length + para_length > chunk_size and current_chunk:
            chunks.append('\n\n'.join(current_chunk))
            
            # オーバーラップ処理
            if overlap > 0:
                overlap_paras = []
                overlap_len = 0
                for p in reversed(current_chunk):
                    if overlap_len + len(p) <= overlap:
                        overlap_paras.insert(0, p)
                        overlap_len += len(p)
                    else:
                        break
                current_chunk = overlap_paras
                current_length = overlap_len
            else:
                current_chunk = []
                current_length = 0
        
        current_chunk.append(para)
        current_length += para_length
    
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))
    
    return chunks

def _chunk_english_text(content: str, chunk_size: int, overlap: int) -> List[str]:
    """
    英語テキストのチャンク分割（既存処理）
    
    Args:
        content: テキスト
        chunk_size: チャンクサイズ
        overlap: オーバーラップ
    
    Returns:
        チャンクのリスト
    """
    paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for para in paragraphs:
        para_length = len(para)
        
        if current_length + para_length > chunk_size and current_chunk:
            chunks.append('\n\n'.join(current_chunk))
            
            if overlap > 0 and current_chunk:
                overlap_text = '\n\n'.join(current_chunk)
                if len(overlap_text) > overlap:
                    overlap_paras = []
                    overlap_len = 0
                    for p in reversed(current_chunk):
                        if overlap_len + len(p) <= overlap:
                            overlap_paras.insert(0, p)
                            overlap_len += len(p)
                        else:
                            break
                    current_chunk = overlap_paras
                    current_length = overlap_len
                else:
                    current_chunk = []
                    current_length = 0
            else:
                current_chunk = []
                current_length = 0
        
        current_chunk.append(para)
        current_length += para_length
    
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))
    
    return chunks
