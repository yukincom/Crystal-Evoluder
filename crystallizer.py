from llama_index.core import Document, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from bs4 import BeautifulSoup
import os
import re
import glob

# グローバルでembedding modelを定義
embed_model = OpenAIEmbedding(model="text-embedding-3-large")

def sanitize_filename(text):
    """ファイル名に使えない文字を削除"""
    return re.sub(r'[<>:"/\\|?*]', '', text)[:50]

def parse_grobid_tei(file_path):
    """GROBIDのTEIファイルをパース"""
    # パス展開
    file_path = os.path.expanduser(file_path)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'xml')
    
    # タイトル
    title_tag = soup.find('titleStmt')
    if title_tag:
        title_tag = title_tag.find('title', level='a', type='main')
    title_text = title_tag.text.strip() if title_tag else "Unknown"
    
    # 著者（ミドルネーム対応）
    authors = []
    for persName in soup.find_all('persName'):
        forenames = [f.text for f in persName.find_all('forename') if f.text]
        surname = persName.find('surname')
        author_name = f"{' '.join(forenames)} {surname.text if surname else ''}".strip()
        if author_name:
            authors.append(author_name)
    
    documents = []
    # 本文（全divを走査）
    for div in soup.find_all('div'):
        head = div.find('head')
        section_title = head.text.strip() if head else "Untitled Section"
        
        text = '\n\n'.join(p.get_text(strip=True) for p in div.find_all('p'))
        if text.strip():
            documents.append(Document(
                text=text,
                metadata={
                    'title': title_text[:200],
                    'authors': ', '.join(authors[:5]),
                    'section': section_title[:100]
                }
            ))
    
    return documents, {'title': title_text, 'authors': authors}

def crystallize_paper(tei_file_path, crystallize_base_path):
    """TEIファイルを処理してノートを生成"""
    # パス展開
    tei_file_path = os.path.expanduser(tei_file_path)
    crystallize_base_path = os.path.expanduser(crystallize_base_path)
    
    # 1. TEIをパース
    print("TEIファイルをパース中...")
    docs, metadata = parse_grobid_tei(tei_file_path)
    
    # 2. インデックス作成または読み込み
    paper_title = sanitize_filename(metadata['title'])
    storage_dir = f"./storage/{paper_title}"
    
    if os.path.exists(storage_dir):
        # 既存インデックスを読み込み
        print(f"既存インデックスを読み込み中: {storage_dir}")
        storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
        index = load_index_from_storage(storage_context, embed_model=embed_model)
    else:
        # 新規作成
        print("新規インデックスを作成中...")
        index = VectorStoreIndex.from_documents(
            docs, 
            embed_model=embed_model,
            transformations=[SentenceSplitter(chunk_size=2048, chunk_overlap=100)]
        )
        index.storage_context.persist(persist_dir=storage_dir)
        print(f"インデックスを保存しました: {storage_dir}")
    
    # 3. ノート生成
    output_dir = f"{crystallize_base_path}/Papers/{paper_title}"
    os.makedirs(output_dir, exist_ok=True)
    
    nodes = list(index.docstore.docs.values())
    
    for i, node in enumerate(nodes):
        section = node.metadata.get('section', 'Untitled')
        
        md_content = f"""---
title: {metadata['title']}
authors: {', '.join(metadata['authors'][:3])}
section: {section}
index: {i}
total: {len(nodes)}
type: paper-section
---

# {section}

{node.text}

---
**Navigation:**
- [[{paper_title}_index|📑 目次に戻る]]
{"- [[" + paper_title + f"_{i-1:03d}|← 前のセクション]]" if i > 0 else ""}
{"- [[" + paper_title + f"_{i+1:03d}|次のセクション →]]" if i < len(nodes)-1 else ""}
"""
        
        filepath = f"{output_dir}/{paper_title}_{i:03d}_{sanitize_filename(section)}.md"
        with open(filepath, "w", encoding='utf-8') as f:
            f.write(md_content)
    
    # 4. 目次ページ作成
    index_content = f"""---
title: {metadata['title']}
type: paper-index
authors: {', '.join(metadata['authors'])}
---

# {metadata['title']}

**著者:** {', '.join(metadata['authors'])}

## セクション一覧

"""
    for i, node in enumerate(nodes):
        section = node.metadata.get('section', 'Untitled')
        index_content += f"{i+1}. [[{paper_title}_{i:03d}_{sanitize_filename(section)}|{section}]]\n"
    
    with open(f"{output_dir}/{paper_title}_index.md", "w", encoding='utf-8') as f:
        f.write(index_content)
    
    print(f"✅ 完了: {len(nodes)}個のノートを生成")
    print(f"📂 場所: {output_dir}")
    
    return index

# 実行例
if __name__ == "__main__":
    # 最新のTEIファイルを自動取得
    tei_files = glob.glob(os.path.expanduser("~/Downloads/*.tei.xml"))
    if tei_files:
        latest_tei = max(tei_files, key=os.path.getctime)
        print(f"処理するファイル: {latest_tei}")
        
        index = crystallize_paper(
            latest_tei,
            "~/CrystalEvoluder/Library"
        )
        
        # 質問もできる
        query_engine = index.as_query_engine()
        response = query_engine.query("この論文の主要な主張を200文字以内で要約してください")
        print(f"\n回答:\n{response}")
    else:
        print("TEIファイルが見つかりません")