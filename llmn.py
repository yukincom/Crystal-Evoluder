from llama_index.core import SimpleDirectoryReader, KnowledgeGraphIndex, StorageContext
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.llms.openai import OpenAI
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding  
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)

FOLDER_PATH = "your_folder_path_here"
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "your_neo4j_password_here"

if not Path(FOLDER_PATH).exists():
    raise FileNotFoundError(f"フォルダが見つかりません: {FOLDER_PATH}")

# Neo4j接続
graph_store = Neo4jGraphStore(
    username=NEO4J_USER,
    password=NEO4J_PASS,
    url=NEO4J_URI,       
)
print("✅ Neo4j接続成功")

# ドキュメント読み込み
print("\n📄 ドキュメント読み込み中...")
documents = SimpleDirectoryReader(
    FOLDER_PATH,
    required_exts=[".md", ".pdf"]
).load_data()
print(f"✅ {len(documents)}個")

llm = OpenAI(model="gpt-4o-mini", timeout=120.0, max_retries=3)
node_parser = SimpleNodeParser.from_defaults(chunk_size=512, chunk_overlap=50)

# StorageContextを明示的に作成
storage_context = StorageContext.from_defaults(graph_store=graph_store)

print("\n🔨 Knowledge Graph構築中...")
index = KnowledgeGraphIndex.from_documents(
    documents,
    storage_context=storage_context, 
    llm=llm,
    transformations=[node_parser],
    embed_model=HuggingFaceEmbedding(model_name="BAAI/bge-m3"),
    show_progress=True,
    max_triplets_per_chunk=10,
)

# 確認
print("\n📊 グラフ統計:")
kg = index.get_networkx_graph()
print(f"  Python側: {len(kg.nodes)}ノード, {len(kg.edges)}エッジ")

print("\n🔍 Neo4j確認:")
with graph_store.client.session() as session:
    result = session.run("MATCH (n) RETURN count(n) as count")
    count = result.single()["count"]
    print(f"  Neo4j側: {count}ノード")
    
    if count == 0:
        print("\n⚠️ まだ空です。手動で保存します...")
        
        # 手動で保存
        for node in kg.nodes():
            session.run("MERGE (n:Entity {id: $id})", id=str(node))
        
        for source, target in kg.edges():
            session.run("""
                MATCH (a:Entity {id: $source})
                MATCH (b:Entity {id: $target})
                MERGE (a)-[r:RELATES_TO]->(b)
            """, source=str(source), target=str(target))
        
        print("✅ 手動保存完了")
        
        # 再確認
        result = session.run("MATCH (n) RETURN count(n) as count")
        count = result.single()["count"]
        print(f"  Neo4j側（再確認）: {count}ノード")

print("\n🎉 Neo4j Browser → http://localhost:7474")