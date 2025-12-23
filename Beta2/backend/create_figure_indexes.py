#!/usr/bin/env python3
"""
Neo4j Figureインデックス作成スクリプト

初回のみ実行してください。
"""

from llama_index.graph_stores.neo4j import Neo4jGraphStore
import os

def create_figure_indexes():
    """Figureノード用のインデックスを作成"""

    # Neo4j接続設定（環境変数から取得）
    neo4j_url = os.environ.get('NEO4J_URI', 'neo4j://localhost:7687')
    neo4j_user = os.environ.get('NEO4J_USER', 'neo4j')
    neo4j_pass = os.environ.get('NEO4J_PASSWORD', 'password')

    # GraphStore初期化
    graph_store = Neo4jGraphStore(
        url=neo4j_url,
        username=neo4j_user,
        password=neo4j_pass
    )

    # インデックス作成クエリ
    queries = [
        "CREATE INDEX figure_id_idx IF NOT EXISTS FOR (f:Figure) ON (f.figure_id);",
        "CREATE INDEX figure_hash_idx IF NOT EXISTS FOR (f:Figure) ON (f.image_hash);",
        "CREATE INDEX figure_page_idx IF NOT EXISTS FOR (f:Figure) ON (f.page);",
        "CREATE INDEX figure_type_idx IF NOT EXISTS FOR (f:Figure) ON (f.type);"
    ]

    print("Creating Figure indexes...")

    for query in queries:
        try:
            graph_store.query(query)
            print(f"✓ {query}")
        except Exception as e:
            print(f"✗ Failed: {query}")
            print(f"  Error: {e}")

    print("Figure indexes creation complete!")

if __name__ == "__main__":
    create_figure_indexes()