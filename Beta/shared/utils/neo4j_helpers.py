"""
Neo4j補助関数
"""
from typing import Any, Dict, List, Optional
from neo4j import GraphDatabase


class Neo4jConnection:
    """Neo4j接続管理クラス"""

    def __init__(self, uri: str, user: str, password: str):
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None

    def connect(self):
        """接続確立"""
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))

    def disconnect(self):
        """接続切断"""
        if self.driver:
            self.driver.close()

    def execute_query(self, query: str, parameters: Optional[Dict] = None) -> List[Dict]:
        """
        クエリ実行

        Args:
            query: Cypherクエリ
            parameters: クエリパラメータ

        Returns:
            結果のリスト
        """
        if not self.driver:
            raise ConnectionError("Neo4j connection not established")

        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()


def execute_query(driver, query: str, parameters: Optional[Dict] = None) -> List[Dict]:
    """
    スタンドアロン関数版クエリ実行

    Args:
        driver: Neo4jドライバー
        query: Cypherクエリ
        parameters: クエリパラメータ

    Returns:
        結果のリスト
    """
    with driver.session() as session:
        result = session.run(query, parameters or {})
        return [record.data() for record in result]
def create_batch_query(node_label: str = "Concept", relationship_type: str = "RELATED") -> str:
    """
    バッチ挿入用のCypherクエリを作成

    Args:
        node_label: ノードラベル
        relationship_type: リレーションシップタイプ

    Returns:
        Cypherクエリ文字列
    """
    query = f"""
    UNWIND $batch AS row
    MERGE (a:{node_label} {{name: row.source}})
    MERGE (b:{node_label} {{name: row.target}})
    MERGE (a)-[r:{relationship_type}]->(b)
    ON CREATE SET r.weight = row.weight
    ON MATCH SET r.weight = row.weight
    """

    return query.strip()