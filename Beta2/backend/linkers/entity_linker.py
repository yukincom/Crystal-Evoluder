"""
エンティティ統合クラス
"""
import numpy as np
from typing import List, Dict, Any, Tuple
from collections import defaultdict

import networkx as nx
from model import ensure_bge_m3


class EntityLinker:
    """エンティティの統合・リンクを担当"""

    def __init__(self, config: dict, logger):
        self.config = config
        self.logger = logger

    def link_entities(
        self,
        kg: nx.Graph,
        similarity_threshold: float = 0.88,
        use_embedding: bool = True
    ) -> Tuple[nx.Graph, Dict[str, str]]:
        """
        同一実体を統合してグラフをクリーンアップ

        Args:
            kg: NetworkXグラフ
            similarity_threshold: 統合する類似度の閾値（0.95推奨）
            use_embedding: True=埋め込み類似度、False=文字列類似度

        Returns:
            (統合後のグラフ, エンティティマッピング)

        例:
            mapping = {
                'Self-Attention': 'self_attention',
                'the attention mechanism': 'self_attention',
                'it': 'self_attention'  # coref解決が必要
            }
        """
        self.logger.info(f"🔗 Starting entity linking (threshold={similarity_threshold})")

        nodes = list(kg.nodes())
        entity_mapping = {}  # old_name -> canonical_name
        clusters = []  # [[類似エンティティのリスト], ...]

        # ============================================================
        # 1. エンティティのクラスタリング
        # ============================================================
        if use_embedding:
            clusters = self._cluster_entities_by_embedding(
                nodes, similarity_threshold
            )
        else:
            clusters = self._cluster_entities_by_string(nodes)

        # ============================================================
        # 2. 各クラスタの代表名を決定
        # ============================================================
        for cluster in clusters:
            if len(cluster) <= 1:
                continue

            # 代表名の選択戦略
            canonical = self._select_canonical_name(cluster, kg)

            for entity in cluster:
                if entity != canonical:
                    entity_mapping[entity] = canonical

        self.logger.info(f"  → {len(entity_mapping)} entities will be merged")

        # ============================================================
        # 3. グラフの統合
        # ============================================================
        merged_kg = self._merge_graph_entities(kg, entity_mapping)

        self.logger.info(
            f"✅ Entity linking complete: "
            f"{len(kg.nodes)} → {len(merged_kg.nodes)} nodes"
        )

        return merged_kg, entity_mapping

    def _cluster_entities_by_embedding(
        self,
        entities: List[str],
        threshold: float
    ) -> List[List[str]]:
        """
        埋め込みベースのクラスタリング

        Returns:
            [[類似エンティティ], [類似エンティティ], ...]
        """
        # エンティティの埋め込み計算
        embeddings = []
        valid_entities = []

        for entity in entities:
            try:
                embed_model = ensure_bge_m3()
                emb = embed_model.get_text_embedding(entity)
                emb = np.array(emb, dtype=np.float32)
                norm = np.linalg.norm(emb)

                if norm > 1e-9:
                    emb = emb / norm
                    embeddings.append(emb)
                    valid_entities.append(entity)
            except Exception as e:
                self.logger.debug(f"Embedding failed for '{entity}': {e}")

        if len(embeddings) == 0:
            return []

        embeddings = np.vstack(embeddings)

        # 類似度マトリクス計算
        sim_matrix = embeddings @ embeddings.T

        # Union-Find でクラスタリング
        parent = {i: i for i in range(len(valid_entities))}

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # 類似度が閾値以上のペアを統合
        for i in range(len(valid_entities)):
            for j in range(i + 1, len(valid_entities)):
                if sim_matrix[i, j] >= threshold:
                    union(i, j)

        # クラスタを構築
        clusters_dict = {}
        for i, entity in enumerate(valid_entities):
            root = find(i)
            if root not in clusters_dict:
                clusters_dict[root] = []
            clusters_dict[root].append(entity)

        clusters = list(clusters_dict.values())

        self.logger.info(
            f"  → Found {len(clusters)} clusters from {len(valid_entities)} entities"
        )

        return clusters

    def _cluster_entities_by_string(
        self,
        entities: List[str]
    ) -> List[List[str]]:
        """
        文字列類似度ベースのクラスタリング（高速だが精度低い）

        使用ケース：
        - "Self-Attention" と "self-attention" を統合
        - "GPT-3" と "GPT3" を統合
        """

        clusters_dict = {}
        normalized = {}

        for entity in entities:
            # 正規化（小文字化、記号除去）
            norm = entity.lower().replace('-', '').replace('_', '').replace(' ', '')
            normalized[entity] = norm
            if norm not in clusters_dict:
                clusters_dict[norm] = []
            clusters_dict[norm].append(entity)

        # 2つ以上のエンティティがある正規化形のみ返す
        clusters = [v for v in clusters_dict.values() if len(v) > 1]

        return clusters

    def _select_canonical_name(
        self,
        cluster: List[str],
        kg: nx.Graph
    ) -> str:
        """
        クラスタの代表名を選択

        戦略：
        1. 最も次数が高い（多くの関係を持つ）
        2. 最も長い名前（情報量が多い）
        3. アルファベット順
        """
        # 次数でスコアリング
        scores = {}
        for entity in cluster:
            degree = kg.degree(entity) if kg.has_node(entity) else 0
            length = len(entity)

            # スコア = 次数 * 10 + 長さ
            scores[entity] = degree * 10 + length

        # スコアが最大のものを選択
        canonical = max(cluster, key=lambda e: scores[e])

        self.logger.debug(
            f"  Cluster: {cluster} → Canonical: '{canonical}'"
        )

        return canonical

    def _merge_graph_entities(
        self,
        kg: nx.Graph,
        entity_mapping: Dict[str, str]
    ) -> nx.Graph:
        """
        エンティティマッピングに従ってグラフを統合

        Args:
            kg: 元のグラフ
            entity_mapping: {old_name: canonical_name}

        Returns:
            統合後のグラフ
        """
        merged_kg = nx.Graph()

        # ノードをコピー（マッピング適用）
        for node, data in kg.nodes(data=True):
            canonical = entity_mapping.get(node, node)

            if merged_kg.has_node(canonical):
                # 既存ノードの属性をマージ
                for key, value in data.items():
                    if key not in merged_kg.nodes[canonical]:
                        merged_kg.nodes[canonical][key] = value
            else:
                merged_kg.add_node(canonical, **data)

        # エッジをコピー（マッピング適用 + 重み統合）
        edge_weights = {}
        
        edge_weights = defaultdict(lambda: {
            'weight': 0.0,
            'intra_raw': 0.0,
            'inter_raw': 0.0,
            'relations': []
        })

        for u, v, data in kg.edges(data=True):
            u_canonical = entity_mapping.get(u, u)
            v_canonical = entity_mapping.get(v, v)

            # 自己ループは除外
            if u_canonical == v_canonical:
                continue

            # 正規化されたエッジキー（方向なし）
            edge_key = tuple(sorted([u_canonical, v_canonical]))

            # 重みを累積
            edge_weights[edge_key]['weight'] += data.get('weight', 0.0)
            edge_weights[edge_key]['intra_raw'] += data.get('intra_raw', 0.0)
            edge_weights[edge_key]['inter_raw'] += data.get('inter_raw', 0.0)

            # 関係タイプを記録
            rel = data.get('relation', 'RELATED')
            if rel not in edge_weights[edge_key]['relations']:
                edge_weights[edge_key]['relations'].append(rel)

        # エッジを追加
        for (u, v), weights in edge_weights.items():
            merged_kg.add_edge(
                u, v,
                weight=weights['weight'],
                intra_raw=weights['intra_raw'],
                inter_raw=weights['inter_raw'],
                relation=weights['relations'][0] if weights['relations'] else 'RELATED',
                relation_types=weights['relations']
            )

        return merged_kg