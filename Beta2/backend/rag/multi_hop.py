"""
Multi-hop探索クラス
"""
import numpy as np
from typing import List, Dict, Any, Set, Tuple

import networkx as nx
from ..model. embed import ensure_bge_m3
from ..builders.retrieval_builder import RetrievalBuilder


class MultiHopExplorer:
    """Multi-hop探索を担当"""

    def __init__(self, config: dict, logger):
        self.config = config
        self.logger = logger
        self.visited_paths = set()

    def explore_multi_hop_paths(
        self,
        kg: nx.Graph,
        query: str,
        retrieval_chunks: List[str] = None,
        max_steps: int = 5,
        top_k_per_hop: int = 3,
        confidence_threshold: float = 0.7,
        extend_on_low_confidence: bool = True
    ) -> Dict[str, Any]:
        """
        Multi-hop探索を実行

        Args:
            kg: NetworkXグラフ
            query: 検索クエリ
            retrieval_chunks: 開始点となるチャンクID（オプション）
            max_steps: 最大ホップ数
            top_k_per_hop: 各ホップで探索する上位K個
            confidence_threshold: 信頼度の閾値
            extend_on_low_confidence: 信頼度が低い場合に探索を拡張するか

        Returns:
            {
                'paths': ランク付けされたパスのリスト,
                'entities': 訪問したエンティティ,
                'confidence': 信頼度スコア,
                'steps_used': 実際に使用したステップ数,
                'evidence': エビデンステキスト
            }
        """
        self.logger.info(f"🔍 Starting multi-hop exploration (max_steps={max_steps})")

        # ============================================================
        # 1. 開始エンティティの決定
        # ============================================================
        start_entities = set()

        if retrieval_chunks:
            # Retrievalで取得したチャンクから開始
            start_entities = self._resolve_entities_from_chunks(retrieval_chunks, kg)

        if not start_entities:
            # フォールバック: クエリに最も関連するエンティティを抽出
            start_entities = self._extract_query_entities(query, kg, top_k=5)

        if not start_entities:
            self.logger.warning("⚠️  No starting entities found")
            return {
                'paths': [],
                'entities': [],
                'confidence': 0.0,
                'steps_used': 0,
                'evidence': []
            }

        self.logger.info(f"  → Starting from {len(start_entities)} entities: {list(start_entities)[:3]}...")

        # ============================================================
        # 2. 各開始エンティティから探索
        # ============================================================
        all_paths = []
        visited_entities = set()
        evidence_texts = []

        for start_entity in list(start_entities)[:top_k_per_hop]:
            if start_entity not in kg.nodes():
                self.logger.debug(f"  Entity '{start_entity}' not in graph, skipping")
                continue

            path_result = self._explore_from_entity(
                kg,
                start_entity,
                query,
                max_steps=max_steps,
                visited=set()
            )

            all_paths.extend(path_result['paths'])
            visited_entities.update(path_result['visited'])

            # エビデンステキストを収集
            for path_info in path_result['paths'][:5]:  # Top 5のみ
                path = path_info['path']
                evidence_texts.append(' → '.join(path))

        # 全体の信頼度を計算
        if all_paths:
            confidence = np.mean([p['score'] for p in all_paths])
        else:
            confidence = 0.0

        current_step = max_steps

        self.logger.info(
            f"  → Found {len(all_paths)} paths with confidence {confidence:.2f}"
        )

        # ============================================================
        # 3. 信頼度が低い場合は拡張
        # ============================================================
        if extend_on_low_confidence and confidence < confidence_threshold:
            extended_steps = max_steps + 2
            self.logger.info(
                f"  → Low confidence ({confidence:.2f} < {confidence_threshold}), "
                f"extending to {extended_steps} steps"
            )

            # 再探索
            extended_paths = []
            for start_entity in list(start_entities)[:top_k_per_hop]:
                if start_entity not in kg.nodes():
                    continue

                path_result = self._explore_from_entity(
                    kg,
                    start_entity,
                    query,
                    max_steps=extended_steps,
                    visited=set()  # リセット
                )

                extended_paths.extend(path_result['paths'])
                confidence = max(confidence, path_result['confidence'])

            if len(extended_paths) > len(all_paths):
                all_paths = extended_paths
                current_step = extended_steps
                self.logger.info(f"  → Extended search found {len(all_paths)} paths")

        # ============================================================
        # 4. パスのスコアリングとランキング
        # ============================================================
        ranked_paths = self._rank_paths(all_paths, query, kg)

        return {
            'paths': ranked_paths[:10],  # Top 10
            'entities': list(visited_entities),
            'confidence': confidence,
            'steps_used': current_step,
            'evidence': evidence_texts
        }

    def _precompute_representative_paths(
        self,
        kg: nx.Graph,
        documents: List[Any],
        num_sample_queries: int = 10
    ) -> None:
        """
        代表的なクエリでパスを事前計算し、グラフに保存

        Args:
            kg: NetworkXグラフ
            documents: ドキュメントリスト
            num_sample_queries: サンプルクエリ数
        """
        self.logger.info(f"Computing representative paths for {num_sample_queries} sample queries...")

        # ============================================================
        # 1. サンプルクエリの生成
        # ============================================================
        sample_queries = self._generate_sample_queries(documents, kg, num_sample_queries)

        if not sample_queries:
            self.logger.warning("  → No sample queries generated, skipping path pre-computation")
            return

        self.logger.info(f"  Generated {len(sample_queries)} sample queries")

        # ============================================================
        # 2. 各クエリでMulti-hop探索を実行
        # ============================================================
        all_paths = []
        path_count = 0

        for i, query in enumerate(sample_queries):
            try:
                result = self.explore_multi_hop_paths(
                    kg=kg,
                    query=query,
                    max_steps=5,
                    top_k_per_hop=3,
                    extend_on_low_confidence=False  # 事前計算では拡張しない
                )

                # 高品質なパスのみ保存（confidence > 0.5）
                for path_info in result['paths']:
                    if path_info.get('final_score', 0) > 0.5:
                        all_paths.append(path_info)
                        path_count += 1

                if (i + 1) % 5 == 0:
                    self.logger.info(f"  Processed {i+1}/{len(sample_queries)} queries...")

            except Exception as e:
                self.logger.debug(f"  Query '{query[:30]}...' failed: {type(e).__name__}")
                continue

        self.logger.info(f"  → Computed {path_count} high-quality paths")

        # ============================================================
        # 3. パス情報をグラフのノード/エッジに保存
        # ============================================================
        self._store_paths_in_graph(kg, all_paths)

    def _generate_sample_queries(
        self,
        documents: List[Any],
        kg: nx.Graph,
        num_queries: int = 10
    ) -> List[str]:
        """
        ドキュメントから代表的なクエリを生成

        Args:
            documents: ドキュメントリスト
            kg: NetworkXグラフ
            num_queries: 生成するクエリ数

        Returns:
            サンプルクエリのリスト
        """
        queries = []

        # ============================================================
        # 戦略1: 中心性の高いノードをクエリにする
        # ============================================================
        try:
            # 次数中心性を計算
            degree_centrality = nx.degree_centrality(kg)

            # 上位ノードを取得
            top_nodes = sorted(
                degree_centrality.items(),
                key=lambda x: x[1],
                reverse=True
            )[:num_queries // 2]

            # ノード名をクエリとして使用
            for node, _ in top_nodes:
                queries.append(f"What is {node}?")
                queries.append(f"How does {node} work?")

        except Exception as e:
            self.logger.debug(f"Centrality-based query generation failed: {e}")

        # ============================================================
        # 戦略2: ドキュメントのメタデータからクエリを生成
        # ============================================================
        for doc in documents[:num_queries // 2]:
            # メタデータに'question'があればそれを使用
            question = doc.metadata.get('question')
            if question:
                queries.append(question)
            else:
                # テキストの最初の文を使用
                text = doc.text.strip()
                if text:
                    first_sentence = text.split('.')[0][:100]
                    if len(first_sentence) > 10:
                        queries.append(first_sentence)

        # ============================================================
        # 戦略3: エンティティペアの関係を問うクエリ
        # ============================================================
        try:
            # 重みの高いエッジを取得
            high_weight_edges = sorted(
                kg.edges(data=True),
                key=lambda x: x[2].get('weight', 0),
                reverse=True
            )[:num_queries // 3]

            for u, v, data in high_weight_edges:
                relation = data.get('relation', 'related to')
                queries.append(f"How is {u} {relation} {v}?")

        except Exception as e:
            self.logger.debug(f"Edge-based query generation failed: {e}")

        # 重複を除去してシャッフル
        queries = list(set(queries))
        import random
        random.shuffle(queries)

        return queries[:num_queries]

    def _store_paths_in_graph(
        self,
        kg: nx.Graph,
        paths: List[Dict]
    ) -> None:
        """
        計算されたパスをグラフのノード/エッジ属性に保存

        Args:
            kg: NetworkXグラフ
            paths: パス情報のリスト
        """
        self.logger.info("  Storing path information in graph...")

        # ============================================================
        # 1. 各ノードが含まれるパス数をカウント
        # ============================================================
        from collections import defaultdict
        node_path_counts = defaultdict(int)
        node_avg_scores = defaultdict(list)

        for path_info in paths:
            path = path_info.get('path', [])
            score = path_info.get('final_score', 0)

            for node in path:
                if kg.has_node(node):
                    node_path_counts[node] += 1
                    node_avg_scores[node].append(score)

        # ノードに属性を追加
        for node in kg.nodes():
            kg.nodes[node]['path_frequency'] = node_path_counts.get(node, 0)

            scores = node_avg_scores.get(node, [])
            if scores:
                kg.nodes[node]['avg_path_score'] = float(np.mean(scores))
            else:
                kg.nodes[node]['avg_path_score'] = 0.0

        # ============================================================
        # 2. 各エッジが含まれるパス数をカウント
        # ============================================================
        edge_path_counts = defaultdict(int)
        edge_avg_scores = defaultdict(list)

        for path_info in paths:
            path = path_info.get('path', [])
            score = path_info.get('final_score', 0)

            # パス内の連続するノードペアをエッジとして記録
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]

                # 無向グラフとして扱う
                edge_key = tuple(sorted([u, v]))
                edge_path_counts[edge_key] += 1
                edge_avg_scores[edge_key].append(score)

        # エッジに属性を追加
        for u, v in kg.edges():
            edge_key = tuple(sorted([u, v]))

            kg[u][v]['path_frequency'] = edge_path_counts.get(edge_key, 0)

            scores = edge_avg_scores.get(edge_key, [])
            if scores:
                kg[u][v]['avg_path_score'] = float(np.mean(scores))
            else:
                kg[u][v]['avg_path_score'] = 0.0

        # 統計情報をログ出力
        total_nodes_with_paths = sum(1 for n in kg.nodes() if kg.nodes[n]['path_frequency'] > 0)
        total_edges_with_paths = sum(1 for u, v in kg.edges() if kg[u][v]['path_frequency'] > 0)

        self.logger.info(
            f"  → {total_nodes_with_paths}/{len(kg.nodes())} nodes and "
            f"{total_edges_with_paths}/{len(kg.edges())} edges have path information"
        )

    def _extract_query_entities(
        self,
        query: str,
        kg: nx.Graph,
        top_k: int = 5
    ) -> Set[str]:
        """
        クエリから関連エンティティを抽出

        Args:
            query: 検索クエリ
            kg: NetworkXグラフ
            top_k: 上位K個のエンティティを返す

        Returns:
            エンティティ名のセット
        """

        embed_model = ensure_bge_m3()

        # クエリの埋め込み
        query_emb = embed_model.get_text_embedding(query)
        query_emb = np.array(query_emb, dtype=np.float32)
        norm = np.linalg.norm(query_emb)
        if norm > 1e-9:
            query_emb = query_emb / norm

        # 全エンティティとの類似度計算
        entity_scores = []

        for entity in kg.nodes():
            try:
                entity_emb = embed_model.get_text_embedding(entity)
                entity_emb = np.array(entity_emb, dtype=np.float32)
                norm = np.linalg.norm(entity_emb)
                if norm > 1e-9:
                    entity_emb = entity_emb / norm

                similarity = float(np.dot(query_emb, entity_emb))
                entity_scores.append((entity, similarity))

            except Exception:
                continue

        # スコア順にソート
        entity_scores.sort(key=lambda x: x[1], reverse=True)

        # Top K を返す
        top_entities = {entity for entity, _ in entity_scores[:top_k]}

        return top_entities

    def _resolve_entities_from_chunks(
        self,
        chunk_ids: Set[str],
        kg: nx.Graph
    ) -> Set[str]:
        """
        チャンクIDから実際のエンティティ名に変換

        Args:
            chunk_ids: チャンクIDのセット
            kg: NetworkXグラフ

        Returns:
            エンティティ名のセット
        """
        entities = set()

        for chunk_id in chunk_ids:
            # chunk_idがすでにエンティティ名の場合
            if chunk_id in kg.nodes():
                entities.add(chunk_id)
                continue

            # ============================================================
            # 2. チャンクIDからエンティティを推定
            # ============================================================

            # パターン1: "doc_X_chunkY_hash" 形式
            # → グラフのノード属性 'chunk_id' を持つノードを検索
            for node, data in kg.nodes(data=True):
                node_chunk_ids = data.get('chunk_ids', [])

                # chunk_ids が文字列の場合もあるので対応
                if isinstance(node_chunk_ids, str):
                    node_chunk_ids = [node_chunk_ids]

                if chunk_id in node_chunk_ids:
                    entities.add(node)

            # パターン2: チャンクID内にエンティティ名が含まれる
            # （例: chunk_id = "attention_mechanism_chunk3"）
            # → グラフ内のノード名がchunk_idに部分一致するか確認
            chunk_id_lower = chunk_id.lower()
            for node in kg.nodes():
                node_lower = node.lower()

                # 部分一致（少なくとも5文字以上）
                if len(node_lower) >= 5 and node_lower in chunk_id_lower:
                    entities.add(node)
                elif len(chunk_id_lower) >= 5 and chunk_id_lower in node_lower:
                    entities.add(node)

        if not entities:
            self.logger.debug(
                f"  Could not resolve entities from {len(chunk_ids)} chunk IDs"
            )

        return entities

    def _explore_from_entity(
        self,
        kg: nx.Graph,
        start_entity: str,
        query: str,
        max_steps: int,
        visited: Set[str]
    ) -> Dict[str, Any]:
        """
        特定エンティティから深さ優先探索

        Returns:
            {
                'paths': [パスのリスト],
                'visited': 訪問ノード,
                'steps': 最大ステップ数,
                'confidence': 信頼度
            }
        """
        paths = []
        visited.add(start_entity)

        embed_model = ensure_bge_m3()

        # クエリの埋め込み
        query_emb = embed_model.get_text_embedding(query)
        query_emb = np.array(query_emb, dtype=np.float32)
        norm = np.linalg.norm(query_emb)
        if norm > 1e-9:
            query_emb = query_emb / norm

        # BFS
        queue = [(start_entity, [start_entity], 0)]  # (current, path, depth)
        # パス数制限
        max_paths = self.config.get('multihop_max_paths', 50)

        while queue and len(paths) < max_paths:
            current, path, depth = queue.pop(0)

            if depth >= max_steps:
                continue

            # 隣接ノードを探索
            neighbors = list(kg.neighbors(current))

            # 各隣接ノードのスコアを計算
            neighbor_scores = []
            for neighbor in neighbors:
                if neighbor in visited:
                    continue

                # エンティティ名の埋め込み
                try:
                    entity_emb = embed_model.get_text_embedding(neighbor)
                    entity_emb = np.array(entity_emb, dtype=np.float32)
                    norm = np.linalg.norm(entity_emb)
                    if norm > 1e-9:
                        entity_emb = entity_emb / norm

                    # クエリとの類似度
                    similarity = float(np.dot(query_emb, entity_emb))

                    # エッジの重み
                    edge_weight = kg[current][neighbor].get('weight', 0.5)

                    # 総合スコア
                    score = similarity * 0.6 + edge_weight * 0.4

                    neighbor_scores.append((neighbor, score))
                except Exception:
                    continue

            # スコア上位を選択
            neighbor_scores.sort(key=lambda x: x[1], reverse=True)
            beam_width = self.config.get('multihop_beam_width', 2)
            top_neighbors = neighbor_scores[:beam_width]  # 3 → 2

            for neighbor, score in top_neighbors:
                new_path = path + [neighbor]

                # パス重複チェック
                path_tuple = tuple(new_path)
                if path_tuple in self.visited_paths:
                    continue
                self.visited_paths.add(path_tuple)

                visited.add(neighbor)

                # パスを保存
                paths.append({
                    'path': new_path,
                    'score': score,
                    'depth': depth + 1
                })

                # キューに追加
                queue.append((neighbor, new_path, depth + 1))

        # 信頼度計算（パスの平均スコア）
        confidence = np.mean([p['score'] for p in paths]) if paths else 0.0

        return {
            'paths': paths,
            'visited': visited,
            'steps': max_steps,
            'confidence': float(confidence)
        }

    def _rank_paths(
        self,
        paths: List[Dict],
        query: str,
        kg: nx.Graph
    ) -> List[Dict]:
        """
        パスをスコアでランキング
        """
        if not paths:
            return []
        embed_model = ensure_bge_m3()

        query_emb = embed_model.get_text_embedding(query)
        query_emb = np.array(query_emb, dtype=np.float32)
        norm = np.linalg.norm(query_emb)
        if norm > 1e-9:
            query_emb = query_emb / norm

        # 各パスに最終スコアを計算
        for path_info in paths:
            path = path_info['path']

            # パスの長さペナルティ（長すぎると信頼度低下）
            length_penalty = 1.0 / (1.0 + 0.1 * len(path))

            # エッジ重みの平均
            edge_weights = []
            for i in range(len(path) - 1):
                if kg.has_edge(path[i], path[i+1]):
                    edge_weights.append(kg[path[i]][path[i+1]].get('weight', 0.5))

            avg_edge_weight = np.mean(edge_weights) if edge_weights else 0.5
            #  パス全体とクエリの関連性スコア
            path_query_relevance = 0.0
            entity_similarities = []

            for entity in path:
                try:
                    entity_emb = embed_model.get_text_embedding(entity)
                    entity_emb = np.array(entity_emb, dtype=np.float32)
                    norm = np.linalg.norm(entity_emb)
                    if norm > 1e-9:
                        entity_emb = entity_emb / norm

                    similarity = float(np.dot(query_emb, entity_emb))
                    entity_similarities.append(similarity)
                except Exception:
                    continue

            if entity_similarities:
                # パス内の最大類似度を使用（最も関連するエンティティを重視）
                path_query_relevance = max(entity_similarities)

            # 最終スコア
            final_score = (
                path_info['score'] * 0.4 +
                avg_edge_weight * 0.25 +
                length_penalty * 0.15 +
                path_query_relevance * 0.2
            )

            path_info['final_score'] = final_score
            path_info['query_relevance'] = path_query_relevance  # デバッグ用に保存

        # スコア順にソート
        paths.sort(key=lambda x: x.get('final_score', 0), reverse=True)

        return paths

    def query_with_multihop(
        self,
        query: str,
        kg: nx.Graph,
        retrieval_store: Dict = None,
        max_steps: int = 5,
        top_k_retrieval: int = 5,
        top_k_paths: int = 10
    ) -> Dict[str, Any]:
        """
        Multi-hop探索を使ったクエリ実行

        Args:
            query: 検索クエリ
            kg: NetworkXグラフ
            retrieval_store: Retrievalストア（オプション）
            max_steps: 最大ホップ数
            top_k_retrieval: Retrieval結果の上位K件
            top_k_paths: 返すパスの上位K件

        Returns:
            {
                'paths': 発見されたパス,
                'retrieval_docs': Retrievalで取得したドキュメント,
                'confidence': 信頼度,
                'answer': 統合された回答（オプション）
            }
        """
        self.logger.info(f"🔍 Query: '{query}'")

        results = {
            'paths': [],
            'retrieval_docs': [],
            'confidence': 0.0,
            'answer': None
        }

        # ============================================================
        # 1. Retrieval（提供されている場合）
        # ============================================================
        retrieval_chunks = []

        if retrieval_store:
            try:
                embed_model = ensure_bge_m3()
                retriever = RetrievalBuilder(embed_model, self.logger)

                retrieval_results = retriever.retrieve(
                    store=retrieval_store,
                    query=query,
                    top_k=top_k_retrieval
                )

                for score, doc, graph_chunk_ids in retrieval_results:
                    results['retrieval_docs'].append({
                        'text': doc.text,
                        'score': score,
                        'metadata': doc.metadata
                    })
                    retrieval_chunks.extend(graph_chunk_ids)

                self.logger.info(
                    f"  → Retrieval: {len(results['retrieval_docs'])} docs, "
                    f"{len(retrieval_chunks)} graph chunks"
                )

            except Exception as e:
                self.logger.warning(f"⚠️  Retrieval failed: {type(e).__name__}")

        # ============================================================
        # 2. Multi-hop探索
        # ============================================================
        try:
            path_result = self.explore_multi_hop_paths(
                kg=kg,
                query=query,
                retrieval_chunks=retrieval_chunks if retrieval_chunks else None,
                max_steps=max_steps,
                top_k_per_hop=3,
                confidence_threshold=0.7,
                extend_on_low_confidence=True
            )

            results['paths'] = path_result['paths'][:top_k_paths]
            results['confidence'] = path_result['confidence']

            self.logger.info(
                f"  → Multi-hop: {len(results['paths'])} paths, "
                f"confidence={results['confidence']:.2f}"
            )

        except Exception as e:
            self.logger.error(f"🚨 Multi-hop exploration failed: {type(e).__name__}")
            self.logger.error(f"   {str(e)[:200]}")

            if self.logger.level <= 10:  # logging.DEBUG
                import traceback
                self.logger.debug(traceback.format_exc())

        # ============================================================
        # 3. 結果の統合（オプション）
        # ============================================================
        if results['paths'] and results['retrieval_docs']:
            results['answer'] = self._synthesize_answer(
                query=query,
                paths=results['paths'],
                retrieval_docs=results['retrieval_docs']
            )

        return results

    def _synthesize_answer(
        self,
        query: str,
        paths: List[Dict],
        retrieval_docs: List[Dict]
    ) -> str:
        """
        パスとRetrievalドキュメントから回答を統合

        Args:
            query: クエリ
            paths: Multi-hopで発見されたパス
            retrieval_docs: Retrievalで取得したドキュメント

        Returns:
            統合された回答文字列
        """
        # 簡易実装（LLMを使った統合は別途実装可能）

        answer_parts = []

        # パスからのエビデンス
        answer_parts.append("**From Knowledge Graph:**")
        for i, path_info in enumerate(paths[:3], 1):
            path = path_info['path']
            score = path_info.get('final_score', 0)
            path_str = ' → '.join(path)
            answer_parts.append(f"{i}. {path_str} (score: {score:.2f})")

        # Retrievalドキュメントからのエビデンス
        answer_parts.append("\n**From Documents:**")
        for i, doc_info in enumerate(retrieval_docs[:3], 1):
            text_preview = doc_info['text'][:150] + "..."
            score = doc_info['score']
            answer_parts.append(f"{i}. {text_preview} (score: {score:.2f})")

        return '\n'.join(answer_parts)