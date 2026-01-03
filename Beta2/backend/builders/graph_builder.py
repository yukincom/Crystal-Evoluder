"""
グラフ構築クラス
"""
import numpy as np
import networkx as nx
from typing import List, Dict, Any, Tuple
from collections import Counter, defaultdict

from llama_index.core import Document, KnowledgeGraphIndex, StorageContext
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.llms.openai import OpenAI

from ..shared import HierarchicalLogger, ErrorCollector, safe_execute
from ..filters.triplet_filter import TripletFilter
from ..linkers.entity_linker import EntityLinker
from ..rag.multi_hop import MultiHopExplorer


class GraphBuilder:
    """ナレッジグラフの構築を担当"""

    def __init__(self, config: dict, embed_model, logger):
        self.config = config
        self.embed_model = embed_model
        self.logger = logger
        self.hlogger = HierarchicalLogger(logger)

    def commit_to_graph(self, documents: List[Document], graph_store):
        """Neo4jにグラフを投入"""
        # 接続確認
        try:
            graph_store.query("RETURN 1")
            self.logger.info("✅ Neo4j connection verified")
        except Exception as e:
            self.logger.error(f"🚨 Neo4j connection failed: {type(e).__name__}")
            raise  # 接続できないなら処理を中断

        # 2. グラフ生成
        try:
            with self.hlogger.section("Graph Generation"):
                llm = OpenAI(
                    model=self.config['llm_model'],
                    timeout=self.config['llm_timeout']
                )

                local_graph_store = SimpleGraphStore()
                local_storage = StorageContext.from_defaults(graph_store=local_graph_store)

                self.logger.info("Building local knowledge graph...")
                index = KnowledgeGraphIndex.from_documents(
                    documents,
                    storage_context=local_storage,
                    llm=llm,
                    transformations=[SimpleNodeParser.from_defaults(chunk_size=512)],
                    embed_model=self.embed_model,
                    show_progress=True,
                    max_triplets_per_chunk=self.config['max_triplets_per_chunk']
                )

                kg = index.get_networkx_graph()
                self.logger.info(f"✅ Initial graph: {len(kg.nodes)} nodes, {len(kg.edges)} edges")

                # トリプレットをメタデータに保存
                all_triples = []

                for subj, obj, data in kg.edges(data=True):
                    rel = data.get('relation', 'RELATED')
                    all_triples.append((subj, rel, obj))

                # rel_map処理
                if hasattr(local_graph_store, 'get_rel_map'):
                    try:
                        rel_map = local_graph_store.get_rel_map()
                        self.logger.debug(f"rel_map structure: {type(rel_map)}")

                        for subj, relations in rel_map.items():
                            # relations が辞書か、リストか確認
                            if isinstance(relations, dict):
                                # 辞書の場合
                                for rel, objs in relations.items():
                                    if isinstance(objs, list):
                                        for obj in objs:
                                            if (subj, rel, obj) not in all_triples:
                                                all_triples.append((subj, rel, obj))
                                    else:
                                        if (subj, rel, objs) not in all_triples:
                                            all_triples.append((subj, rel, objs))
                            elif isinstance(relations, list):
                                # リストの場合
                                for item in relations:
                                    if isinstance(item, tuple) and len(item) == 2:
                                        rel, obj = item
                                        if (subj, rel, obj) not in all_triples:
                                            all_triples.append((subj, rel, obj))
                    except Exception as e:
                        self.logger.warning(f"Could not parse rel_map: {e}")

                self.logger.info(f"Extracted {len(all_triples)} triples (before filtering)")

                # Self-RAG統合
                # 品質フィルタを適用
                if self.config.get('enable_triplet_filter', True):
                    filter_instance = TripletFilter(self.config, self.logger)
                    filtered_triples, rejected_triples, filter_stats = filter_instance.filter_triplets(
                        all_triples,
                        quality_threshold=self.config.get('triplet_quality_threshold', 0.3)
                    )
                    all_triples = filtered_triples

                    self.logger.info(
                        f"After filtering: {len(all_triples)} triples "
                        f"(rejection rate: {filter_stats['rejection_rate']:.1%})"
                    )

                # Self-RAGを適用（チャンクごとに処理）
                if self.config.get('enable_self_rag', False):
                    with self.hlogger.section("Self-RAG Refinement"):
                        filter_instance = TripletFilter(self.config, self.logger)
                        # ドキュメントごとにトリプレットを再生成
                        refined_all_triples = []
                        total_self_rag_stats = {
                            'attempted': 0,
                            'succeeded': 0,
                            'failed': 0
                        }

                        # ドキュメントとそのトリプレットをマッピング
                        doc_triplet_map = filter_instance._map_triplets_to_documents(all_triples, documents)

                        for doc_idx, (doc, doc_triplets) in enumerate(doc_triplet_map.items()):
                            if not doc_triplets:
                                continue

                            try:
                                refined_triplets, stats = filter_instance.self_rag_triplets(
                                    doc_triplets,
                                    doc.text,
                                    llm=llm  # 既存のLLMインスタンスを使用
                                )

                                refined_all_triples.extend(refined_triplets)

                                # 統計を集計
                                if stats.get('self_rag_applied'):
                                    ref_stats = stats['refinement_stats']
                                    total_self_rag_stats['attempted'] += ref_stats['attempted']
                                    total_self_rag_stats['succeeded'] += ref_stats['succeeded']
                                    total_self_rag_stats['failed'] += ref_stats['failed']

                                if (doc_idx + 1) % 10 == 0:
                                    self.logger.info(f"  Processed {doc_idx + 1}/{len(doc_triplet_map)} documents...")

                            except Exception as e:
                                self.logger.warning(f"  Self-RAG failed for doc {doc_idx}: {type(e).__name__}")
                                # 失敗時は元のトリプレットを保持
                                refined_all_triples.extend(doc_triplets)

                        # トリプレットを更新
                        all_triples = refined_all_triples

                        self.logger.info(
                            f"✅ Self-RAG complete: "
                            f"{total_self_rag_stats['succeeded']} improved, "
                            f"{total_self_rag_stats['attempted']} attempted, "
                            f"final count: {len(all_triples)}"
                        )

                # 再度品質フィルタを適用
                if self.config.get('enable_triplet_filter', True):
                    filter_instance = TripletFilter(self.config, self.logger)
                    filtered_triples, rejected_triples, filter_stats = filter_instance.filter_triplets(
                        all_triples,
                        quality_threshold=self.config.get('triplet_quality_threshold', 0.3)
                    )
                    all_triples = filtered_triples

                    # 統計情報を活用
                    self.logger.info(
                        f"After filtering: {len(all_triples)} triples "
                        f"(rejection rate: {filter_stats['rejection_rate']:.1%})"
                    )

                    # 品質が低い場合は警告
                    if filter_stats['avg_quality'] < 0.5:
                        self.logger.warning("⚠️  Low average triplet quality!")

                    # デバッグモードならリジェクト例を表示
                    if rejected_triples and self.logger.level <= 10:  # logging.DEBUG
                        self.logger.debug("Sample rejected triplets:")
                        for s, r, o in rejected_triples[:3]:
                            self.logger.debug(f"  ({s}, {r}, {o})")

                # すべてのドキュメントに全トリプルを共有
                for doc in documents:
                    doc.metadata['triples'] = all_triples

        except Exception as e:
            self.logger.error(
                f"🚨 Graph generation failed: {type(e).__name__}"
            )
            raise

        # Entity Linking
        try:
            with self.hlogger.section("Entity Linking"):
                linker = EntityLinker(self.config, self.logger)
                kg, entity_mapping = linker.link_entities(
                    kg,
                    similarity_threshold=self.config['entity_linking_threshold'],
                    use_embedding=True
                )

                # トリプレット更新
                updated_triples = []
                for s, r, o in all_triples:
                    s_new = entity_mapping.get(s, s)
                    o_new = entity_mapping.get(o, o)
                    if s_new != o_new:  # 自己ループ除外
                        updated_triples.append((s_new, r, o_new))

                # ドキュメントのメタデータを更新
                for doc in documents:
                    doc.metadata['triples'] = updated_triples

                self.logger.info(f"Updated triples: {len(all_triples)} → {len(updated_triples)}")

        except Exception as e:
            self.logger.warning(f"⚠️  Entity linking failed: {e}")
            # Entity Linking失敗でも処理は継続

        # パス情報をグラフに統合
        try:
            with self.hlogger.section("Merging Path Information"):
                self.merge_paths_into_kg(kg, documents)
                self.logger.info(f"✅ Path info merged: {len(kg.nodes)} nodes, {len(kg.edges)} edges")

        except Exception as e:
            self.logger.warning(f"⚠️  Path merging failed: {type(e).__name__} - {str(e)[:100]}")

            self.logger.info("  → Continuing without path information")

        # デバッグ情報を記録
        if self.logger.level <= 10:  # logging.DEBUG
            import traceback
            self.logger.debug(f"Path merge traceback:\n{traceback.format_exc()}")

        # documentsからpaths情報を削除（中途半端なデータを残さない）
        for doc in documents:
            doc.metadata.pop('paths', None)
            doc.metadata.pop('path_distances', None)

        # RAPL最適化
        try:
            with self.hlogger.section("Graph Optimization (RAPL)"):
                optimized_kg = self._optimize_graph_rapl(kg, documents)
                self.logger.info(
                    f"✅ Optimized graph: {len(optimized_kg.nodes)} nodes, "
                    f"{len(optimized_kg.edges)} edges"
                )
        except Exception as e:
            self.logger.error(
                f"🚨 Graph optimization failed: {e}")
            optimized_kg = kg

        # Multi-hop パス探索（サンプルクエリで代表的なパスを計算）
        try:
            with self.hlogger.section("Multi-hop Path Pre-computation"):
                explorer = MultiHopExplorer(self.config, self.logger)
                explorer._precompute_representative_paths(optimized_kg, documents)
                self.logger.info("✅ Representative paths computed and stored")

        except Exception as e:
            self.logger.warning(f"⚠️  Path pre-computation failed: {type(e).__name__} - {str(e)[:100]}")
            self.logger.info("  → Continuing without pre-computed paths")

            if self.logger.level <= 10:  # logging.DEBUG
                import traceback
                self.logger.debug(f"Path pre-computation traceback:\n{traceback.format_exc()}")

        # 最適化されたグラフをNeo4jに反映
        try:
            with self.hlogger.section("Updating Neo4j"):
                result = self._update_neo4j_structure(optimized_kg, graph_store)

                # result が None の場合のフォールバック
                if result is None:
                    result = {'updated': 0, 'skipped': 0, 'failed': 0, 'error_details': []}

                self.logger.warning("⚠️  _update_neo4j_structure returned None")

                # 結果サマリー
                self.logger.info(
                    f"✅ Neo4j update complete:\n"
                    f"   - Updated: {result.get('updated', 0)} edges\n"
                    f"   - Skipped: {result.get('skipped', 0)} edges\n"
                    f"   - Failed: {result.get('failed', 0)} edges"
                )

                # 失敗率が高い場合は警告
                total = result.get('updated', 0) + result.get('failed', 0)
                if total > 0 and result.get('failed', 0) / total > 0.3:
                    self.logger.warning(
                        f"⚠️  High failure rate ({result.get('failed', 0)/total:.1%}). "
                        f"Check Neo4j constraints and data format."
                    )

        except Exception as e:
            self.logger.error(f"🚨 Neo4j update failed: {e}")
            raise

    def merge_paths_into_kg(self, kg, documents: List[Document]):
        """
        kg: networkx.Graph (triples turned into nodes/edges)
        documents: the same documents that have metadata['paths'] etc.
        This will:
          - count how many times each entity appears in top-k paths
          - add edge/node attributes: top_path_count, avg_path_length
        """
        path_entity_counts = Counter()
        entity_path_lengths = defaultdict(list)

        for doc in documents:
            paths = doc.metadata.get('paths', [])  # each path is a str like "A -> B -> C" OR list; adapt if needed
            distances = doc.metadata.get('path_distances', [])
            for i, p in enumerate(paths):
                # normalize path representation
                if isinstance(p, str):
                    nodes = [n.strip() for n in p.split('->') if n.strip()]
                elif isinstance(p, (list, tuple)):
                    nodes = list(p)
                else:
                    continue

                dist = distances[i] if i < len(distances) else len(nodes)-1
                for n in nodes:
                    path_entity_counts[n] += 1
                    entity_path_lengths[n].append(dist)

                # if the path describes relations, you could also add edges for consecutive nodes
                for a, b in zip(nodes, nodes[1:]):
                    if kg.has_edge(a, b):
                        # add a path_support counter on existing edge
                        kg[a][b].setdefault('path_support', 0)
                        kg[a][b]['path_support'] += 1
                    else:
                        kg.add_edge(a, b, relation='path_inferred', path_support=1)

        # inject aggregated attrs to nodes
        for n in kg.nodes():
            cnt = path_entity_counts.get(n, 0)
            lens = entity_path_lengths.get(n, [])
            avg_len = sum(lens)/len(lens) if lens else None
            kg.nodes[n]['path_top_count'] = cnt
            if avg_len is not None:
                kg.nodes[n]['path_avg_length'] = avg_len

    def _optimize_graph_rapl(self, kg, documents):
        """
        RAPL 最適化
        """

        # 1. Triples 抽出
        doc_triples = {}
        for idx, doc in enumerate(documents):
            triples = doc.metadata.get("triples", [])
            if triples:  # 空リストは除外
                doc_triples[idx] = triples

        all_triples = [t for lst in doc_triples.values() for t in lst]

        self.logger.info(f"Total triples: {len(all_triples)}")

        # Weight 格納領域の初期化
        for u, v in kg.edges():
            kg[u][v]["intra_raw"] = 0.0
            kg[u][v]["inter_raw"] = 0.0

        # 2. Intra: 文書内 triple 間相互作用
        self.logger.info("Computing intra-interactions...")
        intra_collector = ErrorCollector(self.logger)
        intra_edges = 0

        for doc_id, triples in doc_triples.items():
            try:
                entities = set()
                for s, _, o in triples:
                    entities.add(s)
                    entities.add(o)

                # Triple 間の相互作用（関係の相性を考慮）
                for i in range(len(triples)):
                    s1, r1, o1 = triples[i]
                    for j in range(i + 1, len(triples)):
                        s2, r2, o2 = triples[j]

                        # 関係の相性
                        try:
                            rel_compat = self._compute_relation_compatibility(r1, r2)

                            # エンティティの共有度
                            shared = len({s1, o1} & {s2, o2})
                            shared_score = shared * 0.5

                            # 統合重み
                            w = rel_compat * 0.6 + shared_score * 0.4

                            if w > 0.3:
                                if kg.has_edge(s1, o1):
                                    kg[s1][o1]["intra_raw"] += w
                                if kg.has_edge(s2, o2):
                                    kg[s2][o2]["intra_raw"] += w
                            intra_collector.add_success()
                        except Exception as e:
                            intra_collector.add_error(
                                context=f"doc_{doc_id}_triple_{i}_{j}",
                                error=e,
                                triple1=(s1, r1, o1),
                                triple2=(s2, r2, o2)
                            )

                # エンティティペア間のエッジ追加
                for e1 in entities:
                    for e2 in entities:
                        if e1 != e2:
                            try:
                                w = self._compute_intra_weight(e1, e2, triples, kg)
                                if w > 0.5:
                                    if kg.has_edge(e1, e2):
                                        kg[e1][e2]["weight"] = kg[e1][e2].get("weight", 0) + w
                                    else:
                                        kg.add_edge(e1, e2, relation="intra_doc", weight=w)
                                        intra_edges += 1
                            except Exception as e:
                                intra_collector.add_error(
                                    context=f"entity_pair_{e1}_{e2}",
                                    error=e
                                )
            except Exception as e:
                self.logger.error(f"Failed to process document {doc_id}: {type(e).__name__}")
                continue

        intra_collector.report("Intra-document processing", threshold=0.3)
        self.logger.info(f"Added {intra_edges} intra-document edges")

        # 3. Inter: 共有エンティティベースの高速化
        self.logger.info("Computing inter-interactions (optimized & sampled)...")
        inter_collector = ErrorCollector(self.logger)

        # 3-1. エンティティ→Triple インデックス構築
        entity_to_triples = defaultdict(set)
        for idx, (s, _, o) in enumerate(all_triples):
            entity_to_triples[s].add(idx)
            entity_to_triples[o].add(idx)

        # 3-2. エンティティを出現頻度でソート（上位のみ処理）
        entity_freq = [(entity, len(triple_indices))
                       for entity, triple_indices in entity_to_triples.items()]
        entity_freq.sort(key=lambda x: x[1], reverse=True)

        # 上位100エンティティのみ処理（調整可能）
        max_entities = min(100, len(entity_freq))
        top_entities = set(entity for entity, _ in entity_freq[:max_entities])

        self.logger.info(
            f"  Sampled {max_entities}/{len(entity_to_triples)} entities "
            f"(covering {sum(freq for _, freq in entity_freq[:max_entities])} triples)"
        )

        # 3-2. 共有エンティティがある Triple ペアのみ計算
        seen_pairs = set()
        inter_count = 0

        for _entity, triple_indices in entity_to_triples.items():
            if _entity not in top_entities:
                continue  # 上位エンティティ以外はスキップ
            if len(triple_indices) < 3:
                continue

            indices = list(triple_indices)
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    idx1, idx2 = indices[i], indices[j]
                    pair = (min(idx1, idx2), max(idx1, idx2))
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)

                    # 重み計算
                    try:
                        t1 = all_triples[idx1]
                        t2 = all_triples[idx2]
                        w = self._compute_inter_weight(t1, t2, kg=kg)

                        if w > self.config['relation_compat_threshold']:
                            s1, _, o1 = t1
                            s2, _, o2 = t2

                            # 双方向に重みを加算
                            if kg.has_edge(s1, o1):
                                kg[s1][o1]["inter_raw"] = kg[s1][o1].get("inter_raw", 0.0) + w
                            if kg.has_edge(s2, o2):
                                kg[s2][o2]["inter_raw"] = kg[s2][o2].get("inter_raw", 0.0) + w

                            inter_count += 1
                        inter_collector.add_success()

                    except Exception as e:
                        inter_collector.add_error(
                            context=f"triple_pair_{idx1}_{idx2}",
                            error=e
                        )

        inter_collector.report("Inter-document processing", threshold=0.3)
        self.logger.info(f"Added {inter_count} meaningful inter-interactions")

        # 4. Document-level linking
        self.logger.info("Computing document-level connections...")

        try:
            entity_docs = {}
            for doc_id, triples in doc_triples.items():
                for s, _, o in triples:
                    entity_docs.setdefault(s, set()).add(doc_id)
                    entity_docs.setdefault(o, set()).add(doc_id)

            doc_pairs = {}
            bridge_entities = []

            for entity_name, doc_set in entity_docs.items():
                if len(doc_set) > 1:
                    docs = list(doc_set)
                    for i, d1 in enumerate(docs):
                        for d2 in docs[i+1:]:
                            pair = (d1, d2)
                            doc_pairs[pair] = doc_pairs.get(pair, 0) + 1

                    if len(doc_set) > 2:
                        bridge_entities.append((entity_name, len(doc_set)))

            # ブリッジエンティティのログ
            if bridge_entities:
                bridge_entities.sort(key=lambda x: x[1], reverse=True)
                self.logger.info("Top bridge entities:")
                for entity_name, count in bridge_entities[:5]:
                    self.logger.info(f"  '{entity_name}': {count} documents")

            inter_doc_count = 0
            for (d1, d2), ct in doc_pairs.items():
                if ct > 2:
                    n1 = f"doc_{d1}"
                    n2 = f"doc_{d2}"

                    if not kg.has_node(n1):
                        kg.add_node(n1, type="document")
                    if not kg.has_node(n2):
                        kg.add_node(n2, type="document")

                    kg.add_edge(n1, n2, relation="inter_doc", weight=ct)
                    inter_doc_count += 1
            self.logger.info(f"Added {inter_doc_count} inter-document links")

        except Exception as e:
            self.logger.error(f"Document linking failed: {type(e).__name__} - {str(e)[:100]}")

        # 統合重み計算
        kg = self._normalize_edge_weights(kg, doc_triples, method='minmax')
        self.logger.info("Finalizing edge weights with normalization...")

        for u, v, d in kg.edges(data=True):
            intra = d.get("intra_normalized", d.get("intra_raw", 0.0))
            inter = d.get("inter_normalized", d.get("inter_raw", 0.0))

            # RAPL論文: intra重視 + inter補完
            d["weight"] = min(0.7 * intra + 0.3 * inter, 1.0)

        self.logger.info(f"Weight calculation complete: {len(kg.edges())} edges")
        return kg

    def _normalize_edge_weights(self, kg: nx.Graph, doc_triples: Dict[int, List[Tuple]], method: str = 'minmax') -> nx.Graph:
        """
        エッジ重みをドキュメントごとに正規化

        Args:
            kg: NetworkXグラフ
            doc_triples: {doc_id: [(s, r, o), ...]} の辞書
            method: 'minmax' または 'zscore'

        Returns:
            正規化されたグラフ
        """
        self.logger.info(f"Normalizing edge weights (method={method})...")

        # ============================================================
        # 1. ドキュメントごとに重みを収集
        # ============================================================
        doc_edge_weights = defaultdict(lambda: {'intra': [], 'inter': []})
        edge_to_docs = defaultdict(set)  # エッジがどのドキュメントに属するか

        for doc_id, triples in doc_triples.items():
            doc_entities = set()
            for s, _, o in triples:
                doc_entities.add(s)
                doc_entities.add(o)

            # このドキュメントに関連するエッジを探す
            for u, v, data in kg.edges(data=True):
                if u in doc_entities or v in doc_entities:
                    edge_key = (u, v)
                    edge_to_docs[edge_key].add(doc_id)

                    intra_raw = data.get('intra_raw', 0.0)
                    inter_raw = data.get('inter_raw', 0.0)

                    if intra_raw > 0:
                        doc_edge_weights[doc_id]['intra'].append(intra_raw)
                    if inter_raw > 0:
                        doc_edge_weights[doc_id]['inter'].append(inter_raw)

        # ============================================================
        # 2. ドキュメントごとに正規化パラメータを計算
        # ============================================================
        norm_params = {}

        for doc_id, weights in doc_edge_weights.items():
            params = {}

            for weight_type in ['intra', 'inter']:
                values = weights[weight_type]

                if not values:
                    params[weight_type] = None
                    continue

                if method == 'minmax':
                    min_val = min(values)
                    max_val = max(values)
                    params[weight_type] = {
                        'min': min_val,
                        'max': max_val,
                        'range': max_val - min_val
                    }

                elif method == 'zscore':
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    params[weight_type] = {
                        'mean': mean_val,
                        'std': std_val if std_val > 0 else 1.0
                    }

            norm_params[doc_id] = params

        # 統計情報をログ出力
        self._log_normalization_stats(doc_edge_weights, norm_params)

        # ============================================================
        # 3. エッジごとに正規化を適用
        # ============================================================
        normalized_count = 0

        for u, v, data in kg.edges(data=True):
            edge_key = (u, v)
            related_docs = edge_to_docs.get(edge_key, set())

            if not related_docs:
                continue

            # このエッジに関連する全ドキュメントの正規化値を平均
            intra_normalized = []
            inter_normalized = []

            for doc_id in related_docs:
                if doc_id not in norm_params:
                    continue

                params = norm_params[doc_id]
                intra_raw = data.get('intra_raw', 0.0)
                inter_raw = data.get('inter_raw', 0.0)

                # Intra正規化
                if params['intra'] and intra_raw > 0:
                    norm_val = self._normalize_value(
                        intra_raw,
                        params['intra'],
                        method
                    )
                    intra_normalized.append(norm_val)

                # Inter正規化
                if params['inter'] and inter_raw > 0:
                    norm_val = self._normalize_value(
                        inter_raw,
                        params['inter'],
                        method
                    )
                    inter_normalized.append(norm_val)

            # 正規化後の値を平均
            if intra_normalized:
                data['intra_normalized'] = np.mean(intra_normalized)
                normalized_count += 1
            else:
                data['intra_normalized'] = data.get('intra_raw', 0.0)

            if inter_normalized:
                data['inter_normalized'] = np.mean(inter_normalized)
            else:
                data['inter_normalized'] = data.get('inter_raw', 0.0)

        self.logger.info(f"  → Normalized {normalized_count} edges")

        return kg

    def _normalize_value(self, value: float, params: dict, method: str) -> float:
        """
        単一の値を正規化

        Args:
            value: 正規化する値
            params: 正規化パラメータ
            method: 'minmax' または 'zscore'

        Returns:
            正規化された値
        """
        if method == 'minmax':
            min_val = params['min']
            max_val = params['max']
            range_val = params['range']

            if range_val < 1e-9:
                return 0.5  # 全て同じ値の場合は中間値

            # [0, 1] に正規化
            normalized = (value - min_val) / range_val
            return max(0.0, min(1.0, normalized))

        elif method == 'zscore':
            mean_val = params['mean']
            std_val = params['std']

            # z-scoreを計算後、sigmoidで [0, 1] に変換
            z = (value - mean_val) / std_val
            sigmoid = 1 / (1 + np.exp(-z))
            return sigmoid

        return value

    def _log_normalization_stats(self, doc_edge_weights: dict, norm_params: dict):
        """正規化統計をログ出力"""
        self.logger.info("  Normalization statistics:")

        for doc_id in list(norm_params.keys())[:3]:  # 最初の3ドキュメント
            params = norm_params[doc_id]

            intra_weights = doc_edge_weights[doc_id]['intra']
            inter_weights = doc_edge_weights[doc_id]['inter']

            if intra_weights:
                self.logger.info(
                    f"    Doc {doc_id} intra: "
                    f"min={min(intra_weights):.3f}, "
                    f"max={max(intra_weights):.3f}, "
                    f"mean={np.mean(intra_weights):.3f}"
                )

            if inter_weights:
                self.logger.info(
                    f"    Doc {doc_id} inter: "
                    f"min={min(inter_weights):.3f}, "
                    f"max={max(inter_weights):.3f}, "
                    f"mean={np.mean(inter_weights):.3f}"
                )

    def _compute_intra_weight(self, e1: str, e2: str, triples: List, kg=None) -> float:
        """
        同一Document内のエンティティ間重み計算

        Args:
            e1, e2: エンティティ名
            triples: (s, r, o) のリスト
            kg: NetworkXグラフ（オプション）
        """
        # ------------------------------------------------------------
        # 1) 共起頻度（基本）
        # ------------------------------------------------------------
        cooccur = sum(
            1 for s, _, o in triples
            if (s == e1 and o == e2) or (s == e2 and o == e1)
        )
        co_norm = min(cooccur / 5.0, 1.0)   # 正規化

        # ------------------------------------------------------------
        # 2) 関係タイプの多様性
        # ------------------------------------------------------------
        rel_pairs = [
            (r, True) for s, r, o in triples
            if (s == e1 and o == e2)
        ] + [
            (r, False) for s, r, o in triples
            if (s == e2 and o == e1)  # 逆向き
        ]

        if not rel_pairs:
            rel_bonus = 0.0
        else:
            # 関係の多様性
            unique_rels = set(r for r, _ in rel_pairs)
            diversity_bonus = min(len(unique_rels) * 0.2, 0.6)

            # 方向の一貫性（同じ向きが多いほど強い関係）
            same_direction_count = sum(1 for _, is_forward in rel_pairs if is_forward)
            opposite_direction_count = len(rel_pairs) - same_direction_count

            # 関係の質（同じ向きか逆向きかで評価）
            if same_direction_count > opposite_direction_count:
                direction_score = same_direction_count / len(rel_pairs)
            else:
                # 逆方向が多い = 双方向の関係（これも有用）
                direction_score = 0.7  # やや高めに評価

            rel_bonus = diversity_bonus * 0.5 + direction_score * 0.5

        # ------------------------------------------------------------
        # 3) パスサポート（kgに path_support があれば）
        # ------------------------------------------------------------
        path_bonus = 0.0
        if kg is not None and kg.has_edge(e1, e2):
            path_bonus = min(kg[e1][e2].get("path_support", 0) * 0.1, 0.5)

        # ------------------------------------------------------------
        # 4) 合成
        # ------------------------------------------------------------
        weight = co_norm * 0.5 + rel_bonus * 0.4 + path_bonus * 0.1
        return min(weight, 1.0)

    def _compute_inter_weight(self, t1: tuple, t2: tuple, kg=None):
        """inter-triple interaction weight計算"""

        s1, r1, o1 = t1
        s2, r2, o2 = t2

        # 共有エンティティ（最重要）
        shared = len({s1, o1} & {s2, o2})
        shared_bonus = min(shared * 0.5, 1.0)
        # 関係の相性計算
        rel_compatibility = safe_execute(
            self._compute_relation_compatibility,
            args=(r1, r2),
            default=0.3,
            logger=self.logger,
            context=f"relation_compatibility({r1}, {r2})"
        )
        # エンティティ類似度
        sim_bonus = 0.0
        try:
            e1 = self.embed_model.get_text_embedding(s1)
            e2 = self.embed_model.get_text_embedding(s2)

            # 正規化済みなので直接内積を計算
            sim = float(np.dot(e1, e2))
            sim_bonus = max(sim, 0) * 0.3

        except Exception as e:
            if not hasattr(self, '_embedding_error_warned'):
                self.logger.warning(f"⚠️  Embedding similarity errors detected")
                self._embedding_error_warned = True

        # 3) graph path-based support（kgが与えられた場合）
        path_bonus = 0.0
        if kg is not None:
            try:
                # 2-hop以内でつながってたら評価
                if kg.has_node(s1) and kg.has_node(s2):
                    length = nx.shortest_path_length(kg, s1, s2)
                    if length <= 2:
                        path_bonus = 0.3 * (1.0 - length / 3.0)  # 近いほど高スコア
            except nx.NetworkXNoPath:
                pass
            except nx.NodeNotFound:
                if self.logger.level <= 10:  # logging.DEBUG
                    self.logger.debug(f"Node not found in graph: {s1} or {s2}")
            except Exception as e:
                if self.logger.level <= 10:  # logging.DEBUG
                    self.logger.debug(f"Path calc failed ({s1}->{s2}): {type(e).__name__}")

        # 4) 総合
        w = (
            shared_bonus * 0.4 +       # 共有エンティティ
            rel_compatibility * 0.3 +   # 関係の相性（ここに統合済み）
            sim_bonus * 0.2 +           # エンティティ類似度
            path_bonus * 0.1            # パス距離
        )

        return min(w, 1.0)

    def _compute_relation_compatibility(self, r1: str, r2: str) -> float:
        """
        関係の相性スコア
        """
        # 正規化（小文字化、アンダースコア統一）
        r1 = r1.lower().replace('-', '_')
        r2 = r2.lower().replace('-', '_')
        # 1. 完全一致
        if r1 == r2:
            return 1.0

        # 2. 逆関係のペア（高スコア）
        inverse_pairs = {
            ("cause_of", "caused_by"),
            ("cause_of", "effect_of"),
            ("part_of", "has_part"),
            ("component_of", "has_component"),
            ("parent_of", "child_of"),
            ("author_of", "written_by"),
            ("owns", "owned_by"),
            ("manages", "managed_by"),
            ("teaches", "taught_by"),
            ("supervises", "supervised_by"),
        }

        if (r1, r2) in inverse_pairs or (r2, r1) in inverse_pairs:
            return 0.9

        # 3. 関連する関係グループ（中スコア）
        related_groups = [
            # 因果関係グループ
            {
                "cause_of", "caused_by", "leads_to", "results_in",
                "triggers", "produces", "generates", "effect_of"
            },

            # 構成要素グループ
            {
                "part_of", "has_part", "component_of", "has_component",
                "contains", "includes", "consists_of", "comprises"
            },

            # 所属グループ
            {
                "member_of", "has_member", "belongs_to", "works_at",
                "employed_by", "affiliated_with"
            },

            # 時間関係グループ
            {
                "before", "after", "during", "precedes", "follows",
                "happens_before", "happens_after"
            },

            # 空間関係グループ
            {
                "located_in", "location_of", "near", "adjacent_to",
                "contains", "inside", "outside"
            },

            # 属性・性質グループ
            {
                "is_a", "type_of", "instance_of", "has_property",
                "characterized_by", "defined_by"
            },

            # 相互作用グループ
            {
                "interacts_with", "collaborates_with", "competes_with",
                "influences", "affected_by"
            },
        ]

        for group in related_groups:
            if r1 in group and r2 in group:
                return 0.7

        # 4. 同じカテゴリ（動詞の性質で判定）
        # 例: action 系、state 系など
        action_verbs = {
            "creates", "builds", "develops", "produces", "makes",
            "constructs", "designs", "implements", "generates",
            "enables", "powers", "leverages", "accelerates"
            # （ML/AI専門）
            "utilizes", "parameterizes", "fine_tunes", "approximates",
            "encodes", "regularizes", "iterates", "optimizes",
            "traverses", "samples", "augments", "normalizes",
            "quantizes", "distills", "ensembles", "prunes",
            "compresses", "aggregates", "fuses", "aligns",
            "projects", "embeds", "transforms", "adapts",

            # CV系
            "detects", "segments", "classifies", "recognizes",
            "extracts", "filters", "convolves", "pools",

            # NLP系
            "tokenizes", "parses", "generates_text", "translates",
            "attends_to", "masks", "predicts",

            # Graph系
            "propagates", "aggregates_neighbors", "diffuses",
            "clusters", "partitions", "samples_neighbors"
        }

        state_verbs = {
            "is", "has", "contains", "includes", "comprises",
            "exists", "represents", "defines", "consists_of",
            "maintains", "preserves", "exhibits", "displays"
        }

        relation_verbs = {
            "relates_to", "associated_with", "connected_to",
            "linked_to", "corresponds_to", "depends_on",
            "derived_from", "based_on", "inspired_by"
        }

        # --- 3-4. 計算動詞 ---
        computational_verbs = {
            "computes", "calculates", "evaluates", "measures",
            "estimates", "infers", "learns", "trains",
            "updates", "backpropagates", "forward_passes"
        }

        # --- 3-5. 比較動詞 ---
        comparison_verbs = {
            "outperforms", "surpasses", "exceeds", "improves_upon",
            "compares_to", "contrasts_with", "benchmarks_against"
        }

        # カテゴリマッチング
        verb_categories = [
            action_verbs,
            state_verbs,
            relation_verbs,
            computational_verbs,
            comparison_verbs
        ]

        for category in verb_categories:
            if r1 in category and r2 in category:
                return 0.5

        # 5. 埋め込みフォールバック（低スコア）
        try:
            emb1 = self.embed_model.get_text_embedding(r1)
            emb2 = self.embed_model.get_text_embedding(r2)
            sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-9)
            return max(0.3, float(sim))
        except Exception:
            return 0.3

    def _update_neo4j_structure(self, kg, graph_store):
        """
        Neo4j更新
        """
        batch_query  = """
        UNWIND $batch AS row
        MERGE (a:Concept {name: row.source})
        MERGE (b:Concept {name: row.target})
        MERGE (a)-[r:RELATED]->(b)
        ON CREATE SET r.weight = row.weight
        ON MATCH SET r.weight = row.weight
        """
        collector = ErrorCollector(self.logger)

        batch = []
        batch_size = 1000  # 1000件ごとに送信

        for s, o, data in kg.edges(data=True):
            weight = data.get('weight', 0.0)

            if weight <= self.config['final_weight_cutoff']:
                collector.add_skip()
                continue

            # バッチに追加
            batch.append({
                'source': s,
                'target': o,
                'weight': float(weight)
            })

        # バッチサイズに達したら送信
        if len(batch) >= batch_size:

            try:
                graph_store.query(batch_query, {'batch': batch})
                collector.add_success(count=len(batch))

                self.logger.debug(f"  Sent batch of {len(batch)} edges")
                batch = []  # バッチをクリア

            except Exception as e:
                collector.add_error(
                    context=f"batch_{len(batch)}_edges",
                    error=e
                )
                # 失敗したバッチは破棄（または個別処理）
                batch = []

        # 残りのバッチを送信
        if batch:
            try:
                graph_store.query(batch_query, {'batch': batch})
                collector.add_success(count=len(batch))
                self.logger.debug(f"  Sent final batch of {len(batch)} edges")

            except Exception as e:
                collector.add_error(
                    context=f"final_batch_{len(batch)}_edges",
                    error=e
                )
        # レポート生成（自動でログ出力）
        collector.report("Neo4j edge update", threshold=0.3)
        # 戻り値も取得可能
        return collector.get_summary()