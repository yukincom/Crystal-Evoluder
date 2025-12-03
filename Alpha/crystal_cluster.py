"""
Crystal Cluster 
Knowledge Graph committer for Neo4j

"""

# ============================================================
# インポート
# ============================================================
import json
import logging
import pickle
import numpy as np
import networkx as nx

from pathlib import Path
from typing import List, Dict, Any, Optional

from llama_index.llms.openai import OpenAI  
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.core import Document, KnowledgeGraphIndex, StorageContext

# 共通モジュール
from shared.logger import setup_logger, HierarchicalLogger
from shared.utils import load_and_validate_paths
from shared.error_handler import ErrorCollector, safe_execute

class CrystalCluster:
    """Crystal Cluster - Neo4j投入専用"""
    
    def __init__(self, log_level: int = logging.INFO):
        self.logger = setup_logger('CrystalCluster', log_level)
        self.hlogger = HierarchicalLogger(self.logger)

        self.embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-m3",
            device="mps",
            embed_batch_size=16,
        )

        from llama_index.core import Settings
        Settings.embed_model = self.embed_model

        self.logger.info("Crystal Cluster v1.1 initialized")

    def load_documents(
        self,
        json_path: str,
        raw_docs: Optional[List[str]] = None,
        path_pickle: Optional[str] = None,
        kg: Optional[nx.Graph] = None) -> List[Document]:
        """
        JSON と 生テキスト両方から Document を作る
        
        Args:
            json_path: JSONファイルのパス
            raw_docs: 生テキストのリスト（オプション）
            path_pickle: パス情報のPickleファイル（オプション）
            kg: ナレッジグラフ（パス情報統合時に必要）
        
        Returns:
            Documentのリスト（パス情報が統合されている場合もある）
        """
        documents = []

        # --- JSON 側 ---
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # json_path の documents を追加
        for i, doc in enumerate(data.get('documents', [])):
            documents.append(
                Document(
                    text=doc['text'],
                    metadata={
                        "source": "json",
                        "json_id": i,
                        **doc.get("metadata", {})
                    }
                )
            )

        # --- 生テキスト側 ---
        if raw_docs:
            for i, text in enumerate(raw_docs):
                documents.append(
                    Document(
                        text=text,
                        metadata={
                            "source": "raw",
                            "raw_id": i
                        }
                    )
                )

        json_count = len(data.get('documents', []))
        raw_count = len(raw_docs) if raw_docs else 0
        
        self.logger.info(
            f"📂 Loaded {len(documents)} documents "
            f"({json_count} from JSON, {raw_count} raw texts)"
        )

        # --- パス情報の統合（オプション）---
        if path_pickle and kg is not None:
            path_dicts = load_and_validate_paths(path_pickle, self.logger)
            if path_dicts:
                self.logger.info("Augmenting documents with path information...")
                documents = self.augment_documents_with_paths(
                    documents, 
                    path_dicts, 
                    kg,
                    entity_embeddings=getattr(self, 'entity_embeddings', None)
                )
                self.logger.info(f"✅ Path information added to {len(documents)} documents")
            else:
                self.logger.warning("Path information could not be loaded, continuing without it")

        return documents

    def augment_documents_with_paths(
        self,
        documents: List[Document], 
        path_dicts: List[Dict], 
        kg: nx.Graph,
        entity_embeddings: Dict[str, np.ndarray] = None,
        match_key='question') -> List[Document]:
        """
        documents に対応する path 情報を注入
        
        Args:
            documents: Documentのリスト
            path_dicts: load_path_dicts の戻り値
            kg: ナレッジグラフ
            entity_embeddings: エンティティの埋め込み（オプション）
            match_key: documents と path_dicts を突き合わせるキー
        
        Returns:
            パス情報が追加された documents

        """
        # defensive
        if entity_embeddings is None:
            entity_embeddings = {}

        # インデックス作成： path_dicts を match_key で引けるようにする
        pd_map = {}
        for p in path_dicts:
            key = p.get(match_key)
            if key is not None:
                pd_map[key] = p

        augmented = []
        matched_count = 0

        for doc in documents:
            meta = dict(getattr(doc, 'metadata', {}) or {})
            doc_key = meta.get(match_key)

            matched = None
            if doc_key is not None and doc_key in pd_map:
                matched = pd_map[doc_key]
                matched_count += 1
            else:
            # フォールバック：テキスト内に match_key の文字列が含まれる path_dict を探す
                text = getattr(doc, 'text', '') or ''
                for k, p in pd_map.items():
                    if isinstance(k, str) and k in text:
                        matched = p
                        matched_count += 1
                        break

            paths_meta = []
            if matched:
                for path in matched.get('translated_paths', []):
                # path: list of node names (entities)
                    path_len = len(path)
                    edge_weights = []
                    path_node_pairs = list(zip(path[:-1], path[1:])) if path_len >= 2 else []
                    for u, v in path_node_pairs:
                        if kg.has_edge(u, v):
                            edge_weights.append(kg[u][v].get('weight', 0.0))
                        elif kg.has_edge(v, u):
                            edge_weights.append(kg[v][u].get('weight', 0.0))
                        else:
                        # edge が存在しない場合は 0.0 を入れておく
                            edge_weights.append(0.0)

                    avg_edge_weight = float(np.mean(edge_weights)) if edge_weights else 0.0
                    sum_edge_weight = float(np.sum(edge_weights)) if edge_weights else 0.0

                # path 内ノードの埋め込みがあれば、ノード間類似度を計算（平均ペア類似度）
                    pair_sims = []
                    for i in range(len(path) - 1):
                        e1 = entity_embeddings.get(path[i])
                        e2 = entity_embeddings.get(path[i + 1])
                        if e1 is not None and e2 is not None:
                        # safe numpy dot / norms
                            denom = (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-9)
                            pair_sims.append(float(np.dot(e1, e2) / denom))
                    avg_pair_sim = float(np.mean(pair_sims)) if pair_sims else None

                # 最短距離（kg 上） — 存在しなければ None
                    shortest = None
                    try:
                        if path_len >= 2:
                        # path の端同士の最短長を計算（例）
                            s1, s2 = path[0], path[-1]
                            if kg.has_node(s1) and kg.has_node(s2):
                                shortest = int(nx.shortest_path_length(kg, s1, s2))
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        pass
                    except Exception:
                        pass

                    paths_meta.append({
                        'path': path,
                        'path_length_nodes': path_len,
                        'avg_edge_weight': avg_edge_weight,
                        'sum_edge_weight': sum_edge_weight,
                        'avg_adjacent_node_sim': avg_pair_sim, 
                        'kg_shortest_between_ends': shortest
                    })

        # attach (既存 metadata を壊さないようにコピー)
                new_meta = dict(meta)
                new_meta['paths'] = paths_meta
            # create a new Document preserving original text & adding metadata (or mutate in place if ok)
                new_doc = Document(text=getattr(doc, 'text', ''), metadata=new_meta)
                augmented.append(new_doc)
                
            return augmented

        # naive match: by ordering if no explicit key available
        if len(path_dicts) == 0:
            return documents

        if len(path_dicts) == len(documents):
            for i, doc in enumerate(documents):
                doc.metadata['paths'] = path_dicts[i].get('translated_paths', [])
                doc.metadata['path_distances'] = path_dicts[i].get('path_distances', [])
        else:
            # fallback: attach top global paths to every doc (still useful)
            sample_paths = path_dicts[0].get('translated_paths', [])
            for doc in documents:
                doc.metadata.setdefault('paths', sample_paths)
                doc.metadata.setdefault('path_distances', path_dicts[0].get('path_distances', []))
        return documents
    

    def commit_to_graph(self, documents: List[Document], graph_store: Neo4jGraphStore):
        """Neo4jにグラフを投入"""
        #　接続確認　===========================================
        try:
            graph_store.query("RETURN 1")
            self.logger.info("✅ Neo4j connection verified")
        except Exception as e:
            self.logger.error(
                f"🚨 Neo4j connection failed: {type(e).__name__}\n"
                f"   Message: {str(e)[:200]}\n"
                f"   Check: URI, username, password, and service status"
            )
            raise  # 接続できないなら処理を中断
        # 2. グラフ生成 ===========================================
        try:
            with self.hlogger.section("Graph Generation"):
                llm = OpenAI(model="gpt-4o-mini", timeout=120.0, output_version="v0")
                storage_context = StorageContext.from_defaults(graph_store=graph_store)
            
                index = KnowledgeGraphIndex.from_documents(
                    documents,
                    storage_context=storage_context,
                    llm=llm,
                    transformations=[SimpleNodeParser.from_defaults(chunk_size=512)],
                    embed_model=HuggingFaceEmbedding(model_name="BAAI/bge-m3"),
                    show_progress=True,
                    max_triplets_per_chunk=10
                )
            
                kg = index.get_networkx_graph()
                self.logger.info(f"✅ Initial graph: {len(kg.nodes)} nodes, {len(kg.edges)} edges")
        except Exception as e:
            self.logger.error(
                f"🚨 Graph generation failed: {type(e).__name__}\n"
                f"   Message: {str(e)[:200]}\n"
                f"   This might be due to: LLM API issues, invalid documents, or embedding model errors"
            )
            raise
        # 3. パス情報をグラフに統合　================================

        try:
            with self.hlogger.section("Merging Path Information"):
                self.merge_paths_into_kg(kg, documents)
                self.logger.info(f"✅ Path info merged: {len(kg.nodes)} nodes, {len(kg.edges)} edges")

        except Exception as e:
            self.logger.warning(
                f"⚠️  Path merging failed: {type(e).__name__} - {str(e)[:100]}\n"
                f"   Continuing without path information..."
            )

        # ============================================================
        # RAPL風最適化: トリプレット間の相互作用を計算
        # ============================================================
        try:
            with self.hlogger.section("Graph Optimization (RAPL)"):
                optimized_kg = self._optimize_graph_rapl(kg, documents)
                self.logger.info(
                    f"✅ Optimized graph: {len(optimized_kg.nodes)} nodes, "
                    f"{len(optimized_kg.edges)} edges"
                )
        except Exception as e:
            self.logger.error(  
                        f"🚨 Graph optimization failed: {type(e).__name__}\n"
                        f"   Message: {str(e)[:200]}\n"
                        f"   Using unoptimized graph instead..."
            )
            optimized_kg = kg 
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
            self.logger.error(
                f"🚨 Neo4j update failed: {type(e).__name__}\n"
                f"   Message: {str(e)[:200]}\n"
                f"   Graph was optimized but not persisted to database!"
            )
            raise

    def merge_paths_into_kg(self, kg, documents: List[Document]):
        """
        kg: networkx.Graph (triples turned into nodes/edges)
        documents: the same documents that have metadata['paths'] etc.
        This will:
          - count how many times each entity appears in top-k paths
          - add edge/node attributes: top_path_count, avg_path_length
        """
        from collections import Counter, defaultdict
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
        完全統合版 RAPL 最適化
        """
        from collections import defaultdict
    
    # ============================================================
    # 1. Triples 抽出
    # ============================================================
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
    
    # ============================================================
    # 2. Intra: 文書内 triple 間相互作用
    # ============================================================
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
    
    # ============================================================
    # 3. Inter: 共有エンティティベースの高速化
    # ============================================================
        self.logger.info("Computing inter-interactions (optimized)...")
        inter_collector = ErrorCollector(self.logger)
    
    # 3-1. エンティティ→Triple インデックス構築
        entity_to_triples = defaultdict(set)
        for idx, (s, r, o) in enumerate(all_triples):
            entity_to_triples[s].add(idx)
            entity_to_triples[o].add(idx)
    
        self.logger.info(f"Built entity index: {len(entity_to_triples)} unique entities")
    
    # 3-2. 共有エンティティがある Triple ペアのみ計算
        seen_pairs = set()
        inter_count = 0
    
        for entity, triple_indices in entity_to_triples.items():
            if len(triple_indices) < 2:
                continue  # 1つの Triple にしか出現しないエンティティはスキップ
        
            indices = list(triple_indices)
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    idx1, idx2 = indices[i], indices[j]
                
                # ペアの重複チェック（同じペアを複数回計算しない）
                    pair = (min(idx1, idx2), max(idx1, idx2))
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                
                # 重み計算
                    try:
                        t1 = all_triples[idx1]
                        t2 = all_triples[idx2]
                        w = self._compute_inter_weight(t1, t2, kg=kg)
                
                        if w > 0.2:
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
    
    # ============================================================
    # 4. Document-level linking
    # ============================================================
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
 
    # ============================================================
    # 5. 統合重み（RAPL式）
    # ============================================================
        self.logger.info("Finalizing edge weights...")
    
        for u, v, d in kg.edges(data=True):
            intra = d.get("intra_raw", 0.0)
            inter = d.get("inter_raw", 0.0)
        
        # RAPL論文: intra重視 + inter補完
            d["weight"] = min(0.7 * intra + 0.3 * inter, 1.0)    
        return kg
    
    def _group_triples_by_document(self, kg, documents):
        """トリプレットをDocument別にグループ化"""
        # 簡易実装: メタデータから推定
        doc_triples = {}
        
        for i, doc in enumerate(documents):
            doc_id = f"doc_{i}"
            doc_triples[doc_id] = []
            
            # エンティティがDocument内に出現するトリプレットを抽出
            for s, o, data in kg.edges(data=True):
                if s in doc.text or o in doc.text:
                    doc_triples[doc_id].append((s, data.get('relation', ''), o))
        return doc_triples
    
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


    def _compute_inter_weight(self, t1: tuple, t2: tuple, kg=None, embed_fn=None) -> float:
        """
        Compute inter-triple interaction weight between two triples.
        t1, t2: (s, r, o)
        kg: optional networkx graph
        embed_fn: optional function to compute embeddings of entity names
        """

        s1, r1, o1 = t1
        s2, r2, o2 = t2

        # ------------------------------------------------------------
        # 1) 共有エンティティ（最重要）
        # ------------------------------------------------------------
        shared = len({s1, o1} & {s2, o2})
        shared_bonus = min(shared * 0.5, 1.0)
        # 関係の相性計算
        rel_compatibility = safe_execute(
            self._compute_relation_compatibility,
            args=(r1, r2),
            default=0.3,  # 失敗時はデフォルト値
            logger=self.logger,
            context=f"relation_compatibility({r1}, {r2})"
        )
        # ------------------------------------------------------------
        # 2) エンティティ類似度（embeddingを渡されたら使う）
        # ------------------------------------------------------------
        sim_bonus = 0.0
        if embed_fn is not None:
            try:        
                e1 = embed_fn(s1)
                e2 = embed_fn(s2)

                # ゼロベクトルチェック
                norm1 = np.linalg.norm(e1)
                norm2 = np.linalg.norm(e2)
            
                if norm1 > 1e-9 and norm2 > 1e-9:
                    sim = (e1 @ e2) / (norm1 * norm2)
                    sim_bonus = max(sim, 0) * 0.3
                else:
                    self.logger.debug(
                        f"Zero embedding detected: {s1} (norm={norm1:.2e}) "
                        f"or {s2} (norm={norm2:.2e})"
                    )
            except Exception as e:
                self.logger.debug(
                    f"Embedding similarity failed for ({s1}, {s2}): "
                    f"{type(e).__name__}"
                )

        # ------------------------------------------------------------
        # 3) graph path-based support（kgが与えられた場合）
        #    2〜3 hop 以内に繋がるかを見る
        # ------------------------------------------------------------
        path_bonus = 0.0
        if kg is not None:
            try:
            # 2-hop以内でつながってたら評価        
                if kg.has_node(s1) and kg.has_node(s2):
                    length = nx.shortest_path_length(kg, s1, s2)
                    if length <= 2:
                        path_bonus = 0.3 * (1.0 - length / 3.0)  # 近いほど高スコア
            except nx.NetworkXNoPath:
                # パスが存在しない場合は0.0のまま（これは正常）
                pass
            except nx.NodeNotFound as e:
                self.logger.debug(f"Node not found in graph: {e}")
            except Exception as e:
                self.logger.debug(
                    f"Path calculation failed ({s1}->{s2}): {type(e).__name__}"
                )

        # ------------------------------------------------------------
        # 4) 総合
        # ------------------------------------------------------------
        w = (
            shared_bonus * 0.4 +       # 共有エンティティ
            rel_compatibility * 0.3 +   # 関係の相性（ここに統合済み）
            sim_bonus * 0.2 +           # エンティティ類似度
            path_bonus * 0.1            # パス距離
        )

        return min(w, 1.0)
    
    def _compute_relation_compatibility(self, r1: str, r2: str) -> float:
        """
        関係の相性スコア（手動ルール + embeddingフォールバック）
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
            "constructs", "designs", "implements", "generates"
        }
    
        state_verbs = {
            "is", "has", "contains", "includes", "comprises",
            "exists", "represents", "defines"
        }
    
        relation_verbs = {
            "relates_to", "associated_with", "connected_to",
            "linked_to", "corresponds_to"
        }
    
        if (r1 in action_verbs and r2 in action_verbs) or \
           (r1 in state_verbs and r2 in state_verbs) or \
           (r1 in relation_verbs and r2 in relation_verbs):
            return 0.5
    
    # 5. それ以外（低スコア）
        try:
            emb1 = self.relation_embedder.get_text_embedding(r1)
            emb2 = self.relation_embedder.get_text_embedding(r2)
            sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-9)
            return max(0.3, float(sim))
        except Exception:
            return 0.3


    def _update_neo4j_structure(self, kg, graph_store):
        query_template = """
        MERGE (a:Concept {name: $source})
        MERGE (b:Concept {name: $target})
        MERGE (a)-[r:RELATED]->(b)
        ON CREATE SET r.weight = $weight
        ON MATCH SET r.weight = $weight
        """
        collector = ErrorCollector(self.logger)

        for s, o, data in kg.edges(data=True):
            weight = data.get('weight', 0.0)

            if weight <= 0: 
                collector.add_skip()
                continue  # 重み0は無視
            
            try:
                graph_store.query(query_template, {
                    'source': s,
                    'target': o,
                    'weight': float(weight)
                })
                collector.add_success()

            except Exception as e:
                collector.add_error(
                    context=f"{s} -> {o}",
                    error=e,
                    weight=weight  # メタデータとして記録
                )
        # レポート生成（自動でログ出力）
        collector.report("Neo4j edge update", threshold=0.3)
    # 戻り値も取得可能
        return collector.get_summary()        

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Crystal Cluster v1.1')
    parser.add_argument('json_file', help='Clean documents JSON file')
    parser.add_argument('--neo4j-uri', default='bolt://localhost:7687')
    parser.add_argument('--neo4j-user', default='neo4j')
    parser.add_argument('--neo4j-pass', required=True)
    parser.add_argument('--debug', action='store_true')
    
    args = parser.parse_args()
    
    print("💾 Crystal Cluster v1.1")
    print("━" * 42)
    
    cluster = CrystalCluster(log_level=logging.DEBUG if args.debug else logging.INFO)
    
    documents = cluster.load_documents(args.json_file)
    
    graph_store = Neo4jGraphStore(
        username=args.neo4j_user,
        password=args.neo4j_pass,
        url=args.neo4j_uri
    )
    
    cluster.commit_to_graph(documents, graph_store)
    
    print("✨ Complete!")