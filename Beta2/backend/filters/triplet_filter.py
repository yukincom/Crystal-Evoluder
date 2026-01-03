"""
トリプレットフィルタリングクラス
"""
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

from ..shared import safe_execute

class TripletFilter:
    """トリプレットの品質管理とSelf-RAGを担当"""

    def __init__(self, config: dict, logger):
        self.config = config
        self.logger = logger
        self.total_self_rag_tokens = 0

        # 🔧 追加: 基本モデル設定を取得
        self.mode = config.get('mode', 'api')
    
        # モードに応じた基本モデルを取得
        if self.mode == 'api':
            self.base_model = config.get('api_model', 'gpt-4o-mini')
        else:
            self.base_model = config.get('ollama_model', '')
    
        # Self-RAG用モデル
        self.critic_model = config.get('self_rag_critic_model') or self.base_model
        self.refiner_model = config.get('self_rag_refiner_model') or self.base_model
    
        self.logger.info(f"TripletFilter initialized:")
        self.logger.info(f"  Base model: {self.base_model}")
        self.logger.info(f"  Critic model: {self.critic_model}")
        self.logger.info(f"  Refiner model: {self.refiner_model}")

        # 関係タイプのブラックリスト
        self.relation_blacklist = {
            'is', 'has', 'are', 'was', 'were',
            'the', 'a', 'an',
            'of', 'in', 'on', 'at',
        }
    # BGE-M3をクラスでロード（1回だけ）
        self.embedder = None
        self.blacklist_embs = None
        self.useful_rel_embs = None

        if hasattr(self, 'embedding_cache') and self.embedding_cache is not None:
            try:
                self.blacklist_embs = [
                    self.embedding_cache.get_embedding(rel.lower())
                    for rel in self.relation_blacklist
                ]

                self.logger.info("✅ Blacklist embeddings precomputed with BGE-M3 cache")

                # 有用関係リスト（例：論文でよく使われる関係を追加）
                useful_relations = [
                    "causes", "affects", "treats", "indicates",
                    "associated_with", "correlates_with", "leads_to"
                ]
                self.useful_rel_embs = [
                    self.embedding_cache.get_embedding(rel.lower())
                    for rel in useful_relations
                ]

                self.logger.info("✅ Useful relation embeddings precomputed with BGE-M3 cache")
            except Exception as e:
                self.logger.warning(f"⚠️ Precomputing embeddings failed: {e}")

    def filter_triplets(
        self,
        triplets: List[Tuple[str, str, str]],
        quality_threshold: float = 0.3
    ) -> Tuple[List[Tuple], List[Tuple], Dict]:
        """
        トリプレットを品質でフィルタリング

        Args:
            triplets: [(subject, relation, object), ...] のリスト
            quality_threshold: 品質スコアの閾値（0.0~1.0）

        Returns:
            (filtered_triplets, rejected_triplets, stats)
        """
        self.logger.info(f"🔍 Filtering {len(triplets)} triplets...")

        filtered = []
        rejected = []
        quality_scores = []

        for s, r, o in triplets:
            # 品質スコア計算
            score = self._compute_triplet_quality(s, r, o)
            quality_scores.append(score)

            if score >= quality_threshold:
                filtered.append((s, r, o))
            else:
                rejected.append((s, r, o))
                self.logger.debug(
                    f"  Rejected: ({s}, {r}, {o}) [score={score:.2f}]"
                )

        # 統計情報
        stats = {
            'original': len(triplets),
            'filtered': len(filtered),
            'rejected': len(rejected),
            'avg_quality': sum(quality_scores) / len(quality_scores) if quality_scores else 0,
            'rejection_rate': len(rejected) / len(triplets) if triplets else 0
        }

        self.logger.info(
            f"  → Kept {len(filtered)} triplets, "
            f"rejected {len(rejected)} ({stats['rejection_rate']:.1%})"
        )
        self.logger.info(f"  → Avg quality: {stats['avg_quality']:.2f}")

        return filtered, rejected, stats

    def self_rag_triplets(
        self,
        triplets: List[Tuple[str, str, str]],
        chunk_text: str,
        ai_router
    ) -> Tuple[List[Tuple], Dict]:
        """
        Self-RAG: トリプレットを評価し、低品質なものを再生成

        """
        if not self.config.get('enable_self_rag', False):
            return triplets, {'self_rag_applied': False}

        # トークン予算チェック
        token_budget = self.config.get('self_rag_token_budget', 100000)

        if self.total_self_rag_tokens >= token_budget:
            self.logger.warning(
                f"⚠️  Self-RAG token budget exhausted "
                f"({self.total_self_rag_tokens}/{token_budget}), skipping"
            )
            return triplets, {
                'self_rag_applied': False,
                'budget_exhausted': True
            }

        self.logger.info(f"🔄 Self-RAG: Evaluating {len(triplets)} triplets...")

        # Critic: トリプレットを評価
        evaluated_triplets = []

        for s, r, o in triplets:
            confidence = self._critic_evaluate_triplet(s, r, o, chunk_text)
            evaluated_triplets.append({
                'triplet': (s, r, o),
                'confidence': confidence,
                'needs_refinement': confidence < self.config['self_rag_confidence_threshold']
            })

        # 統計
        low_confidence_count = sum(1 for t in evaluated_triplets if t['needs_refinement'])
        avg_confidence = np.mean([t['confidence'] for t in evaluated_triplets])

        self.logger.info(
            f"  → Avg confidence: {avg_confidence:.2f}, "
            f"Low confidence: {low_confidence_count}/{len(triplets)}"
        )

        # Refiner: 低品質トリプレットを再生成
        refined_triplets = []
        refinement_stats = {
            'attempted': 0,
            'succeeded': 0,
            'failed': 0,
            'tokens_used': 0
        }

        for triplet_info in evaluated_triplets:
            # 予算チェック
            if self.total_self_rag_tokens >= token_budget:
                self.logger.info("  → Budget limit reached, stopping refinement")
                # 残りは元のトリプレットを保持
                refined_triplets.append(triplet_info['triplet'])
                continue

            if triplet_info['needs_refinement']:
                # 再生成を試みる
                refined, tokens_used = self._refiner_regenerate_triplet(
                    triplet_info['triplet'],
                    chunk_text,
                    ai_router
                )

                refinement_stats['attempted'] += 1
                refinement_stats['tokens_used'] += tokens_used
                self.total_self_rag_tokens += tokens_used

                if refined:
                    # 再評価
                    s, r, o = refined
                    new_confidence = self._critic_evaluate_triplet(s, r, o, chunk_text)

                    if new_confidence > triplet_info['confidence']:
                        # 改善された場合は置き換え
                        refined_triplets.append(refined)
                        refinement_stats['succeeded'] += 1

                        self.logger.debug(
                            f"  ✓ Refined: {triplet_info['triplet']} → {refined} "
                            f"(confidence: {triplet_info['confidence']:.2f} → {new_confidence:.2f})"
                        )
                    else:
                        # 改善されなかった場合は元を保持
                        refined_triplets.append(triplet_info['triplet'])
                        refinement_stats['failed'] += 1
                else:
                    # 再生成失敗時は元を保持
                    refined_triplets.append(triplet_info['triplet'])
                    refinement_stats['failed'] += 1
            else:
                # 高品質なものはそのまま
                refined_triplets.append(triplet_info['triplet'])

        # ============================================================
        # 3. Validator: 最終検証
        # ============================================================
        validated_triplets = self._validator_check_consistency(
            refined_triplets,
            chunk_text
        )

        # 統計情報
        stats = {
            'self_rag_applied': True,
            'original_count': len(triplets),
            'refined_count': len(validated_triplets),
            'avg_confidence': float(avg_confidence),
            'low_confidence_count': low_confidence_count,
            'refinement_stats': refinement_stats,
            'total_tokens_used': self.total_self_rag_tokens
        }

        self.logger.info(
            f"  → Self-RAG complete: "
            f"{refinement_stats['succeeded']} improved, "
            f"{refinement_stats['failed']} kept original"
            f"tokens: {refinement_stats['tokens_used']}"
        )

        return validated_triplets, stats

    def _critic_evaluate_triplet(
        self,
        subject: str,
        relation: str,
        object_: str,
        context: str
    ) -> float:
        """
        トリプレットの品質を評価（信頼度スコア: 0.0～1.0）

        Args:
            subject: 主語
            relation: 関係
            object_: 目的語
            context: 元のテキスト

        Returns:
            信頼度スコア（高いほど高品質）
        """
        scores = []

        # ============================================================
        # 1. エンティティの品質（既存のメソッドを活用）
        # ============================================================
        entity_score = self._score_entities(subject, object_)
        scores.append(('entity', entity_score, 0.3))

        # ============================================================
        # 2. 関係の明確性（既存のメソッドを活用）
        # ============================================================
        relation_score = self._score_relation(relation)
        scores.append(('relation', relation_score, 0.3))

        # ============================================================
        # 3. 文法的正しさ（既存のメソッドを活用）
        # ============================================================
        grammar_score = self._score_grammar(subject, relation, object_)
        scores.append(('grammar', grammar_score, 0.2))

        # ============================================================
        # 4. コンテキストとの整合性（新規）
        # ============================================================
        context_score = self._score_context_alignment(
            subject, relation, object_, context
        )
        scores.append(('context', context_score, 0.2))

        # ============================================================
        # 5. 重み付き平均
        # ============================================================
        total_score = sum(score * weight for _, score, weight in scores)

        # デバッグログ（DEBUG時のみ）
        if self.logger.level <= 10:  # logging.DEBUG
            score_details = ', '.join(f"{name}={score:.2f}" for name, score, _ in scores)
            self.logger.debug(
                f"  Triplet: ({subject[:20]}, {relation}, {object_[:20]}) "
                f"→ {score_details} = {total_score:.2f}"
            )

        return total_score

    def _score_context_alignment(
        self,
        subject: str,
        relation: str,
        object_: str,
        context: str
    ) -> float:
        """
        トリプレットとコンテキストの整合性をスコアリング

        Returns:
            0.0（整合性なし）～ 1.0（完全に整合）
        """
        score = 0.0
        context_lower = context.lower()

        # ============================================================
        # 1. エンティティがコンテキストに存在するか
        # ============================================================
        subject_in_context = subject.lower() in context_lower
        object_in_context = object_.lower() in context_lower

        if subject_in_context and object_in_context:
            score += 0.5
        elif subject_in_context or object_in_context:
            score += 0.3
        else:
            # どちらもコンテキストにない場合は低スコア
            score += 0.1

        # ============================================================
        # 2. 関係がコンテキストの文脈と合致するか
        # ============================================================
        relation_lower = relation.lower().replace('_', ' ')

        # 関係の動詞形がコンテキストに存在するか
        if relation_lower in context_lower:
            score += 0.3
        else:
            # 類似表現をチェック（簡易実装）
            relation_synonyms = self._get_relation_synonyms(relation)
            if any(syn in context_lower for syn in relation_synonyms):
                score += 0.2

        # ============================================================
        # 3. トリプレット全体の近接性
        # ============================================================
        # 主語と目的語がコンテキスト内で近い位置にあるか
        if subject_in_context and object_in_context:
            try:
                subject_pos = context_lower.find(subject.lower())
                object_pos = context_lower.find(object_.lower())

                distance = abs(object_pos - subject_pos)

                # 距離に応じてスコアを調整（近いほど高スコア）
                if distance < 50:
                    score += 0.2
                elif distance < 100:
                    score += 0.1
            except Exception:
                pass

        return min(score, 1.0)

    def _get_relation_synonyms(self, relation: str) -> List[str]:
        """
        関係の同義語・類似表現を返す

        Args:
            relation: 関係名

        Returns:
            同義語のリスト
        """
        # 主要な関係の同義語マップ
        synonym_map = {
            'uses': ['use', 'utilizes', 'employs', 'applies'],
            'causes': ['cause', 'leads to', 'results in', 'triggers'],
            'part_of': ['part of', 'component of', 'belongs to'],
            'is_a': ['is a', 'type of', 'kind of', 'instance of'],
            'has': ['have', 'contains', 'includes', 'comprises'],
            'improves': ['improve', 'enhances', 'optimizes', 'boosts'],
            'based_on': ['based on', 'derived from', 'built on', 'relies on'],
            'enables': ['enable', 'allows', 'permits', 'facilitates'],
            'requires': ['require', 'needs', 'depends on', 'necessitates'],
        }

        relation_lower = relation.lower().replace('_', ' ')

        # 完全一致を探す
        for key, synonyms in synonym_map.items():
            if relation_lower == key.replace('_', ' ') or relation_lower in synonyms:
                return synonyms

        # マッチしない場合は元の関係のみ
        return [relation_lower]

    def _refiner_regenerate_triplet(
        self,
        original_triplet: Tuple[str, str, str],
        chunk_text: str,
        ai_router: Any
    ) -> Tuple[Optional[Tuple[str, str, str]], int]:
        """
        低品質トリプレットを再生成

        Args:
            original_triplet: 元のトリプレット
            chunk_text: 元のテキスト
            ai_router: AIRouterインスタンス

        Returns:
            改善されたトリプレット（失敗時はNone）
        """
        s, r, o = original_triplet

        # LLMが提供されていない場合は初期化
        if ai_router is None:
            self.logger.error("AIRouter not provided for refinement")
            return None, 0


        # ============================================================
        # プロンプト構築
        # ============================================================
        prompt = f"""Given the following text, improve the quality of this knowledge triplet.

    Original triplet:
    - Subject: {s}
    - Relation: {r}
    - Object: {o}

    Text context:
    {chunk_text[:500]}

    Please provide an improved triplet that:
    1. Uses more specific and descriptive entities
    2. Uses a clear and meaningful relation
    3. Accurately reflects the text content
    4. Avoids vague terms like "it", "this", "that"

    Return ONLY the improved triplet in this exact format:
    Subject: [improved subject]
    Relation: [improved relation]
    Object: [improved object]

    If the original triplet cannot be improved, return "NO_IMPROVEMENT".
    """

        # ============================================================
        # LLMで再生成
        # ============================================================
        try:
            # AIRouter経由でRefinerモデルを使って生成！！
            response = ai_router.complete(
                prompt=prompt,
                model=self.refiner_model 
            )
            response_text = response.text.strip()

            # トークン数推定（簡易）
            prompt_tokens = len(prompt) // 4
            response_tokens = len(response_text) // 4
            total_tokens = prompt_tokens + response_tokens

            # "NO_IMPROVEMENT"チェック
            if "NO_IMPROVEMENT" in response_text.upper():
                return None, total_tokens

            # レスポンスをパース
            refined = self._parse_triplet_response(response_text)

            if refined:
                return refined, total_tokens
            else:
                self.logger.debug(f"  Failed to parse refinement response")
                return None, total_tokens

        except Exception as e:
            self.logger.debug(f"  Refinement failed: {type(e).__name__}")
            return None, prompt_tokens

    def _parse_triplet_response(self, response: str) -> Optional[Tuple[str, str, str]]:
        """
        LLMレスポンスからトリプレットを抽出

        Args:
            response: LLMの出力テキスト

        Returns:
            (subject, relation, object) または None
        """
        try:
            lines = response.strip().split('\n')

            subject = None
            relation = None
            object_ = None

            for line in lines:
                line = line.strip()

                if line.startswith('Subject:'):
                    subject = line.replace('Subject:', '').strip()
                elif line.startswith('Relation:'):
                    relation = line.replace('Relation:', '').strip()
                elif line.startswith('Object:'):
                    object_ = line.replace('Object:', '').strip()

            # すべてが抽出できたか確認
            if subject and relation and object_:
                # 空白や記号のみでないか確認
                if (len(subject.strip()) > 1 and
                    len(relation.strip()) > 1 and
                    len(object_.strip()) > 1):
                    return (subject, relation, object_)

            return None

        except Exception as e:
            self.logger.debug(f"  Parse error: {e}")
            return None

    def _validator_check_consistency(
        self,
        triplets: List[Tuple[str, str, str]],
        context: str
    ) -> List[Tuple[str, str, str]]:
        """
        トリプレットの一貫性と矛盾をチェック

        Args:
            triplets: トリプレットのリスト
            context: 元のテキスト

        Returns:
            検証済みトリプレットのリスト（矛盾があるものは除外）
        """
        validated = []
        seen_triplets = set()  # 重複チェック用

        for s, r, o in triplets:
            # ============================================================
            # 1. 重複チェック
            # ============================================================
            triplet_key = (s.lower(), r.lower(), o.lower())
            if triplet_key in seen_triplets:
                self.logger.debug(f"  ⊗ Duplicate: ({s}, {r}, {o})")
                continue

            # ============================================================
            # 2. 自己参照チェック（主語と目的語が同じ）
            # ============================================================
            if s.lower().strip() == o.lower().strip():
                self.logger.debug(f"  ⊗ Self-reference: ({s}, {r}, {o})")
                continue

            # ============================================================
            # 3. 逆関係の矛盾チェック
            # ============================================================
            if self._has_contradictory_relation(s, r, o, validated):
                self.logger.debug(f"  ⊗ Contradictory: ({s}, {r}, {o})")
                continue

            # ============================================================
            # 4. コンテキスト妥当性の最終チェック
            # ============================================================
            if not self._is_contextually_valid(s, r, o, context):
                self.logger.debug(f"  ⊗ Context invalid: ({s}, {r}, {o})")
                continue

            # すべてのチェックをパス
            validated.append((s, r, o))
            seen_triplets.add(triplet_key)

        removed_count = len(triplets) - len(validated)
        if removed_count > 0:
            self.logger.info(f"  → Validator removed {removed_count} inconsistent triplets")

        return validated

    def _has_contradictory_relation(
        self,
        subject: str,
        relation: str,
        object_: str,
        existing_triplets: List[Tuple[str, str, str]]
    ) -> bool:
        """
        既存のトリプレットと矛盾する関係がないかチェック

        Args:
            subject: 主語
            relation: 関係
            object_: 目的語
            existing_triplets: 既に検証済みのトリプレット

        Returns:
            True: 矛盾あり, False: 矛盾なし
        """
        # 矛盾する関係のペア
        contradictory_pairs = [
            # 原因と結果の逆転
            ('causes', 'caused_by'),
            ('creates', 'created_by'),
            ('produces', 'produced_by'),

            # 包含関係の逆転
            ('part_of', 'contains'),
            ('component_of', 'has_component'),
            ('member_of', 'has_member'),

            # 肯定と否定
            ('is', 'is_not'),
            ('has', 'lacks'),
            ('enables', 'prevents'),

            # 時間的矛盾
            ('before', 'after'),
            ('precedes', 'follows'),
        ]

        subject_lower = subject.lower()
        object_lower = object_.lower()
        relation_lower = relation.lower().replace('_', ' ').replace('-', ' ')

        for s_exist, r_exist, o_exist in existing_triplets:
            s_exist_lower = s_exist.lower()
            o_exist_lower = o_exist.lower()
            r_exist_lower = r_exist.lower().replace('_', ' ').replace('-', ' ')

            # 同じエンティティペアで異なる関係
            if ((subject_lower == s_exist_lower and object_lower == o_exist_lower) or
                (subject_lower == o_exist_lower and object_lower == s_exist_lower)):

                # 矛盾する関係のペアをチェック
                for rel1, rel2 in contradictory_pairs:
                    if ((relation_lower == rel1 and r_exist_lower == rel2) or
                        (relation_lower == rel2 and r_exist_lower == rel1)):
                        self.logger.debug(
                            f"  Found contradiction: "
                            f"({subject}, {relation}, {object_}) vs "
                            f"({s_exist}, {r_exist}, {o_exist})"
                        )
                        return True

        return False

    def _is_contextually_valid(
        self,
        subject: str,
        relation: str,
        object_: str,
        context: str,
        min_score: float = 0.3
    ) -> bool:
        """
        トリプレットがコンテキストに対して妥当かチェック

        Args:
            subject: 主語
            relation: 関係
            object_: 目的語
            context: 元のテキスト
            min_score: 最小スコア閾値

        Returns:
            True: 妥当, False: 不適切
        """
        # コンテキストアライメントスコアを使用
        score = self._score_context_alignment(subject, relation, object_, context)

        return score >= min_score

    def _compute_triplet_quality(self, s: str, r: str, o: str) -> float:
        """
        トリプレットの品質をBGE-M3 + ルールベースでスコアリング（API完全排除版）

        Args:
            s: subject (主語)
            r: relation (関係)
            o: object (目的語)

        Returns:
            float: 品質スコア (0.0〜1.0)
        """
        score = 1.0

        # 1. 基本的な文字列チェック（ルールベース、LLM不要）
        s_lower = s.lower().strip()
        r_lower = r.lower().strip()
        o_lower = o.lower().strip()

        # 空・短すぎるチェック
        if len(s_lower) < 2 or len(r_lower) < 2 or len(o_lower) < 2:
            score -= 0.4

        # 数字だけ/記号だけのエンティティ
        if s_lower.isdigit() or o_lower.isdigit() or not any(c.isalnum() for c in s_lower):
            score -= 0.3
        if r_lower in self.relation_blacklist:
            score -= 0.3

        # 自己参照（主語と目的語が同じ）
        if s_lower == o_lower:
            score -= 0.5

        # 主語/関係/目的語が重複
        if s_lower == r_lower or o_lower == r_lower:
            score -= 0.3

        # 2. BGE-M3を使った関係品質チェック
        if hasattr(self, 'embedding_cache') and self.embedding_cache is not None:
            try:
                # 関係文字列を埋め込み
                r_emb = self.embedding_cache.get_embedding(r.lower())
                # ブラックリストとの最大類似度
                if hasattr(self, 'blacklist_embs') and self.blacklist_embs:
                    max_sim = max(
                        np.dot(r_emb, be) / (np.linalg.norm(r_emb) * np.linalg.norm(be) + 1e-9)
                        for be in self.blacklist_embs 
                    )
                    score -= max_sim * 0.8  # 類似度0.8以上で大幅減点
                    # 有用関係リスト（事前定義）との最大類似度（高いほど加点）
                if hasattr(self, 'useful_rel_embs') and self.useful_rel_embs:
                    max_useful_sim = max(
                        np.dot(r_emb, ue) / (np.linalg.norm(r_emb) * np.linalg.norm(ue) + 1e-9)
                        for ue in self.useful_rel_embs
                    )
                    score += max_useful_sim * 0.4
            
                # エンティティの具体性（埋め込みノルム長で簡易判定）
                s_emb = self.embedding_cache.get_embedding(s_lower)
                o_emb = self.embedding_cache.get_embedding(o_lower)
                s_specificity = min(1.0, np.linalg.norm(s_emb) / 0.5)  # スケール調整
                o_specificity = min(1.0, np.linalg.norm(o_emb) / 0.5)
                score += (s_specificity + o_specificity) * 0.3
            except Exception as e:
                self.logger.warning(f"BGE-M3 quality check failed in triplet: {e}. Using rule-based only.")    

        return max(min(score, 1.0), 0.0)

    def _map_triplets_to_documents(
        self,
        triplets: List[Tuple[str, str, str]],
        documents: List[Any]
    ) -> Dict[Any, List[Tuple[str, str, str]]]:
        """
        トリプレットをドキュメントにマッピング

        Args:
            triplets: トリプレットのリスト
            documents: ドキュメントのリスト

        Returns:
            {Document: [triplets]} の辞書
        """
        mapping = {doc: [] for doc in documents}

        # 各トリプレットがどのドキュメントに属するか判定
        for s, r, o in triplets:
            # エンティティがドキュメント内に存在するか確認
            for doc in documents:
                doc_text_lower = doc.text.lower()

                # 主語または目的語がドキュメントに含まれる
                if (s.lower() in doc_text_lower or o.lower() in doc_text_lower):
                    mapping[doc].append((s, r, o))
                    break  # 最初にマッチしたドキュメントに割り当て
            else:
                # どのドキュメントにもマッチしない場合は最初のドキュメントに割り当て
                if documents:
                    mapping[documents[0]].append((s, r, o))

        # 空のエントリを削除
        mapping = {doc: trips for doc, trips in mapping.items() if trips}

        self.logger.info(f"  Mapped {len(triplets)} triplets to {len(mapping)} documents")

        return mapping

    def _score_relation(self, relation: str) -> float:
        """
        関係の明確性をスコアリング

        Returns:
            0.0（最悪）～ 1.0（最良）
        """
        relation_lower = relation.lower().strip()

        # ブラックリスト（即座に0.0）
        if relation_lower in self.relation_blacklist:
            return 0.0

        # 空または短すぎる
        if len(relation_lower) < 2:
            return 0.0

        # 高品質な関係（専門的・具体的）
        high_quality_relations = {
            # 因果関係
            'causes', 'results_in', 'leads_to', 'enables', 'triggers',
            'produces', 'generates', 'influences', 'affects',

            # 構成関係
            'part_of', 'component_of', 'consists_of', 'comprises',
            'contains', 'includes',

            # 使用関係
            'uses', 'utilizes', 'employs', 'applies', 'leverages',
            'implements', 'adopts',

            # 派生関係
            'based_on', 'derived_from', 'inspired_by', 'extends',
            'improves_upon', 'builds_on',

            # 専門関係
            'optimizes', 'parameterizes', 'regularizes', 'approximates',
            'encodes', 'decodes', 'transforms', 'projects',

            # 比較関係
            'outperforms', 'surpasses', 'exceeds', 'improves',
        }

        if relation_lower in high_quality_relations:
            return 1.0

        # 中品質な関係（一般的だが有用）
        medium_quality_relations = {
            'is_a', 'type_of', 'instance_of', 'subclass_of',
            'related_to', 'associated_with', 'connected_to',
            'depends_on', 'requires', 'needs',
        }

        if relation_lower in medium_quality_relations:
            return 0.7

        # 動詞形式（-s, -ed, -ing）なら中程度
        if any(relation_lower.endswith(suffix) for suffix in ['s', 'ed', 'ing']):
            return 0.6

        # それ以外は低品質
        return 0.3

    def _score_entities(self, subject: str, object_: str) -> float:
        """
        エンティティの具体性をスコアリング

        Returns:
            0.0（抽象的・曖昧）～ 1.0（具体的）
        """
        score = 0.0

        # 両方のエンティティをチェック
        for entity in [subject, object_]:
            entity_lower = entity.lower().strip()

            # 空または短すぎる
            if len(entity_lower) < 2:
                continue

            # 代名詞（低品質）
            pronouns = {'it', 'this', 'that', 'these', 'those', 'they', 'them'}
            if entity_lower in pronouns:
                score += 0.0
                continue

            # 単語数（複数単語 = より具体的）
            word_count = len(entity_lower.split())
            if word_count >= 3:
                score += 1.0
            elif word_count == 2:
                score += 0.8
            else:
                score += 0.5

        # 2つのエンティティの平均
        return score / 2.0

    def _score_grammar(
        self,
        subject: str,
        relation: str,
        object_: str
    ) -> float:
        """
        文法的正しさをスコアリング

        Returns:
            0.0（文法的におかしい）～ 1.0（正しい）
        """
        score = 1.0

        # 全て小文字（抽出ミスの可能性）
        if subject.islower() and object_.islower():
            score -= 0.2

        # 数字だけのエンティティ（低品質）
        if subject.isdigit() or object_.isdigit():
            score -= 0.3

        # 記号のみ
        if not any(c.isalnum() for c in subject) or not any(c.isalnum() for c in object_):
            score -= 0.5

        # 主語と目的語が同じ（自己参照）
        if subject.lower() == object_.lower():
            score -= 0.5

        # ------------------------------------------------------------
        # 2. 関係の品質チェック（新規追加）
        # ------------------------------------------------------------

        relation_lower = relation.lower().strip()

        # 関係が空または短すぎる
        if len(relation_lower) < 2:
            score -= 0.4

        # 関係がブラックリストに含まれる（低品質）
        if relation_lower in self.relation_blacklist:
            score -= 0.3

        # 関係が記号のみ
        if not any(c.isalnum() for c in relation):
            score -= 0.4

        # ------------------------------------------------------------
        # 3. トリプレット全体の整合性チェック
        # ------------------------------------------------------------

        # 主語と関係が同じ（例: "uses uses object"）
        if subject.lower() == relation_lower:
            score -= 0.3

        # 目的語と関係が同じ（例: "subject uses uses"）
        if object_.lower() == relation_lower:
            score -= 0.3

        # 3つとも同じ（最悪）
        if subject.lower() == relation_lower == object_.lower():
            score -= 0.5

        return max(score, 0.0)