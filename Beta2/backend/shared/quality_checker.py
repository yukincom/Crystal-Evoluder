"""
データ品質チェッカー（Qwen 8B）
"""
import logging
import json
import re
import csv
import requests

from pathlib import Path
from typing import List, Dict, Any
from llama_index.core import Document
import numpy as np

class DataQualityChecker:
    """データ品質チェック（ローカルLLM: Qwen 8B）"""
    
    def __init__(self, ollama_url: str = 'http://localhost:11434', logger: logging.Logger = None):
        self.ollama_url = ollama_url
        self.logger = logger or logging.getLogger('DataQualityChecker')
        self.ollama_available = self._check_ollama()
        
        self.embedding_cache = None

        if self.ollama_available:
            self.logger.info("✅ Ollama (Qwen 8B) available")
        else:
            self.logger.warning("⚠️  Ollama not available (quality check disabled)")

    def set_embedding_cache(self, embedding_cache):
        """共有キャッシュを設定"""
        self.embedding_cache = embedding_cache
        if self.embedding_cache:
            self.logger.info("✅ BGE-M3 embedding cache set for quality check")
        else:
            self.logger.warning("⚠️ No embedding cache provided for quality check")

    def _check_ollama(self) -> bool:
        """Ollamaが起動しているか確認"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False

    def check_documents(self, documents: List[Document], output_dir: str = './review') -> Dict[str, Any]:
        """
        Documentの品質チェック（BGE-M3ハイブリッド版）
        """
        clean = []
        flagged = []

        for i, doc in enumerate(documents):
            self.logger.info(f"Checking {i+1}/{len(documents)}: {doc.metadata.get('section', 'Unknown')}")

            issues = self._detect_issues(doc)

            if not issues:
                clean.append(doc)
            else:
                severity = self._determine_severity(issues)
                flagged.append({
                    'document': doc,
                    'issues': [issue['type'] for issue in issues],
                    'reasons': [issue['reason'] for issue in issues],
                    'severity': severity
                })

        stats = {
            'total': len(documents),
            'clean': len(clean),
            'flagged': len(flagged)
        }

        if flagged:
            self._save_review_queue(flagged, output_dir)

        return {'clean': clean, 'flagged': flagged, 'stats': stats}

    def _detect_issues(self, doc: Document) -> List[Dict]:
        """BGE-M3で粗チェック + オプションでQwen最終確認"""
        issues = []

        text = doc.text.strip()
        if not text:
            issues.append({'type': 'empty', 'reason': 'Document is empty'})
            return issues

        # BGE-M3粗チェック（オフライン・高速）
        if self.embedding_cache:
            try:
                text_emb = self.embedding_cache.get_embedding(text)
                specificity = np.linalg.norm(text_emb) / 0.5  # 具体性スコア
                if specificity < 0.6:
                    issues.append({'type': 'too_abstract', 'reason': 'Text lacks specificity (low embedding norm)'})

                # 短すぎチェック（埋め込み密度）
                sentences = [s.strip() for s in text.split('\n') if s.strip()]
                if len(sentences) < 3:
                    issues.append({'type': 'too_short', 'reason': 'Too few sentences'})

                # 矛盾チェック（簡易：文間類似度が異常に高い/低い）
                if len(sentences) > 1:
                    embs = [self.embedding_cache.get_embedding(s) for s in sentences[:5]]  # 先頭5文だけ
                    sims = [np.dot(embs[i], embs[i+1]) / (np.linalg.norm(embs[i]) * np.linalg.norm(embs[i+1]) + 1e-9)
                            for i in range(len(embs)-1)]
                    if any(sim > 0.98 for sim in sims):  # ほぼ同一文連続 → 矛盾/重複
                        issues.append({'type': 'potential_contradiction', 'reason': 'High similarity between consecutive sentences'})
            except Exception as e:
                self.logger.warning(f"BGE-M3 coarse check failed: {e}")

        # Qwen 8B最終確認（問題フラグ立ったものだけ）
        if issues and self.ollama_available:
            ollama_issues = self._ollama_check(text)
            issues.extend(ollama_issues)

        return issues

    def _ollama_check(self, documents: List[Document], output_dir: str = './review') -> Dict[str, Any]: 
        """
        Document の品質をチェック

        Returns:
            {
                'clean': [Document, ...],      # 問題なし
                'flagged': [                    # 要レビュー
                    {
                        'document': Document,
                        'issues': ['missing_subject', ...],
                        'reasons': ['主語が欠落', ...],
                        'severity': 'high' | 'medium' | 'low'
                    },
                    ...
                ],
                'stats': {...}
            }
        """
        if not self.ollama_available:
            self.logger.warning("Ollama not available, skipping quality check")
            return {'clean': documents, 'flagged': [], 'stats': {'total': len(documents), 'clean': len(documents), 'flagged': 0}}

        self.logger.info(f"🔍 Checking quality of {len(documents)} documents...")

        clean = []
        flagged = []

        for i, doc in enumerate(documents):
            self.logger.info(f"  Checking {i+1}/{len(documents)}: {doc.metadata.get('section', 'Unknown')}")

            issues = self._detect_issues(doc)

            if issues:
                flagged.append({
                    'document': doc,
                    'issues': [issue['type'] for issue in issues],
                    'reasons': [issue['reason'] for issue in issues],
                    'severity': self._assess_severity(issues),
                    'metadata': doc.metadata
                })
            else:
                clean.append(doc)

        # レビューキュー保存
        if flagged:
            self._save_review_queue(flagged, output_dir)

        stats = {
            'total': len(documents),
            'clean': len(clean),
            'flagged': len(flagged),
            'high_severity': sum(1 for f in flagged if f['severity'] == 'high'),
            'medium_severity': sum(1 for f in flagged if f['severity'] == 'medium'),
            'low_severity': sum(1 for f in flagged if f['severity'] == 'low')
        }

        self.logger.info(
            f"✅ Quality check complete: "
            f"{stats['clean']} clean, {stats['flagged']} flagged "
            f"(high: {stats['high_severity']}, medium: {stats['medium_severity']}, low: {stats['low_severity']})"
        )

        return {
            'clean': clean,
            'flagged': flagged,
            'stats': stats
        }

    def _has_figure_context(self, text: str) -> bool:
        """図表参照に十分なコンテキストがあるか"""
        # 簡易チェック: 図表参照の前後に説明がある
        patterns = [
            r'図\s*\d+.*?[。\.]',
            r'Figure\s*\d+.*?\.',
            r'表\s*\d+.*?[。\.]',
            r'Table\s*\d+.*?\.'
        ]
        for pattern in patterns:
            if re.search(pattern, text):
                return True
        return False

    def _needs_ai_check(self, text: str) -> bool:
        """AIによる詳細チェックが必要か判定"""
        # 構造が複雑な場合のみAIチェック
        suspicious_patterns = [
            '不明', '上記', '前述', '以下',
            '...', '※', '＊',
            len(text.split('。')) > 10,  # 長文
            text.count('、') > 20         # 複雑な構造
        ]
        return any(suspicious_patterns)

    def _ai_deep_check(self, text: str) -> List[Dict[str, str]]:
        """Qwen 8B で詳細チェック"""
        prompt = f"""以下のテキストの品質をチェックしてください。

            テキスト:
            {text[:500]}...

            以下の観点でチェック:
            1. 主語の欠落（文脈が不明確）
            2. 矛盾する記述
            3. 図表参照の欠落
            4. 文章の破損（途中で切れている等）

            問題があればJSON形式で出力:
            {{"issues": [{{"type": "問題タイプ", "reason": "理由"}}]}}

            問題がなければ:
            {{"issues": []}}
            """

        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    'model': 'qwen2.5:32b',
                    'prompt': prompt,
                    'stream': False
                },
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                response_text = result.get('response', '{}')

                # JSON抽出（AIの出力から）
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                    return data.get('issues', [])

        except Exception as e:
            self.logger.warning(f"AI check failed: {e}")

        return []

    def _assess_severity(self, issues: List[Dict[str, str]]) -> str:
        """問題の深刻度を判定"""
        severity_map = {
            'missing_subject': 'high',
            'contradiction': 'high',
            'missing_visual': 'medium',
            'too_short': 'low',
            'structural_damage': 'high'
        }

        max_severity = 'low'
        for issue in issues:
            issue_severity = severity_map.get(issue['type'], 'low')
            if issue_severity == 'high':
                return 'high'
            if issue_severity == 'medium' and max_severity == 'low':
                max_severity = 'medium'

        return max_severity

    def _save_review_queue(self, flagged: List[Dict], output_dir: str):
        """レビューキューをファイル保存"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # CSV形式
        import csv
        csv_path = output_path / 'review_queue.csv'
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Section', 'Severity', 'Issues', 'Reasons', 'Text Preview'])

            for item in flagged:
                writer.writerow([
                    item['metadata'].get('section', 'Unknown'),
                    item['severity'],
                    ', '.join(item['issues']),
                    ', '.join(item['reasons']),
                    item['document'].text[:100] + '...'
                ])

        # JSON形式（詳細情報）
        json_path = output_path / 'review_queue.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump([
                {
                    'section': item['metadata'].get('section'),
                    'severity': item['severity'],
                    'issues': item['issues'],
                    'reasons': item['reasons'],
                    'text': item['document'].text,
                    'metadata': item['metadata']
                }
                for item in flagged
            ], f, ensure_ascii=False, indent=2)

        self.logger.info(f"📝 Review queue saved:")
        self.logger.info(f"   CSV: {csv_path}")
        self.logger.info(f"   JSON: {json_path}")