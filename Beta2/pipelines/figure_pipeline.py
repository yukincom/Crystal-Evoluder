"""
図表解析パイプライン（改善版）

改善点：
- キャプション抽出を上下両方に対応
- 解像度を引数化（デフォルト200dpi）
- チャンク紐付けを領域オーバーラップで改善
- 解析結果キャッシュ機能
- Ollama APIのエラーハンドリング強化
"""

from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import hashlib
import json
import time

from PIL import Image
import fitz  # PyMuPDF
import requests


class FigurePipeline:
    """図表解析パイプライン（改善版）"""

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        vision_model: str = "granite3.2-vision",
        dpi: int = 200,
        use_cache: bool = True,
        logger = None
    ):
        self.ollama_url = ollama_url
        self.vision_model = vision_model
        self.dpi = dpi
        self.use_cache = use_cache
        self.logger = logger

        # 図表保存ディレクトリ
        self.figure_dir = Path("./figures")
        self.figure_dir.mkdir(exist_ok=True)

        # キャッシュディレクトリ
        self.cache_dir = Path("./figures/.cache")
        if use_cache:
            self.cache_dir.mkdir(exist_ok=True)

    # ========================================
    # Step 1: PDF から図表検出
    # ========================================

    def detect_figures(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        PDFから図表を検出

        Args:
            pdf_path: PDFファイルパス

        Returns:
            図表リスト（page, bbox, type, caption含む）
        """
        doc = fitz.open(pdf_path)
        figures = []

        for page_num, page in enumerate(doc):
            # 画像を抽出
            images = page.get_images()

            for img_index, img in enumerate(images):
                xref = img[0]

                # 画像のバウンディングボックスを取得
                rects = page.get_image_rects(xref)

                if not rects:
                    continue

                bbox = rects[0]  # 最初の矩形を使用

                # 小さすぎる画像はスキップ（アイコンなど）
                width = bbox.x1 - bbox.x0
                height = bbox.y1 - bbox.y0
                if width < 50 or height < 50:
                    continue

                # 図表として記録
                figures.append({
                    'page': page_num,
                    'bbox': tuple(bbox),
                    'xref': xref,
                    'type': 'image',
                    'caption': self._extract_nearby_caption(page, bbox)
                })

        doc.close()

        if self.logger:
            self.logger.info(f"Detected {len(figures)} figures")
        return figures

    def _extract_nearby_caption(
        self,
        page,
        bbox: Tuple[float, float, float, float]
    ) -> str:
        """
        図表の上下からキャプションを抽出（改善版）

        Args:
            page: PyMuPDF page
            bbox: 図表のバウンディングボックス

        Returns:
            キャプション文字列
        """
        x0, y0, x1, y1 = bbox

        # 上下両方のエリアをチェック
        upper_area = fitz.Rect(x0, max(0, y0 - 60), x1, y0)
        lower_area = fitz.Rect(x0, y1, x1, min(page.rect.height, y1 + 60))

        upper_text = page.get_text("text", clip=upper_area)
        lower_text = page.get_text("text", clip=lower_area)

        # "Figure X:" や "Table X:" で始まる行を探す（上下両方）
        for text in [lower_text, upper_text]:  # 下を優先
            for line in text.split('\n'):
                line_lower = line.lower().strip()
                if line_lower.startswith(('figure', 'fig.', 'table', 'tab.')):
                    return line.strip()

        # キャプションが見つからない場合は下部テキストを返す
        caption = lower_text.strip() or upper_text.strip()
        return caption[:150]  # 最初の150文字

    # ========================================
    # Step 2: 図表画像を保存
    # ========================================

    def extract_figure_image(
        self,
        pdf_path: str,
        figure: Dict[str, Any],
        output_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        図表画像を抽出して保存

        Args:
            pdf_path: PDFパス
            figure: detect_figures() の返り値
            output_dir: 保存先ディレクトリ

        Returns:
            図表情報（image_path, image_hash追加）
        """
        output_dir = output_dir or self.figure_dir

        doc = fitz.open(pdf_path)
        page = doc[figure['page']]

        # 画像を抽出（解像度は引数化）
        pix = page.get_pixmap(clip=figure['bbox'], dpi=self.dpi)

        # ファイル名生成
        pdf_name = Path(pdf_path).stem
        fig_name = f"{pdf_name}_p{figure['page']}_fig{figure['xref']}.png"
        image_path = output_dir / fig_name

        # 保存
        pix.save(str(image_path))

        # ハッシュ計算
        with open(image_path, 'rb') as f:
            image_hash = hashlib.sha256(f.read()).hexdigest()

        doc.close()

        return {
            'image_path': str(image_path),
            'image_hash': image_hash,
            **figure
        }

    # ========================================
    # Step 3: Vision AIで解析（キャッシュ対応）
    # ========================================

    def analyze_figure_with_vision(
        self,
        image_path: str,
        caption: str = "",
        image_hash: str = None
    ) -> Dict[str, Any]:
        """
        Vision AIで図表を解析（キャッシュ対応）

        Args:
            image_path: 画像パス
            caption: キャプション（ヒントとして使う）
            image_hash: 画像ハッシュ（キャッシュキー）

        Returns:
            解析結果辞書
        """
        # キャッシュチェック
        if self.use_cache and image_hash:
            cached = self._load_cache(image_hash)
            if cached:
                if self.logger:
                    self.logger.info(f"Using cached analysis for {image_hash[:8]}")
                return cached

        # 画像をbase64エンコード
        import base64
        with open(image_path, 'rb') as f:
            image_base64 = base64.b64encode(f.read()).decode()

        # プロンプト構築
        prompt = self._build_vision_prompt(caption)

        # Ollama Vision APIを呼び出し（リトライ付き）
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.vision_model,
                        "prompt": prompt,
                        "images": [image_base64],
                        "stream": False,
                        "options": {
                            "temperature": 0.1,  # より安定した出力
                            "num_predict": 512   # トークン数制限
                        }
                    },
                    timeout=120
                )

                response.raise_for_status()
                result = response.json()

                # JSONパース
                content = result.get('response', '')
                parsed = self._parse_vision_response(content)

                analysis_result = {
                    **parsed,
                    'model': self.vision_model,
                    'extraction_status': 'success'
                }

                # キャッシュ保存
                if self.use_cache and image_hash:
                    self._save_cache(image_hash, analysis_result)

                return analysis_result

            except requests.exceptions.RequestException as e:
                if self.logger:
                    self.logger.warning(
                        f"Vision API attempt {attempt + 1}/{max_retries} failed: {e}"
                    )
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    if self.logger:
                        self.logger.error(f"Vision analysis failed after {max_retries} attempts")

        # フォールバック
        return {
            'type': 'unknown',
            'confidence': 0.0,
            'summary': '',
            'key_values': [],
            'trend': '',
            'model': self.vision_model,
            'extraction_status': 'failed'
        }

    def _build_vision_prompt(self, caption: str) -> str:
        """Vision AIへのプロンプト生成"""

        prompt = """Analyze this figure and respond in JSON format ONLY (no markdown).

Classification:
1. If this is a scientific chart (bar/line/scatter plot, graph, diagram with data):
   - Type: "chart"
   - Provide detailed analysis
2. If this is a decorative illustration, photo, or simple icon:
   - Type: "illustration"
   - Brief description only
3. If irrelevant or too unclear:
   - Type: "irrelevant"

For charts, extract:
- Summary: 2-4 sentences describing the main finding
- Key values: Important numbers/metrics (max 5 items)
- Trend: Overall pattern (e.g., "increasing", "stable", "bimodal")

"""

        if caption:
            prompt += f"\nFigure caption: {caption}\n"

        prompt += """
Response format (pure JSON, no backticks):
{
  "type": "chart",
  "confidence": 0.85,
  "summary": "The chart shows...",
  "key_values": [{"label": "Max value", "value": "42", "metric": "accuracy"}],
  "trend": "increasing"
}
"""

        return prompt

    def _parse_vision_response(self, content: str) -> Dict[str, Any]:
        """Vision AIの応答をパース（改善版）"""

        # JSONを抽出（マークダウン対応）
        content = content.strip()
        if '```json' in content:
            content = content.split('```json')[1].split('```')[0]
        elif '```' in content:
            content = content.split('```')[1].split('```')[0]

        # JSON文字列を探索
        start_idx = content.find('{')
        end_idx = content.rfind('}')

        if start_idx != -1 and end_idx != -1:
            content = content[start_idx:end_idx + 1]

        try:
            parsed = json.loads(content)

            # 必須フィールドの検証
            required_fields = ['type', 'confidence', 'summary']
            for field in required_fields:
                if field not in parsed:
                    parsed[field] = '' if field == 'summary' else 'unknown' if field == 'type' else 0.0

            # デフォルト値設定
            parsed.setdefault('key_values', [])
            parsed.setdefault('trend', '')

            return parsed

        except json.JSONDecodeError as e:
            if self.logger:
                self.logger.warning(f"JSON parse error: {e}")
            return {
                'type': 'unknown',
                'confidence': 0.0,
                'summary': content[:200],
                'key_values': [],
                'trend': ''
            }

    def _load_cache(self, image_hash: str) -> Optional[Dict[str, Any]]:
        """キャッシュから解析結果をロード"""
        cache_file = self.cache_dir / f"{image_hash[:16]}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def _save_cache(self, image_hash: str, result: Dict[str, Any]):
        """解析結果をキャッシュに保存"""
        cache_file = self.cache_dir / f"{image_hash[:16]}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(result, f, indent=2)
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Failed to save cache: {e}")

    # ========================================
    # Step 4: チャンクとの紐付け（改善版）
    # ========================================

    def align_with_chunks(
        self,
        figure: Dict[str, Any],
        chunks: List[Dict[str, Any]]
    ) -> Optional[str]:
        """
        図表を最も関連の高いチャンクに紐付け（領域オーバーラップ考慮）

        Args:
            figure: 図表情報（page, bbox含む）
            chunks: チャンクリスト（page, chunk_id, bbox(optional)含む）

        Returns:
            最も関連の高い chunk_id
        """
        figure_page = figure['page']
        figure_bbox = figure.get('bbox')

        # 同じページのチャンクを探す
        same_page_chunks = [
            c for c in chunks
            if c.get('page') == figure_page
        ]

        if same_page_chunks:
            # チャンクにbbox情報がある場合は領域オーバーラップで判定
            if figure_bbox and any('bbox' in c for c in same_page_chunks):
                best_chunk = self._find_overlapping_chunk(
                    figure_bbox,
                    same_page_chunks
                )
                if best_chunk:
                    return best_chunk['chunk_id']

            # bbox情報がない場合は最初のチャンク
            return same_page_chunks[0]['chunk_id']

        # 隣接ページ（±1ページ）
        adjacent_chunks = [
            c for c in chunks
            if abs(c.get('page', -999) - figure_page) == 1
        ]

        if adjacent_chunks:
            return adjacent_chunks[0]['chunk_id']

        # フォールバック：最も近いページのチャンク
        if chunks:
            return min(
                chunks,
                key=lambda c: abs(c.get('page', -999) - figure_page)
            )['chunk_id']

        return None

    def _find_overlapping_chunk(
        self,
        figure_bbox: Tuple[float, float, float, float],
        chunks: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """領域オーバーラップに基づいて最適なチャンクを探す"""

        fx0, fy0, fx1, fy1 = figure_bbox
        figure_y_center = (fy0 + fy1) / 2

        best_chunk = None
        min_distance = float('inf')

        for chunk in chunks:
            chunk_bbox = chunk.get('bbox')
            if not chunk_bbox:
                continue

            cx0, cy0, cx1, cy1 = chunk_bbox
            chunk_y_center = (cy0 + cy1) / 2

            # Y座標の距離で判定（図表はチャンクの近くにある前提）
            distance = abs(figure_y_center - chunk_y_center)

            if distance < min_distance:
                min_distance = distance
                best_chunk = chunk

        return best_chunk

    # ========================================
    # Step 5: Neo4jに保存
    # ========================================

    def save_to_neo4j(
        self,
        figure: Dict[str, Any],
        chunk_id: str,
        graph_store
    ):
        """
        図表情報をNeo4jに保存

        Args:
            figure: 図表情報（analysis結果含む）
            chunk_id: 紐付けるチャンクID
            graph_store: Neo4jGraphStore
        """

        # Figureノード作成
        query = """
        MERGE (f:Figure {figure_id: $figure_id})
        SET
            f.page = $page,
            f.type = $type,
            f.confidence = $confidence,
            f.summary = $summary,
            f.key_values_json = $key_values_json,
            f.trend = $trend,
            f.caption = $caption,
            f.image_path = $image_path,
            f.image_hash = $image_hash,
            f.extraction_status = $extraction_status,
            f.model = $model,
            f.updated_at = datetime()

        WITH f
        MATCH (c:Chunk {chunk_id: $chunk_id})
        MERGE (f)-[:ALIGNED_WITH {method: 'location_based'}]->(c)
        """

        params = {
            'figure_id': figure['image_hash'][:16],
            'page': figure['page'],
            'type': figure.get('type', 'unknown'),
            'confidence': figure.get('confidence', 0.0),
            'summary': figure.get('summary', ''),
            'key_values_json': json.dumps(figure.get('key_values', [])),
            'trend': figure.get('trend', ''),
            'caption': figure.get('caption', ''),
            'image_path': figure.get('image_path', ''),
            'image_hash': figure.get('image_hash', ''),
            'extraction_status': figure.get('extraction_status', 'unknown'),
            'model': figure.get('model', self.vision_model),
            'chunk_id': chunk_id
        }

        graph_store.query(query, params)

    # ========================================
    # 全体パイプライン
    # ========================================

    def process_pdf(
        self,
        pdf_path: str,
        chunks: List[Dict[str, Any]],
        graph_store
    ) -> Dict[str, Any]:
        """
        PDFの図表を一括処理

        Args:
            pdf_path: PDFパス
            chunks: チャンクリスト
            graph_store: Neo4jGraphStore

        Returns:
            処理結果のサマリ
        """
        if self.logger:
            self.logger.info(f"Processing figures in {pdf_path}")

        # 1. 図表検出
        figures = self.detect_figures(pdf_path)

        if not figures:
            return {
                'total': 0,
                'processed': 0,
                'failed': 0,
                'chart_count': 0,
                'skipped': 0
            }

        results = {
            'total': len(figures),
            'processed': 0,
            'failed': 0,
            'chart_count': 0,
            'skipped': 0
        }

        # 2. 各図表を処理
        for fig in figures:
            try:
                # 画像抽出
                fig_with_image = self.extract_figure_image(pdf_path, fig)

                # Vision解析
                analysis = self.analyze_figure_with_vision(
                    fig_with_image['image_path'],
                    fig_with_image['caption'],
                    fig_with_image['image_hash']
                )

                # 無関係な画像はスキップ
                if analysis['type'] == 'irrelevant':
                    results['skipped'] += 1
                    continue

                # 結合
                complete_figure = {**fig_with_image, **analysis}

                # チャンク紐付け
                chunk_id = self.align_with_chunks(complete_figure, chunks)

                if chunk_id:
                    # Neo4jに保存
                    self.save_to_neo4j(complete_figure, chunk_id, graph_store)
                    results['processed'] += 1

                    if analysis['type'] == 'chart':
                        results['chart_count'] += 1
                else:
                    results['failed'] += 1

            except Exception as e:
                if self.logger:
                    self.logger.error(f"Failed to process figure: {e}")
                results['failed'] += 1

        if self.logger:
            self.logger.info(
                f"Figure processing complete: "
                f"{results['processed']}/{results['total']} processed, "
                f"{results['chart_count']} charts detected, "
                f"{results['skipped']} skipped"
            )

        return results