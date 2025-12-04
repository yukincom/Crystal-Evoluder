# 開発メモ
- 二つのファイルの結合　✅　11/25 → 11/28再分割
- Grobid統合　✅　11/27
- PDF以外のデータ ✅  11/26<br>
   TEI (.tei.xml)<br>
   Markdown (.md) - frontmatter対応、見出し分割<br>
   DOCX (.docx) - 見出しスタイル検出<br>
   HTML (.html) - trafilatura + BeautifulSoup<br>
   TXT (.txt) - SemanticChunker対応<br>
   フォーマット自動判定<br>
- バッチ処理 ✅  11/26
- ログの階層化 ✅  11/26
- エラーリカバリ　✅　11/28
- Neo4jスキーマ　✅　11/28
- データ分割後テスト　✅　12/2
- dual-chunk　✅ 12/4
- マージ追加
- Data Provenance

## 見落としていた問題点　12/4
2024年6月21日[AI生成コンテンツの検索拡張生成 ：調査](https://arxiv.org/html/2402.19473v6)
### トリプレットの欠落・ノイズ
- Dual-Chunkは「チャンクの切り方」しか変えない。gpt-4o-miniが「Self-Attention → uses → Scaled Dot-Product Attention」を見逃したり、「is a type of」と「is based on」を逆に抜いたりはLLMの抽出能力の問題。
- ->Self-RAG / CRITIC風自己修正 ・トリプレット抽出にQwen2.5-32BやClaude-3.5-Sonnetに変更 
- 後処理でrelation_compatibilityで低スコアトリプレットをフィルタ
### 同一実体の別名問題（Coreference）
- 「Self-Attention」「the attention mechanism」「it」が同じものを指すのに別ノードになる。
- Dual-Chunkではチャンク内に収まっていても、実体連結ができない。
- Neo4jに「Self-Attention」が5個ノードできて、重みが分散 → パスが取れなくなる。
- ->Coreference Resolution（NeuralCorefやspacyのcorefモデル） 
- 抽出後にentity linking（nomic-embed-textで類似度0.95以上はマージ）
### Multi-hopの深さ不足
- たとえ重みが完璧でも、agentic_retrieve()のmax_steps=5だと6hop以上の論理は取れない（例：Attention → Scaled Dot-Product → softmax → normalization → stability）。
- 複雑な論文質問で「なぜstableなのか？」が答えられない。
- ->max_stepsを動的に拡張（confidence < 0.7なら+2）
- Graph-S³風のaction spaceに「expand_hop」追加
### 重みのスケールがバラバラ
- intra_rawとinter_rawの絶対値が論文ごとに100倍違うことがある =0.7intra + 0.3interの係数が意味をなさない。
- 小さい論文では重みが全部0.01以下になり、agentが「全部同じ」と勘違い。
- ->論文ごとに重みを正規化（min-max or z-score） 
- 最終重みをlogスケールに変換
[RetrievalとGraphの同期ズレ問題]
- Retrieval storeで取ってきた文脈と、Neo4jのKGが同じチャンクから来ていないと、agentが「文脈はあるのにパスがない」状態になる。
- ->retrieval_docsとgraph_docsに共通のchunk_idを付与 
- retrieve()が返すDocumentにgraph_node_idsリストを持たせる

## この辺を突っ込んだ 12/2
[知識グラフ質問応答のための効率的かつ一般化可能なグラフ検索器の学習](https://arxiv.org/abs/2506.09645 )2025年6月11日提出<br>
[DAMR: LLM ガイド付き MCTS による効率的かつ適応的なコンテキスト認識型知識グラフ質問応答](https://arxiv.org/abs/2508.00719) 2025年8月1日提出<br>
[Graph-S3: 合成段階的監督によるエージェントテキストグラフ検索の強化](https://arxiv.org/abs/2510.03323 ) 2025年10月1日提出<br>

## AI で補完する “ラベル付けモジュール”
これは難しそう。私の手には負えないかも。。。<br>
現状は“ノードとトリプル抽出” までやるけど、下手すると、情報が歪むかも・・・。<br>
### ノード名の正規化
GPT-4o-miniにプロンプトで「同義語/略語を検知してマージ」させる。<br>
embeddingで類似スコア計算（cosine similarity > 0.85でクラスタ）。<br>
Neo4j GDSのk-meansクラスタリング+embeddingで、FastRP/node2vecが専門用語のsemanticを保持（精度85-90%）。<br>
GPT-4o-miniのfew-shotプロンプトでドメイン知識注入（e.g. "CS論文ではGNN=Graph Neural Network"）。<br>
reguloGPTで分子経路のエンティティ正規化成功率92%。<br>

### 関係ラベルの統一
GPT-4o-miniで「類似関係を階層化ラベルにマッピング」。embeddingで関係ベクトル比較。<br>
SF-GPTのEntity Alignment Moduleで、GPTの多レスポンスをフィルタ（ノイズ↓30%）。<br>
https://www.sciencedirect.com/science/article/abs/pii/S0925231224014978
### 類似概念のクラスタリング
embeddingでベクトル距離計算→k-medoidsクラスタ。GPTでクラスタラベル生成。<br>
Neo4j GDSのk-medoidsでグラフ距離ベースクラスタ（embedding不要で説明性↑）。<br>
bge-m3 embeddingのsparseモードでキーワード保持（MLDR nDCG+10%）。 <br>
GPT-4o-miniの速度で10s以内に処理（entity/relation抽出）。<br>
https://medium.com/neo4j/graph-data-models-for-rag-applications-d59c29bb55cc
<br>
➡ ここを GPT-4o-mini / embedding でやると、グラフの質が跳ね上がる。<br>
<br>
### ラベル付けモジュール追加　こんな感じかな？
def label_normalization_module(index, llm=OpenAI(model="gpt-4o-mini")):<br>
1. ノード正規化: embeddingで類似クラスタ<br>
  from sklearn.cluster import KMeans<br>
  embeddings = [node.embedding for node in index.docstore.docs.values()]  bge-m3から<br>
  kmeans = KMeans(n_clusters=10).fit(embeddings)<br>
2. GPTでクラスタラベル生成
  prompt = "Cluster these CS terms: {terms}. Normalize to canonical names."<br>
3. 関係統一: GPTプロンプト
  relations = list(set(edge.relation for edge in index.get_networkx_graph().edges()))<br>
  unified = llm.predict(prompt=f"Unify relations: {relations} into 10 categories.")<br>
4. Neo4jにupsert
  return updated_index

## 将来的に、エクスポート先を増やしたい。
### 推奨（セット済み）
- Neo4j
### プラグインで任意（後で作る。差し込む場所だけ作った。）
- JSON-LD（汎用）
- Blazegraph（RDF 系）
- PuppyGraph（分散高性能が欲しい人向け）
- Stardog（商用なので optional） 
- GroundX On-Prem(企業向けなので、要望があったら)
