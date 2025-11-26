# 開発計画
- 二つのファイルの結合　✅　11/25
- Grobid統合
- PDF以外のデータ ✅  11/26
  ✅ TEI (.tei.xml)<br>
  ✅ Markdown (.md) - frontmatter対応、見出し分割<br>
  ✅ DOCX (.docx) - 見出しスタイル検出<br>
  ✅ HTML (.html) - trafilatura + BeautifulSoup<br>
  ✅ TXT (.txt) - SemanticChunker対応<br>
  ✅ フォーマット自動判定<br>
- バッチ処理 ✅  11/26
- ログの階層化 ✅  11/26
- エラーリカバリ
- Neo4jスキーマ
- Data Provenance
- エクスポート拡張プラグイン
- obsidian　→　DB化　syncのプラグイン
- Markdown粒度調節
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
