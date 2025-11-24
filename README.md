# Crystal-Evoluder

**TEI.xml → Markdown → Neo4j Graph DB Pipeline (LLM-powered)**
Transform structured PDFs into machine-readable knowledge graphs.

Crystal-Evoluder is a two-stage pipeline for converting academic papers into a semantic knowledge graph.
The system uses **GROBID → TEI.xml → Markdown → LLM-driven triplet extraction → Neo4j**.

This repository contains two main scripts:

---

## ✨ Features

* Convert **GROBID TEI.xml** into clean, RAG-ready **Markdown**
* Generate high-quality embeddings using **text-embedding-3-large**
* Extract conceptual relationships (triplets) using **GPT-4o-mini**
* Store nodes/edges inside a **Neo4j** graph database
* Perfect for:

  * literature review automation
  * research knowledge graphs
  * RAG over structured academic content
  * citation network analysis
  * conceptual mapping / world-model construction

---

## 📦 Repository Structure

```
Crystal-Evoluder/
│
├── crystallizer.py   # TEI.xml → Markdown  + Embedding generation
├── llmn.py           # Markdown → Neo4j (LLM-based triplet extraction)
├── requirements.txt
└── README.md
```

---

## 🔧 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Libraries include:

* llama-index-core
* llama-index-readers-file
* llama-index-embeddings-huggingface
* llama-index-graph-stores-neo4j
* llama-index-llms-openai
* neo4j
* logging

---

## 🔐 2. Configure API Keys

Set your OpenAI API key in `~/.zshrc` or `.env`:

```bash
export OPENAI_API_KEY="your-key-here"
```

Then reload:

```bash
source ~/.zshrc
```

---

## 🧱 3. Overview of the Pipeline

### **Step 1 — TEI.xml → Markdown (crystallizer.py)**

* Input: GROBID-generated `tei.xml`
* Output: Clean Markdown file
* Embedding model:
  **OpenAI text-embedding-3-large**

```bash
python3 crystallizer.py
```

### **Step 2 — Markdown → Neo4j (llmn.py)**

* Uses **GPT-4o-mini** for concept extraction
* Generates semantic triplets
* Saves them into a Neo4j database

```bash
python3 llmn.py
```

After import, you can open:

**Neo4j Browser:**
[http://localhost:7474/](http://localhost:7474/)

Example query:

```cypher
MATCH (a)-[r]->(b) RETURN a, r, b LIMIT 100;
```

---

## 🎯 Goals

* Serve as a **research knowledge graph builder**
* Provide a foundation for **RAG over academic papers**
* Enable automated discovery of:

  * conceptual clusters
  * citation networks
  * methodological lineages
  * cross-disciplinary bridges

---

## 📜 License

This project includes **Neo4j Community Edition**,
which is licensed under **GNU GPLv3**.

All other code in this repository follows the same license for compatibility.

---

## 🙌 Acknowledgements

* GROBID (for TEI extraction)
* LlamaIndex
* Neo4j
* OpenAI / GPT-4o-mini

