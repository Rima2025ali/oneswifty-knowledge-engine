# oneswifty-knowledge-engine
# 🚀 OneSwifty: Universal Knowledge Engine

OneSwifty is a high-precision **RAG (Retrieval-Augmented Generation)** platform designed for technical and scientific document analysis. By combining **PostgreSQL (pgvector)** with **OpenAI's LLMs**, OneSwifty moves beyond simple keyword search to provide a deep "Scientific Audit" of your local knowledge base.

## 🌟 Key Features

* **AI Meta-Data Auditor:** Automatically identifies the Title, Author, and Category of PDFs upon upload.
* **Smart Semantic Chunking:** Implements word-boundary awareness with a 50-character overlap to preserve technical context.
* **Technical Formula Detection:** Automatically identifies and renders complex LaTeX equations from scientific papers.
* **Performance Auditing:** Built-in logging system that tracks query timestamps, AI responses, and confidence scores for system tuning.
* **Vectorized Search:** Leverages `text-embedding-3-small` for high-dimensional semantic mapping.

## 🛠️ Tech Stack

* **Language:** Python 3.9+
* **Frontend:** Streamlit
* **Database:** PostgreSQL + pgvector
* **AI Models:** OpenAI GPT-4o (Synthesis) & text-embedding-3-small (Embeddings)
* **PDF Engine:** PyMuPDF (fitz)

## 🏗️ Architecture

1.  **Ingestion:** PDF is processed → AI extracts Metadata → Text is "Smart Chunked" → OpenAI generates 1,536-dimension vectors.
2.  **Storage:** Chunks and vectors are stored in PostgreSQL using the `pgvector` extension.
3.  **Retrieval:** User queries are vectorized → Cosine Similarity search finds top 3 relevant chunks.
4.  **Synthesis:** GPT-4o performs a "Scientific Audit" of the chunks and generates a response with highlighted formulas.

## 🚀 Getting Started

### 1. Database Setup
Ensure you have the `pgvector` extension enabled in your PostgreSQL instance:
```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE oneswifty_knowledge (
    id SERIAL PRIMARY KEY,
    title TEXT,
    author TEXT,
    category TEXT,
    content_text TEXT,
    metadata_source TEXT,
    embedding vector(1536)
);
