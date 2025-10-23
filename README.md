<p align="center">
  <img src="assets/banner.png" width="90%">
</p>
<h1 align="center">🧠 DocuChat Pro</h1>
<p align="center">
  <b>RAG-Powered LLM Chatbot for Intelligent Document Q&A</b><br>
  End-to-end pipeline for document ingestion, retrieval, generation, and evaluation.
</p>

<p align="center">
  <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white" alt="Python"></a>
  <a href="https://fastapi.tiangolo.com"><img src="https://img.shields.io/badge/FastAPI-0.110+-teal?logo=fastapi&logoColor=white" alt="FastAPI"></a>
  <a href="https://openai.com"><img src="https://img.shields.io/badge/OpenAI-API-black?logo=openai" alt="OpenAI"></a>
  <a href="#"><img src="https://img.shields.io/badge/RAGAS-Evaluation-orange" alt="RAGAS"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green" alt="License"></a>
</p>

---

## 🚀 Overview

**DocuChat Pro** is an advanced **Retrieval-Augmented Generation (RAG)** chatbot that allows you to query your own PDFs, text files, or datasets using a Large Language Model (LLM).

It demonstrates the **complete RAG lifecycle** — from data ingestion and vectorization to hybrid retrieval, reranking, LLM-powered generation, and evaluation with **RAGAS metrics**.

---

```markdown
## 🧩 Architecture

mermaid
graph TD
    A[Documents / PDFs] -->|Ingest| B[Chunker & Embedder]
    B --> C[Vector Store (Artifacts)]
    C --> D[Retriever (BM25 / Dense / Hybrid)]
    D --> E[Reranker (optional)]
    E --> F[LLM Generator (OpenAI API)]
    F --> G[Evaluation (RAGAS)]
```

---

## ✨ Features

- 🧠 **RAG End-to-End Pipeline**
- 🔍 **Hybrid Search** (BM25 + Dense + Hybrid + Rerank)
- ⚙️ **FastAPI REST Backend**
- 🧱 **Local Artifacts** for reproducibility
- 🧩 **Guardrails** (Prompt Injection, PII, Dangerous Ops)
- 📊 **RAGAS Evaluation** (Faithfulness, Relevance, Precision, Recall)
- 💬 **Clean modular structure** for easy extension

---

## 🗂️ Project Structure

| Folder | Description |
|---------|--------------|
| `backend/rag/` | Core RAG logic — ingest, retrieve, rerank, and answer |
| `backend/eval/` | Evaluation scripts (RAGAS, Ablation) |
| `backend/guard/` | Guardrails: prompt injection, PII, moderation |
| `data/` | Input data — your source PDFs or text files |
| `artifacts/` | Output — embeddings, chunk indexes, metadata |
| `runtime/` | Temporary runtime storage |

---

## ⚙️ Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/arijanyan94/DocuChat-pro.git
cd docuchat-pro
```

### 2️⃣ Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Configure environment variables
Create a file named `.env` in the project root:
```bash
OPENAI_API_KEY=your_openai_api_key_here
GUARD_USE_OPENAI_MOD=false
OPENAI_MODERATION_MODEL=omni-moderation-latest
```
> ⚠️ Do **not** push your `.env` file to GitHub — `.env.example` is provided for reference only.

---

## 🧠 Usage

### 🧩 1. Ingest Documents
Converts documents into text chunks and embeddings.
```bash
python backend/rag/ingest.py --input data --out artifacts
```

### 🔍 2. Retrieve Information
Query your indexed documents using BM25, Dense, or Hybrid retrieval.
```bash
python backend/rag/retrieve.py --query "refund policy" --mode hybrid
```

### 💬 3. Generate Answers
RAG-powered text generation from the retrieved context.
```bash
python backend/rag/answer.py --query "Explain instruction fine-tuning."
```

### 🧪 4. Evaluate (RAGAS)
Assess pipeline quality using faithfulness, relevance, precision, and recall.
```bash
python backend/eval/ragas_eval.py --samples backend/eval/samples.jsonl --art artifacts
```

### ⚖️ 5. Ablation Study
Compare multiple retrieval configurations:
```bash
python backend/eval/ablation_eval.py --samples backend/eval/samples.jsonl --art artifacts
```

---

## 🧱 Example Evaluation Results

| Method | Faithfulness | Relevance | Precision | Recall |
|---------|---------------|-----------|------------|---------|
| BM25 | 1.0000 | 0.9611 | 0.5708 | 1.0000 |
| Dense | 1.0000 | 0.9404 | 1.0000 | 1.0000 |
| Hybrid | 0.8125 | 0.9518 | 0.8778 | 1.0000 |
| Hybrid + Rerank | 0.8000 | 0.9404 | 0.9383 | 1.0000 |

---

## 🛡️ Guardrails

| Check | Description |
|--------|--------------|
| 🧱 Prompt Injection | Detects jailbreaks like “Ignore all previous instructions” |
| 🔐 PII | Prevents exposure of sensitive information (SSN, passport, credit card) |
| ⚠️ Dangerous Ops | Blocks code execution commands (`rm -rf`, `wget`, etc.) |
| 🧩 Moderation | Optional OpenAI Moderation API check for flagged content |

---

## 🧠 Evaluation Metrics (RAGAS)

| Metric | Description |
|---------|-------------|
| **Faithfulness** | Checks factual grounding of LLM response |
| **Answer Relevance** | Measures alignment of the answer to the query |
| **Context Precision** | Fraction of retrieved chunks that are relevant |
| **Context Recall** | Fraction of relevant context successfully retrieved |

---

## 🧭 Future Improvements

- [ ] 🌐 LangChain integration for pipeline orchestration  
- [ ] 🗄️ FAISS or Chroma persistent vector store  
- [ ] 💬 Frontend (Streamlit / Next.js)  
- [ ] 🧰 Fine-tuning loop from user feedback  
- [ ] 🐳 Docker deployment  

---

## 🧩 Tech Stack

| Category | Technology |
|-----------|-------------|
| Language | Python 3.11 |
| Framework | FastAPI |
| Embeddings | SentenceTransformers |
| Retrieval | BM25 + Dense + Hybrid |
| LLM | OpenAI GPT models |
| Evaluation | RAGAS |
| Moderation | OpenAI Moderation API |
| Logging | Rich + Pydantic models |

---

## 📦 Example Output

> **Query:** “What is instruction fine-tuning?”  
>
> **Answer:**  
> Instruction fine-tuning is a process of aligning a pre-trained language model to follow human-written instructions by further training it on prompt–response pairs. It enhances task generalization and safety across a wide range of NLP tasks.

---

## 📄 License

MIT License © 2025 [Arsen Arijanyan]  
Feel free to use, modify, and share for educational or portfolio purposes.

---

## 🌟 Acknowledgements

This project was built as a hands-on learning experience in **LLM engineering** — implementing a complete **RAG + Evaluation** system from scratch.

**Keywords:** Retrieval-Augmented Generation · LLM · RAGAS · Hybrid Search · OpenAI API · Evaluation

---

<h3 align="center">⭐ If you like this project, consider giving it a star on GitHub!</h3>
<p align="center">
  <a href="https://github.com/arijanyan94/DocuChat-pro/stargazers">
    <img src="https://img.shields.io/github/stars/arijanyan94/DocuChat-pro?style=social" alt="GitHub stars">
  </a>
</p>
