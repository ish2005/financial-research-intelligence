# Financial Research Intelligence

> 🚀 **Live Demo:** https://financial-rag-1060119334978.asia-south1.run.app

AI-powered financial document research system for answering questions over annual reports using hybrid retrieval, cross-encoder reranking, contextual expansion, and citation-grounded LLM generation.

The system is designed for financial research where answers must be traceable back to specific company, year, page, and document chunks.

---

## 🌐 Live Application

The application is deployed on **Google Cloud Run** and can be accessed here:

**https://financial-rag-1060119334978.asia-south1.run.app**

You can ask financial questions about the built-in annual report corpus or upload additional financial PDF documents for analysis.

---

## Overview

Financial reports are long, information-dense documents where relevant information may be expressed using different terminology across sections and years.

A simple vector search system can miss important evidence or retrieve semantically similar but financially irrelevant passages.

This project uses a multi-stage retrieval and generation pipeline:

```text
Financial PDF
     ↓
PDF Parsing
     ↓
Chunking
     ↓
FAISS + BGE Embeddings
     ↓
BM25 Keyword Retrieval
     ↓
Reciprocal Rank Fusion (RRF)
     ↓
Cross-Encoder Reranking
     ↓
Adjacent Context Expansion
     ↓
LLM Generation
     ↓
Citation Validation
     ↓
Grounded Financial Answer
