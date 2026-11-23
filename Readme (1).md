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
```

---

## Key Features

- Hybrid semantic + lexical retrieval
- FAISS vector search using BGE embeddings
- BM25 keyword retrieval
- Reciprocal Rank Fusion (RRF)
- Cross-encoder reranking
- Adjacent chunk context expansion
- Structured LLM responses using Pydantic
- Citation-grounded answers
- Structural citation validation
- Semantic citation validation
- Multi-company and multi-year document search
- Upload and index additional financial PDFs
- Streamlit interface
- Dockerized deployment
- Cloud Run deployment
- Evaluation dataset and retrieval evaluation scripts

---

## Supported Development Corpus

The initial corpus contains annual reports from:

- Apple — 2023, 2024, 2025
- Microsoft — 2023, 2024, 2025
- NVIDIA — 2023, 2024, 2025

The application also supports uploading additional PDF documents for analysis.

---

## Retrieval Architecture

### 1. Document Parsing

PDF documents are parsed page-by-page while preserving metadata such as:

- Company
- Fiscal year
- Page number
- Document identifier

### 2. Chunking

Documents are split into overlapping chunks using a Hugging Face tokenizer.

Current configuration:

```text
Chunk size: 450 tokens
Overlap:     60 tokens
```

The development corpus contains approximately:

```text
1,047 pages
2,276 chunks
```

### 3. Dense Retrieval

Semantic retrieval uses:

```text
BAAI/bge-small-en-v1.5
```

Embeddings are normalized and stored in a FAISS `IndexFlatIP` index.

### 4. BM25 Retrieval

BM25 provides lexical retrieval for queries containing:

- Financial terminology
- Exact phrases
- Company-specific terminology
- Accounting metrics
- Numeric or textual keywords

### 5. Hybrid Retrieval with RRF

Dense and BM25 rankings are combined using Reciprocal Rank Fusion (RRF).

This improves robustness compared with relying on either semantic or keyword retrieval alone.

### 6. Cross-Encoder Reranking

The retrieved candidates are reranked using:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

The reranker evaluates the relevance of each query-document pair rather than relying only on embedding similarity.

### 7. Adjacent Context Expansion

Relevant chunks are expanded with neighboring chunks from the same company and fiscal year.

This helps preserve context when important financial information spans multiple chunks.

---

## Generation

The generation layer uses Groq with:

```text
openai/gpt-oss-120b
```

The model receives only retrieved evidence rather than the entire document corpus.

The generation prompt explicitly requires:

- No unsupported external knowledge
- No invented facts
- No invented citations
- Preservation of financial units
- Explicit acknowledgement when evidence is insufficient
- Citation metadata copied from retrieved evidence

Responses are structured using Pydantic.

Example response structure:

```text
Answer
Citations
Confidence
```

---

## Citation Validation

The system performs two levels of citation validation.

### Structural Validation

Checks that:

- The cited chunk exists
- Company matches
- Year matches
- Page matches

### Semantic Validation

A separate LLM-based judge evaluates whether the cited evidence actually supports the generated answer.

This prevents a response from being considered reliable merely because it contains a syntactically valid citation.

---

## Example

### Question

```text
What was Apple's total net sales in 2023?
```

### Answer

```text
Apple's total net sales in 2023 were $383,285 million.
```

### Sources

```text
Apple 2024 — Page 38
Apple 2023 — Page 31
```

---

## Project Structure

```text
financial-research-intelligence/
│
├── app.py
├── Dockerfile
├── requirements.txt
├── Readme.md
├── .gitignore
├── .dockerignore
│
├── data/
│   ├── evaluation/
│   │   ├── evaluation_questions.json
│   │   └── end_to_end_results.json
│   │
│   └── processed/
│       ├── chunks.jsonl
│       ├── pages.jsonl
│       └── faiss/
│           ├── index.faiss
│           └── metadata.json
│
└── src/
    │
    ├── pipeline.py
    ├── test_pipeline.py
    │
    ├── ingestion/
    │   ├── parser.py
    │   ├── chunker.py
    │   └── pdf_indexer.py
    │
    ├── retrieval/
    │   ├── retriever.py
    │   ├── vector_store.py
    │   ├── bm25.py
    │   ├── hybrid.py
    │   ├── reranker.py
    │   ├── reranker_pipeline.py
    │   └── adjacent.py
    │
    ├── generation/
    │   ├── answer_generator.py
    │   ├── citation_validator.py
    │   └── test_generation.py
    │
    └── evaluation/
        ├── retrieval_eval.py
        ├── end_to_end_eval.py
        ├── candidate_recall.py
        ├── diagnose_misses.py
        ├── verify_gold.py
        └── verify_all_gold.py
```

---

## Tech Stack

### Machine Learning / NLP

- Python
- Sentence Transformers
- BGE embeddings
- Cross-Encoder
- BM25
- FAISS
- Pydantic

### LLM

- Groq
- `openai/gpt-oss-120b`

### Document Processing

- PyMuPDF
- pypdf
- Hugging Face tokenizers

### Application

- Streamlit

### Deployment

- Docker
- Google Cloud Build
- Google Artifact Registry
- Google Cloud Run
- Google Cloud Secret Manager

---

## Running Locally

### 1. Clone the repository

```bash
git clone https://github.com/ish2005/financial-research-intelligence.git
cd financial-research-intelligence
```

### 2. Create a virtual environment

#### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

#### Linux/macOS

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the Groq API key

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
```

Never commit `.env` or API keys to Git.

### 5. Start the application

```bash
streamlit run app.py
```

---

## Docker

Build the image:

```bash
docker build -t financial-rag .
```

Run:

```bash
docker run -p 8080:8080 --env-file .env financial-rag
```

The application will be available at:

```text
http://localhost:8080
```

---

## Evaluation

The repository includes evaluation scripts and a manually constructed question set for testing retrieval and end-to-end answer quality.

Evaluation focuses on:

- Candidate recall
- Retrieval ranking
- Gold evidence retrieval
- End-to-end answer generation
- Citation correctness
- Failure analysis

The evaluation tooling was used during development to diagnose retrieval misses and improve the retrieval pipeline before deployment.

---

## Deployment

The application is containerized and deployed using Google Cloud Run.

High-level deployment architecture:

```text
GitHub
   ↓
Docker
   ↓
Google Cloud Build
   ↓
Artifact Registry
   ↓
Cloud Run
   ↓
Streamlit Application
   ↓
Groq API
```

The Groq API key is provided through Google Cloud Secret Manager rather than being stored inside the Docker image or source code.

---

## Design Decisions

### Why Hybrid Retrieval?

Financial questions often contain both semantic intent and exact terminology.

For example:

```text
"What was NVIDIA's gross margin in fiscal 2025?"
```

Semantic retrieval helps identify relevant financial discussion, while BM25 can strongly match terms such as:

```text
gross margin
fiscal 2025
```

Combining both improves retrieval robustness.

### Why Reranking?

Initial retrieval optimizes for fast candidate discovery.

The cross-encoder provides a more expensive but more accurate relevance assessment on a smaller candidate set.

This creates a practical two-stage retrieval architecture:

```text
Fast retrieval
     ↓
Candidate set
     ↓
Expensive reranking
     ↓
High-quality evidence
```

### Why Adjacent Context?

Important financial statements and explanations can span multiple chunks.

Retrieving neighboring chunks allows the generation model to see the surrounding context without dramatically increasing the initial retrieval size.

### Why Citation Validation?

Financial research requires traceability.

A generated answer is not considered reliable simply because the LLM produced a citation. The citation must correspond to retrieved evidence and actually support the claim.

---

## Security

Secrets are intentionally excluded from the repository.

The project uses:

```text
.env
```

for local development and Google Cloud Secret Manager for deployment.

Do not commit:

```text
.env
API keys
service account credentials
private documents
```

---

## Future Improvements

Potential future extensions include:

- LangGraph-based orchestration
- Query classification and routing
- Query rewriting
- Financial calculation tools
- Multi-document comparison workflows
- Conversation memory
- Streaming responses
- More extensive benchmark datasets
- Automated CI evaluation
- Authentication and user-specific document collections
- Production monitoring and tracing

---

## Author

**Ish Sachdeva**

BSc Statistics (Hons.) — Banaras Hindu University  
BS Data Science — IIT Madras

Interested in AI engineering, machine learning systems, retrieval-augmented generation, and applied data science.

---

## License

This project is intended primarily as a portfolio and educational project.
