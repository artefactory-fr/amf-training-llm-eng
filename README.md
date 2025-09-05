# AMF Training LLM Engineering

This repository is designed as a hands-on course to explore best practices for building Retrieval-Augmented Generation (RAG) and GenAI projects.
It provides a series of practical notebooks (TPs) covering the full workflow — from document parsing and chunking to query augmentation, retrieval, reranking, guardrails, and LLMOps.
Each notebook combines theory with examples to help you learn step by step how to design robust and production-ready RAG pipelines.

---

## Notebooks Overview

### Hands-on 1 — Introduction to RAG
- **1_rag_intro.ipynb**: Setup of a RAG pipeline with LangChain & LangGraph (indexing, generation, LCE, LangGraph).

### Hands-on 2 — Parsing & Chunking
- **2_1_parsing_strategies.ipynb**: Document parsing (PyPDF, Unstructured, Docling, multimodal, LlamaParse).
- **2_2_chunking_strategies.ipynb**: Chunking strategies (hierarchical, recursive, metadata integration).

### Hands-on 3 — Query Augmentation & Retrieval
- **3_1_query_augmentation_strategies.ipynb**: Query reformulation, HyDE, decomposition, step-back prompting.
- **3_2_routing_strategies.ipynb**: Classification, routing, retry mechanisms, response parsing.
- **3_3_retrieval_strategies.ipynb**: Multi-query retrieval, parameter tuning, document examples.

### Hands-on 4 — Reranking
- **4_reranking_strategies.ipynb**: Reranking with embeddings and custom scoring.

### Hands-on 5 — Guardrails & Evaluation
- **5_guardrails_strategies.ipynb**: Guardrails, metrics (faithfulness, relevance), evaluation with Ragas, redteaming.

### Hands-on 6 — LLMOps
- **6_llmops_pipeline.ipynb**: Tracing setup and building an evaluation pipeline.

---

## Installation

### Prerequisites
- Python 3.12 (or compatible version as specified in `pyproject.toml`)
- Have access to an Azure deployment for both LLM and embeddings models.
- For running the noteooks **2_1_parsing_strategies.ipynb**, have a LlamaParse API key > go to https://cloud.llamaindex.ai/ sign in and create one.
- Download the data used in the notebooks from [data](https://drive.google.com/drive/folders/1IAUeIMMfupJRQdjOSg0w-CFwoS5uo6Od) and put the content into the `data/` folder at the root of the project.

### Installation Steps
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd amf-training-llm-eng
   ```

2. Install dependencies (rely on uv):
   ```bash
   make install
   ```

3. Set up environment variables:
   - Create a `.env` file in the root directory.
   - Fill the following variables with your Azure OpenAI credentials (as described in `.env.example`)

---

## Tests

To ensure all notebooks run correctly, this project includes a `pytest` test that executes every notebook under the `notebooks/` folder.

Run the tests with:

```bash
make run_tests
```
