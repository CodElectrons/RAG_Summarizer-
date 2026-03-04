# Project Summary: Fast Modular PDF Summarizer and Q&A

## 1. Project Goal
This project is a local web app that accepts any PDF, builds a semantic index, then:
- generates a structured general summary
- answers user questions grounded in the uploaded document

The design target is practical speed with clean modular code that is easy to extend.

## 2. What Problem It Solves
Long PDFs are hard to read quickly. Traditional summarization with a single prompt can miss details or exceed context size.  
This project solves that by:
- splitting large documents into chunks
- embedding chunks in a vector store for semantic retrieval
- summarizing from the most relevant chunks
- reusing the same index for Q&A

## 3. High-Level Architecture
The app is built as a lightweight RAG-style pipeline:
1. Upload PDF (Gradio UI)
2. Extract text and chunk it
3. Convert chunks to embeddings
4. Store vectors in FAISS
5. Retrieve relevant chunks for summary/Q&A
6. Generate response using Gemini chat model

Core modules:
- `app.py`: UI and orchestration
- `src/config.py`: environment/config management
- `src/llm_factory.py`: Gemini model and embedding clients
- `src/pipelines/ingestion.py`: PDF loading + chunking
- `src/pipelines/vector_store.py`: FAISS index creation
- `src/pipelines/summarization.py`: summary and Q&A prompts

## 4. Why Each Stack Was Used

### Python
- Fast to build and iterate in AI workflows.
- Strong ecosystem for LangChain, FAISS, Gradio, and document tooling.

### Gradio (UI)
- Minimal code for upload + button-based workflows.
- Good for local demos and workshop projects.
- Easy to expose summary and Q&A in a single page.

### LangChain
- Provides reusable abstractions for document loaders, splitters, vector stores, and model invocation.
- Keeps pipeline logic clean and modular instead of writing everything from scratch.

### Gemini API (`langchain-google-genai`)
- Used for both:
  - embeddings (`gemini-embedding-001`)
  - generation (`gemini-2.5-flash` and fallbacks)
- Good speed/quality tradeoff for summarization tasks.
- Integrated through factory functions to allow model switching from `.env`.

### FAISS (Vector Store)
- Very fast local semantic retrieval.
- No external database setup required.
- Ideal for single-document and local-first scenarios.

### PyPDF Loader + Text Splitter
- `PyPDFLoader` extracts PDF text in a simple, reliable way.
- `RecursiveCharacterTextSplitter` creates manageable chunks that balance coverage and latency.

### `python-dotenv`
- Keeps API keys and model configuration out of code.
- Makes local run and deployment cleaner.

## 5. Performance Decisions
The implementation includes practical speed optimizations:
- Larger chunk size to reduce total number of embeddings.
- Hard cap (`max_index_chunks`) to prevent very slow indexing on huge files.
- Re-index skip via file fingerprint (path + size + mtime).
- Reuse indexed FAISS store for both summary and Q&A.
- Chat/embedding model fallback logic to avoid manual failures from unsupported model IDs.

## 6. Robustness and Error Handling
- Clear user messages for:
  - missing upload
  - non-PDF input
  - unsupported embedding models
  - unsupported chat models
- Fallback model candidates for both indexing and generation.
- Port fallback if preferred Gradio port is busy.

## 7. Data Flow (Detailed)
1. User uploads PDF and clicks **Index PDF**.
2. App loads + splits document into chunks.
3. App builds embeddings and stores vectors in FAISS.
4. User clicks **Generate Summary**:
   - app retrieves representative chunks with MMR
   - sends context + structured prompt to chat model
   - returns formatted summary text
5. User asks a question:
   - retriever fetches semantically relevant chunks
   - model answers using context only with page hints

## 8. Current Limitations
- Single in-memory runtime session (intended for local single-user use).
- PDF-only input in this version.
- Summary quality depends on text extraction quality and chosen model quota.

## 9. Why This Design Is Modular
Each concern is isolated:
- UI in `app.py`
- config in `config.py`
- model creation in `llm_factory.py`
- processing pipelines in `src/pipelines/*`

This makes extension straightforward:
- add DOCX ingestion without touching summarization logic
- swap FAISS with another vector DB
- swap Gemini model IDs from `.env`

## 10. Future Enhancements
- Persistent vector index caching across app restarts.
- Multi-document upload and cross-document retrieval.
- Better citations with exact chunk/page mapping in output.
- Async/background indexing for larger documents.
- Optional API layer (FastAPI) for programmatic usage.
