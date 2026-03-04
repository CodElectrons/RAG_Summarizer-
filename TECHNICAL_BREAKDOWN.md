# Technical Breakdown: Fast PDF Summarizer and Q&A

## 1. Project Purpose
This application lets a user upload a PDF, index it semantically, and then:
- generate a structured summary
- ask grounded questions about document content

It is designed for local usage, fast response, and modular maintainability.

## 2. End-to-End Runtime Flow
1. App starts in `app.py`.
2. Configuration is loaded from `.env` via `src/config.py`.
3. Gemini chat + embedding clients are initialized via `src/llm_factory.py`.
4. User uploads PDF and clicks **Index PDF**.
5. `src/pipelines/ingestion.py` extracts and chunks PDF text.
6. `src/pipelines/vector_store.py` creates a FAISS index from chunk embeddings.
7. User clicks **Generate Summary** or asks a question.
8. `src/pipelines/summarization.py` retrieves context and calls chat LLM.
9. Result text is returned to Gradio UI.

## 3. File-by-File Documentation

### `app.py`
Main orchestration and UI layer.

Responsibilities:
- Holds runtime session state.
- Handles upload/index workflow.
- Handles summary and Q&A actions.
- Applies model fallback logic.
- Builds and launches Gradio interface.

#### Class: `SessionArtifacts`
Runtime container for the active app session:
- `config: AppConfig` current application configuration.
- `llm: object` active chat model client.
- `embeddings: object` active embedding model client.
- `vector_store: object | None` FAISS index after indexing.
- `file_name: str | None` indexed document name.
- `chunk_count: int` number of indexed chunks.
- `indexed_fingerprint: str | None` file signature to skip re-indexing unchanged files.

#### Global: `_RUNTIME_SESSION`
Single in-memory session object for local single-user mode.

#### Function: `_bootstrap() -> SessionArtifacts`
Creates initial runtime session:
- loads config
- initializes default LLM and embedding clients

#### Function: `_get_session() -> SessionArtifacts`
Lazy-initializes and returns global runtime session.

#### Function: `ingest_file(file_obj)`
Upload/index handler.

Logic:
- validates file existence and `.pdf` extension
- builds fingerprint (`path + size + mtime`) to avoid duplicate indexing
- chunks document via `load_and_chunk_pdf`
- tries embedding models in fallback order
- builds FAISS store with first working embedding model
- stores vector store and metadata in session
- returns status text with chunk count and elapsed seconds

Why this function matters:
- controls indexing speed and reliability
- shields UI from raw API exceptions

#### Function: `summarize_document()`
Summary handler.

Logic:
- ensures document was indexed
- tries chat models in fallback order
- calls `generate_general_summary`
- returns readable error if no model works

#### Function: `ask_document(question: str)`
Q&A handler.

Logic:
- validates indexed store and question input
- tries chat model fallback order
- calls `answer_question`
- returns readable failure message on model errors

#### Function: `build_ui()`
Builds Gradio app with:
- file upload
- index button
- status box
- summary tab
- Q&A tab

Wires buttons to handlers:
- `ingest_btn -> ingest_file`
- `summary_btn -> summarize_document`
- `ask_btn -> ask_document`

#### Main block
Launches Gradio on preferred port (`GRADIO_SERVER_PORT`, default `7860`) and falls back to random available port if busy.

---

### `src/config.py`
Configuration module.

Responsibilities:
- load environment variables (`dotenv`)
- define app-level defaults
- build immutable runtime config object

#### Class: `AppConfig` (`@dataclass(frozen=True)`)
Configuration structure:
- `google_api_key`
- `chat_model` default `models/gemini-2.5-flash`
- `embedding_model` default `models/gemini-embedding-001`
- `chunk_size` default `3000`
- `chunk_overlap` default `200`
- `max_index_chunks` default `120`
- `summary_chunks` default `10`
- `qa_chunks` default `3`

#### Function: `get_config() -> AppConfig`
- reads key + model ids from environment
- validates `GOOGLE_API_KEY`
- returns `AppConfig` instance

---

### `src/llm_factory.py`
Model factory module.

Responsibilities:
- create Gemini chat model clients
- create Gemini embedding model clients

#### Function: `build_llm(config: AppConfig)`
Creates chat client using configured model.

#### Function: `build_llm_for_model(config: AppConfig, model: str)`
Creates chat client for a specific model id.  
Used for fallback attempts.

#### Function: `build_embeddings(config: AppConfig)`
Creates embedding client using configured embedding model.

#### Function: `build_embeddings_for_model(config: AppConfig, model: str)`
Creates embedding client for a specific model id.  
Used during embedding fallback at indexing time.

---

### `src/pipelines/ingestion.py`
Document ingestion and chunking.

Responsibilities:
- load PDF pages
- split into chunks
- limit chunk count for speed
- assign chunk ids

#### Function: `_select_representative_chunks(chunks, max_chunks)`
If chunks exceed limit, samples evenly across the document.

Purpose:
- keeps full-document coverage
- avoids indexing too many chunks on large PDFs

#### Function: `load_and_chunk_pdf(pdf_path, config)`
Pipeline:
- loads pages with `PyPDFLoader`
- splits text with `RecursiveCharacterTextSplitter`
- filters empty chunks
- caps chunk list with representative sampling
- adds `chunk_id` metadata

Returns list of LangChain `Document` chunks.

---

### `src/pipelines/vector_store.py`
Vector index builder.

Responsibilities:
- convert chunk set into FAISS vector index

#### Function: `build_faiss_index(chunks, embeddings) -> FAISS`
- validates chunk list is not empty
- builds FAISS index from chunk texts and metadata

---

### `src/pipelines/summarization.py`
Summary and Q&A generation logic.

Responsibilities:
- retrieve relevant context from vector store
- build prompts
- normalize LLM output into plain text

#### Function: `_to_text(result) -> str`
Normalizes various LangChain/LLM response shapes:
- string
- object with `.content`
- list-based content blocks

#### Function: `_format_context_docs(docs) -> str`
Creates a readable context string with:
- chunk number
- page number from metadata
- chunk text

#### Function: `generate_general_summary(vector_store, llm, config) -> str`
Workflow:
- runs MMR retrieval using summary query
- retrieves `k=config.summary_chunks`
- builds structured summarization prompt
- invokes chat model
- returns normalized text result

Output sections requested in prompt:
1. Executive summary
2. Main points
3. Important entities/numbers/dates
4. Risks or limitations

#### Function: `answer_question(vector_store, llm, question, config) -> str`
Workflow:
- creates retriever (`search_type="mmr"`)
- gets top question-relevant chunks
- prompts model to answer only from context
- includes page-reference instruction
- returns normalized text

---

### `src/__init__.py` and `src/pipelines/__init__.py`
Package markers for module imports.  
No runtime logic.

---

### `requirements.txt`
Runtime dependencies:
- `gradio`: web UI
- `langchain`: core orchestration
- `langchain-community`: loaders/vector stores
- `langchain-google-genai`: Gemini integration
- `faiss-cpu`: local vector index
- `pypdf`: PDF parsing dependency path
- `python-dotenv`: environment loading

---

### `.env.example`
Environment template:
- `GOOGLE_API_KEY`
- `GOOGLE_CHAT_MODEL`
- `GOOGLE_EMBEDDING_MODEL`

---

### `.gitignore`
Excludes secrets and local artifacts:
- `.env`
- caches/compiled files
- Gradio local files
- temporary log files
- local tooling folder

---

### `README.md`
Quick-start documentation:
- setup
- run commands
- stack overview
- structure overview

---

### `PROJECT_SUMMARY.md`
Higher-level architecture and stack rationale document.

---

## 4. LLM Design Details

### LLM used
- Chat generation via Gemini (`ChatGoogleGenerativeAI`)
- Embeddings via Gemini embeddings model (`GoogleGenerativeAIEmbeddings`)

### Why two model types
- Embedding model transforms chunks to vectors for retrieval.
- Chat model generates natural-language summaries/answers from retrieved context.

### Fallback strategy
Implemented in `app.py`:
- Embedding fallback candidates are tried in sequence until one works.
- Chat fallback candidates are tried in sequence for summary/Q&A.

This handles:
- model id compatibility issues
- per-model quota/rate constraints
- API account differences

## 5. Main Components Summary
- UI component: Gradio app in `app.py`
- Config component: `AppConfig` in `config.py`
- Model factory component: `llm_factory.py`
- Ingestion component: `ingestion.py`
- Indexing component: `vector_store.py`
- Generation component: `summarization.py`
- Documentation component: `README.md`, `PROJECT_SUMMARY.md`, this file

## 6. Extension Points
Where to modify for future features:
- Add DOCX support: `ingestion.py`
- Add persistent index storage: `vector_store.py` + app session logic
- Add API backend: split handlers from `app.py` into service layer
- Add stronger citations: extend `_format_context_docs` + prompt format
- Add authentication/multi-user sessions: replace global runtime session
