# Fast Modular PDF Summarizer (Gemini + LangChain + FAISS + Gradio)

Simple project for uploading **any PDF**, indexing it once, then:
- generating a fast general summary
- asking questions grounded in document chunks

Detailed project explanation:
- See `PROJECT_SUMMARY.md` for architecture, flow, and stack rationale.
- See `TECHNICAL_BREAKDOWN.md` for module-by-module, class-by-class, and function-by-function details.

## Stack
- LLM Pipelines: modular ingestion, indexing, summarization
- LangChain
- Vector Store: FAISS
- UI: Gradio
- LLM/Embeddings: Google Gemini (`langchain-google-genai`)

## Project Structure
```text
app.py
src/
  config.py
  llm_factory.py
  pipelines/
    ingestion.py
    vector_store.py
    summarization.py
```

## Setup
1. Create and activate a virtual environment.
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Add your Gemini key:
```bash
copy .env.example .env
```
Then set:
```env
GOOGLE_API_KEY=your_key_here
```

## Run
```bash
python app.py
```
Open: `http://127.0.0.1:7860`

## Performance Notes
- Fast model default: `gemini-2.5-flash`
- MMR retrieval of representative chunks keeps prompts small and latency low
- Document is indexed once per upload; summary and Q&A re-use the same FAISS index

## Optional Tuning
Edit values in `src/config.py`:
- `chunk_size`, `chunk_overlap`
- `summary_chunks` (lower = faster, higher = broader coverage)
- `qa_chunks`
