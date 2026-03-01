from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from time import perf_counter

import gradio as gr

from src.config import AppConfig, get_config
from src.llm_factory import build_embeddings, build_embeddings_for_model, build_llm, build_llm_for_model
from src.pipelines.ingestion import load_and_chunk_pdf
from src.pipelines.summarization import answer_question, generate_general_summary
from src.pipelines.vector_store import build_faiss_index


@dataclass
class SessionArtifacts:
    config: AppConfig
    llm: object
    embeddings: object
    vector_store: object | None = None
    file_name: str | None = None
    chunk_count: int = 0
    indexed_fingerprint: str | None = None


_RUNTIME_SESSION: SessionArtifacts | None = None


def _bootstrap() -> SessionArtifacts:
    config = get_config()
    return SessionArtifacts(
        config=config,
        llm=build_llm(config),
        embeddings=build_embeddings(config),
    )


def _get_session() -> SessionArtifacts:
    global _RUNTIME_SESSION
    if _RUNTIME_SESSION is None:
        _RUNTIME_SESSION = _bootstrap()
    return _RUNTIME_SESSION


def ingest_file(file_obj):
    session = _get_session()
    if file_obj is None:
        return "Upload a PDF file first."

    file_path = Path(file_obj.name)
    if file_path.suffix.lower() != ".pdf":
        return "Only PDF files are supported in this simple version."

    fingerprint = f"{file_path.resolve()}::{file_path.stat().st_size}::{int(file_path.stat().st_mtime)}"
    if session.vector_store is not None and session.indexed_fingerprint == fingerprint:
        return f"`{file_path.name}` is already indexed with {session.chunk_count} chunks."

    started = perf_counter()
    chunks = load_and_chunk_pdf(file_path, session.config)
    embedding_candidates = [
        session.config.embedding_model,
        "models/gemini-embedding-001",
        "gemini-embedding-001",
    ]
    # Preserve order while removing duplicates.
    embedding_candidates = list(dict.fromkeys(embedding_candidates))

    vector_store = None
    last_error = None
    for emb_model in embedding_candidates:
        try:
            session.embeddings = build_embeddings_for_model(session.config, emb_model)
            vector_store = build_faiss_index(chunks, session.embeddings)
            break
        except Exception as exc:
            last_error = exc

    if vector_store is None:
        return (
            "Indexing failed: no compatible embedding model was found for this API key. "
            "Set GOOGLE_EMBEDDING_MODEL in .env to a model available in your Gemini account. "
            f"Last error: {last_error}"
        )

    elapsed = perf_counter() - started

    session.vector_store = vector_store
    session.file_name = file_path.name
    session.chunk_count = len(chunks)
    session.indexed_fingerprint = fingerprint
    return f"Indexed `{session.file_name}` with {session.chunk_count} chunks in {elapsed:.1f}s."


def summarize_document():
    session = _get_session()
    if session.vector_store is None:
        return "Please upload and index a PDF first."
    chat_candidates = [
        session.config.chat_model,
        "models/gemini-2.5-flash",
        "gemini-2.5-flash",
        "models/gemini-flash-latest",
        "gemini-flash-latest",
        "models/gemini-2.5-flash-lite",
        "gemini-2.5-flash-lite",
        "models/gemini-2.0-flash",
        "gemini-2.0-flash",
    ]
    chat_candidates = list(dict.fromkeys(chat_candidates))
    last_error = None
    for chat_model in chat_candidates:
        try:
            session.llm = build_llm_for_model(session.config, chat_model)
            return generate_general_summary(session.vector_store, session.llm, session.config)
        except Exception as exc:
            last_error = exc
    return (
        "Summary failed: no compatible chat model was found for this API key. "
        "Set GOOGLE_CHAT_MODEL in .env to a model available in your Gemini account. "
        f"Last error: {last_error}"
    )


def ask_document(question: str):
    session = _get_session()
    if session.vector_store is None:
        return "Please upload and index a PDF first."
    if not question or not question.strip():
        return "Enter a question."
    chat_candidates = [
        session.config.chat_model,
        "models/gemini-2.5-flash",
        "gemini-2.5-flash",
        "models/gemini-flash-latest",
        "gemini-flash-latest",
        "models/gemini-2.5-flash-lite",
        "gemini-2.5-flash-lite",
        "models/gemini-2.0-flash",
        "gemini-2.0-flash",
    ]
    chat_candidates = list(dict.fromkeys(chat_candidates))
    last_error = None
    for chat_model in chat_candidates:
        try:
            session.llm = build_llm_for_model(session.config, chat_model)
            return answer_question(session.vector_store, session.llm, question, session.config)
        except Exception as exc:
            last_error = exc
    return (
        "Q&A failed: no compatible chat model was found for this API key. "
        "Set GOOGLE_CHAT_MODEL in .env to a model available in your Gemini account. "
        f"Last error: {last_error}"
    )


def build_ui():
    with gr.Blocks(title="Fast PDF Summarizer (Gemini + LangChain)") as demo:
        gr.Markdown("## Fast PDF Summarizer and Q&A")
        gr.Markdown("Upload any PDF, index once, then get a fast summary or ask questions.")

        with gr.Row():
            file_input = gr.File(label="PDF File", file_types=[".pdf"], type="filepath")
            ingest_btn = gr.Button("Index PDF", variant="primary")

        status_box = gr.Textbox(label="Status", interactive=False)

        with gr.Tab("General Summary"):
            summary_btn = gr.Button("Generate Summary")
            summary_out = gr.Textbox(label="Summary", lines=18)

        with gr.Tab("Q&A"):
            question_in = gr.Textbox(label="Question", placeholder="What is this document about?")
            ask_btn = gr.Button("Ask")
            answer_out = gr.Textbox(label="Answer", lines=10)

        ingest_btn.click(fn=ingest_file, inputs=[file_input], outputs=[status_box])
        summary_btn.click(fn=summarize_document, inputs=[], outputs=[summary_out])
        ask_btn.click(fn=ask_document, inputs=[question_in], outputs=[answer_out])

    return demo


if __name__ == "__main__":
    app = build_ui()
    preferred_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    try:
        app.launch(server_name="0.0.0.0", server_port=preferred_port, inbrowser=True)
    except OSError:
        app.launch(server_name="0.0.0.0", server_port=None, inbrowser=True)
