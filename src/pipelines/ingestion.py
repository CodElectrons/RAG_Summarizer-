from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from src.config import AppConfig


def _select_representative_chunks(chunks: list[Document], max_chunks: int) -> list[Document]:
    if len(chunks) <= max_chunks:
        return chunks
    if max_chunks <= 1:
        return [chunks[0]]

    step = (len(chunks) - 1) / (max_chunks - 1)
    selected = [chunks[round(i * step)] for i in range(max_chunks)]
    return selected


def load_and_chunk_pdf(pdf_path: str | Path, config: AppConfig) -> list[Document]:
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    chunks = [chunk for chunk in chunks if chunk.page_content and chunk.page_content.strip()]
    chunks = _select_representative_chunks(chunks, config.max_index_chunks)
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = idx
    return chunks
