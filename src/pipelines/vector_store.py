from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS


def build_faiss_index(chunks: list[Document], embeddings: Embeddings) -> FAISS:
    if not chunks:
        raise ValueError("No chunks were generated from the document.")
    return FAISS.from_documents(chunks, embeddings)
