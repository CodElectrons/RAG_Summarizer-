import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class AppConfig:
    google_api_key: str
    chat_model: str = "models/gemini-2.5-flash"
    embedding_model: str = "models/gemini-embedding-001"
    chunk_size: int = 3000
    chunk_overlap: int = 200
    max_index_chunks: int = 120
    summary_chunks: int = 10
    qa_chunks: int = 3


def get_config() -> AppConfig:
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is missing. Add it to your environment or .env file.")
    chat_model = os.getenv("GOOGLE_CHAT_MODEL", "models/gemini-2.5-flash").strip()
    embedding_model = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/gemini-embedding-001").strip()
    return AppConfig(
        google_api_key=api_key,
        chat_model=chat_model,
        embedding_model=embedding_model,
    )
