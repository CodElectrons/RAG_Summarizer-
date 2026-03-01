from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from src.config import AppConfig


def build_llm(config: AppConfig) -> ChatGoogleGenerativeAI:
    return build_llm_for_model(config, config.chat_model)


def build_llm_for_model(config: AppConfig, model: str) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=config.google_api_key,
        temperature=0.2,
    )


def build_embeddings(config: AppConfig) -> GoogleGenerativeAIEmbeddings:
    return build_embeddings_for_model(config, config.embedding_model)


def build_embeddings_for_model(config: AppConfig, model: str) -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model=model,
        google_api_key=config.google_api_key,
    )
