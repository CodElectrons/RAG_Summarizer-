from langchain_core.language_models.chat_models import BaseChatModel
from langchain_community.vectorstores import FAISS

from src.config import AppConfig


def _to_text(result) -> str:
    if result is None:
        return ""
    content = getattr(result, "content", result)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
            else:
                parts.append(str(item))
        return "\n".join(parts).strip()
    return str(content)


def _format_context_docs(docs) -> str:
    sections = []
    for i, doc in enumerate(docs, start=1):
        page = doc.metadata.get("page", "n/a")
        sections.append(f"[Chunk {i} | Page {page + 1 if isinstance(page, int) else page}]\n{doc.page_content}")
    return "\n\n".join(sections)


def generate_general_summary(vector_store: FAISS, llm: BaseChatModel, config: AppConfig) -> str:
    docs = vector_store.max_marginal_relevance_search(
        query=(
            "Provide a complete, balanced, high-level summary of this document, "
            "including key topics, key numbers, dates, action items, and conclusions."
        ),
        k=config.summary_chunks,
        fetch_k=max(config.summary_chunks * 3, 20),
    )
    context = _format_context_docs(docs)
    prompt = f"""
You are a precise summarization assistant.
Generate a general summary for the full document using the extracted context.
Be concise, structured, and factual.
If information is missing, say "Not found in extracted chunks".

Context:
{context}

Output format:
1) Executive summary (5-8 lines)
2) Main points (bullet list)
3) Important entities/numbers/dates
4) Risks or limitations mentioned
""".strip()
    result = llm.invoke(prompt)
    text = _to_text(result)
    return text if text else "Summary was empty. Try indexing again or use a different chat model."


def answer_question(vector_store: FAISS, llm: BaseChatModel, question: str, config: AppConfig) -> str:
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": config.qa_chunks, "fetch_k": max(config.qa_chunks * 3, 12)},
    )
    docs = retriever.invoke(question)
    context = _format_context_docs(docs)
    prompt = f"""
Answer the question using only the provided context.
If the answer is not in context, reply: "I couldn't find this in the document."
Always include page references in square brackets like [Page 3] when possible.

Question:
{question}

Context:
{context}
""".strip()
    result = llm.invoke(prompt)
    text = _to_text(result)
    return text if text else "No answer was generated."
