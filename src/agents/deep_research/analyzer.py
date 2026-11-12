import numpy as np
from typing import Any, cast
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# Bi-encoder for chunk embedding (different from CrossEncoder above)
from functools import lru_cache


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer("all-MiniLM-L6-v2")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=450,  # tokens approx — tune this
    chunk_overlap=50,  # overlap so you don't cut mid-sentence context
    separators=["\n\n", "\n", ". ", " "]  # prefer natural breaks
)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return cast(np.ndarray, a @ b.T)


def get_relevant_chunks(page_content: str, query: str, top_k: int = 5, score_threshold: float = 0.4) -> str:
    # 1. Split
    chunks = splitter.split_text(page_content)

    if not chunks:
        return ""

    # 2. Embed everything
    chunk_embeddings = get_embedder().encode(chunks, batch_size=32)  # shape: (N, 384)
    query_embedding = get_embedder().encode([query])  # shape: (1, 384)

    # 3. Score
    scores = cosine_similarity(query_embedding, chunk_embeddings)[0]  # shape: (N,)

    # 4. Take top-k, preserve reading order (important for coherence)
    top_indices = [
        i for i in np.argsort(scores)[-top_k:]
        if scores[i] > score_threshold      # ← drop chunks that scored poorly
    ]
    top_indices = sorted(top_indices)  # ← restore document order

    relevant_text = "\n\n".join(chunks[i] for i in top_indices)
    return relevant_text


def extract_from_page(page_content: str, url: str, query: str, llm: Any) -> str | None:
    relevant_text = get_relevant_chunks(page_content, query, top_k=5)

    if not relevant_text.strip():
        return None

    # Now LLM only sees ~2000 tokens, not 50k
    prompt = f"""
Extract information relevant to: "{query}"

Source: {url}

Content:
{relevant_text}

If this content contains nothing relevant, reply with exactly: IRRELEVANT
Otherwise, extract and summarize the key relevant facts concisely.
"""
    response = llm.invoke(prompt)
    text = response.content.strip()

    return None if text == "IRRELEVANT" else text
