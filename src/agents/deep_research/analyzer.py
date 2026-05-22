import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# Bi-encoder for chunk embedding (different from CrossEncoder above)
embedder = SentenceTransformer("all-MiniLM-L6-v2")  # 80MB, very fast

splitter = RecursiveCharacterTextSplitter(
    chunk_size=450,  # tokens approx — tune this
    chunk_overlap=50,  # overlap so you don't cut mid-sentence context
    separators=["\n\n", "\n", ". ", " "]  # prefer natural breaks
)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return (a @ b.T)


def get_relevant_chunks(page_content: str, query: str, top_k=5) -> str:
    # 1. Split
    chunks = splitter.split_text(page_content)

    if not chunks:
        return ""

    # 2. Embed everything
    chunk_embeddings = embedder.encode(chunks, batch_size=32)  # shape: (N, 384)
    query_embedding = embedder.encode([query])  # shape: (1, 384)

    # 3. Score
    scores = cosine_similarity(query_embedding, chunk_embeddings)[0]  # shape: (N,)

    # 4. Take top-k, preserve reading order (important for coherence)
    top_indices = np.argsort(scores)[-top_k:]
    top_indices = sorted(top_indices)  # ← restore document order

    relevant_text = "\n\n".join(chunks[i] for i in top_indices)
    return relevant_text


def extract_from_page(page_content: str, url: str, query: str, llm) -> str | None:
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
