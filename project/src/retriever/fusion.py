from typing import List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.documents import (
    Document
)
from langchain_community.vectorstores import (
    FAISS
)

from src.retriever.bm25 import (
    clean_and_tokenize
)


def fusion_retrieval(
    query: str,
    vector_store: FAISS,
    bm25,
    all_chunks: List[Document],
    k: int = 10,
    alpha: float = 0.5,
) -> List[Document]:
    """
    Fusion ретривинг контекстов
    """
    epsilon = 1e-8

    bm25_scores = np.array(bm25.get_scores(clean_and_tokenize(query)))
    bm25_norm = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + epsilon)

    dense_results = (vector_store.similarity_search_with_score(query, k=len(all_chunks)))
    dense_scores = np.zeros(len(all_chunks))

    for doc, score in dense_results:
        idx = doc.metadata["chunk_id"]
        dense_scores[idx] = (1 / (1 + score))

    dense_norm = (dense_scores - dense_scores.min()) / (dense_scores.max() - dense_scores.min() + epsilon)

    combined_scores = (alpha * dense_norm + (1 - alpha) * bm25_norm)

    top_indices = np.argsort(combined_scores)[::-1][:k]

    return [all_chunks[i] for i in top_indices]