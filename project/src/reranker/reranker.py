from typing import List
import requests
from langchain_core.documents import Document

from configs.config import settings
from src.utils.logger import logger
from src.utils.metrics import (
    RERANKER_USAGE
)
from src.reranker.reranker_manager import (
    reranker_manager
)


def rerank_docs_api(query, documents, top_k=3):
    """
    Выбор доступной модели reranker или аварийного fallback
    """
    rerankers = reranker_manager.get_rerankers()
    
    for reranker in rerankers:
        try:
            if reranker == "Langsearch":
                reranked_docs = rerank_langsearch(query, documents, top_k)
                RERANKER_USAGE.labels(model="Langsearch").inc()
                return reranked_docs
            if reranker == "Cohere":
                reranked_docs = rerank_cohere(query, documents, top_k)
                RERANKER_USAGE.labels(model="Cohere").inc()
                return reranked_docs

        except Exception as e:
            logger.error(f"{reranker} failed: {e}")

    logger.warning("No reranker models available, using no rerank fallback")
    return documents[:top_k]


def rerank_langsearch(query: str,documents: List[Document],top_k: int = 3) -> List[Document]:
    if not documents:
        return []

    logger.info("Reranking documents with LangSearch API reranker")

    try:
        payload = {
            "model": settings.RERANKER_MODELS[0],
            "query": query,
            "top_n": top_k,
            "return_documents": False,
            "documents": [doc.page_content for doc in documents]
        }

        headers = {
            "Authorization": f"Bearer {settings.RERANKER_LANGSEARCH_API_KEY}",
            "Content-Type": "application/json"
        }

        response = requests.post(
            settings.LANGSEARCH_URL,
            json=payload,
            headers=headers,
            timeout=30
        )

        response.raise_for_status()

        data = response.json()

        reranked_results = data["results"]
        ranked_docs = [(documents[item["index"]], item["relevance_score"]) for item in reranked_results]
        ranked_docs.sort(key=lambda x: x[1], reverse=True)

        logger.info(f"Reranked {len(documents)} docs -> top {len(ranked_docs)}")

        return [doc for (doc, score) in ranked_docs]

    except Exception as e:
        raise e
    

def rerank_cohere(query,documents,top_k):
    if not documents:
        return []

    logger.info("Reranking documents with Cohere API reranker")

    try:
        payload = {
            "model": settings.RERANKER_MODELS[1],
            "query": query,
            "top_n": top_k,
            "documents": [doc.page_content for doc in documents]
        }

        headers = {
            "Authorization": f"Bearer {settings.RERANKER_COHERE_API_KEY}",
            "Content-Type": "application/json"
        }

        response = requests.post(
            settings.COHERE_URL,
            json=payload,
            headers=headers,
            timeout=30
        )
        # logger.error(f"Cohere error {response.status_code}: {response.text}")
        response.raise_for_status()

        data = response.json()

        reranked_results = data["results"]
        ranked_docs = [(documents[item["index"]], item["relevance_score"]) for item in reranked_results]
        ranked_docs.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"Reranked {len(documents)} docs -> top {len(ranked_docs)}")

        return [doc for doc, score in ranked_docs]

    except Exception as e:
        raise e