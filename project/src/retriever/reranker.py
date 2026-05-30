# project/src/reranker/reranker.py

from typing import List

import requests

from langchain_core.documents import Document

from src.utils.config import settings
from src.utils.logger import logger


RERANK_URL = "https://api.langsearch.com/v1/rerank"


def rerank_docs_api(
    query: str,
    documents: List[Document],
    top_k: int = 3,
) -> List[Document]:

    if not documents:
        return []

    logger.info("Reranking documents with LangSearch API...")

    try:

        payload = {
            "model": settings.RERANKER_MODEL_NAME,
            "query": query,
            "top_n": top_k,
            "return_documents": False,
            "documents": [
                doc.page_content for doc in documents
            ]
        }

        headers = {
            "Authorization": (
                f"Bearer {settings.RERANKER_API_KEY}"
            ),
            "Content-Type": "application/json"
        }

        response = requests.post(
            RERANK_URL,
            json=payload,
            headers=headers,
            timeout=30
        )

        response.raise_for_status()

        data = response.json()

        # API возвращает индексы документов
        reranked_results = data["results"]
        ranked_docs = [
            (documents[item["index"]], item["relevance_score"])
            for item in reranked_results
        ]
        ranked_docs.sort(key=lambda x: x[1], reverse=True)
        
        # ranked_docs = []
        # for item in reranked_results:
        #     doc_index = item["index"]
        #     ranked_docs.append(
        #         documents[doc_index]
        #     )

        logger.info(
            f"Reranked {len(documents)} docs -> "
            f"top {len(ranked_docs)}"
        )

        return [doc for doc, score in ranked_docs]
        # return ranked_docs

    except Exception as e:

        logger.error(
            f"Reranker API failed: {e}"
        )

        logger.warning(
            "Returning original documents "
            "without reranking"
        )

        # fallback
        return documents
    



# from typing import List
# from sentence_transformers import CrossEncoder
# from langchain_core.documents import Document

# from src.utils.config import settings
# from src.utils.logger import logger


# _reranker = None


# def get_reranker():
#     global _reranker

#     if _reranker is None:
#         logger.info(f"Loading reranker: {settings.RERANKER_MODEL_NAME}")

#         _reranker = CrossEncoder(
#             settings.RERANKER_MODEL_NAME,
#             device="cuda" if settings.__dict__.get("USE_GPU", False) else "cpu"
#         )

#         logger.info("Reranker loaded")

#     return _reranker


# def rerank_docs(
#     query: str,
#     documents: List[Document],
#     top_k: int = 3,
# ) -> List[Document]:

#     logger.info("Reranking documents...")

#     reranker = get_reranker()

#     pairs = [(query, doc.page_content) for doc in documents]

#     scores = reranker.predict(pairs)

#     ranked = sorted(
#         zip(documents, scores),
#         key=lambda x: x[1],
#         reverse=True
#     )

#     return [doc for doc, _ in ranked[:top_k]]


# def rerank_docs_api(
#     query: str,
#     documents: List[Document],
#     top_k: int = 3,
# ) -> List[Document]:

#     logger.info("Running reranker...")

#     url = f"{settings.BASE_URL}rerank"

#     headers = {
#         "Content-Type": "application/json",
#         "Authorization": (
#             f"Bearer {settings.RERANKER_API_KEY}"
#         ),
#     }

#     doc_strings = [
#         doc.page_content
#         for doc in documents
#     ]

#     payload = {
#         "model": settings.RERANKER_MODEL_NAME,
#         "query": query,
#         "documents": doc_strings,
#     }

#     try:

#         response = requests.post(
#             url,
#             headers=headers,
#             json=payload,
#         )

#         response.raise_for_status()

#         results = response.json().get(
#             "results",
#             []
#         )

#         sorted_results = sorted(
#             results,
#             key=lambda x: x["relevance_score"],
#             reverse=True,
#         )

#         reranked_docs = [
#             documents[r["index"]]
#             for r in sorted_results
#         ]

#         logger.info("Reranking completed")

#         return reranked_docs[:top_k]

#     except Exception as e:

#         logger.error(f"Reranking failed: {e}")

#         return documents[:top_k]