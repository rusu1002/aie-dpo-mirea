from typing import List
import time
import pickle

from langchain_core.documents import Document

from src.retriever.fusion import fusion_retrieval
from src.reranker.reranker import rerank_docs_api
from src.generation.generator import (
    generate_answer
)
from src.retriever.vectorstore import (
    get_embeddings,
    load_vector_store
)
from configs.config import settings
from src.utils.logger import logger
from src.utils.metrics import (
    RETRIEVAL_LATENCY,
    RETRIEVED_DOCS,
    RERANK_LATENCY,
    RERANKED_DOCS,
    GENERATION_LATENCY
)


class RAGPipeline:
    def __init__(self, vector_store, bm25, all_chunks):
        self.vector_store = vector_store
        self.all_chunks = all_chunks
        self.bm25 = bm25

    def retrieve(self, query: str, k: int = 20, alpha: float = 0.6) -> List[Document]:
        """
        Ретривинг контекстов
        """
        logger.info("Starting retrieval...")
        start = time.time()

        retrieved_docs = fusion_retrieval(
            query=query, 
            vector_store=self.vector_store, 
            bm25=self.bm25, 
            all_chunks=self.all_chunks, 
            k=k, 
            alpha=0.5
        )

        RETRIEVAL_LATENCY.observe(time.time() - start)
        RETRIEVED_DOCS.observe(len(retrieved_docs))
        logger.info(f"Retrieved {len(retrieved_docs)} documents")

        return retrieved_docs

    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """
        Реранжирование контекстов
        """
        logger.info("Starting reranking...")
        start = time.time()

        reranked_docs = rerank_docs_api(query=query, documents=documents)

        RERANK_LATENCY.observe(time.time() - start)
        RERANKED_DOCS.observe(len(reranked_docs))
        logger.info(f"Reranked to {len(reranked_docs)} documents")

        return reranked_docs

    def generate(self, query: str, documents: List[Document]) -> str:
        """
        Генерация ответа
        """
        logger.info("Starting generation...")
        start = time.time()

        context = "\n".join([doc.page_content for doc in documents])
        answer = generate_answer(question=query, context=context)

        GENERATION_LATENCY.observe(time.time() - start)

        return answer

    def answer(self, query: str):
        """
        Создание ответа на вопрос пользователя
        """
        start_time = time.time()

        retrieved_docs = self.retrieve(query)
        reranked_docs = self.rerank(query, retrieved_docs)
        answer = self.generate(query, reranked_docs)

        latency = round((time.time() - start_time) * 1000, 2)

        logger.info(f"Pipeline finished in {latency} ms")

        return {
            "question": query,
            "answer": answer,
            "contexts": [doc.page_content for doc in reranked_docs],
            "sources": [doc.metadata for doc in reranked_docs],
            "latency_ms": latency
        }

    # def answer(self, query: str):

    #     start_time = time.time()

    #     retrieved_docs = self.retrieve(query)
    #     reranked_docs = self.rerank(query, retrieved_docs)
    #     answer = self.generate(query, reranked_docs)

    #     latency = round((time.time() - start_time) * 1000, 2)

    #     logger.info(f"Pipeline finished in {latency} ms")

    #     return {
    #         "question": query,
    #         "answer": answer,
    #         "sources": [doc.metadata for doc in reranked_docs],
    #         "latency_ms": latency
    #     }

def build_pipeline():
    """
    Построение пайплайна RAG
    """
    logger.info("Loading embedding model...")
    get_embeddings()

    logger.info("Loading vector store...")
    vector_store = load_vector_store(settings.VECTORSTORE_PATH)

    logger.info("Loading BM25 artifacts...")
    with open("artifacts/bm25.pkl", "rb") as f:
        bm25 = pickle.load(f)
    
    logger.info("Loading chunks...")
    with open("artifacts/all_chunks.pkl", "rb") as f:
        all_chunks = pickle.load(f)

    return RAGPipeline(
        vector_store=vector_store,
        bm25=bm25,
        all_chunks=all_chunks
    )