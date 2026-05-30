from typing import List
import time

from langchain_core.documents import Document

from src.retriever.fusion import fusion_retrieval
from src.retriever.reranker import rerank_docs_api

from src.generation.generator import (
    generate_answer,
)

from src.utils.logger import logger


class RAGPipeline:

    def __init__(
        self,
        vector_store,
        bm25,
        all_chunks,
    ):

        self.vector_store = vector_store
        self.all_chunks = all_chunks
        self.bm25 = bm25

    def retrieve(
        self,
        query: str,
        k: int = 20,
        alpha: float = 0.6,
    ) -> List[Document]:

        logger.info("Starting retrieval...")

        retrieved_docs = (
            fusion_retrieval(
                query=query,
                vector_store=self.vector_store,
                bm25=self.bm25,
                all_chunks=self.all_chunks,
                k=k,
                alpha=0.5,
            )
        )

        logger.info(
            f"Retrieved {len(retrieved_docs)} documents"
        )

        return retrieved_docs

    def rerank(
        self,
        query: str,
        documents: List[Document],
    ) -> List[Document]:

        logger.info("Starting reranking...")

        # resolved_docs = (
        #     resolve_questions_to_original_chunks(
        #         documents,
        #         self.all_chunks,
        #     )
        # )

        reranked_docs = rerank_docs_api(
            query=query,
            documents=documents,
        )

        logger.info(
            f"Reranked to {len(reranked_docs)} documents"
        )

        return reranked_docs

    def generate(
        self,
        query: str,
        documents: List[Document],
    ) -> str:

        context = "\n\n".join(
            [doc.page_content for doc in documents]
        )

        return generate_answer(
            question=query,
            context=context,
        )

    def answer(
        self,
        query: str,
    ):

        start_time = time.time()

        logger.info(f"Question received: {query}")

        retrieved_docs = self.retrieve(query)

        reranked_docs = self.rerank(
            query,
            retrieved_docs,
        )

        answer = self.generate(
            query,
            reranked_docs,
        )

        latency = round(
            (time.time() - start_time) * 1000,
            2,
        )

        logger.info(
            f"Pipeline finished in {latency} ms"
        )

        return {
            "question": query,
            "answer": answer,
            "sources": [
                doc.metadata
                for doc in reranked_docs
            ],
            "latency_ms": latency,
        }