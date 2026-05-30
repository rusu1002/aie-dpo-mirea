import os
import pickle

from src.data.loader import load_docs
from src.data.chunking import split_documents

from src.retriever.vectorstore import (
    build_vector_store,
    save_vector_store,
)

from src.retriever.bm25 import (
    build_bm25_retriever,
)

from src.utils.config import settings
from src.utils.logger import logger


def main():

    logger.info("========== TRAINING STARTED ==========")

    # =========================================
    # 1. LOAD DOCUMENTS
    # =========================================

    logger.info("Loading documents...")

    raw_docs = load_docs(
        settings.TRAIN_DATA_PATH
    )

    logger.info(
        f"Loaded {len(raw_docs)} raw documents"
    )

    # =========================================
    # 2. SPLIT INTO CHUNKS
    # =========================================

    logger.info("Splitting documents into chunks...")

    all_chunks = split_documents(
        raw_docs
    )

    logger.info(
        f"Created {len(all_chunks)} chunks"
    )

    # =========================================
    # 4. BUILD VECTOR STORE
    # =========================================

    logger.info("Building vector store...")

    vector_store = build_vector_store(
        all_chunks
    )

    os.makedirs(
        settings.VECTORSTORE_PATH,
        exist_ok=True,
    )

    save_vector_store(
        vector_store,
        settings.VECTORSTORE_PATH,
    )

    # =========================================
    # 5. BUILD BM25
    # =========================================

    logger.info(
    "Building BM25 retriever..."
    )

    bm25 = build_bm25_retriever(
        all_chunks
    )

    # =========================================
    # 6. SAVE BM25
    # =========================================

    logger.info("Saving TF-IDF artifacts...")

    with open(
    "artifacts/bm25.pkl",
    "wb",
    ) as f:

        pickle.dump(
            bm25,
            f,
        )

    # =========================================
    # 7. SAVE CHUNKS
    # =========================================

    logger.info("Saving chunks...")

    with open(
        "artifacts/all_chunks.pkl",
        "wb",
    ) as f:

        pickle.dump(
            all_chunks,
            f,
        )

    logger.info(
        "========== TRAINING FINISHED =========="
    )


if __name__ == "__main__":
    main()