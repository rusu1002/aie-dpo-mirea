import time
import pickle

from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from src.RAG_pipeline import (
    RAGPipeline,
)

from src.retriever.vectorstore import (
    load_vector_store,
    get_embeddings,
)
from src.generation.manager import model_manager
from src.utils.config import settings
from src.utils.logger import logger


# =========================================
# GLOBAL OBJECTS
# =========================================

pipeline = None
service_start_time = time.time()


# =========================================
# REQUEST/RESPONSE SCHEMAS
# =========================================

class PredictRequest(BaseModel):
    question: str

class PredictResponse(BaseModel):
    question: str
    answer: str
    sources: list
    latency_ms: float


# =========================================
# LIFESPAN
# =========================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    global pipeline

    logger.info(
        "========== SERVICE STARTING =========="
    )

    # =====================================
    # LOAD VECTOR STORE
    # =====================================
    logger.info(
        "Preloading embedding model..."
    )

    get_embeddings()

    logger.info("Loading vector store...")

    vector_store = load_vector_store(
        settings.VECTORSTORE_PATH
    )

    # =====================================
    # LOAD BM25
    # =====================================

    logger.info(
        "Loading BM25 artifacts..."
    )

    with open(
        "artifacts/bm25.pkl",
        "rb",
    ) as f:

        bm25 = pickle.load(f)

    # =====================================
    # LOAD CHUNKS
    # =====================================

    logger.info("Loading chunks...")

    with open(
        "artifacts/all_chunks.pkl",
        "rb",
    ) as f:

        all_chunks = pickle.load(f)

    # =====================================
    # CHECK LLM MODELS AVAILABILITY
    # =====================================
    
    logger.info("Checking LLM models availability...")
    # try:
    model_manager.initialize()  # 👈 Проверяем модели при старте
        # logger.info(f"✅ Active model: {model_manager.get_active_model()}")
    # except Exception as e:
    #     logger.error(f"❌ Model initialization failed: {e}")
    #     logger.warning("Service will start but LLM generation may fail")

    # =====================================
    # INIT PIPELINE
    # =====================================

    logger.info("Initializing RAG pipeline...")

    pipeline = RAGPipeline(
        vector_store=vector_store,
        bm25=bm25,
        all_chunks=all_chunks,
    )

    logger.info(
        "========== SERVICE READY =========="
    )

    yield

    logger.info(
        "========== SERVICE STOPPED =========="
    )


# =========================================
# FASTAPI APP
# =========================================

app = FastAPI(
    title="Financial RAG System",
    description=(
        "RAG система для ответов "
        "на финансовые вопросы"
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# =========================================
# HEALTH ENDPOINT
# =========================================

@app.get("/health")
def health():

    uptime = round(
        time.time() - service_start_time,
        2,
    )

    logger.info("Health check requested")

    return {
        "status": "ok",
        "service": "financial-rag-system",
        "uptime_seconds": uptime,
    }


# =========================================
# PREDICT ENDPOINT
# =========================================

@app.post(
    "/predict",
    response_model=PredictResponse,
)
def predict(
    request: PredictRequest,
):

    logger.info(
        f"Received question: "
        f"{request.question}"
    )

    try:

        result = pipeline.answer(
            request.question
        )

        logger.info("Prediction successful")

        return result

    except Exception as e:

        logger.error(
            f"Prediction failed: {e}"
        )

        return {
            "question": request.question,
            "answer": "Internal server error",
            "sources": [],
            "latency_ms": -1,
        }


# =========================================
# ROOT ENDPOINT
# =========================================

@app.get("/")
def root():

    return {
        "message": (
            "Financial RAG System API"
        ),
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
    }