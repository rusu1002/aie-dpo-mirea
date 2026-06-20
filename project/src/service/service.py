import time
import pickle
import threading
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from prometheus_client import generate_latest
from fastapi.responses import Response

from src.RAG_pipeline import (
    RAGPipeline,
    build_pipeline
)
from src.retriever.vectorstore import (
    load_vector_store,
    get_embeddings
)
from configs.config import settings
from src.utils.logger import logger
from src.utils.metrics import (
    REQUEST_COUNT,
    ERROR_COUNT,
    REQUEST_LATENCY,
    update_system_metrics
)
from src.generation.llm_manager import (
    llm_manager
)
from src.reranker.reranker_manager import (
    reranker_manager
)


pipeline = None
service_start_time = time.time()

class PredictRequest(BaseModel):
    question: str

class PredictResponse(BaseModel):
    question: str
    answer: str
    sources: list
    latency_ms: float


@asynccontextmanager
async def lifespan(app: FastAPI):

    global pipeline

    logger.info("========== SERVICE STARTING ==========")

    # =====================================
    # LOAD VECTOR STORE
    # =====================================
    # logger.info("Preloading embedding model...")
    # get_embeddings()

    # logger.info("Loading vector store...")
    # vector_store = load_vector_store(settings.VECTORSTORE_PATH)

    # # =====================================
    # # LOAD BM25
    # # =====================================

    # logger.info("Loading BM25 artifacts...")
    # with open("artifacts/bm25.pkl", "rb") as f:
    #     bm25 = pickle.load(f)

    # =====================================
    # LOAD CHUNKS
    # =====================================

    # logger.info("Loading chunks...")
    # with open("artifacts/all_chunks.pkl", "rb") as f:
    #     all_chunks = pickle.load(f)

    # =====================================
    # INIT PIPELINE
    # =====================================

    logger.info("Initializing RAG pipeline...")
    # pipeline = RAGPipeline(vector_store=vector_store, bm25=bm25, all_chunks=all_chunks)
    pipeline = build_pipeline()

    logger.info("========== SERVICE READY ==========")
    threading.Thread(target=update_system_metrics, daemon=True).start()
    
    yield

    logger.info("========== SERVICE STOPPED ==========")


# =========================================
# FASTAPI APP
# =========================================

app = FastAPI(
    title="Financial RAG System",
    description="RAG система для ответов на финансовые вопросы",
    version="1.0.0",
    lifespan=lifespan
)


# =========================================
# HEALTH ENDPOINT
# =========================================

@app.get("/health")
def health():

    uptime = round(time.time() - service_start_time, 2)

    logger.info("Health check requested")

    return {
        "status": "ok",
        "service": "financial-rag-system",
        "uptime_seconds": uptime,
        "llm": llm_manager.get_status(),
        "reranker": reranker_manager.get_status()
    }


# =========================================
# PREDICT ENDPOINT
# =========================================

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):

    logger.info(f"Received question: {request.question}")
    REQUEST_COUNT.inc()
    request_start = time.time()
    try:
        result = pipeline.answer(request.question)

        logger.info("Prediction successful")

        return result

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        ERROR_COUNT.inc()

        return {
            "question": request.question,
            "answer": "Internal server error",
            "sources": [],
            "latency_ms": -1,
        }
    finally:
        REQUEST_LATENCY.observe(time.time() - request_start)


# =========================================
# METRICS ENDPOINT
# =========================================

@app.get("/metrics")
def metrics():
    return Response(
        generate_latest(),
        media_type="text/plain"
    )


# =========================================
# ROOT ENDPOINT
# =========================================

@app.get("/")
def root():
    return {
        "message": "Financial RAG System API",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
    }