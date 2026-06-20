# src/monitoring/metrics.py

from prometheus_client import Counter, Histogram, Gauge

import psutil
import time


# ===================================
# Requests
# ===================================

REQUEST_COUNT = Counter(
    "rag_requests_total",
    "Total number of requests"
)
ERROR_COUNT = Counter(
    "rag_errors_total",
    "Total number of failed requests"
)

# ===================================
# Latency
# ===================================

REQUEST_LATENCY = Histogram(
    "rag_request_latency_seconds",
    "Total request latency"
)
RETRIEVAL_LATENCY = Histogram(
    "rag_retrieval_latency_seconds",
    "Retrieval latency"
)
RERANK_LATENCY = Histogram(
    "rag_rerank_latency_seconds",
    "Rerank latency"
)
GENERATION_LATENCY = Histogram(
    "rag_generation_latency_seconds",
    "Generation latency"
)

# ===================================
# Docs
# ===================================

RETRIEVED_DOCS = Histogram(
    "rag_retrieved_docs",
    "Retrieved documents count"
)
RERANKED_DOCS = Histogram(
    "rag_reranked_docs",
    "Reranked documents count"
)

# ===================================
# Model usage
# ===================================

LLM_USAGE = Counter(
    "rag_llm_usage_total",
    "LLM usage",
    ["model"]
)
RERANKER_USAGE = Counter(
    "rag_reranker_usage_total",
    "Reranker usage",
    ["model"]
)


# ===================================
# CPU, RAM USAGE
# ===================================

CPU_USAGE = Gauge(
    "rag_cpu_percent",
    "CPU usage percent"
)
MEMORY_USAGE = Gauge(
    "rag_memory_percent",
    "Memory usage percent"
)

def update_system_metrics():
    while True:
        CPU_USAGE.set(psutil.cpu_percent())
        MEMORY_USAGE.set(psutil.virtual_memory().percent)

        time.sleep(5)