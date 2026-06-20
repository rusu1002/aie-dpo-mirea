# ToDo - Добавить проверку для judge модели

import json
import requests
from pathlib import Path

from configs.config import settings
from src.utils.logger import logger


def check_llm(model: str) -> bool:
    """
    Проверка OpenRouter LLM models на доступность
    """
    try:
        response = requests.post(
            settings.OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {settings.LLM_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 1,
                "temperature": 0,
            },
            timeout=10
        )
        
        if response.status_code == 200:
            logger.info(f"Response status code: {response.status_code}")
            return True
        logger.warning(f"Response status code: {response.status_code}")
        logger.warning(f"Response text: {response.text}")
        return False
    
    except Exception:
        return False


def check_langsearch() -> bool:
    """
    Проверка Langsearch Reranker на доступность
    """
    try:
        response = requests.post(
            settings.LANGSEARCH_URL,
            headers={
                "Authorization": f"Bearer {settings.RERANKER_LANGSEARCH_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": settings.RERANKER_MODELS[0],
                "query": "test",
                "documents": ["test"],
                "top_n": 1,
            },
            timeout=10,
        )

        if response.status_code == 200:
            logger.info(f"Response status code: {response.status_code}")
            return True
        logger.warning(f"Response status code: {response.status_code}")
        logger.warning(f"Response text: {response.text}")
        return False

    except Exception:
        return False


def check_cohere() -> bool:
    """
    Проверка Cohere Reranker на доступность
    """
    try:
        response = requests.post(
            settings.COHERE_URL,
            headers={
                "Authorization": f"Bearer {settings.RERANKER_COHERE_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": settings.RERANKER_MODELS[1],
                "query": "test",
                "documents": ["test"],
                "top_n": 1,
            },
            timeout=10,
        )

        if response.status_code == 200:
            logger.info(f"Response status code: {response.status_code}")
            return True
        logger.warning(f"Response status code: {response.status_code}")
        logger.warning(f"Response text: {response.text}")
        return False

    except Exception:
        return False


def main():

    llms = []
    for model in settings.LLM_MODELS:

        logger.info(f"Checking {model}...")
        if check_llm(model):
            llms.append(model)

    rerankers = []

    logger.info(f"Checking Langsearch reranker...")
    if check_langsearch():
        rerankers.append("Langsearch")

    logger.info(f"Checking Cohere reranker...")
    if check_cohere():
        rerankers.append("Cohere")

    result = {
        "llm": llms,
        "reranker": rerankers
    }

    Path("artifacts").mkdir(exist_ok=True)

    with open(settings.API_STATUS_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved service status to {settings.API_STATUS_FILE}")


if __name__ == "__main__":
    main()