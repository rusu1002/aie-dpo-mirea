from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings


BASE_DIR = Path(__file__).resolve().parents[2]

ENV_PATH = BASE_DIR / ".env"

load_dotenv(ENV_PATH)


class Settings(BaseSettings):

    LLM_API_KEY: str
    RERANKER_LANGSEARCH_API_KEY: str
    RERANKER_COHERE_API_KEY: str

    OPENROUTER_URL: str = "https://openrouter.ai/api/v1/chat/completions"
    LANGSEARCH_URL: str = "https://api.langsearch.com/v1/rerank"
    COHERE_URL: str = "https://api.cohere.com/v2/rerank"

    LLM_MODELS: list = [
        "openai/gpt-oss-120b:free",
        "qwen/qwen3-next-80b-a3b-instruct:free",
        "google/gemma-4-31b-it:free",
    ]

    EMBEDDER_MODEL: str = "ai-forever/sbert_large_nlu_ru"

    RERANKER_MODELS: list = [
        "langsearch-reranker-v1",
        "rerank-v3.5"
    ]

    COHERE_JUDGE_MODEL: str = "command-r-plus-08-2024"
    OPENROUTER_JUDGE_MODEL: str = "openai/gpt-oss-20b:free"

    TRAIN_DATA_PATH: str = "data/train_data.csv"

    QUESTIONS_PATH: str = "data/questions.csv"

    VECTORSTORE_PATH: str = "artifacts/vectorstore"

    LOG_DIR: str = "artifacts/logs"

    API_STATUS_FILE: str = "artifacts/services_status.json"

    EVAL_ANSWERS_FILE: str = "artifacts/rag_answers.json"
    EVAL_RESULTS_PATH: str = "artifacts/deepeval_results.json"
    EVAL_REPORT_DIR: str = "artifacts/evaluation_report"

    class Config:
        env_file = "configs/.env"
        extra = "ignore"


settings = Settings()