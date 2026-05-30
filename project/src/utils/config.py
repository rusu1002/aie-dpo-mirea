from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings


BASE_DIR = Path(__file__).resolve().parents[2]

ENV_PATH = BASE_DIR / ".env"

load_dotenv(ENV_PATH)


class Settings(BaseSettings):

    # BASE_URL: str = (
    #     ""
    # )

    # LLM_API_KEY: str
    # EMBEDDER_API_KEY: str
    # RERANKER_API_KEY: str
    LLM_API_KEY: str
    RERANKER_API_KEY: str

    LLM_MODEL_NAME: str = (
        "Qwen/Qwen2.5-1.5B-Instruct"
    )

    EMBEDDER_MODEL_NAME: str = (
        "ai-forever/sbert_large_nlu_ru"
        # "BAAI/bge-m3"
        # "intfloat/multilingual-e5-base"
    )

    RERANKER_MODEL_NAME: str = (
        "langsearch-reranker-v1"
        # "BAAI/bge-reranker-base"
    )

    TRAIN_DATA_PATH: str = (
        "data/train_data.csv"
    )

    QUESTIONS_PATH: str = (
        "data/questions.csv"
    )

    VECTORSTORE_PATH: str = (
        "artifacts/vectorstore"
    )

    LOG_DIR: str = (
        "artifacts/logs"
    )

    class Config:
        env_file = "configs/.env.example"
        extra = "ignore"


settings = Settings()