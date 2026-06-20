import json
from pathlib import Path

from configs.config import settings


class RerankerManager:
    def __init__(self):
        self.rerankers = []
        self.load()

    def load(self):
        if Path(settings.API_STATUS_FILE).exists():
            with open(settings.API_STATUS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)

        self.rerankers = data.get("reranker", [])

    def get_rerankers(self):
        return self.rerankers

    def get_status(self):
        return {
            "rerankers": self.rerankers,
            "reranker_available": len(self.rerankers) > 0
        }

reranker_manager = RerankerManager()