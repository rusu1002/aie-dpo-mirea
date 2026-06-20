import json
import pandas as pd

from pathlib import Path

from src.RAG_pipeline import build_pipeline
from configs.config import settings
from src.utils.logger import logger


def main():
    pipeline = build_pipeline()

    df = pd.read_csv(settings.QUESTIONS_PATH)

    results = []
    for _, row in df.iterrows():

        question = row["Вопрос"]
        result = pipeline.answer(question)
        results.append(result)

        logger.info(f"Done: {question[:60]}")

    Path("artifacts").mkdir(exist_ok=True)

    with open(settings.EVAL_ANSWERS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(results)} answers")


if __name__ == "__main__":
    main()