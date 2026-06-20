from typing import Optional
from pydantic import BaseModel
import json
import random
from pathlib import Path

import cohere

from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.models import OpenRouterModel
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRelevancyMetric
)
from deepeval.test_case import LLMTestCase

from evaluation.analyse_results import generate_evaluation_report
from configs.config import settings
from src.utils.logger import logger


class CohereWithFallbackModel(DeepEvalBaseLLM):
    def __init__(
        self,
        cohere_api_key: str,
        openrouter_api_key: str,
        cohere_model: str = settings.COHERE_JUDGE_MODEL,
        openrouter_model: str = settings.OPENROUTER_JUDGE_MODEL
    ):

        self.client = cohere.ClientV2(api_key=cohere_api_key)
        self.cohere_model = cohere_model
        self.openrouter = OpenRouterModel(
            model=openrouter_model,
            api_key=openrouter_api_key,
            base_url="https://openrouter.ai/api/v1"
        )

    def load_model(self):
        return self.client

    def get_model_name(self):
        return f"Cohere({self.cohere_model})"

    def generate(self, prompt: str, schema: Optional[type[BaseModel]] = None):
        try:
            kwargs = {}
            if schema is not None:
                kwargs["response_format"] = {"type": "json_object"}
                prompt = ("Return ONLY a valid JSON object matching the requested schema.\n\n" + prompt)

            response = self.client.chat(
                model=self.cohere_model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                **kwargs
            )

            text = response.message.content[0].text

            if schema is not None:
                return schema.model_validate_json(text)

            return text

        except Exception as e:
            logger.error(f"Cohere failed: {e}")
            logger.warning("Fallback to OpenRouter")

            return self.openrouter.generate(prompt, schema=schema)

    async def a_generate(self, prompt: str, schema: Optional[BaseModel] = None):
        return self.generate(prompt, schema=schema)


def main():
    logger.info("Starting evaluation...")

    SAMPLE_SIZE = 100

    if Path(settings.EVAL_RESULTS_PATH).exists():
        with open(Path(settings.EVAL_RESULTS_PATH), "r", encoding="utf-8") as f:
            results = json.load(f)
    else:
        results = []

    # processed_ids = {r["question_id"] for r in results}
    processed_questions = {r["question"] for r in results}
    logger.info(f"Loaded {len(results)} existing results")

    with open("artifacts/rag_answers.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    random.seed(42)

    sampled_data = random.sample(data, SAMPLE_SIZE)
    logger.info(f"Loaded {len(sampled_data)} samples")

    judge_model = CohereWithFallbackModel(cohere_api_key=settings.RERANKER_COHERE_API_KEY, openrouter_api_key=settings.LLM_API_KEY)

    metrics = {
        "faithfulness":FaithfulnessMetric(model=judge_model, threshold=0.7),
        "answer_relevancy":AnswerRelevancyMetric(model=judge_model, threshold=0.7),
        "contextual_relevancy":ContextualRelevancyMetric(model=judge_model, threshold=0.7)
    }

    for idx, item in enumerate(sampled_data):
        if item["question"] in processed_questions:
            logger.info(f"Skipping already processed question: {item['question'][:50]}...")
            continue

        logger.info(f"\nProcessing {idx+1}/{len(sampled_data)}")

        test_case = LLMTestCase(
            input=item["question"],
            actual_output=item["answer"],
            retrieval_context=item["contexts"]
        )
        row = {"question_id": idx, "question": item["question"]}

        for metric_name, metric in metrics.items():
            try:
                metric.measure(test_case)
                row[metric_name] = {"score": metric.score, "reason": metric.reason}

            except Exception as e:
                row[metric_name] = {"score": None, "reason": str(e)}

        results.append(row)

        with open(Path(settings.EVAL_RESULTS_PATH), "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved progress: {idx+1}")
    
    logger.info(f"\nEvaluation complete!")
    generate_evaluation_report()

if __name__ == "__main__":
    main()