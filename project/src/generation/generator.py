import requests
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import re

from src.generation.llm_manager import llm_manager
from configs.config import settings
from src.utils.logger import logger
from src.utils.metrics import LLM_USAGE


def generate_answer(question: str, context: str) -> str:
    """
    Выбор доступной модели LLM или аварийного fallback
    """
    llms = llm_manager.get_llms()

    for llm in llms:
        try:
            answer = generate_answer_api(question, context, llm)

            LLM_USAGE.labels(model=llm).inc()
            
            return answer

        except Exception as e:
            logger.error(f"LLM {llm} failed: {e}")

    logger.warning("No LLM models available, using fallback generation from context")
    return generate_answer_from_context(question, context)


def generate_answer_api(question: str, context: str, model: str) -> str:
    # current_model = llm_manager.get_active_model()
    logger.info(f"Generating answer with OpenRouter API LLM: {model}")

    messages = [
        {
            "role": "system",
            "content": (
                "Ты финансовый AI-ассистент. "
                "Отвечай подробно и структурированно, но не переписывай весь контекст. Ответь максимум 10 предложениями. Пиши ответ в разговорном уважительном стиле. Если точного ответа на вопрос нет, скажи: 'К сожалению, я не могу ответить на ваш вопрос'. НЕЛЬЗЯ использовать markdown, НИКАКИХ спецсимволов, только текст. Нельзя упоминать контекст, данный текст, предоставленные данные."
            )
        },
        {
            "role": "user",
            "content": f"""
Контекст:
{context}

Вопрос:
{question}
"""
        }
    ]
    response = requests.post(
        settings.OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {settings.LLM_API_KEY}",
            "Content-Type": "application/json"
            # "HTTP-Referer": "http://localhost",  # можно оставить так
            # "X-Title": "RAG Project"
        },
        data=json.dumps({
            "model": model,
            "messages": messages,
            "temperature": 0.5,
            "max_tokens": 512
        }
        )
    )

    response.raise_for_status()
    result = response.json()

    # logger.info(f"Full API response: {json.dumps(result, ensure_ascii=False)[:500]}")
    
    if "choices" not in result:
        logger.error(f"Unexpected response structure: {result.keys()}")
        return "Ошибка: неожиданный формат ответа от API"
    
    choice = result["choices"][0]
    message = choice.get("message", {})
    
    answer = message.get("content")
    
    if answer is None:
        answer = message.get("text")
        
    if answer is None and "reasoning" in choice:
        answer = choice["reasoning"]
        
    if answer is None:
        logger.error(f"No answer field found. Message keys: {message.keys() if message else 'empty'}")
        logger.error(f"Full choice: {choice}")
        return "Ошибка: модель не вернула ответ"
    
    logger.info("Answer generated successfully")
    return answer

    # answer = result["choices"][0]["message"]["content"]
    # logger.info("Answer generated successfully")
    # return answer


def split_into_sentences(text: str) -> list:
    """Разделение текста на предложения"""
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

def generate_answer_from_context(query: str, context: str, max_sentences: int = 2) -> str:
    """
    Генерация ответа выбором наиболее релевантных предложений из контекста.
    Используется как fallback, когда API модели недоступен.
    """
    # logger.info("Using fallback: generating answer from context without LLM")
    
    raw_lines = [line.strip() for line in context.splitlines() if line.strip()]
    content_lines = [line for line in raw_lines]

    sentence_pool = []
    for line in content_lines:
        sentence_pool.extend(split_into_sentences(line))

    # Фильтруем слишком короткие предложения
    sentence_pool = [s for s in sentence_pool if len(s.split()) >= 4]

    if not sentence_pool:
        return "Недостаточно контекста для построения ответа."

    try:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        matrix = vectorizer.fit_transform([query] + sentence_pool).toarray().astype(np.float32)

        query_vec = matrix[0]
        sentence_vecs = matrix[1:]

        query_norm = np.linalg.norm(query_vec) + 1e-12
        sent_norms = np.linalg.norm(sentence_vecs, axis=1) + 1e-12
        scores = (sentence_vecs @ query_vec) / (sent_norms * query_norm)

        ranked_idx = np.argsort(-scores)
        selected_sentences = []
        used_normalized = set()

        for idx in ranked_idx:
            sentence = sentence_pool[idx]
            normalized = sentence.lower().strip()
            if scores[idx] <= 0:
                continue
            if normalized in used_normalized:
                continue
            used_normalized.add(normalized)
            selected_sentences.append(sentence)
            if len(selected_sentences) >= max_sentences:
                break

        if not selected_sentences:
            return "В найденном контексте нет достаточно релевантного фрагмента для уверенного ответа."

        logger.info(f"Fallback selected {len(selected_sentences)} sentences")
        return " ".join(selected_sentences)
    
    except Exception as e:
        logger.error(f"Error in fallback generation: {e}")
        # Если что-то пошло не так, возвращаем первые несколько предложений
        return "Не удалось сформировать ответ из контекста."