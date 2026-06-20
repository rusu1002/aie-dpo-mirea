from typing import List
import re
from rank_bm25 import BM25Okapi
import nltk
from nltk.corpus import stopwords
from langchain_core.documents import Document

from src.utils.logger import logger

nltk.download('stopwords', quiet=True)
russian_stopwords = set(
    stopwords.words("russian")
)

extra_stopwords = {
    'это', 'эти', 'этому', 'этих', 'этого', 'этим', 'этими', 'этом', 'этих', 'этом', 'эта', 'этот',
    'так', 'вот', 'быть', 'как', 'также', 'только', 'еще', 'уже',
    'даже', 'может', 'можно', 'нельзя', 'нужно', 'надо', 'весь', 'свой',
    'который', 'такой', 'там', 'тут', 'здесь', 'теперь', 'потом', 'тогда',
    'поэтому', 'поскольку', 'однако', 'зато', 'либо', 'кроме', 'через',
    'между', 'среди', 'внутри', 'снаружи', 'вдоль', 'вокруг', 'мимо',
    'согласно', 'благодаря', 'несмотря', 'несколько', 'много', 'мало',
    'больше', 'меньше', 'часто', 'редко', 'всегда', 'никогда', 'иногда',
    'обычно', 'конечно', 'действительно', 'абсолютно', 'совершенно',
    'почти', 'примерно', 'около', 'более', 'менее', 'наиболее', 'наименее'
}

number_stopwords = {
    'один', 'два', 'три', 'четыре', 'пять', 'шесть', 'семь', 'восемь', 'девять', 'десять',
    'первый', 'второй', 'третий', 'четвертый', 'пятый', 'шестой', 'седьмой', 'восьмой',
    'девятый', 'десятый', 'одна', 'одно', 'одни', 'одних', 'одним', 'одной',
    'тысяча', 'миллион', 'миллиард', 'рубль', 'рублей', 'копейка', 'копеек',
    'год', 'лет', 'месяц', 'месяца', 'месяцев', 'день', 'дня', 'дней',
    'час', 'часа', 'часов', 'минута', 'минуты', 'минут'
}

all_stopwords = russian_stopwords.union(extra_stopwords).union(number_stopwords)


def clean_and_tokenize(text):
    """
    Очистка текста и токенизация с фильтрацией стоп-слов и цифр
    """
    text = text.lower()
    text = re.sub(r'\b\d+\b', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    tokens = text.split()
    
    filtered_tokens = [
        token for token in tokens 
        if token not in all_stopwords
        and len(token) > 2
        and any(c.isalpha() for c in token)
    ]
    
    return filtered_tokens


def build_bm25_retriever(all_chunks: List[Document]):
    """
    Построение BM25 ретривера
    """
    logger.info("Building BM25 retriever...")

    corpus = [clean_and_tokenize(doc.page_content) for doc in all_chunks]
    bm25 = BM25Okapi(corpus)

    logger.info("BM25 retriever built")

    return bm25