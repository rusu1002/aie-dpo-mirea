from typing import List
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS

from configs.config import settings
from src.utils.logger import logger


_tokenizer = None
_embedding_model = None
_embeddings_instance = None

_device = ("cuda" if torch.cuda.is_available() else "cpu")


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]

    input_mask_expanded = (attention_mask .unsqueeze(-1) .expand(token_embeddings.size()) .float())

    return (
        torch.sum( token_embeddings * input_mask_expanded, 1)
        / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    )

# LOAD MODEL
def get_embedding_model():
    global _tokenizer
    global _embedding_model

    if _embedding_model is None:
        logger.info(f"Loading embedding model: {settings.EMBEDDER_MODEL}")

        _tokenizer = AutoTokenizer.from_pretrained(settings.EMBEDDER_MODEL)
        _embedding_model = (AutoModel.from_pretrained(settings.EMBEDDER_MODEL).to(_device))
        _embedding_model.eval()

        logger.info("Embedding model loaded")

    return _tokenizer, _embedding_model

# EMBEDDINGS WRAPPER
class HFEmbeddingsWrapper(Embeddings):

    def __init__(self):
        self.tokenizer, self.model = (get_embedding_model())

    def _encode(self, texts: List[str]) -> List[List[float]]:

        encoded_input = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        encoded_input = {k: v.to(_device) for k, v in encoded_input.items()}

        with torch.no_grad():
            model_output = self.model(**encoded_input)

        sentence_embeddings = mean_pooling(
            model_output,
            encoded_input["attention_mask"]
        )

        sentence_embeddings = F.normalize(
            sentence_embeddings,
            p=2,
            dim=1,
        )

        return (sentence_embeddings.cpu().tolist())

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._encode(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._encode([text])[0]

# SINGLETON EMBEDDINGS
def get_embeddings():
    global _embeddings_instance

    if _embeddings_instance is None:
        logger.info("Creating embeddings wrapper...")
        _embeddings_instance = (HFEmbeddingsWrapper())

    return _embeddings_instance

# VECTOR STORE
def build_vector_store(all_chunks: List[Document], batch_size: int = 100) -> FAISS:
    logger.info("Building FAISS vector store...")

    embeddings = get_embeddings()

    vector_store = None
    for i in range(0, len(all_chunks), batch_size):

        batch = all_chunks[i:i + batch_size]
        logger.info(f"Processing batch {i // batch_size + 1}")

        if vector_store is None:
            vector_store = (FAISS.from_documents(batch, embeddings))
        else:
            vector_store.add_documents(batch)

    logger.info("FAISS vector store built")

    return vector_store


def save_vector_store(vector_store: FAISS, save_path: str):
    logger.info(f"Saving vector store to {save_path}")
    vector_store.save_local(save_path)
    logger.info("Vector store saved")


def load_vector_store(load_path: str) -> FAISS:
    logger.info(f"Loading vector store from {load_path}")
    embeddings = get_embeddings()

    vector_store = FAISS.load_local(
        load_path,
        embeddings,
        allow_dangerous_deserialization=True
    )
    logger.info("Vector store loaded")

    return vector_store