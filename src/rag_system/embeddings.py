import torch
from langchain_huggingface import HuggingFaceEmbeddings

from src.rag_system.settings import EMBEDDING_MODEL_NAME


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": get_device()},
    )
