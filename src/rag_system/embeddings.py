# Copyright (c) 2025 Raul Hernandez Lopez
#
# This file is part of the project and is licensed under the
# Creative Commons Attribution-ShareAlike 4.0 International License (CC BY-SA 4.0).
#
# You are free to share and adapt this file under the terms of the CC BY-SA 4.0 license.
# Full license: https://creativecommons.org/licenses/by-sa/4.0/legalcode

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
