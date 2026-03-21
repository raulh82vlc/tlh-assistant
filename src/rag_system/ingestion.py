# Copyright (c) 2025 Raul Hernandez Lopez
#
# This file is part of the project and is licensed under the
# Creative Commons Attribution-ShareAlike 4.0 International License (CC BY-SA 4.0).
#
# You are free to share and adapt this file under the terms of the CC BY-SA 4.0 license.
# Full license: https://creativecommons.org/licenses/by-sa/4.0/legalcode

import shutil
import time
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from src.rag_system.embeddings import build_embeddings, get_device
from src.rag_system.settings import COLLECTION_NAME, DATA_PATH, DB_PATH


def feed_vector_db_with_docs():
    if not DATA_PATH.exists():
        DATA_PATH.mkdir(parents=True, exist_ok=True)
        print(f"Carpeta '{DATA_PATH}' creada para leer PDFs.")
        return

    print(f"Dispositivo: {get_device().upper()}")
    print("Cargando documentos...")

    loader = DirectoryLoader(str(DATA_PATH), glob="**/*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()

    if not docs:
        print("No hay documentos en la carpeta.")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    print(f"Generados {len(chunks)} fragmentos.")

    embeddings = build_embeddings()

    if DB_PATH.exists():
        print("Eliminando base de datos anterior...")
        shutil.rmtree(DB_PATH)
        time.sleep(1)

    client = QdrantClient(path=str(DB_PATH))
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )

    qdrant = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )
    qdrant.add_documents(chunks)

    print(f"Base de datos guardada en {DB_PATH}")
