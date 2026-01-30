# Copyright (c) 2025 Raul Hernandez Lopez
#
# This file is part of the project and is licensed under the
# Creative Commons Attribution-ShareAlike 4.0 International License (CC BY-SA 4.0).
#
# You are free to share and adapt this file under the terms of the CC BY-SA 4.0 license.
# Full license: https://creativecommons.org/licenses/by-sa/4.0/legalcode

import os
import time
import shutil
import torch
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from qdrant_client.models import Distance, VectorParams

DATA_PATH = './data'
DB_PATH = './qdrant_db'
COLLECTION_NAME = 'tlh_rag'


def feed_vector_db_with_docs():
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        print(f"Carpeta '{DATA_PATH}' creada para leer PDFs.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo: {device.upper()}")

    print("Cargando documentos...")
    loader = DirectoryLoader(DATA_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()

    if not docs:
        print("No hay documentos en la carpeta.")
        return

    # Splitting documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    print(f"Generados {len(chunks)} fragmentos.")

    # Vectorizing documents
    print("Vectorizando...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        model_kwargs={'device': device}
    )

    # Clean old DB if exists
    if os.path.exists(DB_PATH):
        print("Eliminando base de datos anterior...")
        shutil.rmtree(DB_PATH)
        time.sleep(1)

    # Create Qdrant client and collection (768 dimens for the model)
    client = QdrantClient(path=DB_PATH)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE)
    )

    qdrant = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings
    )
    qdrant.add_documents(chunks)

    print(f"Base de datos guardada en {DB_PATH}")


if __name__ == "__main__":
    feed_vector_db_with_docs()