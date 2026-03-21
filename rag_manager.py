# Copyright (c) 2025 Raul Hernandez Lopez
#
# This file is part of the project and is licensed under the
# Creative Commons Attribution-ShareAlike 4.0 International License (CC BY-SA 4.0).
#
# You are free to share and adapt this file under the terms of the CC BY-SA 4.0 license.
# Full license: https://creativecommons.org/licenses/by-sa/4.0/legalcode

from langchain_ollama import ChatOllama
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from qdrant_client import QdrantClient
import torch
from constants import DB_PATH, COLLECTION_NAME

PROMPT_TEMPLATE = """Eres un asistente académico especializado en Tecnologías del Lenguaje Humano.
Tu ÚNICA fuente de información es el contexto proporcionado abajo.

REGLAS:
1. Responde ÚNICAMENTE con información del contexto.
2. Si la respuesta NO está en el contexto, di: "No encuentro esa información en los documentos proporcionados."
3. NO inventes datos, fechas, nombres ni conceptos.
4. Cita o parafrasea el contenido del contexto.
5. Responde siempre en español.

CONTEXTO:
{context}

PREGUNTA:
{question}

RESPUESTA:
"""


def run_rag():
    # Embeddings
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        model_kwargs={'device': device}
    )
    # Qdrant local vector DB
    client = QdrantClient(path=DB_PATH)
    doc_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings
    )
    # LLM local / Ollama using llama3
    llm = ChatOllama(
        model="llama3",
        temperature=0,
        keep_alive="1h" # keeps model loaded in RAM during 1 hour
    )

    # Custom prompt
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    retriever = doc_store.as_retriever(search_kwargs={"k": 6})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    lcel_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    class RAGChain:
        def invoke(self, inputs):
            query = inputs["query"]
            source_documents = retriever.invoke(query)
            result = lcel_chain.invoke(query)
            return {"result": result, "source_documents": source_documents}

    return RAGChain()