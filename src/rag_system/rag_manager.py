from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from src.rag_system.embeddings import build_embeddings
from src.rag_system.settings import COLLECTION_NAME, DB_PATH, OLLAMA_MODEL_NAME

PROMPT_TEMPLATE = """Eres un asistente académico especializado en Tecnologías del Lenguaje Humano.
Tu ÚNICA fuente de información es el contexto proporcionado abajo.

REGLAS:
1. Responde ÚNICAMENTE con información del contexto.
2. Si la respuesta NO está en el contexto, di: \"No encuentro esa información en los documentos proporcionados.\"
3. NO inventes datos, fechas, nombres ni conceptos.
4. Cita o parafrasea el contenido del contexto.
5. Responde siempre en español.

CONTEXTO:
{context}

PREGUNTA:
{question}

RESPUESTA:
"""


class RAGChain:
    def __init__(self, retriever, lcel_chain):
        self.retriever = retriever
        self.lcel_chain = lcel_chain

    def invoke(self, inputs):
        query = inputs["query"]
        source_documents = self.retriever.invoke(query)
        result = self.lcel_chain.invoke(query)
        return {"result": result, "source_documents": source_documents}


def run_rag():
    embeddings = build_embeddings()

    client = QdrantClient(path=str(DB_PATH))
    doc_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )

    llm = ChatOllama(
        model=OLLAMA_MODEL_NAME,
        temperature=0,
        keep_alive="1h",
    )

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"],
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

    return RAGChain(retriever=retriever, lcel_chain=lcel_chain)
