import random
import re

import pandas as pd
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

from src.rag_system.settings import DATA_PATH, OLLAMA_MODEL_NAME, TESTSET_CSV_PATH

NUM_TEST_SAMPLES = 10
MIN_QUESTION_LENGTH = 20

GENERATION_PROMPT = """Eres un experto creando exámenes para estudiantes de Tecnologías del Lenguaje Humano.

CONTEXTO:
{context}

REGLAS:
- La respuesta debe ser precisa y técnica
- NO mencionar \"según el texto\" ni \"el fragmento dice\"
- Usar sinónimos técnicos cuando sea posible
- Escribir en español

RESPONDE EXACTAMENTE EN ESTE FORMATO (sin texto adicional antes ni después):
PREGUNTA: [tu pregunta aquí]
RESPUESTA_IDEAL: [tu respuesta aquí]
"""


def parse_qa(text: str):
    pregunta_match = re.search(
        r"PREGUNTA[:\s]+(.+?)(?=RESPUESTA[_\s]?IDEAL|RESPUESTA[:\s]|$)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    respuesta_match = re.search(
        r"RESPUESTA[_\s]?IDEAL[:\s]+(.+?)$|RESPUESTA[:\s]+(.+?)$",
        text,
        re.DOTALL | re.IGNORECASE,
    )

    question = pregunta_match.group(1).strip() if pregunta_match else ""
    ground_truth = (respuesta_match.group(1) or respuesta_match.group(2) or "").strip() if respuesta_match else ""

    question = re.sub(r"\*\*", "", question)
    ground_truth = re.sub(r"\*\*", "", ground_truth)

    question = re.sub(r"^(PREGUNTA|Pregunta)[:\s]+", "", question, flags=re.IGNORECASE)
    ground_truth = re.sub(
        r"^(RESPUESTA[_\s]?IDEAL|RESPUESTA|Respuesta)[:\s]+",
        "",
        ground_truth,
        flags=re.IGNORECASE,
    )

    question = " ".join(question.split())
    ground_truth = " ".join(ground_truth.split())

    return question, ground_truth


def generate_testset():
    print("Cargando documentos...")
    loader = DirectoryLoader(str(DATA_PATH), glob="**/*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    selected_chunks = random.sample(chunks, min(len(chunks), NUM_TEST_SAMPLES))

    llm = ChatOllama(model=OLLAMA_MODEL_NAME, temperature=0.7)
    prompt = PromptTemplate(template=GENERATION_PROMPT, input_variables=["context"])
    chain = prompt | llm

    data = []
    print(f"Generando {len(selected_chunks)} pares de evaluación...")

    for chunk in tqdm(selected_chunks):
        try:
            response = chain.invoke({"context": chunk.page_content})
            question, ground_truth = parse_qa(response.content)

            if len(question) >= MIN_QUESTION_LENGTH and len(ground_truth) >= 10:
                data.append(
                    {
                        "question": question,
                        "ground_truth": ground_truth,
                        "evolution_type": "simple",
                    }
                )
            else:
                print(f"Descartado (muy corto): Q='{question[:50]}...' A='{ground_truth[:50]}...'")
        except Exception as error:
            print(f"Error: {error}")

    df = pd.DataFrame(data)
    df.to_csv(TESTSET_CSV_PATH, index=False)
    print(f"\nDataset guardado: {TESTSET_CSV_PATH} ({len(data)} preguntas)")
    print(df.head())
