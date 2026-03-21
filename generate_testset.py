# Copyright (c) 2025 Raul Hernandez Lopez
#
# This file is part of the project and is licensed under the
# Creative Commons Attribution-ShareAlike 4.0 International License (CC BY-SA 4.0).
#
# You are free to share and adapt this file under the terms of the CC BY-SA 4.0 license.
# Full license: https://creativecommons.org/licenses/by-sa/4.0/legalcode

import pandas as pd
import random
import re
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from tqdm import tqdm
from constants import DATA_PATH, TESTSET_CSV_PATH

NUM_TEST_SAMPLES = 10

print("Cargando documentos...")
loader = DirectoryLoader(DATA_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader)
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(docs)

selected_chunks = random.sample(chunks, min(len(chunks), NUM_TEST_SAMPLES))

llm = ChatOllama(model="llama3", temperature=0.7)

GENERATION_PROMPT = """Eres un experto creando exámenes para estudiantes de Tecnologías del Lenguaje Humano.

CONTEXTO:
{context}

REGLAS:
- La respuesta debe ser precisa y técnica
- NO mencionar "según el texto" ni "el fragmento dice"
- Usar sinónimos técnicos cuando sea posible
- Escribir en español

RESPONDE EXACTAMENTE EN ESTE FORMATO (sin texto adicional antes ni después):
PREGUNTA: [tu pregunta aquí]
RESPUESTA_IDEAL: [tu respuesta aquí]
"""

prompt = PromptTemplate(template=GENERATION_PROMPT, input_variables=["context"])
chain = prompt | llm

data = []
print(f"Generando {len(selected_chunks)} pares de evaluación...")

def parse_qa(text):
   # Extract question and answer from the generated text
    pregunta_match = re.search(
        r'PREGUNTA[:\s]+(.+?)(?=RESPUESTA[_\s]?IDEAL|RESPUESTA[:\s]|$)',
        text, re.DOTALL | re.IGNORECASE
    )
    respuesta_match = re.search(
        r'RESPUESTA[_\s]?IDEAL[:\s]+(.+?)$|RESPUESTA[:\s]+(.+?)$',
        text, re.DOTALL | re.IGNORECASE
    )
    
    question = ""
    if pregunta_match:
        question = pregunta_match.group(1).strip()
    
    ground_truth = ""
    if respuesta_match:
        ground_truth = (respuesta_match.group(1) or respuesta_match.group(2) or "").strip()
    
    question = re.sub(r'\*\*', '', question)
    ground_truth = re.sub(r'\*\*', '', ground_truth)
    
    # Remove residual prefixes
    question = re.sub(r'^(PREGUNTA|Pregunta)[:\s]+', '', question, flags=re.IGNORECASE)
    ground_truth = re.sub(r'^(RESPUESTA[_\s]?IDEAL|RESPUESTA|Respuesta)[:\s]+', '', ground_truth, flags=re.IGNORECASE)
    
    question = ' '.join(question.split())
    ground_truth = ' '.join(ground_truth.split())
    
    return question, ground_truth

MIN_QUESTION_LENGTH = 20

for chunk in tqdm(selected_chunks):
    try:
        response = chain.invoke({"context": chunk.page_content})
        text = response.content

        question, ground_truth = parse_qa(text)

        if len(question) >= MIN_QUESTION_LENGTH and len(ground_truth) >= 10:
            data.append({
                "question": question,
                "ground_truth": ground_truth,
                "evolution_type": "simple"
            })
        else:
            print(f"Descartado (muy corto): Q='{question[:50]}...' A='{ground_truth[:50]}...'")
            print(f"  Raw: {text[:200]}...")

    except Exception as e:
        print(f"Error: {e}")

df = pd.DataFrame(data)
df.to_csv(TESTSET_CSV_PATH, index=False)
print(f"\nDataset guardado: {TESTSET_CSV_PATH} ({len(data)} preguntas)")
print(df.head())