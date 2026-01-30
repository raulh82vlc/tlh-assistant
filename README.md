<!--
  Copyright (c) 2025 Raul Hernandez Lopez

  This file is part of the project and is licensed under the
  Creative Commons Attribution-ShareAlike 4.0 International License (CC BY-SA 4.0).

  You are free to share and adapt this file under the terms of the CC BY-SA 4.0 license.
  Full license: https://creativecommons.org/licenses/by-sa/4.0/legalcode
-->

# RAG System - TLH Assistant

Sistema de Recuperación Aumentada por Generación (RAG) para el asistente TLH. [TLH demo](https://www.youtube.com/watch?v=AVxlBk3FwV0)

## Estructura de Directorios

```
RAG/
├── data/                    # PDFs fuente (documentos a vectorizar)
├── qdrant_db/              # Base de datos vectorial Qdrant
│   └── collection/
│       └── tlh_rag/
├── feed_db_docs.py         # PROCESO OFFLINE - Vectorización de PDFs
├── TLH_assistant.py        # Interfaz Streamlit del asistente
├── rag_manager.py          # Gestión del sistema RAG
├── generate_testset.py     # Generación del dataset de pruebas
└── evaluate_ragas.py       # Evaluación con RAGAS
```

## Scripts Principales

### `feed_db_docs.py` - PASO MÁS IMPORTANTE
Proceso offline de vectorización de documentos.

Este script:
- Lee los PDFs de la carpeta `data/`
- Procesa y divide los documentos en chunks
- Genera embeddings vectoriales
- Almacena los vectores en la base de datos Qdrant

Ejecutar primero antes de usar el sistema RAG.

### `TLH_assistant.py`
Interfaz del asistente con Streamlit.
> streamlit run TLH_assistant.py

### `rag_manager.py`
Módulo principal que gestiona:
- Conexión con la base de datos vectorial
- Búsqueda de documentos similares
- Integración con el modelo LLM
- Generación de respuestas

### `generate_testset.py`
Genera el dataset de pruebas (testset) para evaluación.

Ejecutar antes de evaluar con RAGAS.

### `evaluate_ragas.py`
Evalúa el rendimiento del sistema RAG utilizando el framework RAGAS.

Prerrequisito: Generar el testset primero con `generate_testset.py`

## Flujo de Trabajo

1. Vectorización (Offline)
   - Colocar PDFs en `data/`
   - Ejecutar `feed_db_docs.py`

2. Uso del Asistente (Online en localhost)
   - Ejecutar `TLH_assistant.py` con Streamlit

3. Evaluación (Opcional)
   - Generar testset: `generate_testset.py`
   - Evaluar: `evaluate_ragas.py`

## Requisitos Previos

- Documentos PDF de TLH en la carpeta `data/`
- Dependencias Python instaladas, se necesitan las siguientes dependencias:
```bash
pip install torch
pip install langchain
pip install langchain-community
pip install langchain-huggingface
pip install langchain-ollama
pip install langchain-qdrant
pip install qdrant-client
pip install sentence-transformers
pip install pypdf
pip install streamlit
pip install pandas
pip install ragas
pip install tqdm
```

- Ollama instalado con el modelo `llama3`, pasos una vez instalado:
    - `ollama pull llama3` - Descarga el modelo llama3
    - `ollama list` - Verifica que el modelo está instalado
    - `ollama run llama3` - Prueba el modelo interactivamente
    - Nota: El servidor Ollama debe estar ejecutándose
- Base de datos Qdrant (se crea automáticamente al ejecutar `feed_db_docs.py`)

---

Nota: La parte más crítica del sistema es la vectorización offline de documentos mediante `feed_db_docs.py`, ya que sin este paso la base de datos vectorial estará vacía y el RAG no funcionará.
