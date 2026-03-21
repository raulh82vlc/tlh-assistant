# RAG System - TLH Assistant
```
@Author: Raul Hernandez Lopez.
```
[raulh82vlc's GitHub](https://github.com/raulh82vlc)

Retrieval-Augmented Generation (RAG) system for the TLH assistant. [TLH demo](https://www.youtube.com/watch?v=AVxlBk3FwV0)

## Directory Structure

```
RAG/
├── data/                      # Source PDFs (documents to vectorize)
├── qdrant_db/                 # Qdrant vector database
├── src/
│   └── rag_system/
│       ├── settings.py        # Shared paths and model settings
│       ├── embeddings.py      # Embedding builder and device resolution
│       ├── ingestion.py       # Offline PDF vectorization logic
│       ├── rag_manager.py     # Retrieval + generation orchestration
│       ├── streamlit_app.py   # Streamlit UI implementation
│       ├── testset_generation.py
│       └── ragas_evaluation.py
├── feed_db_docs.py            # CLI wrapper (offline vectorization)
├── TLH_assistant.py           # CLI wrapper (Streamlit app)
├── rag_manager.py             # Compatibility wrapper
├── generate_testset.py        # CLI wrapper (testset generation)
└── evaluate_ragas.py          # CLI wrapper (RAGAS evaluation)
```

## Main Scripts

### `feed_db_docs.py` - MOST IMPORTANT STEP
Offline document vectorization process.

This script:
- Reads PDFs from the `data/` folder
- Processes and splits documents into chunks
- Generates vector embeddings
- Stores vectors in the Qdrant database

Run this first before using the RAG system.

### `TLH_assistant.py`
Assistant interface built with Streamlit.
> make run

### `rag_manager.py`
Main module that handles:
- Connection to the vector database
- Similar document retrieval
- Integration with the LLM model
- Response generation

### `generate_testset.py`
Generates the evaluation test dataset (testset).

Run this before evaluating with RAGAS.

### `evaluate_ragas.py`
Evaluates RAG system performance using the RAGAS framework.

Prerequisite: Generate the testset first with `make testset`

## Workflow

1. Vectorization (Offline)
   - Place PDFs in `data/`
   - Run `make feed`

2. Assistant Usage (Online on localhost)
   - Run `make run`

3. Evaluation (Optional)
   - Generate testset: `make testset`
   - Evaluate: `make evaluate`

## Prerequisites

- TLH PDF documents in the `data/` folder
- GNU Make installed
- Python 3 available as `python3`

### Makefile hints (recommended)

```bash
make install   # create .venv and install dependencies
make feed      # build vector database from PDFs
make run       # start Streamlit assistant
```

### Optional local `.env` configuration

You can customize the Makefile behavior by creating a local `.env` file:

```bash
cp .env.example .env
```

Example variables:

```bash
PYTHONUNBUFFERED=1
VENV_DIR=.venv
MAIN_SCRIPT=TLH_assistant.py
```

- `PYTHONUNBUFFERED=1`: forces immediate Python log flushing.
- `VENV_DIR`: virtual environment folder used by `make venv/install/run/...`.
- `MAIN_SCRIPT`: Streamlit entry file used by `make run`.

Other useful commands:

```bash
make testset   # generate evaluation dataset
make evaluate  # run RAGAS evaluation
make clean     # remove generated artifacts
```

- Ollama installed with the `llama3` model. Steps after installation:
   - `ollama pull llama3` - Download the llama3 model
   - `ollama list` - Verify the model is installed
   - `ollama run llama3` - Test the model interactively
   - Note: The Ollama server must be running
- Qdrant database (created automatically when running `feed_db_docs.py`)

---

Note: The most critical part of the system is offline document vectorization via `feed_db_docs.py`. Without this step, the vector database will be empty and RAG will not work.

## License

This project is licensed under the **Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)**.

See [LICENSE](LICENSE) for the full text and official links.
