# Copyright (c) 2025 Raul Hernandez Lopez
#
# This file is part of the project and is licensed under the
# Creative Commons Attribution-ShareAlike 4.0 International License (CC BY-SA 4.0).
#
# You are free to share and adapt this file under the terms of the CC BY-SA 4.0 license.
# Full license: https://creativecommons.org/licenses/by-sa/4.0/legalcode

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = BASE_DIR / "data"
DB_PATH = BASE_DIR / "qdrant_db"
COLLECTION_NAME = "tlh_rag"

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
OLLAMA_MODEL_NAME = "llama3"

TESTSET_CSV_PATH = BASE_DIR / "rag_testset.csv"
RAGAS_REPORT_CSV_PATH = BASE_DIR / "ragas_full_report.csv"
