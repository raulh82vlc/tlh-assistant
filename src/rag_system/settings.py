from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = BASE_DIR / "data"
DB_PATH = BASE_DIR / "qdrant_db"
COLLECTION_NAME = "tlh_rag"

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
OLLAMA_MODEL_NAME = "llama3"

TESTSET_CSV_PATH = BASE_DIR / "rag_testset.csv"
RAGAS_REPORT_CSV_PATH = BASE_DIR / "ragas_full_report.csv"
