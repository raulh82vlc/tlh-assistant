-include .env

VENV_DIR ?= .venv
PYTHON ?= $(VENV_DIR)/bin/python
PIP ?= $(VENV_DIR)/bin/pip
MAIN_SCRIPT ?= TLH_assistant.py
PYTHONUNBUFFERED ?= 1

export PYTHONUNBUFFERED

.PHONY: help venv install feed run testset evaluate clean

help:
	@echo "Available targets:"
	@echo "  make venv      - Create local virtual environment (.venv)"
	@echo "  make install   - Install/update dependencies from requirements.txt"
	@echo "  make feed      - Build vector database from PDFs"
	@echo "  make run       - Run Streamlit assistant"
	@echo "  make testset   - Generate evaluation testset"
	@echo "  make evaluate  - Run RAGAS evaluation"
	@echo "  make clean     - Remove generated artifacts"

venv:
	python3 -m venv $(VENV_DIR)
	$(PIP) install --upgrade pip

install: venv
	$(PIP) install -r requirements.txt

feed:
	$(PYTHON) feed_db_docs.py

run:
	$(PYTHON) -m streamlit run $(MAIN_SCRIPT)

testset:
	$(PYTHON) generate_testset.py

evaluate:
	$(PYTHON) evaluate_ragas.py

clean:
	rm -rf __pycache__
	rm -rf qdrant_db
	rm -f rag_testset.csv
	rm -f ragas_full_report.csv
