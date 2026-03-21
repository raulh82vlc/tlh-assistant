PYTHON := .venv/bin/python
PIP := .venv/bin/pip

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
	python3 -m venv .venv
	$(PIP) install --upgrade pip

install: venv
	$(PIP) install -r requirements.txt

feed:
	$(PYTHON) feed_db_docs.py

run:
	$(PYTHON) -m streamlit run TLH_assistant.py

testset:
	$(PYTHON) generate_testset.py

evaluate:
	$(PYTHON) evaluate_ragas.py

clean:
	rm -rf __pycache__
	rm -rf qdrant_db
	rm -f rag_testset.csv
	rm -f ragas_full_report.csv
