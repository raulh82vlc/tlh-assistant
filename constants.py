# Copyright (c) 2025 Raul Hernandez Lopez
#
# This file is part of the project and is licensed under the
# Creative Commons Attribution-ShareAlike 4.0 International License (CC BY-SA 4.0).
#
# You are free to share and adapt this file under the terms of the CC BY-SA 4.0 license.
# Full license: https://creativecommons.org/licenses/by-sa/4.0/legalcode

from src.rag_system.settings import (
	BASE_DIR,
	COLLECTION_NAME,
	DATA_PATH,
	DB_PATH,
	RAGAS_REPORT_CSV_PATH,
	TESTSET_CSV_PATH,
)

# Backward-compatible string exports for legacy imports.
DATA_PATH = str(DATA_PATH)
DB_PATH = str(DB_PATH)
TESTSET_CSV_PATH = str(TESTSET_CSV_PATH)
RAGAS_REPORT_CSV_PATH = str(RAGAS_REPORT_CSV_PATH)
