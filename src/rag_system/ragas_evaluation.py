# Copyright (c) 2025 Raul Hernandez Lopez
#
# This file is part of the project and is licensed under the
# Creative Commons Attribution-ShareAlike 4.0 International License (CC BY-SA 4.0).
#
# You are free to share and adapt this file under the terms of the CC BY-SA 4.0 license.
# Full license: https://creativecommons.org/licenses/by-sa/4.0/legalcode

import pandas as pd
from langchain_ollama import ChatOllama
from ragas import EvaluationDataset, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    FactualCorrectness,
    LLMContextPrecisionWithoutReference,
    LLMContextRecall,
    ResponseRelevancy,
)

from src.rag_system.embeddings import build_embeddings
from src.rag_system.rag_manager import run_rag
from src.rag_system.settings import OLLAMA_MODEL_NAME, RAGAS_REPORT_CSV_PATH, TESTSET_CSV_PATH


def run_evaluation():
    judge_llm = ChatOllama(model=OLLAMA_MODEL_NAME, temperature=0)
    evaluator_llm = LangchainLLMWrapper(judge_llm)

    evaluator_embeddings = LangchainEmbeddingsWrapper(build_embeddings())

    print("Cargando dataset de prueba...")
    try:
        test_df = pd.read_csv(TESTSET_CSV_PATH)
    except FileNotFoundError:
        print(f"No encuentro '{TESTSET_CSV_PATH}'. Ejecuta primero generate_testset.py")
        return

    chain = run_rag()
    results = []
    print(f"Evaluando {len(test_df)} casos...")

    for index, row in test_df.iterrows():
        question = row["question"]
        ground_truth = row["ground_truth"]
        print(f"({index + 1}/{len(test_df)}): {question[:60]}...")

        response = chain.invoke({"query": question})
        results.append(
            {
                "user_input": question,
                "response": response["result"],
                "retrieved_contexts": [doc.page_content for doc in response["source_documents"]],
                "reference": ground_truth,
            }
        )

    evaluation_dataset = EvaluationDataset.from_list(results)
    print("Calculando métricas...")

    metrics = [
        Faithfulness(),
        ResponseRelevancy(embeddings=evaluator_embeddings),
        LLMContextPrecisionWithoutReference(),
        LLMContextRecall(),
        FactualCorrectness(),
    ]

    scores = evaluate(dataset=evaluation_dataset, metrics=metrics, llm=evaluator_llm)

    df_scores = scores.to_pandas()
    df_scores.to_csv(RAGAS_REPORT_CSV_PATH, index=False)
    print("\nResultados:")

    # RAGAS can version metric columns with suffixes, e.g. "factual_correctness(mode=f1)".
    # Build a robust preview with whichever metric names are available.
    available_columns = list(df_scores.columns)

    def find_first(prefix: str):
        for column in available_columns:
            if column == prefix or column.startswith(f"{prefix}("):
                return column
        return None

    preview_columns = ["user_input"]
    for metric_prefix in ["factual_correctness", "faithfulness", "context_recall"]:
        metric_column = find_first(metric_prefix)
        if metric_column is not None:
            preview_columns.append(metric_column)

    print(df_scores[preview_columns].head())
    print(f"Reporte guardado en {RAGAS_REPORT_CSV_PATH}")
