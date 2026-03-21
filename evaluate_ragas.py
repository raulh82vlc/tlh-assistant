# Copyright (c) 2025 Raul Hernandez Lopez
#
# This file is part of the project and is licensed under the
# Creative Commons Attribution-ShareAlike 4.0 International License (CC BY-SA 4.0).
#
# You are free to share and adapt this file under the terms of the CC BY-SA 4.0 license.
# Full license: https://creativecommons.org/licenses/by-sa/4.0/legalcode

import pandas as pd
from rag_manager import run_rag
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate, EvaluationDataset
from ragas.metrics import (
    LLMContextRecall,
    LLMContextPrecisionWithoutReference,
    Faithfulness,
    FactualCorrectness,
    ResponseRelevancy,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from constants import TESTSET_CSV_PATH, RAGAS_REPORT_CSV_PATH

judge_llm = ChatOllama(model="llama3", temperature=0)
evaluator_llm = LangchainLLMWrapper(judge_llm)

hf_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)
evaluator_embeddings = LangchainEmbeddingsWrapper(hf_embeddings)


def run_evaluation():
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
        q = row['question']
        gt = row['ground_truth']
        print(f"({index+1}/{len(test_df)}): {q[:60]}...")

        response = chain.invoke({"query": q})
        results.append({
            "user_input": q,
            "response": response['result'],
            "retrieved_contexts": [doc.page_content for doc in response['source_documents']],
            "reference": gt
        })

    evaluation_dataset = EvaluationDataset.from_list(results)
    print("Calculando métricas...")

    metrics = [
        Faithfulness(),
        ResponseRelevancy(embeddings=evaluator_embeddings),
        LLMContextPrecisionWithoutReference(),
        LLMContextRecall(),
        FactualCorrectness(),
    ]

    scores = evaluate(
        dataset=evaluation_dataset,
        metrics=metrics,
        llm=evaluator_llm,
    )

    df_scores = scores.to_pandas()
    df_scores.to_csv(RAGAS_REPORT_CSV_PATH, index=False)
    print("\nResultados:")
    print(df_scores[['user_input', 'factual_correctness', 'faithfulness', 'context_recall']].head())
    print(f"Reporte guardado en {RAGAS_REPORT_CSV_PATH}")


if __name__ == "__main__":
    run_evaluation()