import json
from pathlib import Path

from src.retrieval.retriever import load_vector_store
from src.retrieval.bm25 import build_bm25
from src.retrieval.hybrid import rrf_search


QUESTIONS_FILE = Path("data/evaluation/evaluation_questions.json")
CANDIDATE_K = 20


def load_questions():
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    print("Loading retrieval systems...")

    vector_index, vector_metadata, vector_model = load_vector_store()
    bm25, bm25_chunks = build_bm25()

    questions = load_questions()

    print("\nRRF candidate misses:")
    print("=" * 70)

    for question in questions:

        results = rrf_search(
            query=question["question"],
            vector_index=vector_index,
            vector_metadata=vector_metadata,
            vector_model=vector_model,
            bm25=bm25,
            bm25_chunks=bm25_chunks,
            top_k=CANDIDATE_K,
            candidate_k=CANDIDATE_K,
        )

        retrieved_ids = {
            result["chunk_id"]
            for result in results
        }

        gold_chunks = set(question["gold_chunks"])

        if not retrieved_ids.intersection(gold_chunks):
            print(
                f"\nQ{question['id']}: "
                f"{question['question']}"
            )

            print(
                "Gold:",
                ", ".join(question["gold_chunks"])
            )


if __name__ == "__main__":
    main()