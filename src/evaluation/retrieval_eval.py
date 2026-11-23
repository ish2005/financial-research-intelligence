from pathlib import Path
import json

from src.retrieval.retriever import load_vector_store, search
from src.retrieval.bm25 import build_bm25, search_bm25
from src.retrieval.hybrid import rrf_search
from src.retrieval.reranker import load_reranker, rerank


QUESTIONS_FILE = Path("data/evaluation/evaluation_questions.json")

TOP_K = 5
CANDIDATE_K = 20


def load_questions():
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def is_relevant(result, question):
    gold_chunks = question["gold_chunks"]
    return result["chunk_id"] in gold_chunks


def evaluate_results(results, question, top_k=TOP_K):
    results = results[:top_k]

    relevant_ranks = []

    for rank, result in enumerate(results, start=1):
        if is_relevant(result, question):
            relevant_ranks.append(rank)

    # Recall@K:
    # Did we retrieve at least one acceptable gold chunk?
    recall = 1.0 if relevant_ranks else 0.0

    # Precision@K:
    # How many retrieved chunks were gold chunks?
    precision = len(relevant_ranks) / top_k

    # MRR:
    # Rank of the first relevant chunk.
    mrr = (
        1.0 / relevant_ranks[0]
        if relevant_ranks
        else 0.0
    )

    return {
        "recall": recall,
        "precision": precision,
        "mrr": mrr,
    }


def main():

    print("Loading retrieval systems...")

    vector_index, vector_metadata, vector_model = load_vector_store()
    bm25, bm25_chunks = build_bm25()
    reranker = load_reranker()

    questions = load_questions()

    print(f"Loaded {len(questions)} evaluation questions.")

    results = {
        "dense": [],
        "bm25": [],
        "rrf": [],
        "rrf_reranker": [],
    }

    for i, question in enumerate(questions, start=1):

        query = question["question"]

        print(
            f"\n[{i}/{len(questions)}] "
            f"{query}"
        )

        # --------------------------------------------------
        # 1. Dense retrieval
        # --------------------------------------------------

        dense_results = search(
            query=query,
            index=vector_index,
            metadata=vector_metadata,
            model=vector_model,
            top_k=TOP_K,
        )

        results["dense"].append(
            evaluate_results(
                dense_results,
                question,
            )
        )

        # --------------------------------------------------
        # 2. BM25
        # --------------------------------------------------

        bm25_results = search_bm25(
            query=query,
            bm25=bm25,
            chunks=bm25_chunks,
            top_k=TOP_K,
        )

        results["bm25"].append(
            evaluate_results(
                bm25_results,
                question,
            )
        )

        # --------------------------------------------------
        # 3. RRF hybrid
        # --------------------------------------------------

        rrf_results = rrf_search(
            query=query,
            vector_index=vector_index,
            vector_metadata=vector_metadata,
            vector_model=vector_model,
            bm25=bm25,
            bm25_chunks=bm25_chunks,
            top_k=TOP_K,
            candidate_k=CANDIDATE_K,
        )

        results["rrf"].append(
            evaluate_results(
                rrf_results,
                question,
            )
        )

        # --------------------------------------------------
        # 4. RRF + Cross Encoder reranker
        # --------------------------------------------------

        rrf_candidates = rrf_search(
            query=query,
            vector_index=vector_index,
            vector_metadata=vector_metadata,
            vector_model=vector_model,
            bm25=bm25,
            bm25_chunks=bm25_chunks,
            top_k=CANDIDATE_K,
            candidate_k=CANDIDATE_K,
        )

        reranked_results = rerank(
            query=query,
            results=rrf_candidates,
            model=reranker,
            top_k=TOP_K,
        )

        results["rrf_reranker"].append(
            evaluate_results(
                reranked_results,
                question,
            )
        )

    # --------------------------------------------------
    # Final evaluation
    # --------------------------------------------------

    print("\n")
    print("=" * 70)
    print("RETRIEVAL EVALUATION")
    print("=" * 70)

    for method, metrics in results.items():

        recall = (
            sum(x["recall"] for x in metrics)
            / len(metrics)
        )

        precision = (
            sum(x["precision"] for x in metrics)
            / len(metrics)
        )

        mrr = (
            sum(x["mrr"] for x in metrics)
            / len(metrics)
        )

        print(f"\n{method.upper()}")

        print(f"Recall@5:    {recall:.3f}")
        print(f"Precision@5: {precision:.3f}")
        print(f"MRR:         {mrr:.3f}")


if __name__ == "__main__":
    main()