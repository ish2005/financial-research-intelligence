import json
from pathlib import Path

from src.retrieval.retriever import load_vector_store, search
from src.retrieval.bm25 import build_bm25, search_bm25
from src.retrieval.hybrid import rrf_search


QUESTIONS_FILE = Path("data/evaluation/evaluation_questions.json")
TOP_K = 20


def load_questions():
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def get_chunk_ids(results):
    return {result["chunk_id"] for result in results}


def main():
    print("Loading retrieval systems...")

    vector_index, vector_metadata, vector_model = load_vector_store()
    bm25, bm25_chunks = build_bm25()

    questions = load_questions()

    print(f"Loaded {len(questions)} questions.")

    dense_hits = 0
    bm25_hits = 0
    rrf_hits = 0

    misses = []

    for i, question in enumerate(questions, start=1):
        query = question["question"]
        company = question.get("company")
        year = question.get("year")
        gold_chunks = set(question["gold_chunks"])

        print(f"[{i:02d}/{len(questions)}] {query}")

        # --------------------------------------------------
        # Dense
        # --------------------------------------------------

        dense_results = search(
            query=query,
            index=vector_index,
            metadata=vector_metadata,
            model=vector_model,
            top_k=TOP_K,
        )

        dense_results = [
            r for r in dense_results
            if r.get("company", "").lower() == company.lower()
            and int(r.get("year")) == int(year)
        ]

        dense_ids = get_chunk_ids(dense_results)

        # --------------------------------------------------
        # BM25
        # --------------------------------------------------

        bm25_results = search_bm25(
            query=query,
            bm25=bm25,
            chunks=bm25_chunks,
            top_k=TOP_K,
        )

        bm25_results = [
            r for r in bm25_results
            if r.get("company", "").lower() == company.lower()
            and int(r.get("year")) == int(year)
        ]

        bm25_ids = get_chunk_ids(bm25_results)

        # --------------------------------------------------
        # RRF
        # --------------------------------------------------

        rrf_results = rrf_search(
            query=query,
            vector_index=vector_index,
            vector_metadata=vector_metadata,
            vector_model=vector_model,
            bm25=bm25,
            bm25_chunks=bm25_chunks,
            company=company,
            year=year,
            top_k=TOP_K,
            candidate_k=TOP_K,
        )

        rrf_ids = get_chunk_ids(rrf_results)

        # --------------------------------------------------
        # Recall
        # --------------------------------------------------

        dense_hit = bool(gold_chunks & dense_ids)
        bm25_hit = bool(gold_chunks & bm25_ids)
        rrf_hit = bool(gold_chunks & rrf_ids)

        dense_hits += dense_hit
        bm25_hits += bm25_hit
        rrf_hits += rrf_hit

        if not rrf_hit:
            misses.append(
                {
                    "id": question["id"],
                    "question": query,
                    "company": company,
                    "year": year,
                    "gold_chunks": list(gold_chunks),
                    "dense_results": dense_results,
                    "bm25_results": bm25_results,
                    "rrf_results": rrf_results,
                }
            )

    # ------------------------------------------------------
    # Summary
    # ------------------------------------------------------

    total = len(questions)

    print("\n")
    print("=" * 70)
    print(f"CANDIDATE RECALL@{TOP_K}")
    print("=" * 70)

    print(f"DENSE    {dense_hits}/{total} ({dense_hits / total:.3f})")
    print(f"BM25     {bm25_hits}/{total} ({bm25_hits / total:.3f})")
    print(f"RRF      {rrf_hits}/{total} ({rrf_hits / total:.3f})")

    # ------------------------------------------------------
    # RRF misses
    # ------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("RRF MISSES")
    print("=" * 70)

    if not misses:
        print("No RRF misses.")
        return

    for miss in misses:
        print("\n" + "-" * 70)
        print(f"Q{miss['id']}: {miss['question']}")
        print(f"Company: {miss['company']}")
        print(f"Year: {miss['year']}")
        print(f"Gold: {miss['gold_chunks']}")

        print("\nTop RRF results:")

        for rank, result in enumerate(miss["rrf_results"], start=1):
            print(
                f"\n{rank}. "
                f"{result['chunk_id']} | "
                f"page {result.get('page')} | "
                f"RRF={result.get('rrf_score', 0):.6f}"
            )

            print(
                f"Dense rank: {result.get('dense_rank')} | "
                f"BM25 rank: {result.get('bm25_rank')}"
            )

            print(result["text"][:500].replace("\n", " "))

        print("\nGold chunk details:")

        for gold_id in miss["gold_chunks"]:
            gold = next(
                (
                    r
                    for r in vector_metadata
                    if r["chunk_id"] == gold_id
                ),
                None,
            )

            if gold:
                print(
                    f"\n{gold_id} | "
                    f"page {gold.get('page')}"
                )
                print(gold["text"][:1000].replace("\n", " "))


if __name__ == "__main__":
    main()