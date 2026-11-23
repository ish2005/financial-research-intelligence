from src.retrieval.retriever import load_vector_store
from src.retrieval.bm25 import build_bm25
from src.retrieval.hybrid import rrf_search
from src.retrieval.reranker_pipeline import load_reranker, rerank
from src.evaluation.inspect_questions import load_questions


MISS_IDS = {4, 9, 10, 11, 12, 13, 15, 20}

CANDIDATE_K = 20


def main():
    print("=" * 100)
    print("DIAGNOSING RETRIEVAL MISSES")
    print("=" * 100)

    # --------------------------------------------------
    # Load retrieval pipeline
    # --------------------------------------------------

    print("\nLoading vector store...")
    vector_index, vector_metadata, vector_model = load_vector_store()

    print("Loading BM25...")
    bm25, bm25_chunks = build_bm25()

    print("Loading reranker...")
    reranker = load_reranker()

    questions = load_questions()

    # --------------------------------------------------
    # Diagnose each miss
    # --------------------------------------------------

    for i, q in enumerate(questions, start=1):

        if i not in MISS_IDS:
            continue

        query = q["question"]
        company = q.get("company")
        year = q.get("year")
        gold_chunks = q.get("gold_chunks", [])

        print("\n" + "=" * 100)
        print(f"Q{i}")
        print("=" * 100)

        print(f"Question: {query}")
        print(f"Company:  {company}")
        print(f"Year:     {year}")
        print(f"Gold:     {gold_chunks}")

        # --------------------------------------------------
        # Hybrid RRF: get 20 candidates
        # --------------------------------------------------

        candidates = rrf_search(
            query=query,
            vector_index=vector_index,
            vector_metadata=vector_metadata,
            vector_model=vector_model,
            bm25=bm25,
            bm25_chunks=bm25_chunks,
            company=company,
            year=year,
            top_k=CANDIDATE_K,
            candidate_k=CANDIDATE_K,
        )

        print(f"\nRRF candidates: {len(candidates)}")

        # Check whether gold is already missing before reranking
        rrf_gold_rank = None

        for rank, result in enumerate(candidates, start=1):
            if result["chunk_id"] in gold_chunks:
                rrf_gold_rank = rank
                break

        if rrf_gold_rank is None:
            print("RRF GOLD: NOT FOUND IN TOP 20")
        else:
            print(f"RRF GOLD: FOUND AT RANK {rrf_gold_rank}")

        # --------------------------------------------------
        # Rerank the 20 candidates
        # --------------------------------------------------

        reranked = rerank(
            query=query,
            results=candidates,
            top_k=CANDIDATE_K,
            model=reranker,
        )

        print("\nRERANKED TOP 20")
        print("-" * 100)

        rerank_gold_rank = None

        for rank, result in enumerate(reranked, start=1):

            chunk_id = result["chunk_id"]

            if chunk_id in gold_chunks:
                rerank_gold_rank = rank
                marker = "  <<< GOLD"
            else:
                marker = ""

            score = result.get(
                "rerank_score",
                result.get("score", 0.0)
            )

            print(
                f"{rank:2d}. "
                f"{chunk_id} | "
                f"page={result.get('page')} | "
                f"score={score:.4f}"
                f"{marker}"
            )

            text = result.get("text", "").replace("\n", " ")
            print(f"    {text[:250]}")

        if rerank_gold_rank is None:
            print("\nRERANK GOLD: NOT FOUND IN TOP 20")
        else:
            print(f"\nRERANK GOLD: FOUND AT RANK {rerank_gold_rank}")

        # --------------------------------------------------
        # Diagnosis
        # --------------------------------------------------

        print("\nDIAGNOSIS")

        if rrf_gold_rank is None:
            print("❌ Retrieval/RRF problem:")
            print("   Gold chunk never entered the RRF candidate pool.")

        elif rerank_gold_rank is None:
            print("❌ Reranker problem:")
            print("   Gold chunk entered RRF but disappeared during reranking.")

        elif rerank_gold_rank > 5:
            print("⚠️ Reranker ranking problem:")
            print(f"   Gold chunk is at reranked position {rerank_gold_rank}.")

        else:
            print("✅ Gold chunk is inside final top 5.")


if __name__ == "__main__":
    main()