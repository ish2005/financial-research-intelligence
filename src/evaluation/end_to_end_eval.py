import json
from pathlib import Path

from src.retrieval.retriever import (
    load_vector_store,
)

from src.retrieval.bm25 import (
    build_bm25,
)

from src.retrieval.hybrid import (
    rrf_search,
)

from src.retrieval.reranker_pipeline import (
    load_reranker,
    rerank,
)

from src.retrieval.adjacent import (
    expand_adjacent_chunks,
)

from src.generation.answer_generator import (
    load_llm,
    generate_answer,
)

from src.generation.citation_validator import (
    validate_citations,
    load_citation_judge,
    validate_citation_support,
)


# ============================================================
# Configuration
# ============================================================

QUESTIONS_FILE = Path(
    "data/evaluation/evaluation_questions.json"
)

CANDIDATE_K = 20
TOP_K = 5
ADJACENT_WINDOW = 1


# ============================================================
# Helpers
# ============================================================

def load_questions():
    with open(
        QUESTIONS_FILE,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def get_retrieved_chunk_ids(results):
    return [
        result["chunk_id"]
        for result in results
    ]


def calculate_recall(results, gold_chunks):
    retrieved_ids = set(
        get_retrieved_chunk_ids(results)
    )

    gold_ids = set(gold_chunks)

    return bool(
        retrieved_ids.intersection(gold_ids)
    )


def get_best_gold_rank(results, gold_chunks):
    gold_chunks = set(gold_chunks)

    for rank, result in enumerate(
        results,
        start=1,
    ):
        if result["chunk_id"] in gold_chunks:
            return rank

    return None


def print_question_header(question):
    print("\n" + "=" * 100)
    print(f"Q{question['id']}")
    print("=" * 100)

    print(
        f"Question: {question['question']}"
    )

    print(
        f"Company:  {question['company']}"
    )

    print(
        f"Year:     {question['year']}"
    )

    print(
        f"Gold:     {question['gold_chunks']}"
    )


# ============================================================
# Main
# ============================================================

def main():

    questions = load_questions()

    # ========================================================
    # Load retrieval components
    # ========================================================

    print("=" * 100)
    print("LOADING RETRIEVAL PIPELINE")
    print("=" * 100)

    print("\nLoading vector store...")

    (
        vector_index,
        vector_metadata,
        vector_model,
    ) = load_vector_store()

    print("Vector store loaded.")

    print("\nBuilding BM25 index...")

    bm25, bm25_chunks = build_bm25()

    print("BM25 index built.")

    print("\nLoading cross-encoder reranker...")

    reranker = load_reranker()

    # ========================================================
    # Load generation components
    # ========================================================

    print("\nLoading answer-generation LLM...")

    llm = load_llm()

    print("\nLoading citation judge...")

    citation_judge = load_citation_judge()

    # ========================================================
    # Metrics
    # ========================================================

    retrieval_hits = 0
    expanded_context_hits = 0

    structural_valid_count = 0
    semantic_supported_count = 0

    answer_count = 0

    results_summary = []

    # ========================================================
    # Evaluate questions
    # ========================================================

    for question in questions:

        print_question_header(question)

        # ====================================================
        # STEP 1 — Hybrid RRF retrieval
        # ====================================================

        rrf_results = rrf_search(
            query=question["question"],
            vector_index=vector_index,
            vector_metadata=vector_metadata,
            vector_model=vector_model,
            bm25=bm25,
            bm25_chunks=bm25_chunks,
            company=question["company"],
            year=question["year"],
            candidate_k=CANDIDATE_K,
            top_k=CANDIDATE_K,
        )

        # ====================================================
        # STEP 2 — Cross-encoder reranking
        # ====================================================

        results = rerank(
            query=question["question"],
            results=rrf_results,
            model=reranker,
            top_k=TOP_K,
        )

        # ====================================================
        # STEP 3 — Raw Recall@5
        # ====================================================

        recall_hit = calculate_recall(
            results,
            question["gold_chunks"],
        )

        best_gold_rank = get_best_gold_rank(
            results,
            question["gold_chunks"],
        )

        if recall_hit:
            retrieval_hits += 1

        print(
            f"\nRaw Retrieval Recall@{TOP_K}: "
            f"{recall_hit}"
        )

        print(
            f"Best gold rank: "
            f"{best_gold_rank}"
        )

        # ====================================================
        # STEP 4 — Adjacent chunk expansion
        # ====================================================

        expanded_results = expand_adjacent_chunks(
            results=results,
            metadata=vector_metadata,
            window=ADJACENT_WINDOW,
        )

        expanded_context_hit = calculate_recall(
            expanded_results,
            question["gold_chunks"],
        )

        expanded_gold_rank = get_best_gold_rank(
            expanded_results,
            question["gold_chunks"],
        )

        if expanded_context_hit:
            expanded_context_hits += 1

        print(
            f"Expanded Context Hit: "
            f"{expanded_context_hit}"
        )

        print(
            f"Expanded Gold Rank: "
            f"{expanded_gold_rank}"
        )

        print(
            f"Evidence chunks supplied to LLM: "
            f"{len(expanded_results)}"
        )

        # ====================================================
        # Show added chunks
        # ====================================================

        original_ids = {
            result["chunk_id"]
            for result in results
        }

        added_chunks = [
            result["chunk_id"]
            for result in expanded_results
            if result["chunk_id"] not in original_ids
        ]

        if added_chunks:

            print("\nAdjacent chunks added:")

            for chunk_id in added_chunks:
                print(
                    f"  + {chunk_id}"
                )

        # ====================================================
        # STEP 5 — Generate answer
        # ====================================================

        try:

            answer = generate_answer(
                query=question["question"],
                results=expanded_results,
                llm=llm,
            )

            answer_count += 1

            print("\nAnswer:")
            print(answer.answer)

            print("\nGenerated citations:")

            for citation in answer.citations:

                print(
                    f"  {citation.chunk_id} | "
                    f"{citation.company} | "
                    f"{citation.year} | "
                    f"page {citation.page}"
                )

            print(
                f"\nConfidence: "
                f"{answer.confidence}"
            )

        except Exception as e:

            print(
                f"\nANSWER GENERATION FAILED: {e}"
            )

            results_summary.append(
                {
                    "id": question["id"],
                    "question": question["question"],
                    "retrieval_hit": recall_hit,
                    "gold_rank": best_gold_rank,
                    "expanded_context_hit": (
                        expanded_context_hit
                    ),
                    "expanded_gold_rank": (
                        expanded_gold_rank
                    ),
                    "answer_generated": False,
                    "structural_valid": False,
                    "semantic_supported": False,
                }
            )

            continue

        # ====================================================
        # STEP 6 — Structural citation validation
        # ====================================================

        structural_validation = validate_citations(
            answer=answer,
            retrieved_results=expanded_results,
        )

        structural_valid = (
            structural_validation.all_citations_valid
        )

        if structural_valid:
            structural_valid_count += 1

        print(
            f"\nStructural citation validation: "
            f"{structural_valid}"
        )

        # ====================================================
        # STEP 7 — Semantic citation validation
        # ====================================================

        semantic_validation = (
            validate_citation_support(
                answer=answer,
                retrieved_results=expanded_results,
                judge=citation_judge,
            )
        )

        semantic_supported = (
            semantic_validation.all_citations_supported
        )

        if semantic_supported:
            semantic_supported_count += 1

        print(
            f"Semantic citation support: "
            f"{semantic_supported}"
        )

        # ====================================================
        # Store result
        # ====================================================

        results_summary.append(
            {
                "id": question["id"],
                "question": question["question"],
                "retrieval_hit": recall_hit,
                "gold_rank": best_gold_rank,
                "expanded_context_hit": (
                    expanded_context_hit
                ),
                "expanded_gold_rank": (
                    expanded_gold_rank
                ),
                "answer_generated": True,
                "answer": answer.answer,
                "confidence": answer.confidence,
                "citations": [
                    citation.chunk_id
                    for citation in answer.citations
                ],
                "structural_valid": structural_valid,
                "semantic_supported": semantic_supported,
            }
        )

    # ========================================================
    # Final metrics
    # ========================================================

    total = len(questions)

    retrieval_recall = (
        retrieval_hits / total
        if total
        else 0
    )

    expanded_context_recall = (
        expanded_context_hits / total
        if total
        else 0
    )

    answer_generation_rate = (
        answer_count / total
        if total
        else 0
    )

    structural_rate = (
        structural_valid_count / total
        if total
        else 0
    )

    semantic_rate = (
        semantic_supported_count / total
        if total
        else 0
    )

    # ========================================================
    # Final summary
    # ========================================================

    print("\n\n")
    print("=" * 100)
    print("FINAL EVALUATION SUMMARY")
    print("=" * 100)

    print(
        f"\nTotal questions: "
        f"{total}"
    )

    print(
        f"\nRaw Retrieval Recall@{TOP_K}: "
        f"{retrieval_hits}/{total} "
        f"({retrieval_recall:.1%})"
    )

    print(
        f"\nExpanded Context Recall: "
        f"{expanded_context_hits}/{total} "
        f"({expanded_context_recall:.1%})"
    )

    print(
        f"\nAnswer generation: "
        f"{answer_count}/{total} "
        f"({answer_generation_rate:.1%})"
    )

    print(
        f"\nStructural citation validity: "
        f"{structural_valid_count}/{total} "
        f"({structural_rate:.1%})"
    )

    print(
        f"\nSemantic citation support: "
        f"{semantic_supported_count}/{total} "
        f"({semantic_rate:.1%})"
    )

    # ========================================================
    # Questions requiring attention
    # ========================================================

    print("\n" + "=" * 100)
    print("QUESTIONS REQUIRING ATTENTION")
    print("=" * 100)

    failures = []

    for result in results_summary:

        if (
            not result.get(
                "retrieval_hit",
                False,
            )
            or not result.get(
                "answer_generated",
                False,
            )
            or not result.get(
                "structural_valid",
                False,
            )
            or not result.get(
                "semantic_supported",
                False,
            )
        ):
            failures.append(result)

    if not failures:

        print("\nNo failures.")

    else:

        for result in failures:

            print(
                f"\nQ{result['id']}: "
                f"{result.get('question', '')}"
            )

            print(
                f"  Retrieval hit: "
                f"{result.get('retrieval_hit')}"
            )

            print(
                f"  Gold rank: "
                f"{result.get('gold_rank')}"
            )

            print(
                f"  Expanded context hit: "
                f"{result.get('expanded_context_hit')}"
            )

            print(
                f"  Expanded gold rank: "
                f"{result.get('expanded_gold_rank')}"
            )

            print(
                f"  Answer generated: "
                f"{result.get('answer_generated')}"
            )

            print(
                f"  Structural valid: "
                f"{result.get('structural_valid')}"
            )

            print(
                f"  Semantic supported: "
                f"{result.get('semantic_supported')}"
            )

    # ========================================================
    # Save results
    # ========================================================

    output_file = Path(
        "data/evaluation/end_to_end_results.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            {
                "total_questions": total,
                "retrieval_recall_at_5": (
                    retrieval_recall
                ),
                "expanded_context_recall": (
                    expanded_context_recall
                ),
                "answer_generation_rate": (
                    answer_generation_rate
                ),
                "structural_citation_rate": (
                    structural_rate
                ),
                "semantic_citation_rate": (
                    semantic_rate
                ),
                "results": results_summary,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"\nDetailed results saved to: "
        f"{output_file}"
    )


if __name__ == "__main__":
    main()