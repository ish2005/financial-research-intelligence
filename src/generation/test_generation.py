from src.retrieval.reranker_pipeline import (
    load_pipeline,
    search,
)

from src.generation.answer_generator import (
    load_llm,
    generate_answer,
)

from src.generation.citation_validator import (
    validate_citations,
    print_validation,
    load_citation_judge,
    validate_citation_support,
    print_semantic_validation,
)

QUERY = "What was Apple's total net sales in 2023?"
COMPANY = "Apple"
YEAR = 2023

CANDIDATE_K = 20
TOP_K = 5


def main():

    # --------------------------------------------------
    # Load retrieval + reranking pipeline
    # --------------------------------------------------

    print("Loading retrieval pipeline...")

    (
        vector_index,
        vector_metadata,
        vector_model,
        bm25,
        bm25_chunks,
        reranker,
    ) = load_pipeline()

    # --------------------------------------------------
    # Load LangChain Groq LLM
    # --------------------------------------------------

    print("Loading LangChain Groq...")

    llm = load_llm()

    # --------------------------------------------------
    # Retrieval + Reranking
    # --------------------------------------------------

    print("\nRunning retrieval + reranking...")

    results = search(
        query=QUERY,
        vector_index=vector_index,
        vector_metadata=vector_metadata,
        vector_model=vector_model,
        bm25=bm25,
        bm25_chunks=bm25_chunks,
        reranker=reranker,
        company=COMPANY,
        year=YEAR,
        candidate_k=CANDIDATE_K,
        top_k=TOP_K,
    )

    print("\nTop evidence:")

    for rank, result in enumerate(results, start=1):

        print(
            f"\n--- Evidence {rank} ---"
        )

        print(
            f"Chunk: {result['chunk_id']}"
        )

        print(
            f"Page: {result['page']}"
        )

        print(
            f"Reranker score: "
            f"{result['reranker_score']:.4f}"
        )

        print(
            f"Text:\n"
            f"{result['text'][:800]}"
        )

    # --------------------------------------------------
    # Generate answer
    # --------------------------------------------------

    print("\nGenerating answer...")

    answer = generate_answer(
        query=QUERY,
        results=results,
        llm=llm,
    )

    # --------------------------------------------------
    # Display structured result
    # --------------------------------------------------

    print("\n" + "=" * 80)
    print("FINAL ANSWER")
    print("=" * 80)

    print(answer.answer)

    print("\n" + "=" * 80)
    print("CITATIONS")
    print("=" * 80)

    for citation in answer.citations:

        print(
            f"{citation.chunk_id} | "
            f"{citation.company} | "
            f"{citation.year} | "
            f"page {citation.page}"
        )

        print("\n" + "=" * 80)
    print("CONFIDENCE")
    print("=" * 80)
    print(answer.confidence)

    validation = validate_citations(
        answer=answer,
        retrieved_results=results,
    )

    print_validation(validation)

    print("\nLoading citation judge...")

    citation_judge = load_citation_judge()

    semantic_validation = validate_citation_support(
        answer=answer,
        retrieved_results=results,
        judge=citation_judge,
    )

    print_semantic_validation(
        semantic_validation
    )

if __name__ == "__main__":
    main()