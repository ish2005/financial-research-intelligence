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
    load_citation_judge,
    validate_citations,
    validate_citation_support,
)


CANDIDATE_K = 20
RERANK_TOP_K = 5

ADJACENT_WINDOW = 1
MAX_EVIDENCE = 8


class FinancialRAGPipeline:

    def __init__(
        self,
        vector_index=None,
        vector_metadata=None,
        vector_model=None,
        bm25=None,
        bm25_chunks=None,
    ):

        # ====================================================
        # Static development corpus
        # ====================================================

        if vector_index is None:

            print(
                "Loading vector store..."
            )

            (
                self.vector_index,
                self.vector_metadata,
                self.vector_model,
            ) = load_vector_store()

        else:

            self.vector_index = (
                vector_index
            )

            self.vector_metadata = (
                vector_metadata
            )

            self.vector_model = (
                vector_model
            )

        # ====================================================
        # BM25
        # ====================================================

        if bm25 is None:

            print(
                "Building BM25 index..."
            )

            (
                self.bm25,
                self.bm25_chunks,
            ) = build_bm25()

        else:

            self.bm25 = bm25

            self.bm25_chunks = (
                bm25_chunks
            )

        # ====================================================
        # Models
        # ====================================================

        print(
            "Loading reranker..."
        )

        self.reranker = load_reranker()

        print(
            "Loading LLM..."
        )

        self.llm = load_llm()

        print(
            "Loading citation judge..."
        )

        self.citation_judge = (
            load_citation_judge()
        )

        self.last_citation_validation = None

        self.last_semantic_validation = None

        print(
            "Pipeline ready."
        )

    # ========================================================
    # Create a pipeline using already-loaded models
    #
    # This prevents duplicate BGE/reranker/LLM loading.
    # ========================================================

    @classmethod
    def from_existing_models(
        cls,
        vector_index,
        vector_metadata,
        vector_model,
        bm25,
        bm25_chunks,
        base_pipeline,
    ):

        pipeline = cls.__new__(cls)

        pipeline.vector_index = (
            vector_index
        )

        pipeline.vector_metadata = (
            vector_metadata
        )

        pipeline.vector_model = (
            vector_model
        )

        pipeline.bm25 = bm25

        pipeline.bm25_chunks = (
            bm25_chunks
        )

        pipeline.reranker = (
            base_pipeline.reranker
        )

        pipeline.llm = (
            base_pipeline.llm
        )

        pipeline.citation_judge = (
            base_pipeline.citation_judge
        )

        pipeline.last_citation_validation = None

        pipeline.last_semantic_validation = None

        return pipeline

    # ========================================================
    # Retrieval
    # ========================================================

    def retrieve(
        self,
        query,
        company=None,
        year=None,
    ):

        candidates = rrf_search(
            query=query,
            vector_index=self.vector_index,
            vector_metadata=self.vector_metadata,
            vector_model=self.vector_model,
            bm25=self.bm25,
            bm25_chunks=self.bm25_chunks,
            company=company,
            year=year,
            top_k=CANDIDATE_K,
            candidate_k=CANDIDATE_K,
        )

        reranked = rerank(
            query=query,
            results=candidates,
            model=self.reranker,
            top_k=RERANK_TOP_K,
        )

        expanded = expand_adjacent_chunks(
            results=reranked,
            metadata=self.vector_metadata,
            window=ADJACENT_WINDOW,
            max_results=MAX_EVIDENCE,
        )

        return expanded

    # ========================================================
    # Ask
    # ========================================================

    def ask(
        self,
        query,
        company=None,
        year=None,
    ):

        results = self.retrieve(
            query=query,
            company=company,
            year=year,
        )

        if not results:

            raise ValueError(
                "No relevant evidence was retrieved."
            )

        answer = generate_answer(
            query=query,
            results=results,
            llm=self.llm,
        )

        # ====================================================
        # Structural citation validation
        # ====================================================

        self.last_citation_validation = (
            validate_citations(
                answer=answer,
                retrieved_results=results,
            )
        )

        # ====================================================
        # Semantic citation validation
        # ====================================================

        self.last_semantic_validation = (
            validate_citation_support(
                answer=answer,
                retrieved_results=results,
                judge=self.citation_judge,
            )
        )

        print(
            "\n"
            + "=" * 70
        )

        print(
            "CITATION VALIDATION"
        )

        print(
            "=" * 70
        )

        print(
            "Structural validation:",
            self.last_citation_validation
            .all_citations_valid,
        )

        print(
            "Semantic validation:",
            self.last_semantic_validation
            .all_citations_supported,
        )

        return (
            answer,
            results,
        )


# ============================================================
# Default static pipeline
# ============================================================

_pipeline = None


def get_pipeline():

    global _pipeline

    if _pipeline is None:

        _pipeline = (
            FinancialRAGPipeline()
        )

    return _pipeline


def ask(
    query,
    company=None,
    year=None,
):

    pipeline = get_pipeline()

    return pipeline.ask(
        query=query,
        company=company,
        year=year,
    )