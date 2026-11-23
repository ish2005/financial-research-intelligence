from sentence_transformers import CrossEncoder


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def load_reranker():
    print("Loading cross-encoder reranker...")
    model = CrossEncoder(MODEL_NAME)
    print("Reranker loaded.")
    return model


def rerank(
    query,
    results,
    model,
    top_k=5,
):
    """
    Rerank hybrid candidates using the cross-encoder while
    retaining a small amount of the original RRF signal.

    This prevents the cross-encoder from completely overriding
    strong hybrid retrieval results.
    """

    if not results:
        return []

    pairs = [
        (query, result["text"])
        for result in results
    ]

    scores = model.predict(
        pairs,
        batch_size=2,
        show_progress_bar=False,
    )

    reranked = []

    for result, score in zip(results, scores):
        item = result.copy()

        item["rerank_score"] = float(score)

        # Normalize the RRF score approximately using rank.
        # RRF rank 1 gets the strongest prior.
        rrf_rank = (
            result.get("dense_rank")
            if result.get("dense_rank") is not None
            else 20
        )

        bm25_rank = (
            result.get("bm25_rank")
            if result.get("bm25_rank") is not None
            else 20
        )

        dense_prior = 1.0 / (60 + rrf_rank)
        bm25_prior = 1.0 / (60 + bm25_rank)

        rrf_prior = dense_prior + bm25_prior

        # Cross-encoder remains the dominant signal,
        # but RRF provides a retrieval-quality prior.
        item["final_score"] = (
            float(score) + 100.0 * rrf_prior
        )

        reranked.append(item)

    reranked.sort(
        key=lambda x: x["final_score"],
        reverse=True
    )

    return reranked[:top_k]