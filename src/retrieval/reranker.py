from sentence_transformers import CrossEncoder

# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_NAME= "cross-encoder/ms-marco-MiniLM-L-6-v2"

TOP_K=5

# --------------------------------------------------
# Load reranker
# --------------------------------------------------

def load_reranker():
    """
    Load the pretrained cross-encoder reranker.
    """

    print("Loading cross-encoder reranker...")

    model=CrossEncoder(
        MODEL_NAME
    )

    print("Reranker loaded.")

    return model

# --------------------------------------------------
# Rerank results
# --------------------------------------------------

def rerank(
        query,
        results,
        model,
        top_k=TOP_K
):
    """
    Rerank retrieved documents using a cross-encoder.

    Each candidate is scored using:
        [query,document]

    The cross-encoder jointly processes the query 
    and document instead of embedding them seprately
    """

    if not results:
        return[]

    # --------------------------------------------------
    # Create query-document pairs
    # --------------------------------------------------

    pairs=[
        [query, result["text"]]
        for result in results
    ]

    # --------------------------------------------------
    # Get cross-encoder scores
    # --------------------------------------------------

    scores=model.predict(
        pairs
    )

    # --------------------------------------------------
    # Attach reranker scores
    # --------------------------------------------------

    reranked_results=[]

    for result, score in zip(
        results,
        scores,
    ):
        result_copy= result.copy()

        result_copy["reranker_score"]= float(
            score
        )

        reranked_results.append(
            result_copy
        )
    # --------------------------------------------------
    # Sort by reranker score
    # --------------------------------------------------

    reranked_results.sort(
        key=lambda x: x["reranker_score"],
        reverse=True
    )

    return reranked_results[:top_k]

# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    model=load_reranker()

    print("\nReranker test loaded successfully.")

    # Simple sanity test 
    query="What was Apple's total net sales in 2023?"

    documents=[
        {
            "chunk_id":"test_1",
            "text": (
                "Apple's total net sales were "
                "$383.285 billion in 2023."
            ),
        },
        {
            "chunk_id":"test_2",
            "text":(
                "Apple increased its research and "
                "development spending in 2023."
            ),
        },
        {
            "chunk_id":"test_3",
            "text":(
                "Microsoft reported revenue growth "
                "from its cloud services."
            ),
        },
    ]

    results=rerank(
        query=query,
        results=documents,
        model=model,
        top_k=3,
    )

    print("\n" + "="*80)
    print("RERANKER TEST")
    print("="*80)

    print(f"\nQuery: {query}")

    for rank,result in enumerate(
        results,
        start=1,
    ):
        print(
            f"\n--- Result {rank} ---"
        )

        print(
            f"Reranker Score: "
            f"{result['reranker_score']:.4f}"
        )

        print(
            f"Chunk ID: "
            f"{result['chunk_id']}"
        )

        print(
            f"Text: "
            f"{result['text']}"
        )

# --------------------------------------------------
# Entry point
# --------------------------------------------------

if __name__=="__main__":
    main()