from pathlib import Path
import json

from src.retrieval.retriever import load_vector_store, search


QUESTIONS_FILE = Path(
    "data/evaluation/evaluation_questions.json"
)

TOP_K = 5


def load_questions():

    with open(
        QUESTIONS_FILE,
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


def main():

    print("Loading vector store...")

    index, metadata, model = load_vector_store()

    questions = load_questions()

    for i, question in enumerate(
        questions,
        start=1
    ):

        query = question["question"]

        print("\n" + "=" * 80)
        print(f"[{i}/24] {query}")
        print("=" * 80)

        results = search(
            query=query,
            index=index,
            metadata=metadata,
            model=model,
            top_k=TOP_K,
        )

        for rank, result in enumerate(
            results,
            start=1
        ):

            print(
                f"\n--- Rank {rank} ---"
            )

            print(
                f"Company: {result['company']}"
            )

            print(
                f"Year: {result['year']}"
            )

            print(
                f"Page: {result['page']}"
            )

            print(
                f"Chunk: {result['chunk_id']}"
            )

            print(
                f"Score: {result['score']:.4f}"
            )

            print(
                "\nText:"
            )

            print(
                result["text"][:1200]
            )


if __name__ == "__main__":
    main()