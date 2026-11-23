from src.pipeline import ask


def main():
    questions = [
        "What was Apple's total net sales in 2024?",
        "What was Microsoft's Intelligent Cloud revenue in 2024?",
        "What was NVIDIA's Data Center revenue in fiscal year 2024?",
        "What are Microsoft's major risk factors?",
        "How much did Apple spend on research and development in 2024?",
    ]

    for i, question in enumerate(questions, start=1):
        print("\n" + "=" * 70)
        print(f"QUESTION {i}")
        print("=" * 70)
        print(question)

        try:
            answer, evidence = ask(question)

            print("\nANSWER:")
            print(answer.answer)

            print("\nCONFIDENCE:")
            print(answer.confidence)

            print("\nCITATIONS:")
            for citation in answer.citations:
                print(
                    f"- {citation.chunk_id} | "
                    f"{citation.company} {citation.year} | "
                    f"page {citation.page}"
                )

            print("\nEVIDENCE CHUNKS:")
            for result in evidence:
                print(
                    f"- {result['chunk_id']} | "
                    f"{result.get('company')} "
                    f"{result.get('year')} | "
                    f"page {result.get('page')}"
                )

        except Exception as e:
            print(f"\nERROR: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()