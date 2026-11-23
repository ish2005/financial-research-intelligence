import json
from pathlib import Path

QUESTIONS_FILE = Path("data/evaluation/evaluation_questions.json")
CHUNKS_FILE = Path("data/processed/chunks.jsonl")


def load_questions():
    with QUESTIONS_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_chunks():
    chunks = []

    with CHUNKS_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    return chunks


def find_chunk(chunks, chunk_id):
    for chunk in chunks:
        if chunk["chunk_id"] == chunk_id:
            return chunk

    return None


def print_chunk(chunk):
    if chunk is None:
        print("  NOT FOUND")
        return

    print(f"  Chunk ID: {chunk['chunk_id']}")
    print(f"  Company: {chunk.get('company')}")
    print(f"  Year: {chunk.get('year')}")
    print(f"  Page: {chunk.get('page')}")
    print(f"  Section: {chunk.get('section')}")
    print()
    print("  Text:")
    print("  " + chunk["text"].replace("\n", "\n  "))


def main():
    questions = load_questions()
    chunks = load_chunks()

    chunk_lookup = {
        chunk["chunk_id"]: chunk
        for chunk in chunks
    }

    print("=" * 75)
    print("CURRENT GOLD VERIFICATION")
    print("=" * 75)

    for question in questions:
        qid = question.get("id")
        query = question.get("question")

        print()
        print("=" * 75)
        print(f"Q{qid}")
        print("=" * 75)
        print(query)

        gold_chunks = question.get("gold_chunks", [])

        if isinstance(gold_chunks, str):
            gold_chunks = [gold_chunks]

        print("\nGold chunks:")

        for chunk_id in gold_chunks:
            print()
            print_chunk(chunk_lookup.get(chunk_id))

    # ------------------------------------------------------------
    # Specific investigation for Q20
    # ------------------------------------------------------------

    print()
    print("=" * 75)
    print("Q20 INVESTIGATION")
    print("=" * 75)

    q20 = None

    for question in questions:
        if str(question.get("id")) == "20":
            q20 = question
            break

    if q20 is None:
        print("Q20 not found.")
        return

    print("Question:")
    print(q20.get("question"))

    print("\nCurrent gold:")
    current_gold = q20.get("gold_chunks", [])

    if isinstance(current_gold, str):
        current_gold = [current_gold]

    for chunk_id in current_gold:
        print()
        print_chunk(chunk_lookup.get(chunk_id))

    # Search all NVIDIA FY2024 chunks for Data Center revenue
    print()
    print("-" * 75)
    print("NVIDIA 2024 chunks containing relevant Data Center / revenue evidence")
    print("-" * 75)

    keywords = [
        "47.5",
        "47,500",
        "data center revenue",
        "data center revenues",
        "data center",
    ]

    candidates = []

    for chunk in chunks:
        if (
            chunk.get("company", "").lower() == "nvidia"
            and int(chunk.get("year", 0)) == 2024
        ):
            text = chunk.get("text", "").lower()

            matched = [
                keyword
                for keyword in keywords
                if keyword.lower() in text
            ]

            if matched:
                candidates.append((chunk, matched))

    for chunk, matched in candidates:
        print()
        print(
            f"{chunk['chunk_id']} | "
            f"page {chunk.get('page')} | "
            f"matched: {', '.join(matched)}"
        )
        print("-" * 75)
        print(chunk["text"])

    print()
    print("=" * 75)
    print("DONE")
    print("=" * 75)


if __name__ == "__main__":
    main()