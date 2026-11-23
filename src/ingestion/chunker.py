from pathlib import Path
import json

from transformers import AutoTokenizer


INPUT_FILE = Path("data/processed/pages.jsonl")
OUTPUT_FILE = Path("data/processed/chunks.jsonl")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 450
CHUNK_OVERLAP = 60


def load_pages() -> list[dict]:
    pages = []

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            pages.append(json.loads(line))

    return pages


def create_chunks(page: dict, tokenizer) -> list[dict]:
    text = page["text"]

    tokens = tokenizer.encode(
        text,
        add_special_tokens=False,
        truncation=False
    )

    chunks = []

    start = 0
    chunk_number = 0

    while start < len(tokens):

        end = min(
            start + CHUNK_SIZE,
            len(tokens)
        )

        chunk_tokens = tokens[start:end]

        chunk_text = tokenizer.decode(
            chunk_tokens,
            skip_special_tokens=True
        ).strip()

        if chunk_text:

            chunk_id = (
                f"{page['company'].lower()}_"
                f"{page['year']}_"
                f"p{page['page']}_"
                f"c{chunk_number}"
            )

            chunk = {
                "chunk_id": chunk_id,
                "text": chunk_text,
                "company": page["company"],
                "year": page["year"],
                "document": page["document"],
                "page": page["page"],
                "section": page["section"],
            }

            chunks.append(chunk)

        chunk_number += 1

        if end == len(tokens):
            break

        start = end - CHUNK_OVERLAP

    return chunks


def main():

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        model_max_length=100000
    )

    pages = load_pages()

    print(f"Loaded {len(pages)} pages")

    all_chunks = []

    for i, page in enumerate(pages, start=1):

        chunks = create_chunks(
            page,
            tokenizer
        )

        all_chunks.extend(chunks)

        if i % 100 == 0:
            print(
                f"Processed {i}/{len(pages)} pages"
            )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        for chunk in all_chunks:

            f.write(
                json.dumps(
                    chunk,
                    ensure_ascii=False
                )
                + "\n"
            )

    print("\nDone!")
    print(f"Total chunks: {len(all_chunks)}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()