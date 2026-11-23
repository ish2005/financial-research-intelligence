import json

CHUNKS_FILE = "data/processed/chunks.jsonl"

TARGETS = {
    "Q13": ["risk factors"],
    "Q14": ["29,510", "29510"],
}

with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    chunks = [json.loads(line) for line in f if line.strip()]

for q, terms in TARGETS.items():
    print("\n" + "=" * 70)
    print(q)
    print("=" * 70)

    found = 0

    for chunk in chunks:
        text = chunk["text"].lower().replace(",", "")

        if q == "Q13":
            if chunk["company"].lower() != "microsoft":
                continue
            if "item 1a" not in text and "risk factors" not in text:
                continue

        elif q == "Q14":
            if chunk["company"].lower() != "microsoft":
                continue
            if "29510" not in text:
                continue

        print(
            f"{chunk['chunk_id']} | "
            f"page {chunk['page']}"
        )
        print(chunk["text"][:500].replace("\n", " "))
        print()

        found += 1

        if found >= 5:
            break