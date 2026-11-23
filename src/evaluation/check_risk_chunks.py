import json

with open("data/processed/chunks.jsonl", "r", encoding="utf-8") as f:
    chunks = [json.loads(line) for line in f if line.strip()]

signals = [
    "cybersecurity",
    "security vulnerabilities",
    "privacy",
    "artificial intelligence",
    "regulatory",
    "competition",
    "intellectual property",
    "business interruption",
]

found = 0

for chunk in chunks:
    if chunk["company"].lower() != "microsoft":
        continue
    if int(chunk["year"]) != 2024:
        continue

    text = chunk["text"].lower()

    matches = [s for s in signals if s in text]

    if matches:
        print(
            f"{chunk['chunk_id']} | p{chunk['page']} | "
            f"{', '.join(matches)}"
        )

        found += 1

        if found >= 15:
            break