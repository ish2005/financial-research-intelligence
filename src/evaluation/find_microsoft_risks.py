import json

with open("data/processed/chunks.jsonl", "r", encoding="utf-8") as f:
    chunks = [json.loads(line) for line in f if line.strip()]

for chunk in chunks:
    if chunk["company"].lower() != "microsoft":
        continue

    if int(chunk["year"]) != 2024:
        continue

    text = chunk["text"].lower()

    signals = [
        "risk factors",
        "risks we face",
        "could adversely affect",
        "may adversely affect",
        "adversely affect",
        "risks related to",
    ]

    if any(signal in text for signal in signals):
        print(
            f"{chunk['chunk_id']} | p{chunk['page']} | "
            + chunk["text"][:350].replace("\n", " ")
        )