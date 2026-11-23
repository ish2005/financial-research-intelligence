import json
import re

with open("data/processed/chunks.jsonl", "r", encoding="utf-8") as f:
    chunks = [json.loads(line) for line in f if line.strip()]

for chunk in chunks:
    if chunk["company"].lower() != "microsoft":
        continue

    if int(chunk["year"]) != 2024:
        continue

    text = chunk["text"].lower()

    if "research and development" in text:
        numbers = re.findall(r"\$?\s*\d[\d,\s]*\d", chunk["text"])

        print(
            f"{chunk['chunk_id']} | p{chunk['page']} | "
            f"numbers={numbers[:10]}"
        )