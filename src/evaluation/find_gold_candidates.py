from pathlib import Path
import json
import re

QUESTIONS_FILE = Path("data/evaluation/evaluation_questions.json")
CHUNKS_FILE = Path("data/processed/chunks.jsonl")
MAX_RESULTS = 5


def load_questions():
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def load_chunks():
    chunks = []
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def normalize(text):
    text = text.lower()
    text = text.replace(",", "")
    text = text.replace("$", "")
    text = text.replace("%", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def question_year(question):
    match = re.search(r"\b(20\d{2})\b", question["question"])
    return int(match.group(1)) if match else None


def find_candidates(question, chunks):
    company = question["company"].lower()
    keywords = [
        normalize(keyword)
        for keyword in question.get("keywords", [])
    ]

    target_year = question_year(question)
    candidates = []

    for chunk in chunks:

        if chunk["company"].lower() != company:
            continue

        if target_year is not None:
            if int(chunk["year"]) != target_year:
                continue

        text = normalize(chunk["text"])

        # Special handling for risk-factor questions
        if "risk factors" in question["question"].lower():

            risk_signals = [
                "item 1a. risk factors",
                "item 1a risk factors",
                "risks related to",
                "could adversely affect",
                "may adversely affect",
            ]

            matched = [
                signal for signal in risk_signals
                if signal in text
            ]

        else:
            matched = [
                keyword for keyword in keywords
                if keyword in text
            ]

        if not matched:
            continue

        candidates.append({
            "chunk_id": chunk["chunk_id"],
            "company": chunk["company"],
            "year": chunk["year"],
            "page": chunk["page"],
            "section": chunk.get("section"),
            "match_count": len(matched),
        })

    candidates.sort(
        key=lambda x: (
            x["match_count"],
            -int(x["page"])
        ),
        reverse=True,
    )

    return candidates[:MAX_RESULTS]


def main():
    questions = load_questions()
    chunks = load_chunks()

    print(f"Loaded {len(questions)} questions, {len(chunks)} chunks\n")

    for i, question in enumerate(questions, start=1):

        candidates = find_candidates(question, chunks)

        print(f"Q{i}: {question['question']}")

        if not candidates:
            print("  NO CANDIDATES")
            continue

        for rank, candidate in enumerate(candidates, start=1):
            print(
                f"  {rank}. "
                f"{candidate['chunk_id']} | "
                f"p{candidate['page']} | "
                f"matches={candidate['match_count']}"
            )

        print()


if __name__ == "__main__":
    main()