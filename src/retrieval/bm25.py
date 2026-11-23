import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

INPUT_FILE=Path("data/processed/chunks.jsonl")

def tokenize(text):
    """
    Convert text into tokens for BM25.

    Example:
    "Microsoft's  revenue increased 12%"
    ->
    ["microsoft","s","revenue","increased","12"]
    """
    return re.findall(r"\b\w+\b",text.lower())
def load_chunks():
    """
    Load all chunks from chunks.jsonl.
    """
    chunks=[]

    with INPUT_FILE.open("r",encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks

def build_bm25():
    """
    Build a BM25 index over all chunk text.
    """
    chunks=load_chunks()
    corpus=[
        tokenize(chunk["text"])
        for chunk in chunks
    ]
    bm25=BM25Okapi(corpus=corpus)

    return bm25, chunks

def search_bm25(query,bm25,chunks,top_k=5):
    """
    Search the BM25 index and return the top_k chunks.
    """
    query_tokens=tokenize(query)
    scores=bm25.get_scores(query_tokens)
    ranked_indices=scores.argsort()[::-1][:top_k]
    results=[]
    for index in ranked_indices:
        result=chunks[index].copy()
        result["score"]=float(scores[index])
        results.append(result)
    return results
if __name__=="__main__":
    bm25,chunks=build_bm25()
    queries=[
        "What was Apple's total net sales in 2023?",
        "How did NVIDIA'S revenue change between 2024 and 2025?",
        "What are Microsoft's major risk factors?",
    ]
    for query in queries:
        print("\n"+"="*80)
        print(f"QUERY: {query}")
        print("="*80)
        results=search_bm25(
            query=query,
            bm25=bm25,
            chunks=chunks,
            top_k=5,
        )
        for i,result in enumerate(results,start=1):
            print(f"\nResult {1}")
            print(f"Score: {result['score']:4f}")
            print(
                f"Company: {result['company']} |"
                f"Year: {result['year']} | "
                f"Page: {result['page']}"
            )
            print(f"Chunk ID: {result['chunk_id']}")
            print(result["text"[:500]].replace("\n"," "))