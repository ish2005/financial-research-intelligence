from pathlib import Path
import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

INPUT_FILE=Path("data/processed/chunks.jsonl")

INDEX_DIR=Path("data/processed/faiss")
INDEX_FILE=INDEX_DIR / "index.faiss"
METADATA_FILE= INDEX_DIR / "metadata.json"

MODEL_NAME= "BAAI/bge-small-en-v1.5"

def load_chunks() -> list[dict]:
    chunks=[]
    with open(INPUT_FILE,"r",encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    return chunks

def create_embeddings(
        chunks: list[dict],
        model: SentenceTransformer
) -> np.ndarray:
    texts=[chunk["text"] for chunk in chunks]

    embeddings=model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    return np.asanyarray(embeddings,dtype="float32")

def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:

    dimension = embeddings.shape[1]

    index=faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index

def save_index(
        index: faiss.Index,
        chunks: list[dict]
) -> None:

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    faiss.write_index(
        index,
        str(INDEX_FILE)
    )

    metadata=[]

    for chunk in chunks:
        metadata.append({
            "chunk_id":chunk["chunk_id"],
            "company": chunk["company"],
            "year": chunk["year"],
            "document": chunk["document"],
            "page": chunk["page"],
            "section": chunk["section"],
            "text": chunk["text"],
        })
    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=2
        )
def main():
    print("Loading chunks...")
    chunks=load_chunks()
    print(f"Loaded {len(chunks)} chunks")
    print("\nLoading embedding model...")
    model=SentenceTransformer(MODEL_NAME)
    print("Creating embeddings...")
    embeddings=create_embeddings(
        chunks,
        model
    )
    print(f"Embedding shape: {embeddings.shape}")
    print("\nBuilding FAISS index...")
    index=build_faiss_index(embeddings)
    print(f"Faiss Index Contains"
          f"{index.ntotal} vectors"
          )
    print("\nSaving index...")
    save_index(
        index,
        chunks
    )
    print("\nDone")

    print(f"Index: {INDEX_FILE}")
    print(f"Metadata: {METADATA_FILE}")

if __name__=="__main__":
    main()