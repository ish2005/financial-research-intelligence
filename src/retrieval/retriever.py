from pathlib import Path
import json

import faiss
from sentence_transformers import SentenceTransformer

INDEX_FILE=Path("data/processed/faiss/index.faiss")
METADATA_FILE=Path("data/processed/faiss/metadata.json")

MODEL_NAME="BAAI/bge-small-en-v1.5"

def load_vector_store():
    index=faiss.read_index(
        str(INDEX_FILE)
    )
    with open(
        METADATA_FILE,
        "r",
        encoding="utf-8"
    ) as f:
        metadata=json.load(f)
    model=SentenceTransformer(
        MODEL_NAME
    )
    return index,metadata,model
def search(
        query:str,
        index,
        metadata,
        model,
        top_k: int=5
):
    query_embedding=model.encode(
        [query],
        normalize_embeddings=True
    )
    scores,indices=index.search(
        query_embedding,
        top_k
    )

    results=[]

    for score,index_id in zip(
        scores[0],
        indices[0]
    ):
        result=metadata[index_id].copy()
        result["score"]=float(score)
        results.append(result)
    return results

def main():

    print("Loading vector store...")
    index,metadata,model=load_vector_store()
    print(
        f"Loaded {index.ntotal} vectors"
    )
    query=input(
        "\nEnter your question: "
    )
    results=search(
        query=query,
        index=index,
        metadata=metadata,
        model=model,
        top_k=5
    )

    print("\n" + "="*80)
    print("RETRIEVED RESULTS")
    print("="*80)

    for rank,result in enumerate(
        results,
        start=1
    ):
        print(
            f"\n--Result {rank} ---"
        )
        print(
            f"Score: {result['score']:.4f}"
        )
        print(
            f"Year: {result['year']}"
        )
        print(
            f"Page: {result['page']}"
        )
        print(
            f"Chunk ID: {result['chunk_id']}"
        )
        print(
            f"\n{result['text'][:1000]}"
        )
if __name__=="__main__":
    main()