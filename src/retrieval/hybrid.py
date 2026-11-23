import numpy as np

from src.retrieval.retriever import load_vector_store, search
from src.retrieval.bm25 import build_bm25, search_bm25


RRF_K = 60
CANDIDATE_K = 20
TOP_K = 5


def filter_by_metadata(results, company=None, year=None):
    """
    Keep only chunks matching the requested company/year.
    """
    filtered = []

    for result in results:
        if company is not None:
            if result.get("company", "").lower() != company.lower():
                continue

        if year is not None:
            if int(result.get("year")) != int(year):
                continue

        filtered.append(result)

    return filtered


def reciprocal_rank_fusion(
    dense_results,
    bm25_results,
    top_k=TOP_K,
    rrf_k=RRF_K,
):
    combined = {}

    for rank, result in enumerate(dense_results, start=1):
        chunk_id = result["chunk_id"]

        if chunk_id not in combined:
            combined[chunk_id] = {
                "data": result,
                "dense_rank": None,
                "bm25_rank": None,
                "dense_rrf": 0.0,
                "bm25_rrf": 0.0,
            }

        combined[chunk_id]["dense_rank"] = rank
        combined[chunk_id]["dense_rrf"] = 1.0 / (rrf_k + rank)

    for rank, result in enumerate(bm25_results, start=1):
        chunk_id = result["chunk_id"]

        if chunk_id not in combined:
            combined[chunk_id] = {
                "data": result,
                "dense_rank": None,
                "bm25_rank": None,
                "dense_rrf": 0.0,
                "bm25_rrf": 0.0,
            }

        combined[chunk_id]["bm25_rank"] = rank
        combined[chunk_id]["bm25_rrf"] = 1.0 / (rrf_k + rank)

    results = []

    for item in combined.values():
        result = item["data"].copy()

        result["dense_rank"] = item["dense_rank"]
        result["bm25_rank"] = item["bm25_rank"]
        result["dense_rrf"] = item["dense_rrf"]
        result["bm25_rrf"] = item["bm25_rrf"]

        result["rrf_score"] = (
            item["dense_rrf"] +
            item["bm25_rrf"]
        )

        results.append(result)

    results.sort(
        key=lambda x: x["rrf_score"],
        reverse=True
    )

    return results[:top_k]


def rrf_search(
    query,
    vector_index,
    vector_metadata,
    vector_model,
    bm25,
    bm25_chunks,
    company=None,
    year=None,
    top_k=TOP_K,
    candidate_k=CANDIDATE_K,
):
    # Retrieve a larger pool first.
    dense_results = search(
        query=query,
        index=vector_index,
        metadata=vector_metadata,
        model=vector_model,
        top_k=len(vector_metadata),
    )

    bm25_results = search_bm25(
        query=query,
        bm25=bm25,
        chunks=bm25_chunks,
        top_k=len(bm25_chunks),
    )

    # Apply metadata filtering BEFORE RRF.
    dense_results = filter_by_metadata(
        dense_results,
        company=company,
        year=year,
    )

    bm25_results = filter_by_metadata(
        bm25_results,
        company=company,
        year=year,
    )

    # Only now take the candidate pool.
    dense_results = dense_results[:candidate_k]
    bm25_results = bm25_results[:candidate_k]

    return reciprocal_rank_fusion(
        dense_results=dense_results,
        bm25_results=bm25_results,
        top_k=top_k,
        rrf_k=RRF_K,
    )