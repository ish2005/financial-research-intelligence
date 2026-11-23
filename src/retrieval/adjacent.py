import re


def get_chunk_number(chunk_id):
    match = re.search(r"_c(\d+)$", chunk_id)

    if match:
        return int(match.group(1))

    return 0


def build_chunk_lookup(metadata):
    grouped = {}

    for item in metadata:
        key = (
            item.get("company"),
            int(item.get("year")),
        )

        grouped.setdefault(key, []).append(item)

    for chunks in grouped.values():
        chunks.sort(
            key=lambda x: (
                int(x.get("page", 0)),
                get_chunk_number(x["chunk_id"]),
            )
        )

    return grouped


def expand_adjacent_chunks(
    results,
    metadata,
    window=1,
    max_results=8,
):
    """
    Expand the top reranked results with nearby chunks.

    Original reranked results are always kept first.
    Adjacent chunks are added only until max_results is reached.
    """

    if not results:
        return []

    grouped_chunks = build_chunk_lookup(metadata)

    metadata_by_id = {
        item["chunk_id"]: item
        for item in metadata
    }

    expanded = {}
    original_ids = set()

    # --------------------------------------------------
    # 1. Keep original reranked results first
    # --------------------------------------------------

    for result in results:
        chunk_id = result["chunk_id"]

        if chunk_id not in expanded:
            expanded[chunk_id] = result.copy()

        original_ids.add(chunk_id)

    # --------------------------------------------------
    # 2. Add adjacent chunks
    # --------------------------------------------------

    for result in results:
        if len(expanded) >= max_results:
            break

        chunk_id = result["chunk_id"]

        metadata_item = metadata_by_id.get(chunk_id)

        if metadata_item is None:
            continue

        key = (
            metadata_item.get("company"),
            int(metadata_item.get("year")),
        )

        chunks = grouped_chunks.get(key, [])

        try:
            current_index = next(
                i
                for i, chunk in enumerate(chunks)
                if chunk["chunk_id"] == chunk_id
            )
        except StopIteration:
            continue

        start = max(
            0,
            current_index - window,
        )

        end = min(
            len(chunks),
            current_index + window + 1,
        )

        for neighbor in chunks[start:end]:

            if len(expanded) >= max_results:
                break

            neighbor_id = neighbor["chunk_id"]

            if neighbor_id in expanded:
                continue

            neighbor_copy = neighbor.copy()

            neighbor_copy["expanded_from"] = chunk_id

            expanded[neighbor_id] = neighbor_copy

    # --------------------------------------------------
    # 3. Return originals first
    # --------------------------------------------------

    original_results = [
        item
        for item in expanded.values()
        if item["chunk_id"] in original_ids
    ]

    expanded_results = [
        item
        for item in expanded.values()
        if item["chunk_id"] not in original_ids
    ]

    return (
        original_results + expanded_results
    )[:max_results]