import re

import faiss
import fitz

from rank_bm25 import BM25Okapi


CHUNK_SIZE = 450
CHUNK_OVERLAP = 60


# ============================================================
# Tokenization
# ============================================================

def tokenize(text):
    return re.findall(
        r"\b\w+\b",
        text.lower(),
    )


# ============================================================
# PDF parsing
# ============================================================

def parse_pdf(pdf_path):

    document = fitz.open(pdf_path)

    pages = []

    for page_number, page in enumerate(
        document,
        start=1,
    ):

        text = page.get_text("text").strip()

        if not text:
            continue

        pages.append(
            {
                "page": page_number,
                "text": text,
            }
        )

    document.close()

    return pages


# ============================================================
# Metadata extraction
# ============================================================

def _clean_company_name(value):

    value = re.sub(
        r"\s+",
        " ",
        value,
    ).strip()

    value = value.strip(
        " :-–—|,.;"
    )

    return value


def _looks_like_company_name(value):

    if not value:
        return False

    value_lower = value.lower()

    company_terms = [
        "limited",
        "ltd",
        "inc",
        "incorporated",
        "corporation",
        "corp",
        "plc",
        "llc",
        "private limited",
    ]

    return any(
        term in value_lower
        for term in company_terms
    )


def extract_company_name(pages):

    if not pages:
        return "Uploaded Document"

    first_pages = "\n".join(
        page["text"]
        for page in pages[:8]
    )

    lines = [
        re.sub(
            r"\s+",
            " ",
            line,
        ).strip()
        for line in first_pages.splitlines()
        if line.strip()
    ]

    patterns = [
        r"([A-Z][A-Za-z0-9&.,'()\- ]{3,120}"
        r"(?:Limited|Ltd\.?|Private Limited|"
        r"Inc\.?|Incorporated|Corporation|"
        r"Corp\.?|PLC|LLC))",

        r"([A-Z][A-Za-z0-9&.,'()\- ]{3,120}"
        r"(?:LIMITED|LTD\.?|PRIVATE LIMITED))",
    ]

    for line in lines:

        if len(line) > 160:
            continue

        for pattern in patterns:

            match = re.search(
                pattern,
                line,
            )

            if match:

                candidate = _clean_company_name(
                    match.group(1)
                )

                if _looks_like_company_name(
                    candidate
                ):
                    return candidate

    # Check lines around "Annual Report"

    for index, line in enumerate(lines):

        if "annual report" not in line.lower():
            continue

        nearby = []

        if index > 0:
            nearby.append(
                lines[index - 1]
            )

        if index + 1 < len(lines):
            nearby.append(
                lines[index + 1]
            )

        for candidate in nearby:

            candidate = _clean_company_name(
                candidate
            )

            if (
                len(candidate) <= 160
                and _looks_like_company_name(candidate)
            ):
                return candidate

    # Fallback

    candidates = []

    for line in lines[:100]:

        if len(line) > 120:
            continue

        if _looks_like_company_name(line):
            candidates.append(line)

    if candidates:

        candidates.sort(
            key=len,
            reverse=True,
        )

        return _clean_company_name(
            candidates[0]
        )

    return "Uploaded Document"


def extract_reporting_year(pages):

    if not pages:
        return 2026

    first_pages = "\n".join(
        page["text"]
        for page in pages[:10]
    )

    patterns = [

        # year ended 31 March 2025
        r"(?:year|financial year|fiscal year)"
        r"[^\n]{0,80}?"
        r"(?:ended|ending)"
        r"[^\n]{0,40}?"
        r"(?:31\s+March|March\s+31)"
        r"[,\s]+(20\d{2})",

        # 31 March 2025
        r"(?:31\s+March|March\s+31)"
        r"[,\s]+(20\d{2})",

        # Annual Report 2025
        r"annual\s+report"
        r"[^\n]{0,50}?"
        r"(20\d{2})",

        # Financial Year 2025
        r"(?:financial|fiscal)\s+year"
        r"[^\n]{0,30}?"
        r"(20\d{2})",

        # FY2025
        r"\bFY\s*[-:]?\s*(20\d{2})\b",

        # 2024-25
        r"\b(20\d{2})\s*[-–]\s*(\d{2})\b",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            first_pages,
            flags=re.IGNORECASE,
        )

        if not match:
            continue

        groups = match.groups()

        if len(groups) == 1:

            try:
                return int(groups[0])
            except ValueError:
                continue

        if len(groups) == 2:

            try:

                start_year = int(groups[0])
                end_suffix = int(groups[1])

                century = (
                    start_year // 100
                ) * 100

                return century + end_suffix

            except ValueError:
                continue

    # Fallback

    years = re.findall(
        r"\b(20\d{2})\b",
        first_pages,
    )

    if years:

        return max(
            int(year)
            for year in years
        )

    return 2026


def extract_document_metadata(pages):

    return {
        "company": extract_company_name(
            pages
        ),
        "year": extract_reporting_year(
            pages
        ),
    }


# ============================================================
# Chunking
# ============================================================

def chunk_text(
    text,
    tokenizer,
):

    tokens = tokenizer.encode(
        text,
        add_special_tokens=False,
    )

    chunks = []

    start = 0

    while start < len(tokens):

        end = min(
            start + CHUNK_SIZE,
            len(tokens),
        )

        chunk_tokens = tokens[start:end]

        chunk_value = tokenizer.decode(
            chunk_tokens,
            skip_special_tokens=True,
        ).strip()

        if chunk_value:
            chunks.append(
                chunk_value
            )

        if end >= len(tokens):
            break

        start = end - CHUNK_OVERLAP

    return chunks


# ============================================================
# Build chunks
# ============================================================

def build_chunks(
    pages,
    company,
    year,
    model,
):

    tokenizer = model.tokenizer

    chunks = []

    safe_company = re.sub(
        r"[^a-z0-9]+",
        "_",
        company.lower(),
    ).strip("_")

    for page_data in pages:

        page_number = page_data["page"]
        text = page_data["text"]

        page_chunks = chunk_text(
            text,
            tokenizer,
        )

        for chunk_number, chunk in enumerate(
            page_chunks
        ):

            chunks.append(
                {
                    "chunk_id": (
                        f"{safe_company}_"
                        f"{year}_"
                        f"p{page_number}_"
                        f"c{chunk_number}"
                    ),
                    "company": company,
                    "year": year,
                    "page": page_number,
                    "text": chunk,
                }
            )

    return chunks


# ============================================================
# Vector store
# ============================================================

def build_vector_store(
    chunks,
    model,
):

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=8,
    )

    embeddings = embeddings.astype(
        "float32"
    )

    index = faiss.IndexFlatIP(
        embeddings.shape[1]
    )

    index.add(
        embeddings
    )

    return index


# ============================================================
# BM25
# ============================================================

def build_bm25_index(chunks):

    corpus = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]

    return BM25Okapi(
        corpus
    )


# ============================================================
# MULTI-PDF INDEX
#
# IMPORTANT:
# The embedding model is passed into this function.
# We DO NOT load another SentenceTransformer here.
# ============================================================

def build_uploaded_index(
    pdf_documents,
    model,
):

    if not pdf_documents:

        raise ValueError(
            "No PDF documents were supplied."
        )

    print(
        f"Indexing {len(pdf_documents)} document(s)..."
    )

    all_chunks = []

    document_metadata = []

    # --------------------------------------------------------
    # Process every PDF
    # --------------------------------------------------------

    for pdf_document in pdf_documents:

        pdf_path = pdf_document["path"]

        supplied_company = (
            pdf_document.get("company")
        )

        supplied_year = (
            pdf_document.get("year")
        )

        print(
            f"\nParsing: {pdf_document.get('filename', pdf_path)}"
        )

        pages = parse_pdf(
            pdf_path
        )

        if not pages:

            raise ValueError(
                f"No readable text found in: "
                f"{pdf_path}"
            )

        detected = extract_document_metadata(
            pages
        )

        company = (
            supplied_company
            if supplied_company
            else detected["company"]
        )

        year = (
            supplied_year
            if supplied_year
            else detected["year"]
        )

        print(
            f"Detected: {company} — {year}"
        )

        chunks = build_chunks(
            pages=pages,
            company=company,
            year=year,
            model=model,
        )

        all_chunks.extend(
            chunks
        )

        document_metadata.append(
            {
                "filename": pdf_document.get(
                    "filename",
                    str(pdf_path),
                ),
                "company": company,
                "year": year,
                "pages": len(pages),
                "chunks": len(chunks),
            }
        )

    if not all_chunks:

        raise ValueError(
            "No chunks were generated from "
            "the uploaded PDFs."
        )

    print(
        f"\nTotal chunks: {len(all_chunks)}"
    )

    # --------------------------------------------------------
    # Combined FAISS index
    # --------------------------------------------------------

    vector_index = build_vector_store(
        chunks=all_chunks,
        model=model,
    )

    # --------------------------------------------------------
    # Combined BM25 index
    # --------------------------------------------------------

    bm25 = build_bm25_index(
        all_chunks
    )

    return {
        "vector_index": vector_index,
        "vector_metadata": all_chunks,
        "vector_model": model,
        "bm25": bm25,
        "bm25_chunks": all_chunks,
        "document_metadata": document_metadata,
    }