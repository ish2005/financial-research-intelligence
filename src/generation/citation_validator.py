from typing import List

from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


MODEL_NAME = "openai/gpt-oss-120b"


# ============================================================
# Structured output schemas
# ============================================================

class CitationValidation(BaseModel):
    chunk_id: str
    valid: bool
    reason: str


class SemanticCitationValidation(BaseModel):
    chunk_id: str
    supported: bool
    reason: str


class ValidationResult(BaseModel):
    all_citations_valid: bool
    validations: List[CitationValidation]


class SemanticValidationResult(BaseModel):
    all_citations_supported: bool
    validations: List[SemanticCitationValidation]


# ============================================================
# Structural citation validation
# ============================================================

def validate_citations(answer, retrieved_results):
    """
    Validate that every citation returned by the LLM:

    1. Exists in the retrieved evidence.
    2. Has matching metadata.
    3. Does not refer to a hallucinated chunk.
    """

    evidence_by_chunk = {
        result["chunk_id"]: result
        for result in retrieved_results
    }

    validations = []

    for citation in answer.citations:
        chunk_id = citation.chunk_id

        # Check whether chunk exists
        if chunk_id not in evidence_by_chunk:
            validations.append(
                CitationValidation(
                    chunk_id=chunk_id,
                    valid=False,
                    reason=(
                        "Citation chunk was not present "
                        "in retrieved evidence."
                    ),
                )
            )
            continue

        evidence = evidence_by_chunk[chunk_id]

        # Check metadata
        metadata_errors = []

        if citation.company != evidence.get("company"):
            metadata_errors.append(
                f"company mismatch: "
                f"citation={citation.company}, "
                f"evidence={evidence.get('company')}"
            )

        if int(citation.year) != int(evidence.get("year")):
            metadata_errors.append(
                f"year mismatch: "
                f"citation={citation.year}, "
                f"evidence={evidence.get('year')}"
            )

        if int(citation.page) != int(evidence.get("page")):
            metadata_errors.append(
                f"page mismatch: "
                f"citation={citation.page}, "
                f"evidence={evidence.get('page')}"
            )

        if metadata_errors:
            validations.append(
                CitationValidation(
                    chunk_id=chunk_id,
                    valid=False,
                    reason="; ".join(metadata_errors),
                )
            )
            continue

        validations.append(
            CitationValidation(
                chunk_id=chunk_id,
                valid=True,
                reason=(
                    "Citation exists in retrieved evidence "
                    "and metadata matches."
                ),
            )
        )

    all_valid = all(
        validation.valid
        for validation in validations
    )

    return ValidationResult(
        all_citations_valid=all_valid,
        validations=validations,
    )


# ============================================================
# Semantic citation validation
# ============================================================

def load_citation_judge():
    """
    Load the LLM used to judge whether a citation
    actually supports the generated answer.
    """

    llm = ChatGroq(
        model=MODEL_NAME,
        temperature=0,
        api_key=None,
    )

    return llm.with_structured_output(
        SemanticCitationValidation
    )


def validate_citation_support(
    answer,
    retrieved_results,
    judge,
):
    """
    Check whether each citation actually supports
    the generated answer.

    This is different from structural validation.

    Structural validation:
        Does the citation exist?

    Semantic validation:
        Does the cited text actually support the answer?
    """

    evidence_by_chunk = {
        result["chunk_id"]: result
        for result in retrieved_results
    }

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a strict citation verification judge.

Your task is to determine whether the supplied
annual-report evidence actually supports the supplied
answer.

Rules:

1. Judge ONLY the supplied answer and evidence.
2. Do not use outside knowledge.
3. The evidence must contain enough information to
   support the answer.
4. If the evidence merely mentions the topic but does
   not support the answer, mark it unsupported.
5. If the answer contains a number, the evidence must
   support that number.
6. Do not assume missing information.
7. Return supported=true only when the evidence clearly
   supports the answer.
8. Keep the reason concise.
"""
            ),
            (
                "human",
                """
Answer:
{answer}

Citation chunk ID:
{chunk_id}

Evidence:
{evidence}

Does this evidence actually support the answer?
"""
            ),
        ]
    )

    chain = prompt | judge

    validations = []

    for citation in answer.citations:
        chunk_id = citation.chunk_id

        # A structurally invalid citation cannot be
        # semantically valid.
        if chunk_id not in evidence_by_chunk:
            validations.append(
                SemanticCitationValidation(
                    chunk_id=chunk_id,
                    supported=False,
                    reason=(
                        "Citation was not present in "
                        "retrieved evidence."
                    ),
                )
            )
            continue

        evidence = evidence_by_chunk[chunk_id]

        result = chain.invoke(
            {
                "answer": answer.answer,
                "chunk_id": chunk_id,
                "evidence": evidence["text"],
            }
        )

        validations.append(result)

    all_supported = all(
        validation.supported
        for validation in validations
    )

    return SemanticValidationResult(
        all_citations_supported=all_supported,
        validations=validations,
    )


# ============================================================
# Printing helpers
# ============================================================

def print_validation(result):
    print("\n" + "=" * 80)
    print("CITATION VALIDATION")
    print("=" * 80)

    print(
        f"\nAll citations valid: "
        f"{result.all_citations_valid}"
    )

    for validation in result.validations:
        status = (
            "VALID"
            if validation.valid
            else "INVALID"
        )

        print(f"\n[{status}] {validation.chunk_id}")
        print(f"Reason: {validation.reason}")


def print_semantic_validation(result):
    print("\n" + "=" * 80)
    print("SEMANTIC CITATION VALIDATION")
    print("=" * 80)

    print(
        f"\nAll citations supported: "
        f"{result.all_citations_supported}"
    )

    for validation in result.validations:
        status = (
            "SUPPORTED"
            if validation.supported
            else "NOT SUPPORTED"
        )

        print(f"\n[{status}] {validation.chunk_id}")
        print(f"Reason: {validation.reason}")
