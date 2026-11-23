from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()

MODEL_NAME = "openai/gpt-oss-120b"


class Citation(BaseModel):
    chunk_id: str
    page: int
    company: str
    year: int


class FinancialAnswer(BaseModel):
    answer: str
    citations: List[Citation]
    confidence: str = Field(
        description="Confidence level: high, medium, or low."
    )


SYSTEM_PROMPT = """
You are a financial research assistant.

Answer the user's question using ONLY the supplied annual-report
evidence.

Rules:

1. Never use outside knowledge.
2. Never invent facts, numbers, or citations.
3. Every factual claim must be supported by the supplied evidence.
4. Citations must use only chunk IDs present in the evidence.
5. Preserve financial units exactly as stated.
6. If the evidence is insufficient, say so clearly.
7. Keep the answer concise.
8. Confidence must be high, medium, or low.

CRITICAL CITATION RULE:

For every citation, copy the metadata EXACTLY from the supplied
evidence for that chunk.

The citation fields must be:

- chunk_id = exact Chunk ID
- company = exact Company metadata
- year = exact Year metadata
- page = exact Page metadata

DO NOT infer the page number from the text.

DO NOT use the page number printed inside the annual report text.

The "Page" metadata supplied above the text is authoritative.

Return the answer using the required structured schema.
"""


def load_llm():

    llm = ChatGroq(
        model=MODEL_NAME,
        temperature=0,
        api_key=None,
    )

    return llm.with_structured_output(FinancialAnswer)


def build_evidence(results):

    evidence = []

    for i, result in enumerate(results, start=1):

        evidence.append(
            f"""
Evidence {i}

Chunk ID: {result["chunk_id"]}
Company: {result.get("company")}
Year: {result.get("year")}
Page: {result.get("page")}

Text:
{result["text"]}
"""
        )

    return "\n".join(evidence)


def generate_answer(query, results, llm):

    evidence = build_evidence(results)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            (
                "human",
                """
Question:
{query}

Evidence:
{evidence}

Answer the question using only this evidence.

Remember:
citation page numbers MUST come from the supplied Page metadata,
not from page numbers appearing inside the document text.
""",
            ),
        ]
    )

    chain = prompt | llm

    return chain.invoke(
        {
            "query": query,
            "evidence": evidence,
        }
    )