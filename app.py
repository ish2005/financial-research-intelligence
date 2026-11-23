import tempfile
from pathlib import Path

import streamlit as st

from src.pipeline import (
    FinancialRAGPipeline,
    get_pipeline,
)

from src.ingestion.pdf_indexer import (
    build_uploaded_index,
)


st.set_page_config(
    page_title="Financial Research Intelligence",
    page_icon="📊",
    layout="wide",
)


st.title(
    "📊 Financial Research Intelligence"
)

st.caption(
    "AI-powered financial document research with "
    "hybrid retrieval, reranking, and citation-grounded answers."
)


# --------------------------------------------------
# Session state
# --------------------------------------------------

if "uploaded_pipeline" not in st.session_state:
    st.session_state.uploaded_pipeline = None

if "uploaded_files_signature" not in st.session_state:
    st.session_state.uploaded_files_signature = None

if "uploaded_documents" not in st.session_state:
    st.session_state.uploaded_documents = []

if "last_question" not in st.session_state:
    st.session_state.last_question = None

if "last_answer" not in st.session_state:
    st.session_state.last_answer = None

if "last_evidence" not in st.session_state:
    st.session_state.last_evidence = None


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.header("📄 Documents")

    uploaded_files = st.file_uploader(
        "Upload financial PDF reports",
        type=["pdf"],
        accept_multiple_files=True,
        help=(
            "Upload one or more annual reports "
            "or financial documents."
        ),
    )

    if uploaded_files:

        current_signature = tuple(
            (
                file.name,
                file.size,
            )
            for file in uploaded_files
        )

        if (
            st.session_state.uploaded_files_signature
            != current_signature
        ):

            with st.spinner(
                "Parsing and indexing documents..."
            ):

                temp_files = []

                try:

                    for uploaded_file in uploaded_files:

                        tmp = tempfile.NamedTemporaryFile(
                            delete=False,
                            suffix=".pdf",
                        )

                        tmp.write(
                            uploaded_file.getbuffer()
                        )

                        tmp.close()

                        temp_files.append(
                            {
                                "path": tmp.name,
                                "filename": uploaded_file.name,
                            }
                        )

                    base_pipeline = get_pipeline()

                    indexed = build_uploaded_index(
                        pdf_documents=temp_files,
                        model=base_pipeline.vector_model,
                    )

                    st.session_state.uploaded_pipeline = (
                        FinancialRAGPipeline.from_existing_models(
                            vector_index=(
                                indexed["vector_index"]
                            ),
                            vector_metadata=(
                                indexed["vector_metadata"]
                            ),
                            vector_model=(
                                indexed["vector_model"]
                            ),
                            bm25=(
                                indexed["bm25"]
                            ),
                            bm25_chunks=(
                                indexed["bm25_chunks"]
                            ),
                            base_pipeline=base_pipeline,
                        )
                    )

                    st.session_state.uploaded_documents = (
                        indexed["document_metadata"]
                    )

                    st.session_state.uploaded_files_signature = (
                        current_signature
                    )

                    st.session_state.last_question = None
                    st.session_state.last_answer = None
                    st.session_state.last_evidence = None

                finally:

                    for item in temp_files:

                        try:

                            Path(
                                item["path"]
                            ).unlink(
                                missing_ok=True
                            )

                        except Exception:
                            pass

        if st.session_state.uploaded_documents:

            st.success(
                f"{len(st.session_state.uploaded_documents)} "
                f"document(s) indexed"
            )

            for document in (
                st.session_state.uploaded_documents
            ):

                st.markdown(
                    f"**{document['company']}**"
                )

                st.caption(
                    f"{document['year']} · "
                    f"{document['pages']} pages · "
                    f"{document['chunks']} chunks"
                )

    else:

        st.info(
            "No uploaded documents. "
            "The built-in development corpus will be used."
        )


# --------------------------------------------------
# Select pipeline
# --------------------------------------------------

if (
    st.session_state.uploaded_pipeline
    is not None
):

    pipeline = (
        st.session_state.uploaded_pipeline
    )

    st.info(
        "🔎 Searching uploaded documents"
    )

else:

    with st.spinner(
        "Loading Financial RAG pipeline..."
    ):

        pipeline = get_pipeline()


# --------------------------------------------------
# Question interface
# --------------------------------------------------

st.subheader(
    "Ask a financial question"
)

question = st.text_area(
    "Question",
    placeholder=(
        "Examples:\n"
        "• What was Apple's revenue in 2025?\n"
        "• What was Microsoft's net income in 2024?\n"
        "• What are NVIDIA's major business risks?\n"
        "• Compare revenue between Apple and Microsoft."
    ),
    height=120,
    label_visibility="collapsed",
)


# --------------------------------------------------
# Clear previous answer
# --------------------------------------------------

if st.session_state.last_answer is not None:

    if st.button(
        "Clear previous answer",
        use_container_width=True,
    ):

        st.session_state.last_question = None
        st.session_state.last_answer = None
        st.session_state.last_evidence = None
        st.rerun()


# --------------------------------------------------
# Ask
# --------------------------------------------------

if st.button(
    "Ask",
    type="primary",
    use_container_width=True,
):

    normalized_question = question.strip()

    if not normalized_question:

        st.warning(
            "Please enter a question."
        )

    elif (
        st.session_state.last_question
        == normalized_question
        and st.session_state.last_answer
        is not None
    ):

        st.info(
            "Using the completed result for this question."
        )

    else:

        with st.spinner(
            "Researching financial documents..."
        ):

            try:

                answer, evidence = pipeline.ask(
                    query=normalized_question
                )

                st.session_state.last_question = (
                    normalized_question
                )

                st.session_state.last_answer = answer

                st.session_state.last_evidence = evidence

            except Exception as error:

                st.error(
                    f"Unable to process the question: "
                    f"{type(error).__name__}: {error}"
                )


# --------------------------------------------------
# Display completed answer
# --------------------------------------------------

if st.session_state.last_answer is not None:

    answer = st.session_state.last_answer

    st.subheader(
        "Answer"
    )

    st.write(
        answer.answer
    )

    # --------------------------------------------------
    # Confidence
    # --------------------------------------------------

    if answer.confidence == "high":

        st.success(
            "Confidence: HIGH"
        )

    elif answer.confidence == "medium":

        st.warning(
            "Confidence: MEDIUM"
        )

    else:

        st.error(
            "Confidence: LOW"
        )

    # --------------------------------------------------
    # Sources
    # --------------------------------------------------

    st.subheader(
        "Sources"
    )

    if not answer.citations:

        st.info(
            "No citations were returned."
        )

    else:

        for citation in answer.citations:

            st.markdown(
                f"**{citation.company} "
                f"{citation.year}** · "
                f"Page {citation.page}"
            )

            st.caption(
                citation.chunk_id
            )
