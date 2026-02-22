import streamlit as st
import time
import logging
from PyPDF2 import PdfReader

from retriever import retrieve_top_chunks
from reranker import rerank_chunks
from scorer import compute_structured_score

# -----------------------------
# Streamlit Config
# -----------------------------
st.set_page_config(
    page_title="Resume Intelligence RAG System",
    layout="wide"
)

logging.basicConfig(level=logging.INFO)

# -----------------------------
# Header Branding
# -----------------------------
st.markdown(
    """
    # 🚀 Resume Intelligence RAG System  
    ### Built by **Trinadh Kolluboyina**  
    AI Engineer | GenAI | RAG Systems | Agentic AI  
    ---
    """
)

# -----------------------------
# Inputs
# -----------------------------
uploaded_file = st.file_uploader("Upload Resume (PDF)", type="pdf")

job_description = st.text_area(
    "Paste Job Description",
    placeholder="Paste the full job description here..."
)

analyze_button = st.button(
    "🔍 Analyze Resume",
    disabled=not (uploaded_file and job_description.strip())
)

# -----------------------------
# Caching Layer
# -----------------------------
@st.cache_data(show_spinner=False)
def cached_retrieval(resume_text, job_description):
    return retrieve_top_chunks(resume_text, job_description)

@st.cache_data(show_spinner=False)
def cached_reranking(job_description, retrieved_chunks):
    return rerank_chunks(job_description, retrieved_chunks)

# -----------------------------
# Main Pipeline
# -----------------------------
if analyze_button:

    start_total = time.time()

    # ---- Parse Resume ----
    try:
        reader = PdfReader(uploaded_file)
        resume_text = ""

        for page in reader.pages:
            resume_text += page.extract_text() or ""

        if not resume_text.strip():
            st.warning("Resume extraction failed.")
            st.stop()

        st.success("Resume successfully parsed.")

    except Exception as e:
        st.error(f"PDF Parsing Error: {str(e)}")
        st.stop()

    # ---- Retrieval ----
    start_retrieval = time.time()
    retrieved_chunks = cached_retrieval(resume_text, job_description)
    retrieval_time = time.time() - start_retrieval

    if not retrieved_chunks:
        st.warning("No resume sections found.")
        st.stop()

    # ---- Reranking ----
    start_rerank = time.time()
    ranked_chunks = cached_reranking(job_description, retrieved_chunks)
    rerank_time = time.time() - start_rerank

    # ---- Structured Scoring ----
    scores = compute_structured_score(ranked_chunks)

    st.subheader("🎯 Final Resume Match Evaluation")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Match Score",
            value=f"{scores['percentage_score']}%"
        )

    with col2:
        st.metric(
            label="Match Rating",
            value=scores["rating"]
        )

    with col3:
        st.metric(
            label="Confidence",
            value=f"{scores['confidence']}%"
        )

    # ---- Rating Message ----
    if scores["rating"] == "Excellent Match":
        st.success("Your resume is strongly aligned with this role.")
    elif scores["rating"] == "Good Match":
        st.info("Your resume is well aligned. Minor improvements could strengthen it further.")
    elif scores["rating"] == "Moderate Match":
        st.warning("Your resume partially matches this role. Consider refining skills and experience.")
    else:
        st.error("Your resume has low alignment with this job description.")

    # ---- Latency Metrics ----
    st.subheader("⏱ System Performance Metrics")
    st.write({
        "retrieval_time_sec": round(retrieval_time, 3),
        "rerank_time_sec": round(rerank_time, 3),
        "total_pipeline_time_sec": round(time.time() - start_total, 3)
    })

    # ---- Top Relevant Resume Sections ----
    st.subheader("🔎 Top Relevant Resume Sections")

    for chunk, score in ranked_chunks[:5]:
        st.markdown("---")
        st.write(chunk)

# -----------------------------
# Footer
# -----------------------------
st.markdown(
    """
    ---
    Built with ❤️ using SentenceTransformers, FAISS, Cross-Encoders, and Streamlit  
    © 2026 Trinadh Kolluboyina
    """
)
