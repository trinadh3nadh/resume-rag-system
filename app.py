import streamlit as st
import time
import logging
from PyPDF2 import PdfReader

from retriever import retrieve_top_chunks
from reranker import rerank_chunks
from scorer import compute_structured_score
from llm_feedback import generate_rag_feedback

# -----------------------------
# Streamlit Config
# -----------------------------
st.set_page_config(
    page_title="Resume Intelligence RAG System",
    layout="wide"
)

logging.basicConfig(level=logging.INFO)

st.title("🚀 Resume Intelligence RAG System")

# -----------------------------
# File Upload + JD Input
# -----------------------------
uploaded_file = st.file_uploader("Upload Resume (PDF)", type="pdf")
job_description = st.text_area("Paste Job Description")

# -----------------------------
# Cached Retrieval
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
if uploaded_file and job_description:

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

    st.subheader("📊 Structured Scores")
    st.json(scores)

    # ---- Latency Display ----
    st.subheader("⏱ Latency Metrics")
    st.write({
        "retrieval_time_sec": round(retrieval_time, 3),
        "rerank_time_sec": round(rerank_time, 3)
    })

    # ---- Top Resume Sections ----
    st.subheader("🔎 Top Relevant Resume Sections")

    for chunk, score in ranked_chunks[:5]:
        st.markdown(f"**Relevance Score:** {round(float(score), 3)}")
        st.write(chunk)
        st.markdown("---")

    # ---- LLM Evaluation ----
    if st.button("Generate AI Evaluation"):
        st.info("LLM evaluation is available in the local version only.")