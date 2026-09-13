from __future__ import annotations

import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

from src.file_utils import read_uploaded_file
from src.rag_engine import (
    create_documents,
    split_documents,
    build_vectorstore,
    run_career_coach,
    generate_complete_report,
)

# Page configuration
st.set_page_config(
    page_title="AI Career Coach - RAG Intelligence",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load environment
PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env")

# -------------------------------------------------------------
# Custom Modern CSS Design System
# -------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Hero Header */
    .hero-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f766e 100%);
        padding: 32px 36px;
        border-radius: 20px;
        color: #ffffff;
        margin-bottom: 25px;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .hero-badge {
        display: inline-block;
        padding: 6px 14px;
        background: rgba(45, 212, 191, 0.15);
        color: #2dd4bf;
        font-size: 13px;
        font-weight: 700;
        border-radius: 30px;
        border: 1px solid rgba(45, 212, 191, 0.3);
        margin-bottom: 12px;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .hero-title {
        font-size: 32px;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
        color: #ffffff;
    }
    .hero-subtitle {
        font-size: 15px;
        color: #cbd5e1;
        margin-top: 8px;
        max-width: 800px;
        line-height: 1.5;
    }

    /* Metric Badges */
    .metric-container {
        display: flex;
        gap: 15px;
        margin: 15px 0;
    }
    .metric-pill {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 12px 18px;
        flex: 1;
        text-align: center;
    }
    .metric-value {
        font-size: 22px;
        font-weight: 800;
        color: #0f766e;
    }
    .metric-label {
        font-size: 12px;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
    }

    /* Response Box */
    .response-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 5px solid #0d9488;
        border-radius: 14px;
        padding: 24px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04);
        margin-top: 15px;
        line-height: 1.6;
        color: #1e293b;
    }

    /* Pipeline Step */
    .pipeline-step {
        padding: 10px 14px;
        background: #f1f5f9;
        border-radius: 10px;
        margin-bottom: 8px;
        font-size: 13px;
        font-weight: 500;
        color: #334155;
        border-left: 3px solid #0ea5e9;
    }

    /* Buttons & Controls */
    .stButton>button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------
# App State Management
# -------------------------------------------------------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""
if "jd_text" not in st.session_state:
    st.session_state.jd_text = ""

# -------------------------------------------------------------
# Sidebar Configuration
# -------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Engine Settings")

    api_key_input = st.text_input(
        "Groq API Key",
        value=os.getenv("GROQ_API_KEY", ""),
        type="password",
        help="Enter your Groq API key. If empty, falls back to the .env file.",
    )

    selected_model = st.selectbox(
        "LLM Inference Model",
        [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
            "qwen/qwen3-32b",
        ],
        index=0,
    )

    st.markdown("---")
    st.markdown("### 🎛️ RAG Parameters")
    chunk_size = st.slider("Chunk Size (Tokens/Chars)", 300, 1500, 800, 100)
    chunk_overlap = st.slider("Chunk Overlap", 0, 400, 150, 50)
    retrieval_k = st.slider("Top-K Retrieved Chunks", 2, 10, 5, 1)

    st.markdown("---")
    st.markdown("### 🧬 Architecture Flow")
    st.markdown(
        """
        <div class="pipeline-step">1. 📄 Multi-Doc Ingestion (PDF / DOCX)</div>
        <div class="pipeline-step">2. ✂️ Recursive Semantic Chunking</div>
        <div class="pipeline-step">3. 🧠 HuggingFace MiniLM Embeddings</div>
        <div class="pipeline-step">4. 🗄️ ChromaDB In-Memory Vector Store</div>
        <div class="pipeline-step">5. 🔍 Cosine Similarity Top-K Retrieval</div>
        <div class="pipeline-step">6. 🚀 Groq Ultra-Fast LLM Generation</div>
        """,
        unsafe_allow_html=True,
    )

# -------------------------------------------------------------
# Hero Banner
# -------------------------------------------------------------
st.markdown(
    """
    <div class="hero-container">
        <span class="hero-badge">⚡ Production RAG Pipeline</span>
        <h1 class="hero-title">AI Career Coach & Resume Intelligence</h1>
        <p class="hero-subtitle">
            Upload your resume and target job description. Our Retrieval-Augmented Generation (RAG) system 
            analyzes technical skill gaps, evaluates ATS compatibility, and prepares tailored interview strategies.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------
# Sample Data Loader Helper
# -------------------------------------------------------------
sample_col1, sample_col2 = st.columns([3, 1])
with sample_col2:
    if st.button("⚡ Load Sample Data", use_container_width=True):
        sample_resume_path = PROJECT_DIR / "data" / "sample_resume.txt"
        sample_jd_path = PROJECT_DIR / "data" / "sample_job_description.txt"

        if sample_resume_path.exists() and sample_jd_path.exists():
            st.session_state.resume_text = sample_resume_path.read_text(encoding="utf-8")
            st.session_state.jd_text = sample_jd_path.read_text(encoding="utf-8")
            st.toast("Sample Resume and Job Description loaded!", icon="✅")
        else:
            st.error("Sample files not found in data/ directory.")

# -------------------------------------------------------------
# Input Cards (Resume & JD)
# -------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📄 1. Candidate Resume")
    resume_tab1, resume_tab2 = st.tabs(["📁 File Upload", "✍️ Paste Text"])
    with resume_tab1:
        resume_file = st.file_uploader(
            "Upload Resume (.pdf, .docx, .txt)",
            type=["pdf", "docx", "txt"],
            key="resume_uploader",
        )
        if resume_file:
            st.session_state.resume_text = read_uploaded_file(resume_file)
    with resume_tab2:
        st.session_state.resume_text = st.text_area(
            "Resume Content",
            value=st.session_state.resume_text,
            height=200,
            placeholder="Paste raw resume text here...",
        )

    if st.session_state.resume_text:
        word_count = len(st.session_state.resume_text.split())
        st.caption(f"📊 Resume Loaded: ~{word_count} words ({len(st.session_state.resume_text)} characters)")

with col2:
    st.markdown("### 💼 2. Target Job Description")
    jd_tab1, jd_tab2 = st.tabs(["📁 File Upload", "✍️ Paste Text"])
    with jd_tab1:
        jd_file = st.file_uploader(
            "Upload Job Description (.pdf, .docx, .txt)",
            type=["pdf", "docx", "txt"],
            key="jd_uploader",
        )
        if jd_file:
            st.session_state.jd_text = read_uploaded_file(jd_file)
    with jd_tab2:
        st.session_state.jd_text = st.text_area(
            "Job Description Content",
            value=st.session_state.jd_text,
            height=200,
            placeholder="Paste target job requirements, skills, and qualifications here...",
        )

    if st.session_state.jd_text:
        jd_word_count = len(st.session_state.jd_text.split())
        st.caption(f"📊 JD Loaded: ~{jd_word_count} words ({len(st.session_state.jd_text)} characters)")

# -------------------------------------------------------------
# Build RAG Index Action
# -------------------------------------------------------------
st.markdown("---")

btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])
with btn_col2:
    build_index_btn = st.button(
        "🚀 Build & Index Knowledge Base",
        type="primary",
        use_container_width=True,
    )

if build_index_btn:
    if not st.session_state.resume_text.strip() or not st.session_state.jd_text.strip():
        st.error("⚠️ Please provide both Candidate Resume and Target Job Description before indexing.")
    else:
        with st.status("🔄 Initializing RAG Pipeline...", expanded=True) as status:
            st.write("📥 Step 1: Converting documents into LangChain Document schema...")
            docs = create_documents(st.session_state.resume_text, st.session_state.jd_text)

            st.write(f"✂️ Step 2: Splitting documents (Chunk size: {chunk_size}, Overlap: {chunk_overlap})...")
            chunks = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

            st.write("🧠 Step 3: Generating HuggingFace sentence embeddings & building ChromaDB vector index...")
            vectorstore = build_vectorstore(chunks)

            st.session_state.vectorstore = vectorstore
            st.session_state.chunks = chunks

            status.update(label="✅ RAG Knowledge Base Successfully Built!", state="complete", expanded=False)

# Show RAG Metrics when vectorstore is available
if st.session_state.vectorstore:
    st.markdown(
        f"""
        <div class="metric-container">
            <div class="metric-pill">
                <div class="metric-value">2</div>
                <div class="metric-label">Source Documents</div>
            </div>
            <div class="metric-pill">
                <div class="metric-value">{len(st.session_state.chunks)}</div>
                <div class="metric-label">Indexed Chunks</div>
            </div>
            <div class="metric-pill">
                <div class="metric-value">all-MiniLM-L6</div>
                <div class="metric-label">Embedding Model</div>
            </div>
            <div class="metric-pill">
                <div class="metric-value">ChromaDB</div>
                <div class="metric-label">Vector Store</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------
    # Insights & Coaching Tabs
    # -------------------------------------------------------------
    tab_report, tab_qa, tab_context = st.tabs(
        ["📊 Executive Career Report", "💬 Interactive RAG Coach", "🔍 Retrieved Context Inspector"]
    )

    # Tab 1: Comprehensive Career Report
    with tab_report:
        st.markdown("#### 📑 Comprehensive Fitment & Strategy Report")
        st.write("Generate an end-to-end evaluation covering ATS score, skill gaps, resume bullet upgrades, and project recommendations.")

        if st.button("✨ Generate Full Career Report", type="primary", key="gen_report_btn"):
            with st.spinner(f"Running multi-stage retrieval with {selected_model}..."):
                try:
                    report, sources = generate_complete_report(
                        st.session_state.vectorstore,
                        st.session_state.resume_text,
                        st.session_state.jd_text,
                        model=selected_model,
                        api_key=api_key_input,
                    )
                    st.markdown(f'<div class="response-card">{report}</div>', unsafe_allow_html=True)
                    st.download_button(
                        label="📥 Download Report (.md)",
                        data=report,
                        file_name="ai_career_coach_report.md",
                        mime="text/markdown",
                    )
                except Exception as e:
                    st.error(f"Error generating report: {e}")

    # Tab 2: Interactive Q&A
    with tab_qa:
        st.markdown("#### 🎯 Ask Targeted Career Questions")

        quick_prompts = [
            "How well does this resume match the job description? Give an estimated match percentage.",
            "What critical technical and soft skills are missing from this resume?",
            "How can I rewrite my resume experience bullets to better target this role?",
            "Suggest 3 high-impact portfolio projects that will bridge the skill gaps.",
            "Generate 5 challenging technical and behavioral interview questions based on the gaps.",
        ]

        selected_prompt = st.selectbox("⚡ Quick Inquiry Presets", quick_prompts)
        custom_query = st.text_input("💬 Or Ask a Custom Question", placeholder="e.g., Does my background qualify for the senior seniority level?")
        final_query = custom_query.strip() if custom_query.strip() else selected_prompt

        if st.button("🤖 Get Career Coach Advice", type="primary", key="ask_coach_btn"):
            with st.spinner("Retrieving relevant context from ChromaDB & querying Groq LLM..."):
                try:
                    answer, sources = run_career_coach(
                        st.session_state.vectorstore,
                        st.session_state.resume_text,
                        st.session_state.jd_text,
                        final_query,
                        model=selected_model,
                        api_key=api_key_input,
                    )
                    st.markdown(f'<div class="response-card">{answer}</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error querying AI Coach: {e}")

    # Tab 3: RAG Grounding & Context Inspector
    with tab_context:
        st.markdown("#### 🔍 Ground Truth & Retrieved Chunks")
        st.caption("Inspect the exact text chunks retrieved from ChromaDB that grounded the LLM's answers.")

        if st.session_state.chunks:
            for i, chunk in enumerate(st.session_state.chunks[:6], 1):
                with st.expander(f"📦 Chunk {i} | Source: {chunk.metadata.get('source', 'Unknown')} ({chunk.metadata.get('doc_type', 'doc')})"):
                    st.code(chunk.page_content, language="markdown")
        else:
            st.info("Build the RAG index to inspect chunk embeddings.")

else:
    st.info("💡 **Getting Started**: Upload your Resume & Job Description above (or click **'⚡ Load Sample Data'**), then click **'🚀 Build & Index Knowledge Base'**.")
