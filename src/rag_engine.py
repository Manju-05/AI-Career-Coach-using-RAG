import os
import tempfile
from pathlib import Path
from typing import List, Tuple
from functools import lru_cache

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Disable ChromaDB telemetry errors
os.environ["ANONYMIZED_TELEMETRY"] = "False"

PROJECT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_DIR / ".env")

DB_DIR = "career_coach_chroma_db"

def get_llm(model: str | None = None, temperature: float = 0.2, api_key: str | None = None):
    key = (api_key or os.getenv("GROQ_API_KEY", "")).strip()
    if not key:
        raise ValueError("GROQ_API_KEY not found. Please provide a valid Groq API key.")
    model = model or os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip()
    return ChatGroq(model=model, temperature=temperature, groq_api_key=key)


@lru_cache(maxsize=1)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# -----------------------------
# Stage 1: Load Documents
# -----------------------------
def load_text_file(file_path: str, source_name: str, doc_type: str) -> List[Document]:
    path = Path(file_path)
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [Document(page_content=text, metadata={"source": source_name, "doc_type": doc_type})]


def create_documents(resume_text: str, jd_text: str) -> List[Document]:
    return [
        Document(page_content=resume_text, metadata={"source": "uploaded_resume", "doc_type": "resume"}),
        Document(page_content=jd_text, metadata={"source": "uploaded_job_description", "doc_type": "job_description"}),
    ]


# -----------------------------
# Stage 2: Split Documents
# -----------------------------
def split_documents(docs: List[Document], chunk_size: int = 800, chunk_overlap: int = 150) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    return splitter.split_documents(docs)


# -----------------------------
# Stage 3 + 4: Embeddings + Vector DB
# -----------------------------
def build_vectorstore(chunks: List[Document], persist_directory: str = DB_DIR):
    base_directory = Path(persist_directory)
    base_directory.parent.mkdir(parents=True, exist_ok=True)
    build_directory = tempfile.mkdtemp(
        prefix=f"{base_directory.name}_",
        dir=base_directory.parent,
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=build_directory,
        collection_name="career_coach_rag",
    )
    return vectorstore


# -----------------------------
# Stage 5: Retrieve Context
# -----------------------------
def retrieve_context(vectorstore, query: str, k: int = 5) -> Tuple[str, List[Document]]:
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(query)
    context = "\n\n".join([f"SOURCE: {d.metadata}\nCONTENT:\n{d.page_content}" for d in docs])
    return context, docs


# -----------------------------
# Stage 6: Generate Answer
# -----------------------------
def run_career_coach(vectorstore, resume_text: str, jd_text: str, question: str, model: str | None = None, api_key: str | None = None):
    llm = get_llm(model=model, api_key=api_key)

    retrieval_query = f"""
    Resume content and job description content relevant to this career coaching question:
    {question}
    """

    context, source_docs = retrieve_context(vectorstore, retrieval_query, k=6)

    prompt = ChatPromptTemplate.from_template("""
You are an expert AI Career Coach for students, freshers and working professionals.
Use ONLY the given context from the resume and job description.
Do not invent skills, experience or job requirements.

CONTEXT:
{context}

USER QUESTION:
{question}

Give a clear, practical answer with these sections when relevant:
1. Current Match Summary
2. Strengths
3. Missing Skills / Gaps
4. Recommended Improvements
5. Suggested Projects
6. Interview Preparation Tips

Keep the answer simple, actionable and beginner-friendly.
""")

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    return answer, source_docs


def generate_complete_report(vectorstore, resume_text: str, jd_text: str, model: str | None = None, api_key: str | None = None):
    question = """
    Analyze this resume against this job description. Provide ATS-style score, skill match, missing skills,
    resume improvement suggestions, project suggestions, and interview questions.
    """
    return run_career_coach(vectorstore, resume_text, jd_text, question, model=model, api_key=api_key)
