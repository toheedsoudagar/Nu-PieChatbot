# ingest.py
from pathlib import Path
import os
from collections import defaultdict
import textwrap

# Try to import streamlit for secrets (preferred in your Streamlit app).
# If streamlit is not available, fall back to environment variables.
try:
    import streamlit as st
    _HAS_STREAMLIT = True
except Exception:
    _HAS_STREAMLIT = False

# LangChain / Chroma imports
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    UnstructuredExcelLoader
)
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_unstructured.document_loaders import UnstructuredLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from tempfile import TemporaryDirectory

# Optional OCR imports (used only if fallback is needed)
try:
    from pdf2image import convert_from_path
    import pytesseract
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

# ---------- Defaults (can be overridden by Streamlit secrets / env) ----------
DOCS_DIR = "docs"
DB_DIR = "chroma_db"
EMBEDDING_MODEL = "nomic-embed-text"        # default embedding model (from screenshot)
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 400
# ---------------------------------------------------------------------------

# Streamlit secrets or env variables for Ollama cloud
if _HAS_STREAMLIT:
    # Expect these keys in .streamlit/secrets.toml
    # OLLAMA_HOST, OLLAMA_API_KEY, EMBEDDING_MODEL (optional), CHROMA_DB_DIR (optional)
    OLLAMA_BASE_URL = st.secrets.get("OLLAMA_HOST")
    OLLAMA_API_KEY = st.secrets.get("OLLAMA_API_KEY")
    EMBEDDING_MODEL = st.secrets.get("EMBEDDING_MODEL", EMBEDDING_MODEL)
    DB_DIR = st.secrets.get("CHROMA_DB_DIR", DB_DIR)
else:
    OLLAMA_BASE_URL = os.environ.get("OLLAMA_HOST") or os.environ.get("OLLAMA_BASE_URL")
    OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY")
    EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", EMBEDDING_MODEL)
    DB_DIR = os.environ.get("CHROMA_DB_DIR", DB_DIR)


def ocr_pdf_to_documents(pdf_path):
    """
    Convert PDF pages to text via OCR and return a list of LangChain Documents.
    Requires pdf2image + pytesseract and a Tesseract binary installed on the system.
    """
    docs = []
    if not OCR_AVAILABLE:
        print(f"[ingest][ocr] OCR dependencies not available (pdf2image/pytesseract). Skipping OCR for {pdf_path}.")
        return docs

    print(f"[ingest][ocr] Running OCR fallback on {pdf_path} .")
    try:
        with TemporaryDirectory() as tmpdir:
            images = convert_from_path(pdf_path, dpi=200, output_folder=tmpdir)
            for i, img in enumerate(images):
                try:
                    text = pytesseract.image_to_string(img)
                except Exception as e:
                    print(f"[ingest][ocr] pytesseract failed on page {i}: {e}")
                    text = ""
                if text and text.strip():
                    docs.append(Document(page_content=text, metadata={"source": Path(pdf_path).name}))
    except Exception as e:
        print(f"[ingest][ocr] Failed to OCR {pdf_path}: {e}")
    print(f"[ingest][ocr] OCR produced {len(docs)} page documents for {pdf_path}")
    return docs


def load_all_documents():
    docs = []
    path = Path(DOCS_DIR)

    if not path.exists():
        raise ValueError(f"Docs directory {DOCS_DIR} does not exist. Create it and add documents.")

    for file in sorted(path.iterdir()):
        try:
            suffix = file.suffix.lower()

            if suffix == ".pdf":
                print(f"[ingest] Loading PDF → {file}")
                pdf_docs = PyPDFLoader(str(file)).load()

                # If extraction looks sparse, fallback to OCR (useful for scanned PDFs)
                non_empty_pages = sum(1 for d in pdf_docs if (d.page_content or "").strip())
                if non_empty_pages < max(1, len(pdf_docs) // 2):
                    print("[ingest] PDF text extraction is sparse — attempting OCR fallback.")
                    ocr_docs = ocr_pdf_to_documents(str(file))
                    if ocr_docs:
                        pdf_docs = ocr_docs

                for d in pdf_docs:
                    d.metadata["source"] = file.name
                docs.extend(pdf_docs)

            elif suffix == ".txt":
                print(f"[ingest] Loading TXT → {file}")
                txt_docs = TextLoader(str(file)).load()
                for d in txt_docs:
                    d.metadata["source"] = file.name
                docs.extend(txt_docs)

            elif suffix == ".csv":
                print(f"[ingest] Loading CSV → {file}")
                csv_docs = CSVLoader(str(file)).load()
                for d in csv_docs:
                    d.metadata["source"] = file.name
                docs.extend(csv_docs)

            elif suffix in [".xls", ".xlsx"]:
                print(f"[ingest] Loading XLSX/XLS (Unstructured) → {file}")
                excel_docs = UnstructuredExcelLoader(str(file)).load()
                for d in excel_docs:
                    d.metadata["source"] = file.name
                docs.extend(excel_docs)

            elif suffix in [".doc", ".docx", ".ppt", ".pptx", ".html", ".htm"]:
                print(f"[ingest] Loading {suffix.upper()} (Unstructured) → {file}")
                other_docs = UnstructuredLoader(str(file)).load()
                for d in other_docs:
                    d.metadata["source"] = file.name
                docs.extend(other_docs)

        except Exception as e:
            print(f"[ingest] Failed to load {file}: {e}")

    if len(docs) == 0:
        raise ValueError(f"No documents found or extracted in ./{DOCS_DIR} folder!")

    print(f"[ingest] Loaded {len(docs)} raw document sections.")
    return docs


def split_into_chunks(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)
    print(f"[ingest] Created {len(chunks)} chunks.")

    # Report summary by source (counts + preview)
    report_chunk_stats(chunks)
    return chunks


def report_chunk_stats(chunks, max_preview_chars=300):
    """
    Print how many chunks were created per source file and show a short preview.
    """
    by_source = defaultdict(list)
    for i, ch in enumerate(chunks):
        src = ch.metadata.get("source", "unknown")
        by_source[src].append((i, ch.page_content or ""))

    print("[ingest] Chunk statistics by source:")
    for src, items in by_source.items():
        print(f" - {src}: {len(items)} chunks")
        idx, text = items[0]
        preview = textwrap.shorten((text or "").replace("\n", " "), width=max_preview_chars, placeholder="...")
        print(f"    preview (chunk #{idx}): {preview}")
    print("[ingest] End chunk statistics.\n")


def build_vectorstore(chunks):
    # Use Ollama embeddings
    print(f"[ingest] Building embeddings using model: {EMBEDDING_MODEL}")

    # Build kwargs for optional cloud/remote settings (backwards compatible)
    ollama_kwargs = {}
    if OLLAMA_BASE_URL:
        ollama_kwargs["base_url"] = OLLAMA_BASE_URL
    if OLLAMA_API_KEY:
        ollama_kwargs["api_key"] = OLLAMA_API_KEY

    # Instantiate embeddings with optional remote/cloud args
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, **ollama_kwargs)

    vectordb = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=DB_DIR
    )
    print("[ingest] Chroma DB built and persisted.")
    return vectordb


def run_ingestion_if_needed():
    # If DB does not exist or empty, run ingestion
    if not os.path.exists(DB_DIR) or len(os.listdir(DB_DIR)) == 0:
        print("[ingest] No DB found → Running ingestion...")
        docs = load_all_documents()
        chunks = split_into_chunks(docs)

        # Filter complex metadata before building the vector store
        print("[ingest] Filtering complex metadata for Chroma compatibility...")
        chunks = filter_complex_metadata(chunks)

        build_vectorstore(chunks)
        print("[ingest] Ingestion complete.")
    else:
        print("[ingest] DB already exists → Skipping ingestion.")


if __name__ == "__main__":
    run_ingestion_if_needed()
