# ingest.py
"""
Simple ingestion: walk a source_dir, load .txt/.md/.pdf files into Documents,
compute embeddings using Ollama cloud embedding model, and persist a Chroma DB.

Usage:
    python ingest.py  # uses default folders below
or
    from ingest import run_ingest; run_ingest(source_dir="docs", persist_directory="chroma_db")
"""

import os
import traceback
from pathlib import Path

# read secrets from streamlit if available, else env
try:
    import streamlit as st
    _st = True
except Exception:
    st = None
    _st = False

def _get_secret(name, default=""):
    if _st:
        try:
            val = st.secrets.get(name)
            if val:
                return val
        except Exception:
            pass
    return os.environ.get(name, default)

EMBEDDING_MODEL = _get_secret("EMBEDDING_MODEL", os.environ.get("EMBEDDING_MODEL", "nomic-embed-text"))
OLLAMA_HOST = _get_secret("OLLAMA_HOST", os.environ.get("OLLAMA_HOST", ""))
OLLAMA_API_KEY = _get_secret("OLLAMA_API_KEY", os.environ.get("OLLAMA_API_KEY", ""))
CHROMA_DB_DIR = _get_secret("CHROMA_DB_DIR", os.environ.get("CHROMA_DB_DIR", "chroma_db"))

# ensure env for clients
if OLLAMA_HOST:
    os.environ["OLLAMA_HOST"] = OLLAMA_HOST
if OLLAMA_API_KEY:
    os.environ["OLLAMA_API_KEY"] = OLLAMA_API_KEY
os.environ.setdefault("EMBEDDING_MODEL", EMBEDDING_MODEL)

# dynamic imports (so envs are set first)
try:
    from langchain_ollama import OllamaEmbeddings
except Exception:
    OllamaEmbeddings = None

try:
    from langchain_chroma import Chroma
except Exception:
    Chroma = None

# Document class import fallback
try:
    from langchain_core.documents import Document
except Exception:
    class Document:
        def __init__(self, page_content, metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}

# PDF reader try
_pdf_reader_available = False
try:
    # prefer pypdf
    import pypdf
    _pdf_reader_available = True
    def _read_pdf(path):
        text = []
        reader = pypdf.PdfReader(path)
        for page in reader.pages:
            try:
                text.append(page.extract_text() or "")
            except Exception:
                pass
        return "\n".join(text)
except Exception:
    try:
        import PyPDF2 as pypdf2
        _pdf_reader_available = True
        def _read_pdf(path):
            text = []
            with open(path, "rb") as fh:
                reader = pypdf2.PdfReader(fh)
                for p in reader.pages:
                    try:
                        text.append(p.extract_text() or "")
                    except Exception:
                        pass
            return "\n".join(text)
    except Exception:
        _pdf_reader_available = False
        def _read_pdf(path):
            raise RuntimeError("PDF reader not available; install pypdf or PyPDF2 to ingest PDFs.")

def _load_text_file(path: Path):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read()
    except Exception:
        try:
            with open(path, "r", encoding="latin-1") as fh:
                return fh.read()
        except Exception:
            return ""

def collect_documents(source_dir: str):
    """
    Walk source_dir and return list of Document(page_content, metadata).
    Supports .txt .md .pdf. Skip others.
    """
    docs = []
    p = Path(source_dir)
    if not p.exists():
        print("Source dir does not exist:", source_dir)
        return docs

    for f in sorted(p.rglob("*")):
        if f.is_dir():
            continue
        suffix = f.suffix.lower()
        try:
            if suffix in [".txt", ".md"]:
                text = _load_text_file(f)
            elif suffix == ".pdf":
                if _pdf_reader_available:
                    text = _read_pdf(str(f))
                else:
                    print("Skipping PDF (no reader installed):", f)
                    continue
            else:
                # skip other file types
                continue

            if not text or len(text.strip()) == 0:
                continue

            # chunking: simple naive split by paragraphs if very long (optional)
            # For simplicity, we store each file as one document; for better RAG, chunk into smaller docs.
            meta = {"source": str(f)}
            docs.append(Document(page_content=text, metadata=meta))
            print("Collected:", f)
        except Exception:
            print("Error reading file:", f)
            traceback.print_exc()
    return docs

def run_ingest(source_dir: str = "docs", persist_directory: str = CHROMA_DB_DIR):
    """
    Main ingestion flow:
      - collects documents
      - creates OllamaEmbeddings client (cloud) and checks connectivity
      - persists Chroma vectorstore
    """
    print("=== run_ingest ===")
    print("Source dir:", source_dir, "Persist dir:", persist_directory)
    docs = collect_documents(source_dir)
    print("Collected docs:", len(docs))

    if len(docs) == 0:
        print("No documents found; nothing to ingest.")
        return

    if OllamaEmbeddings is None:
        raise RuntimeError("langchain_ollama not installed / importable. Install and retry.")

    # create embedding client
    try:
        try:
            emb = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=os.environ.get("OLLAMA_HOST"), api_key=os.environ.get("OLLAMA_API_KEY"))
        except TypeError:
            emb = OllamaEmbeddings(model=EMBEDDING_MODEL)
        # connectivity check
        _ = emb.embed_query("ping")
        print("Ollama embeddings reachable.")
    except Exception:
        print("Ollama embeddings not reachable. Check OLLAMA_HOST and OLLAMA_API_KEY.")
        traceback.print_exc()
        raise

    # create Chroma vectorstore
    if Chroma is None:
        raise RuntimeError("langchain_chroma not installed / importable. Install and retry.")

    try:
        # many Chroma wrappers provide from_documents staticmethod
        try:
            vs = Chroma.from_documents(documents=docs, embedding=emb, persist_directory=persist_directory)
            print("Chroma.from_documents succeeded.")
        except Exception:
            # fallback: construct Chroma client and try add_documents
            store = Chroma(persist_directory=persist_directory, embedding_function=emb)
            # if Chroma wrapper provides add_documents:
            try:
                store.add_documents(docs)
                store.persist()
                vs = store
                print("Chroma created via add_documents.")
            except Exception:
                print("Failed to add documents to Chroma.")
                raise
        print("Ingestion complete; persisted to:", persist_directory)
    except Exception:
        print("Chroma ingestion failed.")
        traceback.print_exc()
        raise

if __name__ == "__main__":
    # quick CLI entry
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="docs", help="Source documents directory")
    parser.add_argument("--persist", default=CHROMA_DB_DIR, help="Chroma persist directory")
    args = parser.parse_args()
    run_ingest(source_dir=args.source, persist_directory=args.persist)
