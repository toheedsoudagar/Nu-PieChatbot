# ingest.py
"""
Ingest documents into a Chroma vectorstore using Ollama Cloud embeddings.
Reads credentials from streamlit.secrets (preferred) or os.environ.

Usage:
    python ingest.py --source docs --persist chroma_db
or:
    from ingest import run_ingest
    run_ingest(source_dir="docs", persist_directory="chroma_db")
"""

import os
import sys
import traceback
from pathlib import Path
import argparse

# Prefer streamlit.secrets when available
try:
    import streamlit as st
    _st = True
except Exception:
    st = None
    _st = False

def _get_secret(name, default=""):
    if _st:
        try:
            v = st.secrets.get(name)
            if v:
                return v
        except Exception:
            pass
    return os.environ.get(name, default)

# Config (prefer secrets)
OLLAMA_HOST = _get_secret("OLLAMA_HOST", os.environ.get("OLLAMA_HOST", ""))
OLLAMA_API_KEY = _get_secret("OLLAMA_API_KEY", os.environ.get("OLLAMA_API_KEY", ""))
EMBEDDING_MODEL = _get_secret("EMBEDDING_MODEL", os.environ.get("EMBEDDING_MODEL", "nomic-embed-text"))
CHROMA_DB_DIR = _get_secret("CHROMA_DB_DIR", os.environ.get("CHROMA_DB_DIR", "chroma_db"))

# make sure env visible before imports that might read them
if OLLAMA_HOST:
    os.environ["OLLAMA_HOST"] = OLLAMA_HOST
if OLLAMA_API_KEY:
    os.environ["OLLAMA_API_KEY"] = OLLAMA_API_KEY
os.environ.setdefault("EMBEDDING_MODEL", EMBEDDING_MODEL)

# Optional PDF support
_pdf_available = False
try:
    import pypdf
    _pdf_available = True
    def _read_pdf(path: str) -> str:
        text_parts = []
        reader = pypdf.PdfReader(path)
        for p in reader.pages:
            try:
                text_parts.append(p.extract_text() or "")
            except Exception:
                pass
        return "\n".join(text_parts)
except Exception:
    try:
        import PyPDF2 as pypdf2
        _pdf_available = True
        def _read_pdf(path: str) -> str:
            text_parts = []
            with open(path, "rb") as fh:
                reader = pypdf2.PdfReader(fh)
                for p in reader.pages:
                    try:
                        text_parts.append(p.extract_text() or "")
                    except Exception:
                        pass
            return "\n".join(text_parts)
    except Exception:
        _pdf_available = False
        def _read_pdf(path: str) -> str:
            raise RuntimeError("PDF reader not available. Install pypdf or PyPDF2 to ingest PDFs.")

# Try imports that don't require ollama connectivity
try:
    from langchain_chroma import Chroma
except Exception:
    Chroma = None

# Document class fallback
try:
    from langchain_core.documents import Document
except Exception:
    class Document:
        def __init__(self, page_content, metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}

# Delay importing Ollama embedding wrapper until env is set
try:
    from langchain_ollama import OllamaEmbeddings
except Exception:
    OllamaEmbeddings = None

def _load_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        try:
            return path.read_text(encoding="latin-1")
        except Exception:
            return ""

def collect_documents(source_dir: str):
    """Walk source_dir and collect Documents for supported file types."""
    docs = []
    p = Path(source_dir)
    if not p.exists():
        print("Source directory does not exist:", source_dir)
        return docs

    for f in sorted(p.rglob("*")):
        if f.is_dir():
            continue
        suffix = f.suffix.lower()
        try:
            if suffix in [".txt", ".md"]:
                text = _load_text_file(f)
            elif suffix == ".pdf":
                if _pdf_available:
                    text = _read_pdf(str(f))
                else:
                    print("Skipping PDF (no PDF reader installed):", f)
                    continue
            else:
                # skip unsupported types
                continue

            if not text or not text.strip():
                continue

            meta = {"source": str(f)}
            docs.append(Document(page_content=text, metadata=meta))
            print("Collected:", f)
        except Exception:
            print("Error reading file:", f)
            traceback.print_exc()
    return docs

def run_ingest(source_dir: str = "docs", persist_directory: str = CHROMA_DB_DIR):
    """
    Collect documents, create/open Chroma vectorstore and persist embeddings.
    """
    print("=== run_ingest ===")
    print("Source:", source_dir)
    print("Persist dir:", persist_directory)
    docs = collect_documents(source_dir)
    print("Documents collected:", len(docs))

    if len(docs) == 0:
        print("No documents found — skipping ingestion.")
        return

    if OllamaEmbeddings is None:
        raise RuntimeError("langchain_ollama (Ollama wrapper) is not importable. Install it in your environment.")

    # Create embedding client — do NOT pass api_key/base_url as kwargs in many versions
    try:
        try:
            emb = OllamaEmbeddings(model=EMBEDDING_MODEL)
        except Exception:
            # If the straightforward constructor fails, we optionally try with explicit args
            # (some versions accept them). We try safe constructor first to avoid Pydantic errors.
            try:
                emb = OllamaEmbeddings(model=EMBEDDING_MODEL,
                                       base_url=os.environ.get("OLLAMA_HOST"),
                                       api_key=os.environ.get("OLLAMA_API_KEY"))
            except Exception:
                raise
        # connectivity probe
        try:
            _ = emb.embed_query("ping")
            print("Embedding client reachable (ping OK).")
        except Exception:
            print("Embedding client construction succeeded but embed_query failed. Trace:")
            traceback.print_exc()
            raise RuntimeError("Embed probe failed — check OLLAMA_HOST and OLLAMA_API_KEY.")
    except Exception as e:
        print("Failed to initialize Ollama embeddings client.")
        traceback.print_exc()
        raise

    if Chroma is None:
        raise RuntimeError("langchain_chroma (Chroma wrapper) is not importable. Install it to persist vector DBs.")

    # Create or populate Chroma vectorstore
    try:
        # Prefer high-level convenience API if available
        try:
            print("Attempting Chroma.from_documents(...)")
            vs = Chroma.from_documents(documents=docs, embedding=emb, persist_directory=persist_directory)
            print("Chroma.from_documents succeeded.")
        except Exception:
            print("Chroma.from_documents not available or failed; falling back to manual creation.")
            store = Chroma(persist_directory=persist_directory, embedding_function=emb)
            # some wrappers provide add_documents
            try:
                store.add_documents(docs)
                store.persist()
                vs = store
                print("Added documents and persisted.")
            except Exception:
                # Last-resort: iterate and add vectors manually (if interface differs)
                print("store.add_documents failed; attempting manual add loop.")
                try:
                    for d in docs:
                        # embed text explicitly and call add
                        emb_vec = emb.embed_documents([d.page_content])[0]
                        # different Chroma wrappers expect different add signatures; try a few common ones
                        try:
                            store.add_texts([d.page_content], metadatas=[d.metadata], embedding=emb_vec)
                        except Exception:
                            try:
                                store.add_documents([d])
                            except Exception:
                                # if all attempts fail, raise
                                raise
                    store.persist()
                    vs = store
                    print("Manual add loop succeeded.")
                except Exception:
                    print("Failed to add documents to Chroma using fallback methods.")
                    raise
        print("Ingestion completed; persisted to:", persist_directory)
        return vs
    except Exception:
        print("Chroma ingestion failed.")
        traceback.print_exc()
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="docs", help="Source directory containing docs")
    parser.add_argument("--persist", default=CHROMA_DB_DIR, help="Chroma DB persist directory")
    args = parser.parse_args()
    run_ingest(source_dir=args.source, persist_directory=args.persist)
