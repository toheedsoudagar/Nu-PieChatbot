# ingest.py
"""
Ingest documents from ./docs into a persistent Chroma DB (./chroma_db)
using Ollama Cloud embeddings.

Usage:
  python ingest.py

Notes:
- Replace OLLAMA_HOST / OLLAMA_API_KEY placeholders if you want inline creds.
- Ensure poppler-utils and tesseract are installed for PDF/OCR support.
- Matching EMBEDDING_MODEL is critical: use same model in rag.py and ingest.py.
"""

import os
import sys
import traceback
from pathlib import Path
from typing import List

# inline placeholders (replace if you want inline credentials)
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "https://ollama.com/api")
OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY", "51e3006b663948fda90df90f4885af72.wjBXcfuUkzz128XvGbrCrQf_")

# config
DOCS_DIR = Path("docs")
CHROMA_DIR = Path("chroma_db")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "nomic-embed-text")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 1200))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 200))
FORCE_REINGEST = os.environ.get("FORCE_REINGEST", "0") in ("1", "true", "True")

# Make sure libs are available
try:
    from langchain_ollama import OllamaEmbeddings
    from langchain_chroma import Chroma
    from langchain_core.documents import Document
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    # unstructured for improved parsing
    from unstructured.partition.auto import partition
except Exception as e:
    print("Missing required packages. Please install requirements.txt and try again.")
    traceback.print_exc()
    sys.exit(1)


def set_ollama_env():
    # ensure env for underlying libs
    if OLLAMA_HOST:
        os.environ["OLLAMA_HOST"] = OLLAMA_HOST
    if OLLAMA_API_KEY:
        os.environ["OLLAMA_API_KEY"] = OLLAMA_API_KEY


def load_file(path: Path) -> str:
    """Try to extract text from many filetypes using unstructured; fallback to plain read."""
    try:
        elems = partition(str(path))
        parts = [e.get("text", "") if isinstance(e, dict) else (getattr(e, "text", "") or str(e)) for e in elems]
        text = "\n\n".join(p for p in parts if p)
        return text
    except Exception:
        # fallback for plain text files
        try:
            return path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return ""


def gather_documents(docs_dir: Path) -> List[Document]:
    docs: List[Document] = []
    for p in sorted(docs_dir.rglob("*")):
        if p.is_file():
            txt = load_file(p)
            if not txt or len(txt.strip()) == 0:
                continue
            metadata = {"source": str(p.relative_to(docs_dir.parent))}
            docs.append(Document(page_content=txt, metadata=metadata))
            print(f"Loaded: {p} (chars={len(txt)})")
    return docs


def split_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    out: List[Document] = []
    for d in docs:
        chunks = splitter.split_text(d.page_content)
        for i, c in enumerate(chunks):
            md = dict(d.metadata)
            md["chunk"] = i
            out.append(Document(page_content=c, metadata=md))
    print(f"Split into {len(out)} chunks")
    return out


def get_model_emb_dim(emb):
    try:
        if hasattr(emb, "embed_query"):
            e = emb.embed_query("test")
            return len(e) if e else None
        elif hasattr(emb, "embed"):
            e = emb.embed(["test"])[0]
            return len(e) if e else None
    except Exception:
        return None


def get_stored_dim(db: Chroma):
    try:
        res = db.get(include=["embeddings"]) or {}
        embs = res.get("embeddings", [])
        return len(embs[0]) if embs else None
    except Exception:
        return None


def run_ingest():
    set_ollama_env()

    # create embeddings client (do not pass api_key kwarg to avoid pydantic extra errors)
    try:
        emb = OllamaEmbeddings(model=EMBEDDING_MODEL)
        print("Created OllamaEmbeddings model:", EMBEDDING_MODEL)
    except Exception:
        print("Failed to create OllamaEmbeddings:")
        traceback.print_exc()
        emb = None

    # If FORCE_REINGEST is set, remove existing chroma DB
    if FORCE_REINGEST and CHROMA_DIR.exists():
        print("FORCE_REINGEST enabled — deleting existing chroma_db")
        import shutil
        shutil.rmtree(CHROMA_DIR, ignore_errors=True)

    # Initialize Chroma (may create empty DB)
    try:
        db = Chroma(persist_directory=str(CHROMA_DIR), embedding_function=emb)
        print("Opened Chroma at", CHROMA_DIR)
    except Exception:
        print("Failed to open Chroma client:")
        traceback.print_exc()
        db = None

    # If DB exists, check dims
    if db and emb:
        model_dim = get_model_emb_dim(emb)
        stored_dim = get_stored_dim(db)
        print("Embedding dims -> model:", model_dim, "stored:", stored_dim)
        if stored_dim and model_dim and stored_dim != model_dim:
            print("Dimension mismatch between stored DB and current model.")
            print("Either re-create DB with this embedding model or use the model that created it.")
            # Do not automatically delete here unless FORCE_REINGEST True.
            if not FORCE_REINGEST:
                print("Aborting ingestion to avoid corrupting DB. Set FORCE_REINGEST=1 to force reingest.")
                return

    # Gather & split documents
    docs = gather_documents(DOCS_DIR)
    if not docs:
        print("No documents found in", DOCS_DIR)
        return
    docs_split = split_documents(docs)

    # Convert to langchain documents if needed and persist to chroma
    try:
        # Use Chroma.from_documents for initial ingestion
        vect = None
        try:
            # prefer to use the Chroma vectorstore helper to write docs
            vect = Chroma.from_documents(docs_split, embedding=emb, persist_directory=str(CHROMA_DIR))
            vect.persist()
            print("Persisted documents to Chroma (from_documents).")
        except Exception:
            # fallback: use db.add_documents if available
            if db:
                # some wrappers provide add_documents
                try:
                    db.add_documents(docs_split)
                    print("Added documents to Chroma (add_documents).")
                except Exception:
                    # last resort: iterate and add (rare)
                    print("Failed to add documents via helper methods — aborting.")
                    traceback.print_exc()
                    return
    except Exception:
        print("Error writing documents to Chroma:")
        traceback.print_exc()
        return

    print("Ingestion complete.")


if __name__ == "__main__":
    run_ingest()

