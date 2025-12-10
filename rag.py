# rag.py (fixed) — do NOT pass api_key/base_url into Ollama wrapper constructors
import os
import re
import traceback
import sys
from collections import defaultdict
import numpy as np
import hashlib

# streamlit secrets helper
try:
    import streamlit as st
    _st_available = True
except Exception:
    st = None
    _st_available = False

def _get_secret(name, default=""):
    if _st_available:
        try:
            val = st.secrets.get(name)
            if val:
                return val
        except Exception:
            pass
    return os.environ.get(name, default)

# config
OLLAMA_HOST = _get_secret("OLLAMA_HOST", os.environ.get("OLLAMA_HOST", ""))
OLLAMA_API_KEY = _get_secret("OLLAMA_API_KEY", os.environ.get("OLLAMA_API_KEY", ""))
EMBEDDING_MODEL = _get_secret("EMBEDDING_MODEL", os.environ.get("EMBEDDING_MODEL", "nomic-embed-text"))
LLM_MODEL = _get_secret("LLM_MODEL", os.environ.get("LLM_MODEL", "gpt-oss:120b-cloud"))
CHROMA_DB_DIR = _get_secret("CHROMA_DB_DIR", os.environ.get("CHROMA_DB_DIR", "chroma_db"))

SCORE_THRESHOLD = float(_get_secret("SCORE_THRESHOLD", os.environ.get("SCORE_THRESHOLD", "0.40")))
CANDIDATE_POOL = int(_get_secret("CANDIDATE_POOL", os.environ.get("CANDIDATE_POOL", "20")))
FINAL_K = int(_get_secret("FINAL_K", os.environ.get("FINAL_K", "6")))
LLM_TEMPERATURE = float(_get_secret("LLM_TEMPERATURE", os.environ.get("LLM_TEMPERATURE", "0.2")))

# ensure env visible before imports that might read them
if OLLAMA_HOST:
    os.environ["OLLAMA_HOST"] = OLLAMA_HOST
if OLLAMA_API_KEY:
    os.environ["OLLAMA_API_KEY"] = OLLAMA_API_KEY
os.environ.setdefault("EMBEDDING_MODEL", EMBEDDING_MODEL)
os.environ.setdefault("LLM_MODEL", LLM_MODEL)

# optional imports
try:
    from langchain_chroma import Chroma
except Exception:
    Chroma = None

try:
    from langchain_core.documents import Document
except Exception:
    class Document:
        def __init__(self, page_content, metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}

# deterministic fallback embeddings (dev)
class DeterministicFallbackEmbeddings:
    def __init__(self, dim=512):
        self.dim = dim

    def _text_to_vector(self, text):
        if text is None:
            text = ""
        h = hashlib.sha256(text.encode("utf-8")).digest()
        repeats = (self.dim * 4 + len(h) - 1) // len(h)
        blob = (h * repeats)[: self.dim * 4]
        vals = []
        for i in range(0, self.dim * 4, 4):
            chunk = blob[i : i + 4]
            v = int.from_bytes(chunk, "big", signed=False)
            vals.append(((v / 2**32) * 2) - 1)
        return vals

    def embed_query(self, text):
        return self._text_to_vector(text)

    def embed_documents(self, texts):
        return [self._text_to_vector(t) for t in texts]

class RAGPipeline:
    def __init__(self, chroma_dir: str = CHROMA_DB_DIR):
        print("=== RAGPipeline init ===")
        print("OLLAMA_HOST present?:", bool(OLLAMA_HOST), "value:", OLLAMA_HOST or "(empty)")
        print("EMBEDDING_MODEL:", EMBEDDING_MODEL)
        print("LLM_MODEL:", LLM_MODEL)
        sys.stdout.flush()

        self.embeddings = None
        self.llm = None
        self.db = None

        # Import langchain_ollama AFTER environment is set
        try:
            from langchain_ollama import OllamaEmbeddings, ChatOllama
            print("Imported langchain_ollama.")
        except Exception:
            print("Could not import langchain_ollama; embeddings/LLM will be disabled.")
            traceback.print_exc()
            self.embeddings = DeterministicFallbackEmbeddings(dim=512)
            self.llm = None
            # still attempt to init Chroma later if available
            if Chroma is not None:
                try:
                    self.db = Chroma(persist_directory=chroma_dir, embedding_function=self.embeddings)
                except Exception:
                    self.db = None
            return

        # IMPORTANT: many versions of langchain_ollama/ollama do NOT accept api_key/base_url kwargs.
        # Construct without passing api_key/base_url so Pydantic validation won't reject extra fields.
        try:
            try:
                # Preferred: call with just the model name
                self.embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
            except Exception:
                # If that fails for any reason, try the constructor with explicit args (some versions accept it).
                # We try the safe order: no kwargs first, then explicit (reverse of earlier code).
                try:
                    self.embeddings = OllamaEmbeddings(
                        model=EMBEDDING_MODEL,
                        base_url=os.environ.get("OLLAMA_HOST"),
                        api_key=os.environ.get("OLLAMA_API_KEY"),
                    )
                except Exception:
                    raise

            # Connectivity probe (small embed)
            try:
                _ = self.embeddings.embed_query("ping")
                print("Ollama embeddings ping OK.")
            except Exception:
                print("Embedding ping failed after construction — raising to surface the error.")
                traceback.print_exc()
                raise RuntimeError("Embedding client not reachable.")
        except Exception:
            print("Failed to initialize OllamaEmbeddings. Ensure OLLAMA_HOST=https://api.ollama.ai and OLLAMA_API_KEY are set and valid.")
            traceback.print_exc()
            # fallback to deterministic embeddings to keep app alive (optional)
            self.embeddings = DeterministicFallbackEmbeddings(dim=512)

        # Initialize Chroma if present
        if Chroma is not None:
            try:
                self.db = Chroma(persist_directory=chroma_dir, embedding_function=self.embeddings)
                print("Chroma DB loaded at:", chroma_dir)
            except Exception:
                print("Chroma init failed; continuing without DB.")
                traceback.print_exc()
                self.db = None
        else:
            print("Chroma not available; db=None")
            self.db = None

        # Initialize ChatOllama LLM (construct without api_key/base_url first)
        try:
            try:
                self.llm = ChatOllama(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
            except Exception:
                # try with explicit args only if the simple constructor fails
                self.llm = ChatOllama(
                    model=LLM_MODEL,
                    temperature=LLM_TEMPERATURE,
                    base_url=os.environ.get("OLLAMA_HOST"),
                    api_key=os.environ.get("OLLAMA_API_KEY"),
                )
            print("ChatOllama initialized.")
        except Exception:
            print("Failed to initialize ChatOllama; LLM will be disabled.")
            traceback.print_exc()
            self.llm = None

    @staticmethod
    def _cosine(a, b):
        a = np.array(a, dtype=float)
        b = np.array(b, dtype=float)
        if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
            return 0.0
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    @staticmethod
    def _compact_text(text, max_chars=1500):
        if not text:
            return ""
        t = re.sub(r"\s+", " ", text).strip()
        return t[:max_chars]

    @staticmethod
    def _dedupe_sentences(text, max_output_chars=1200):
        if not text:
            return ""
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        seen = set()
        out = []
        for p in parts:
            s = p.strip()
            if not s or s.lower() in seen:
                continue
            seen.add(s.lower())
            out.append(s)
            if sum(len(x) for x in out) > max_output_chars:
                break
        return " ".join(out)

    def retrieve(self, query, k: int = FINAL_K):
        if self.embeddings is None:
            raise RuntimeError("Embeddings are not initialized.")

        try:
            q_emb = self.embeddings.embed_query(query)
        except Exception:
            print("embed_query failed; printing traceback and returning empty.")
            traceback.print_exc()
            return []

        if self.db is None:
            print("No Chroma DB available; returning empty list.")
            return []

        try:
            candidates = self.db.similarity_search(query, k=CANDIDATE_POOL)
        except Exception:
            print("Chroma similarity_search failed.")
            traceback.print_exc()
            return []

        scored = []
        for c in candidates:
            emb_meta = None
            try:
                emb_meta = c.metadata.get("_embedding")
            except Exception:
                emb_meta = None
            if emb_meta:
                score = self._cosine(q_emb, emb_meta)
            else:
                try:
                    doc_emb = self.embeddings.embed_documents([c.page_content])[0]
                    score = self._cosine(q_emb, doc_emb)
                except Exception:
                    score = 0.0
            if score >= SCORE_THRESHOLD:
                scored.append((c, score))
            if len(scored) >= k:
                break

        scored_sorted = sorted(scored, key=lambda x: x[1], reverse=True)
        return [s[0] for s in scored_sorted]

    def format_context(self, docs):
        if not docs:
            return ""
        grouped = defaultdict(list)
        for d in docs:
            src = d.metadata.get("source", "unknown")
            grouped[src].append(self._compact_text(d.page_content))
        out = []
        for src, pieces in grouped.items():
            merged = " ".join(pieces)
            deduped = self._dedupe_sentences(merged)
            out.append(f"[{src}]\n{deduped}\n")
        return "\n".join(out)

    def ask(self, query):
        docs = self.retrieve(query)
        context = self.format_context(docs)
        if not context.strip():
            return "I don't have enough information in the documents to answer that question.", docs

        messages = [
            {"role": "system", "content": "You are a RAG assistant. Use only the provided context and cite sources."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]

        if self.llm is None:
            return "LLM not initialized; check ChatOllama availability.", docs

        try:
            if hasattr(self.llm, "invoke"):
                resp = self.llm.invoke(messages)
                ans = getattr(resp, "content", str(resp))
            elif hasattr(self.llm, "generate"):
                gen = self.llm.generate(messages)
                try:
                    ans = gen.generations[0][0].text
                except Exception:
                    ans = str(gen)
            else:
                ans = str(self.llm(messages))
        except Exception:
            print("LLM invocation error:")
            traceback.print_exc()
            ans = "There was an error generating the answer."

        return ans, docs

if __name__ == "__main__":
    p = RAGPipeline()
    a, s = p.ask("hello")
    print("Answer:", a)
    print("Sources returned:", len(s))
