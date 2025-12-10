# rag.py
"""
RAGPipeline for Streamlit + Ollama Cloud (delayed-import + fallback).
Place this file in your repo. Ensure Streamlit secrets contain:
OLLAMA_HOST, OLLAMA_API_KEY (do not commit keys to repo).
"""

import os
import re
import traceback
import importlib
from collections import defaultdict
import numpy as np
import hashlib
import sys

# Try to read streamlit secrets if running in Streamlit
try:
    import streamlit as st  # optional
    _st_available = True
except Exception:
    st = None
    _st_available = False

def _get_secret(name, default=""):
    """Prefer streamlit secrets when available, else environment variable."""
    if _st_available:
        try:
            # st.secrets acts like a dict; avoid KeyError
            val = st.secrets.get(name)
            if val:
                return val
        except Exception:
            pass
    return os.environ.get(name, default)

# Load configuration (prefer st.secrets)
OLLAMA_HOST = _get_secret("OLLAMA_HOST", os.environ.get("OLLAMA_HOST", ""))
OLLAMA_API_KEY = _get_secret("OLLAMA_API_KEY", os.environ.get("OLLAMA_API_KEY", ""))
EMBEDDING_MODEL = _get_secret("EMBEDDING_MODEL", os.environ.get("EMBEDDING_MODEL", "embeddinggemma:latest"))
LLM_MODEL = _get_secret("LLM_MODEL", os.environ.get("LLM_MODEL", "gpt-oss:120b-cloud"))
CHROMA_DB_DIR = _get_secret("CHROMA_DB_DIR", os.environ.get("CHROMA_DB_DIR", "chroma_db"))

SCORE_THRESHOLD = float(_get_secret("SCORE_THRESHOLD", os.environ.get("SCORE_THRESHOLD", "0.40")))
CANDIDATE_POOL = int(_get_secret("CANDIDATE_POOL", os.environ.get("CANDIDATE_POOL", "20")))
FINAL_K = int(_get_secret("FINAL_K", os.environ.get("FINAL_K", "6")))
LLM_TEMPERATURE = float(_get_secret("LLM_TEMPERATURE", os.environ.get("LLM_TEMPERATURE", "0.2")))

# Ensure env visible for libs that read env at import time
if OLLAMA_HOST:
    os.environ["OLLAMA_HOST"] = OLLAMA_HOST
if OLLAMA_API_KEY:
    os.environ["OLLAMA_API_KEY"] = OLLAMA_API_KEY
if EMBEDDING_MODEL:
    os.environ.setdefault("EMBEDDING_MODEL", EMBEDDING_MODEL)
if LLM_MODEL:
    os.environ.setdefault("LLM_MODEL", LLM_MODEL)

# Fallback deterministic embeddings (fast, deterministic, dev-friendly)
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

# Try to import Chroma (optional)
try:
    from langchain_chroma import Chroma
except Exception:
    Chroma = None

class RAGPipeline:
    """
    RAGPipeline: instantiate in your Streamlit app.
    Usage:
        from rag import RAGPipeline
        rag = RAGPipeline()
        answer, docs = rag.ask("your question")
    """
    def __init__(self):
        print("=== RAGPipeline init ===")
        print("OLLAMA_HOST:", bool(OLLAMA_HOST), " value present:", OLLAMA_HOST or "(empty)")
        print("OLLAMA_API_KEY present?:", bool(OLLAMA_API_KEY))
        print("EMBEDDING_MODEL:", EMBEDDING_MODEL)
        print("LLM_MODEL:", LLM_MODEL)
        sys.stdout.flush()

        # Embeddings/LLM placeholders
        self.embeddings = None
        self.llm = None
        self.db = None

        # Attempt to import Ollama wrappers now that env is set
        try:
            from langchain_ollama import OllamaEmbeddings, ChatOllama
            print("Imported langchain_ollama.")
            try:
                # Try explicit host/api_key if wrapper accepts them
                try:
                    self.embeddings = OllamaEmbeddings(
                        model=EMBEDDING_MODEL,
                        base_url=os.environ.get("OLLAMA_HOST"),
                        api_key=os.environ.get("OLLAMA_API_KEY"),
                    )
                except TypeError:
                    # wrapper may not accept base_url/api_key
                    self.embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

                # lightweight connectivity probe
                try:
                    _ = self.embeddings.embed_query("ping")
                    print("Ollama embeddings ping OK — Ollama reachable.")
                except Exception:
                    print("Ollama embeddings ping failed; falling back to local deterministic embeddings.")
                    traceback.print_exc()
                    self.embeddings = DeterministicFallbackEmbeddings(dim=512)
            except Exception:
                print("Error initializing OllamaEmbeddings — using fallback.")
                traceback.print_exc()
                self.embeddings = DeterministicFallbackEmbeddings(dim=512)

            # Try to initialize LLM
            try:
                try:
                    self.llm = ChatOllama(
                        model=LLM_MODEL,
                        temperature=LLM_TEMPERATURE,
                        base_url=os.environ.get("OLLAMA_HOST"),
                        api_key=os.environ.get("OLLAMA_API_KEY"),
                    )
                except TypeError:
                    self.llm = ChatOllama(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
                print("ChatOllama initialized.")
            except Exception:
                print("ChatOllama initialization failed — LLM disabled.")
                traceback.print_exc()
                self.llm = None

        except Exception:
            print("langchain_ollama not importable — using deterministic fallback embeddings.")
            traceback.print_exc()
            self.embeddings = DeterministicFallbackEmbeddings(dim=512)
            self.llm = None

        # Initialize Chroma DB if available and if embeddings provided
        if Chroma is not None:
            try:
                # If embeddings object is from Ollama wrapper and Chroma expects a specific interface,
                # the embedding_function param may be accepted. Otherwise Chroma may call embed internally.
                self.db = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=self.embeddings)
                print("Chroma DB loaded at:", CHROMA_DB_DIR)
            except Exception:
                print("Failed to initialize Chroma DB (it may still work without persistent db).")
                traceback.print_exc()
                self.db = None
        else:
            print("langchain_chroma not available; vector DB disabled.")
            self.db = None

    @staticmethod
    def _cosine(v1, v2):
        v1 = np.array(v1, dtype=float)
        v2 = np.array(v2, dtype=float)
        if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
            return 0.0
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

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

    def retrieve(self, query, k=FINAL_K):
        print(f"[retrieve] query: {query}")
        # embed query
        try:
            q_emb = self.embeddings.embed_query(query)
        except Exception:
            print("embed_query failed; switching to fallback embedding and retrying.")
            traceback.print_exc()
            if not isinstance(self.embeddings, DeterministicFallbackEmbeddings):
                self.embeddings = DeterministicFallbackEmbeddings(dim=512)
            q_emb = self.embeddings.embed_query(query)

        # if no DB present, return empty list
        if self.db is None:
            print("No vector DB present (Chroma missing or failed). Returning empty results.")
            return []

        # use Chroma similarity_search if available
        try:
            candidates = self.db.similarity_search(query, k=CANDIDATE_POOL)
        except Exception:
            print("Chroma similarity_search failed; will attempt to score manually if embeddings present.")
            traceback.print_exc()
            candidates = []

        final = []
        for c in candidates:
            # try to read precomputed embedding from metadata
            emb_meta = None
            try:
                emb_meta = c.metadata.get("_embedding")
            except Exception:
                emb_meta = None

            if emb_meta:
                score = self._cosine(q_emb, emb_meta)
            else:
                # embed the doc content to score
                try:
                    doc_emb = self.embeddings.embed_documents([c.page_content])[0]
                    score = self._cosine(q_emb, doc_emb)
                except Exception:
                    score = 0.0

            if score >= SCORE_THRESHOLD:
                final.append((c, score))
            if len(final) >= k:
                break

        final_sorted = sorted(final, key=lambda x: x[1], reverse=True)
        return [f[0] for f in final_sorted]

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
            {
                "role": "system",
                "content": (
                    "You are a RAG assistant. Use ONLY the provided context and cite sources in brackets."
                ),
            },
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ]

        # If no LLM available, return a helpful message + docs
        if self.llm is None:
            return "LLM is not available. Check that langchain_ollama is installed and Ollama is reachable.", docs

        try:
            if hasattr(self.llm, "invoke"):
                response = self.llm.invoke(messages)
                answer = getattr(response, "content", str(response))
            elif hasattr(self.llm, "generate"):
                gen = self.llm.generate(messages)
                try:
                    answer = gen.generations[0][0].text
                except Exception:
                    answer = str(gen)
            else:
                answer = str(self.llm(messages))
        except Exception:
            print("LLM invocation error:")
            traceback.print_exc()
            answer = "There was an error generating the answer."

        return answer, docs

# Quick self-test when run directly (not in import)
if __name__ == "__main__":
    rp = RAGPipeline()
    a, s = rp.ask("hello")
    print("Answer:", a)
    print("Sources returned:", len(s))
