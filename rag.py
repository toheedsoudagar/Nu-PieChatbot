# rag.py — Ollama Cloud / delayed-import version
# Saves: move any langchain_ollama imports until after env vars are set
# Based on original provided in the repo (edited to delay imports).
import os
import re
import traceback
import importlib
from collections import defaultdict
import numpy as np

# NOTE: do NOT import langchain_ollama at module top-level — delay until after env set.
# We still import Chroma and Document since they don't depend on Ollama env vars at import-time.
try:
    from langchain_chroma import Chroma
    from langchain_core.documents import Document
except Exception:
    # If these are missing, we let the code fail later with a clear message
    print("Warning: langchain_chroma/langchain_core not importable at module import time.")

# ============================================================
# INLINE OLLAMA CLOUD CREDENTIALS — REPLACE THESE VALUES
# ============================================================
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "https://ollama.com/api")    # <-- Replace if needed
OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY", "51e3006b663948fda90df90f4885af72.wjBXcfuUkzz128XvGbrCrQf_")  # <-- Set via env preferable
# ============================================================

# ---------- Configuration ----------
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "embeddinggemma:latest")
CHROMA_DB_DIR = os.environ.get("CHROMA_DB_DIR", "chroma_db")
SCORE_THRESHOLD = float(os.environ.get("SCORE_THRESHOLD", 0.40))
MAX_PER_SOURCE = int(os.environ.get("MAX_PER_SOURCE", 6))
CANDIDATE_POOL = int(os.environ.get("CANDIDATE_POOL", 20))
FINAL_K = int(os.environ.get("FINAL_K", 6))
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-oss:120b-cloud")
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", 0.2))


class RAGPipeline:
    def __init__(self):
        # 1) ensure env vars are set BEFORE importing ollama client libs
        if OLLAMA_HOST:
            os.environ["OLLAMA_HOST"] = OLLAMA_HOST
        if OLLAMA_API_KEY:
            os.environ["OLLAMA_API_KEY"] = OLLAMA_API_KEY

        # 2) dynamically import ingest module AFTER env is set to avoid import-time embedding client creation
        try:
            ingest = importlib.import_module("ingest")
        except Exception:
            print("Failed to import ingest module. Ensure ingest.py is on PYTHONPATH.")
            traceback.print_exc()
            ingest = None

        # 3) run ingestion (ingest.run_ingest will call set_ollama_env internally)
        if ingest:
            try:
                # Prefer explicit function call; it's safe — ingest.run_ingest sets env internally too.
                ingest.run_ingest()
            except Exception:
                print("Ingestion error:")
                traceback.print_exc()

        # 4) Now import the Ollama client wrappers (after env is set)
        try:
            from langchain_ollama import OllamaEmbeddings, ChatOllama
        except Exception:
            print("Failed to import langchain_ollama. Make sure package is installed.")
            traceback.print_exc()
            raise

        # --------------------------
        # Embeddings (Cloud)
        # --------------------------
        self.embeddings = None
        try:
            # Prefer passing base_url/api_key explicitly if supported by the wrapper
            # Some wrappers expect env vars only — passing explicit args is safer if allowed.
            # If the wrapper doesn't accept base_url/api_key kwargs, it will ignore them or raise.
            try:
                self.embeddings = OllamaEmbeddings(
                    model=EMBEDDING_MODEL,
                    base_url=os.environ.get("OLLAMA_HOST"),
                    api_key=os.environ.get("OLLAMA_API_KEY")
                )
            except TypeError:
                # Fallback if the wrapper doesn't accept base_url/api_key in constructor
                self.embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
            print("OllamaEmbeddings initialized (cloud mode).")
        except Exception:
            print("Failed to initialize embeddings:")
            traceback.print_exc()
            self.embeddings = None

        # --------------------------
        # Chroma DB
        # --------------------------
        try:
            self.db = Chroma(
                persist_directory=CHROMA_DB_DIR,
                embedding_function=self.embeddings
            )
            print("Chroma DB loaded successfully.")
        except Exception:
            print("Chroma DB initialization failed:")
            traceback.print_exc()
            self.db = None

        # --------------------------
        # LLM (Cloud)
        # --------------------------
        self.llm = None
        try:
            try:
                self.llm = ChatOllama(
                    model=LLM_MODEL,
                    temperature=LLM_TEMPERATURE,
                    api_key=os.environ.get("OLLAMA_API_KEY"),
                    base_url=os.environ.get("OLLAMA_HOST")
                )
            except TypeError:
                # fallback to constructor without explicit api args if wrapper doesn't accept them
                self.llm = ChatOllama(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
            print("ChatOllama LLM initialized (cloud).")
        except Exception:
            print("Failed to initialize LLM:")
            traceback.print_exc()
            self.llm = None

    # -------------------- helpers --------------------
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
        t = re.sub(r'\s+', ' ', text).strip()
        return t[:max_chars]

    @staticmethod
    def _dedupe_sentences(text, max_output_chars=1200):
        if not text:
            return ""
        parts = re.split(r'(?<=[.!?])\s+', text.strip())
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

    # -------------------- retrieval --------------------
    def retrieve(self, query, k=FINAL_K):
        print(f"[retrieve] Query: {query}")

        # 1. embed query
        try:
            q_emb = self.embeddings.embed_query(query)
        except Exception:
            print("Embedding failed:")
            traceback.print_exc()
            return []

        # 2. fetch documents from Chroma
        try:
            candidates = self.db.similarity_search(query, k=CANDIDATE_POOL)
        except Exception:
            print("Chroma similarity_search failed:")
            traceback.print_exc()
            return []

        # 3. scoring manually
        final = []
        for c in candidates:
            score = self._cosine(q_emb, c.metadata.get("_embedding", q_emb))
            if score >= SCORE_THRESHOLD:
                final.append(c)
            if len(final) >= k:
                break

        return final

    # -------------------- format context --------------------
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

    # -------------------- ask --------------------
    def ask(self, query):
        docs = self.retrieve(query)
        context = self.format_context(docs)

        if not context.strip():
            return "I don't have enough information in the documents to answer that question.", docs

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an offline RAG assistant. Use ONLY the provided context.\n"
                    "Format your answer in clean Markdown with headings and bullet points.\n"
                    "If a fact comes from a document, mention its source in brackets.\n"
                )
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}"
            }
        ]

        try:
            # ChatOllama wrapper may expose different call signatures; adapt as needed.
            # Many wrappers provide an .invoke or .generate or .__call__ API. Here we use `.invoke`
            # because your original code used self.llm.invoke(messages)
            response = None
            if hasattr(self.llm, "invoke"):
                response = self.llm.invoke(messages)
                # response.content as used in original code
                answer = getattr(response, "content", str(response))
            elif hasattr(self.llm, "generate"):
                gen = self.llm.generate(messages)
                # extract text depending on wrapper structure
                try:
                    answer = gen.generations[0][0].text
                except Exception:
                    answer = str(gen)
            else:
                # fallback attempt: call as a function
                answer = str(self.llm(messages))
        except Exception:
            print("LLM error:")
            traceback.print_exc()
            answer = "There was an error generating the answer."

        return answer, docs


if __name__ == "__main__":
    p = RAGPipeline()
    ans, src = p.ask("hello")
    print(ans)
