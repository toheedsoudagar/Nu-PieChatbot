# rag.py — Ollama Cloud Version (with placeholders)

import os
import re
import traceback
from collections import defaultdict
import numpy as np

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.documents import Document

from ingest import run_ingest as run_ingestion_if_needed


# ============================================================
# INLINE OLLAMA CLOUD CREDENTIALS — REPLACE THESE TWO VALUES
# ============================================================

OLLAMA_HOST = "https://ollama.com/api"    # <-- Replace with your Ollama Cloud base URL
OLLAMA_API_KEY = "51e3006b663948fda90df90f4885af72.wjBXcfuUkzz128XvGbrCrQf_"  # <-- Replace with your API key

# ============================================================


# ---------- Configuration ----------
EMBEDDING_MODEL = "embeddinggemma:latest"
CHROMA_DB_DIR = "chroma_db"
SCORE_THRESHOLD = 0.40
MAX_PER_SOURCE = 6
CANDIDATE_POOL = 20
FINAL_K = 6
LLM_MODEL = "gpt-oss:120b-cloud"
LLM_TEMPERATURE = 0.2


class RAGPipeline:
    def __init__(self):

        # make cloud credentials visible to underlying libraries
        os.environ["OLLAMA_HOST"] = OLLAMA_HOST
        os.environ["OLLAMA_API_KEY"] = OLLAMA_API_KEY

        # ingestion step
        try:
            run_ingestion_if_needed()
        except Exception:
            print("Ingestion error:")
            traceback.print_exc()

        # --------------------------
        # Embeddings (Cloud)
        # --------------------------
        self.embeddings = None
        try:
            self.embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
            print("OllamaEmbeddings initialized (cloud mode).")
        except Exception:
            print("Failed to initialize embeddings:")
            traceback.print_exc()

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
            self.llm = ChatOllama(
                model=LLM_MODEL,
                temperature=LLM_TEMPERATURE,
                api_key=OLLAMA_API_KEY,
                base_url=OLLAMA_HOST
            )
            print("ChatOllama LLM initialized (cloud).")
        except Exception:
            print("Failed to initialize LLM:")
            traceback.print_exc()

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
            response = self.llm.invoke(messages)
            answer = response.content
        except Exception:
            print("LLM error:")
            traceback.print_exc()
            answer = "There was an error generating the answer."

        return answer, docs


if __name__ == "__main__":
    p = RAGPipeline()
    ans, src = p.ask("hello")
    print(ans)


