# rag.py
import os
import re
from collections import defaultdict
import numpy as np

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.documents import Document

from ingest import run_ingestion_if_needed

# ---------- Default Configuration (can be overridden by Streamlit secrets) ----------
EMBEDDING_MODEL = "embeddinggemma:latest"
CHROMA_DB_DIR = "chroma_db"
SCORE_THRESHOLD = 0.40
MAX_PER_SOURCE = 6
CANDIDATE_POOL = 20
FINAL_K = 6
LLM_MODEL = "gemma3:1b"
LLM_TEMPERATURE = 0.1
# ------------------------------------------------------------------------------

# --- Streamlit secrets (preferred) with an os.environ fallback ---
try:
    import streamlit as st  # type: ignore
    _secrets = st.secrets
    OLLAMA_BASE_URL = _secrets.get("OLLAMA_HOST") or _secrets.get("OLLAMA_BASE_URL") or os.environ.get("OLLAMA_BASE_URL")
    OLLAMA_API_KEY = _secrets.get("OLLAMA_API_KEY") or os.environ.get("OLLAMA_API_KEY")
    EMBEDDING_MODEL = _secrets.get("EMBEDDING_MODEL", EMBEDDING_MODEL)
    LLM_MODEL = _secrets.get("LLM_MODEL", LLM_MODEL)
    CHROMA_DB_DIR = _secrets.get("CHROMA_DB_DIR", CHROMA_DB_DIR)
except Exception:
    OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL") or os.environ.get("OLLAMA_HOST")
    OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY")
    EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", EMBEDDING_MODEL)
    LLM_MODEL = os.environ.get("LLM_MODEL", LLM_MODEL)
    CHROMA_DB_DIR = os.environ.get("CHROMA_DB_DIR", CHROMA_DB_DIR)
# ------------------------------------------------------------------------------

# Build kwargs for Ollama client
ollama_client_kwargs = {}
if OLLAMA_BASE_URL:
    ollama_client_kwargs["base_url"] = OLLAMA_BASE_URL
if OLLAMA_API_KEY:
    ollama_client_kwargs["api_key"] = OLLAMA_API_KEY

class RAGPipeline:
    def __init__(self):
        # Ensure ingestion
        run_ingestion_if_needed()

        # Embeddings (must match ingestion)
        print(f"[RAG] Initializing Ollama embeddings (model={EMBEDDING_MODEL}) with kwargs={ollama_client_kwargs}")
        self.embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, **ollama_client_kwargs)

        # Chroma DB
        self.db = Chroma(
            persist_directory=CHROMA_DB_DIR,
            embedding_function=self.embeddings
        )

        # LLM (ChatOllama) — pass same kwargs so it uses cloud/remote if configured
        print(f"[RAG] Initializing ChatOllama model={LLM_MODEL} temperature={LLM_TEMPERATURE} kwargs={ollama_client_kwargs}")
        self.llm = ChatOllama(model=LLM_MODEL, temperature=LLM_TEMPERATURE, **ollama_client_kwargs)

        self.chat_history = []

    # ---------------- helpers ----------------
    @staticmethod
    def _cosine(v1, v2):
        v1 = np.array(v1, dtype=float)
        v2 = np.array(v2, dtype=float)
        n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (n1 * n2))

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
        parts = re.split(r'(?<=[\.\?\!])\s+', text.strip())
        seen = set()
        out = []
        for p in parts:
            s = p.strip()
            if not s:
                continue
            key = s.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(s)
            if sum(len(x) for x in out) > max_output_chars:
                break
        return " ".join(out)

    # ---------------- retrieval ----------------
    def retrieve(self, query, k=FINAL_K, debug=True, force_source=None, source_boost=None,
                 score_threshold=SCORE_THRESHOLD, max_per_source=MAX_PER_SOURCE,
                 candidate_pool=CANDIDATE_POOL):

        if debug:
            print(f"[retrieve] Query={query!r} k={k} force_source={force_source} score_threshold={score_threshold}")

        # embed query
        q_emb = self.embeddings.embed_query(query)

        # try Chroma similarity_search to get candidates
        try:
            candidates = self.db.similarity_search(query, k=candidate_pool)
        except Exception:
            # fallback: read raw arrays from DB
            if debug:
                print("[retrieve] similarity_search failed; falling back to db.get()")
            res = self.db.get(include=["documents", "embeddings", "metadatas"])
            docs_list = res.get("documents", [])
            embs_list = res.get("embeddings", [])
            metas_list = res.get("metadatas", [])
            candidates = []
            for i in range(len(docs_list)):
                class _D: pass
                d = _D()
                d.page_content = docs_list[i]
                d.metadata = metas_list[i] if metas_list else {}
                d._embedding = embs_list[i]
                candidates.append(d)

        # collect texts, embeddings, metas
        texts = []
        embeddings = []
        metas = []
        extracted = 0
        for c in candidates:
            emb = getattr(c, "_embedding", None)
            if emb is not None:
                embeddings.append(emb)
                texts.append(getattr(c, "page_content", "") or "")
                metas.append(getattr(c, "metadata", {}) or {})
                extracted += 1

        if extracted == 0:
            res = self.db.get(include=["documents", "embeddings", "metadatas"])
            texts = res.get("documents", [])
            embeddings = res.get("embeddings", [])
            metas = res.get("metadatas", []) or []

        if len(texts) == 0 or len(embeddings) == 0:
            if debug:
                print("[retrieve] No docs/embs available")
            return []

        scored = []
        for idx in range(len(texts)):
            meta = metas[idx] if metas else {}
            src = meta.get("source", "unknown") if isinstance(meta, dict) else "unknown"
            if force_source and (force_source.lower() not in src.lower()):
                continue
            base_score = self._cosine(q_emb, embeddings[idx])
            boost = 1.0
            if source_boost and isinstance(source_boost, dict):
                for key, factor in source_boost.items():
                    if key.lower() in src.lower():
                        try:
                            boost = float(factor)
                        except Exception:
                            boost = 1.0
                        break
            final_score = base_score * boost
            scored.append({"score": final_score, "text": texts[idx], "meta": meta, "src": src, "idx": idx})

        scored.sort(key=lambda x: x["score"], reverse=True)
        filtered = [s for s in scored if s["score"] >= score_threshold]

        if debug:
            print(f"[retrieve] scored: {len(scored)} ; after threshold: {len(filtered)}")

        by_source_count = defaultdict(int)
        final_items = []
        for item in filtered:
            if by_source_count[item["src"]] >= max_per_source:
                continue
            by_source_count[item["src"]] += 1
            final_items.append(item)
            if len(final_items) >= k:
                break

        if debug:
            print("[retrieve] Final items:")
            for i, it in enumerate(final_items, 1):
                preview = (it["text"] or "")[:200].replace("\n", " ")
                print(f" {i:02d}. score={it['score']:.4f} src={it['src']} preview={preview}")

        docs_out = []
        for it in final_items:
            docs_out.append(Document(page_content=it["text"], metadata=it["meta"] or {}))

        return docs_out

    # ---------------- format context ----------------
    def format_context(self, docs):
        if not docs:
            return ""
        grouped = defaultdict(list)
        for d in docs:
            src = (d.metadata.get("source") if isinstance(d.metadata, dict) else "") or "unknown"
            compacted = self._compact_text(d.page_content, max_chars=2000)
            grouped[src].append(compacted)

        sections = []
        for src, pieces in grouped.items():
            merged = " ".join(pieces)
            deduped = self._dedupe_sentences(merged)
            sections.append(f"[{src}]\n{deduped}\n")

        return "\n\n".join(sections)

    # ---------------- ask ----------------
    def ask(self, query):
        docs = self.retrieve(query, k=FINAL_K, debug=True,
                             force_source=None, source_boost=None,
                             score_threshold=SCORE_THRESHOLD,
                             max_per_source=MAX_PER_SOURCE,
                             candidate_pool=CANDIDATE_POOL)

        context = self.format_context(docs)

        if len(context.strip()) == 0:
            answer = "I don't have enough information in the documents to answer that question."
            return answer, docs

        messages = [
            {"role": "system", "content":
                "You are a RAG assistant. Use ONLY the provided context. "
                "Format your answer in clean Markdown with headings and bullet points.\n"
                "If a fact comes from a document, mention its source in brackets.\n"
                "Do NOT copy verbatim; synthesize and merge information from different pieces of context. "
                "If the context doesn't contain the answer, say you don't have enough information."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]

        response = self.llm.invoke(messages)
        raw_answer = response.content

        # dedupe answer lines
        lines = [ln.strip() for ln in raw_answer.splitlines() if ln.strip()]
        out_lines = []
        seen = set()
        for ln in lines:
            key = ln.lower()
            if key in seen:
                continue
            seen.add(key)
            out_lines.append(ln)
        final_answer = "\n".join(out_lines)

        self.chat_history.append({"user": query, "assistant": final_answer})
        return final_answer, docs


if __name__ == "__main__":
    p = RAGPipeline()
    a, ds = p.ask("hello")
    print(a)
