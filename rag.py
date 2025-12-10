# rag_fixed.py — full final RAG pipeline
"""
This file is the full, ready-to-run RAG pipeline with the fixes you requested.

- DOES NOT pass api_key to OllamaEmbeddings (avoids pydantic extra_forbidden).
- Passes credentials to ChatOllama (LLM) if environment provides them.
- Detects embedding-dimension mismatches and offers to re-ingest (optional AUTO_REINGEST env).
- Keeps verbose HTTP logging helpers for debugging in VS Code.

Run:
  python rag_fixed.py

If you want this as the main file, rename to rag.py or replace your existing file.
"""

import os
import re
import shutil
import traceback
import logging
from collections import defaultdict

import numpy as np
import requests

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.documents import Document

from ingest import run_ingestion_if_needed

# ---------- Configuration (env-overridable) ----------
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "embeddinggemma:latest")
CHROMA_DB_DIR = os.environ.get("CHROMA_DB_DIR", "chroma_db")
SCORE_THRESHOLD = float(os.environ.get("SCORE_THRESHOLD", 0.40))
MAX_PER_SOURCE = int(os.environ.get("MAX_PER_SOURCE", 6))
CANDIDATE_POOL = int(os.environ.get("CANDIDATE_POOL", 20))
FINAL_K = int(os.environ.get("FINAL_K", 6))
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-oss:120b-cloud")
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", 0.2))
DEBUG = os.environ.get("RAG_DEBUG", "1") in ("1", "true", "True")
AUTO_REINGEST = os.environ.get("RAG_AUTO_REINGEST_ON_DIM_MISMATCH", "0") in ("1", "true", "True")
# -----------------------------------


# ---------------- helper debug utilities ----------------
def enable_verbose_http_logging():
    import http.client as http_client
    http_client.HTTPConnection.debuglevel = 1
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)
    requests_log = logging.getLogger("urllib3")
    requests_log.setLevel(logging.DEBUG)
    requests_log.propagate = True


def print_env_presence():
    keys = [
        "OLLAMA_API_KEY",
        "OLLAMA_HOST",
        "CHROMA_API_KEY",
        "CHROMA_SERVER_HOST",
        "HTTP_PROXY",
        "HTTPS_PROXY",
    ]
    #print("\n=== ENV PRESENCE (True=present, False=missing) ===")
    for k in keys:
        print(f"{k}: {bool(os.environ.get(k))}")


def test_ollama_endpoint():
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    url = host + "/v1/models"
    headers = {}
    api_key = os.environ.get("OLLAMA_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        #print(f"Testing Ollama endpoint: {url}")
        r = requests.get(url, headers=headers, timeout=10)
        #print("Status:", r.status_code)
        #print("Response headers:", {k: v for k, v in r.headers.items() if k.lower() in ("content-type", "www-authenticate")})
        #print("Body (first 800 chars):", (r.text or "")[:800])
        return r.status_code, r.text
    except Exception:
        print("Ollama endpoint test failed:")
        traceback.print_exc()
        return None, None


# ---------------- RAG Pipeline ----------------
class RAGPipeline:
    def __init__(self):
        if DEBUG:
            print_env_presence()
            enable_verbose = os.environ.get("RAG_ENABLE_HTTP_LOG", "1") in ("1", "true", "True")
            if enable_verbose:
                #print("Enabling verbose HTTP logging (urllib3/http.client)")
                try:
                    enable_verbose_http_logging()
                except Exception:
                    pass

        # Ensure ingestion (safe to call)
        try:
            run_ingestion_if_needed()
        except Exception:
            print("run_ingestion_if_needed() raised an exception:")
            traceback.print_exc()

        # Create embeddings (no api_key kwarg)
        self.embeddings = None
        try:
            self.embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
            print("OllamaEmbeddings() created successfully")
        except Exception:
            print("Failed to create OllamaEmbeddings():")
            traceback.print_exc()

        # Open Chroma DB
        self.db = None
        try:
            self.db = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=self.embeddings)
            print(f"Chroma DB opened (persist dir = {CHROMA_DB_DIR})")
        except Exception:
            print("Failed to open Chroma DB:")
            traceback.print_exc()
            try:
                self.db = Chroma(embedding_function=self.embeddings)
                print("Opened fallback Chroma client (in-memory)")
            except Exception:
                print("Fallback Chroma client also failed. Setting db=None")
                traceback.print_exc()
                self.db = None

        # Create LLM and pass creds if available
        self.llm = None
        llm_kwargs = {}
        if os.environ.get("OLLAMA_API_KEY"):
            llm_kwargs["51e3006b663948fda90df90f4885af72.wjBXcfuUkzz128XvGbrCrQf_"] = os.environ.get("OLLAMA_API_KEY")
        if os.environ.get("OLLAMA_HOST"):
            llm_kwargs["base_url"] = os.environ.get("OLLAMA_HOST")

        try:
            if llm_kwargs:
                try:
                    self.llm = ChatOllama(model=LLM_MODEL, temperature=LLM_TEMPERATURE, **llm_kwargs)
                except TypeError:
                    self.llm = ChatOllama(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
            else:
                self.llm = ChatOllama(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
            print("ChatOllama() created successfully")
        except Exception:
            print("Failed to create ChatOllama():")
            traceback.print_exc()

        # Check embedding dims
        try:
            self._check_embedding_dims_and_handle()
        except Exception:
            traceback.print_exc()

        self.chat_history = []

    # ---------------- dimension helpers ----------------
    def _get_model_embedding_dim(self):
        try:
            if self.embeddings is None:
                return None
            if hasattr(self.embeddings, "embed_query"):
                e = self.embeddings.embed_query("__r_a_g_sample__")
                return len(e) if e is not None else None
            if hasattr(self.embeddings, "embed"):
                e = self.embeddings.embed(["__r_a_g_sample__"])
                return len(e[0]) if e else None
        except Exception:
            return None

    def _get_stored_embedding_dim(self):
        try:
            if self.db is None:
                return None
            res = self.db.get(include=["embeddings"]) or {}
            embs = res.get("embeddings", [])
            return len(embs[0]) if embs else None
        except Exception:
            return None

    def _check_embedding_dims_and_handle(self):
        model_dim = self._get_model_embedding_dim()
        stored_dim = self._get_stored_embedding_dim()
        print("Embedding dims: model->", model_dim, "stored->", stored_dim)
        if stored_dim and model_dim and stored_dim != model_dim:
            print("\nWARNING: embedding dimension mismatch (stored != model).")
            print("Your existing Chroma DB was created with vectors of length", stored_dim,
                  "but the current embedding model produces vectors of length", model_dim)
            print("Options:")
            print("  1) Re-run ingestion to recreate the DB using the current embedding model (recommended).")
            print("  2) Switch EMBEDDING_MODEL to one that produces the stored vector dimension.")
            if AUTO_REINGEST:
                print("AUTO_REINGEST enabled — deleting chroma_db and re-running ingestion...")
                try:
                    shutil.rmtree(CHROMA_DB_DIR, ignore_errors=True)
                    run_ingestion_if_needed()
                    self.db = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=self.embeddings)
                    print("Recreated Chroma DB with current embedding model.")
                except Exception:
                    print("Auto reingest failed:")
                    traceback.print_exc()
        else:
            print("Embedding dimensions OK (or unavailable).")

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
        q_emb = None
        try:
            if self.embeddings is None:
                raise RuntimeError("Embeddings client not available")
            if hasattr(self.embeddings, "embed_query"):
                q_emb = self.embeddings.embed_query(query)
            elif hasattr(self.embeddings, "embed"):
                q_emb = self.embeddings.embed([query])[0]
            else:
                raise AttributeError("No known embed method on embeddings client")
        except Exception:
            print("Embedding query failed — printing traceback:")
            traceback.print_exc()
            q_emb = None

        # try Chroma similarity_search to get candidates
        candidates = []
        try:
            if self.db is None:
                raise RuntimeError("DB client not available")
            candidates = self.db.similarity_search(query, k=candidate_pool)
        except Exception:
            if debug:
                print("[retrieve] similarity_search failed; falling back to db.get()")
                traceback.print_exc()
            # fallback: read raw arrays from DB
            try:
                res = self.db.get(include=["documents", "embeddings", "metadatas"]) if self.db else {}
                docs_list = res.get("documents", [])
                embs_list = res.get("embeddings", [])
                metas_list = res.get("metadatas", [])
                class _D: pass
                candidates = []
                for i in range(len(docs_list)):
                    d = _D()
                    d.page_content = docs_list[i]
                    d.metadata = metas_list[i] if metas_list else {}
                    d._embedding = embs_list[i] if i < len(embs_list) else None
                    candidates.append(d)
            except Exception:
                print("Fallback DB read also failed:")
                traceback.print_exc()

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
            try:
                res = self.db.get(include=["documents", "embeddings", "metadatas"]) if self.db else {}
                texts = res.get("documents", [])
                embeddings = res.get("embeddings", [])
                metas = res.get("metadatas", []) or []
            except Exception:
                texts = []
                embeddings = []
                metas = []

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
            base_score = self._cosine(q_emb, embeddings[idx]) if q_emb is not None else 0.0
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
        docs = self.retrieve(query, k=FINAL_K, debug=DEBUG,
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
        "You are a RAG assistant. Use ONLY the provided context.\n"
"Format your answers using clean Markdown:\n"
"- Use **bold** for important terms\n"
"- Use headings like: ## Sales Module\n"
"- Use bullet points and spacing\n"
"- Keep paragraphs readable\n"
"Do NOT invent structure not supported by the text.\n"
"If a fact comes from a document, include its source in brackets (e.g. [boldfit.pdf])."
},
    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
]

        raw_answer = ""
        try:
            if self.llm is None:
                raise RuntimeError("LLM client not available")
            response = self.llm.invoke(messages)
            raw_answer = response.content
        except Exception:
            print("LLM invocation failed — traceback:")
            traceback.print_exc()
            raw_answer = "I couldn't call the language model due to an internal error."

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
    if DEBUG:
        print("Running rag_fixed.py in debug mode")
        print_env_presence()
        test_ollama_endpoint()

    p = RAGPipeline()
    a, ds = p.ask("hello")
    print("\n=== ANSWER ===\n")
    print(a)