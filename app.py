# app.py — clean chat UI + small Maintenance controls
import streamlit as st
from rag import RAGPipeline
from ingest import run_ingestion_if_needed
import traceback
import shutil
import os
from pathlib import Path

st.set_page_config(page_title="RAG Chat", layout="centered")
st.title("RAG Chat")
st.caption("Chat with your documents (Ollama Cloud models configured via Streamlit secrets)")

# ---------- Utility: get chroma db dir from secrets or default ----------
CHROMA_DB_DIR = None
try:
    CHROMA_DB_DIR = st.secrets.get("CHROMA_DB_DIR", "chroma_db")
except Exception:
    CHROMA_DB_DIR = os.environ.get("CHROMA_DB_DIR", "chroma_db")
# ----------------------------------------------------------------------

# --- Pipeline Initialization (cached) ---
@st.cache_resource
def initialize_pipeline():
    with st.spinner("Initializing retrieval pipeline..."):
        return RAGPipeline()

# Try to initialize pipeline (show fatal error if fails)
try:
    pipeline = initialize_pipeline()
except Exception as e:
    st.error("Failed to initialize RAG pipeline. Check logs.")
    st.exception(traceback.format_exc())
    st.stop()

# --- Session State for chat ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Hello! I'm your RAG assistant. Ask me anything about your documents."
    })

# Top minimal controls row
cols = st.columns([1, 6, 1])
with cols[0]:
    pass
with cols[1]:
    if st.button("Clear chat"):
        st.session_state.messages = [{
            "role": "assistant",
            "content": "Hello! I'm your RAG assistant. Ask me anything about your documents."
        }]
        st.experimental_rerun()
with cols[2]:
    pass

# --------- Maintenance (compact, unobtrusive) ----------
with st.expander("Maintenance", expanded=False):
    st.write("Small admin actions for maintaining the vector DB and pipeline cache.")
    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("Rebuild Vector DB (force)"):
            # Confirm and run: delete DB dir then re-ingest
            try:
                db_path = Path(CHROMA_DB_DIR)
                if db_path.exists() and db_path.is_dir():
                    with st.spinner(f"Deleting existing DB at {db_path}..."):
                        shutil.rmtree(db_path)
                # run ingestion (will create DB)
                with st.spinner("Running ingestion to rebuild vector DB..."):
                    run_ingestion_if_needed()
                st.success("Vector DB rebuilt successfully.")
                # Clear cached pipeline so new DB is used
                try:
                    st.cache_resource.clear()
                    st.info("Cleared cached pipeline — it will reinitialize on next request.")
                except Exception:
                    pass
            except Exception as e:
                st.error(f"Failed to rebuild DB: {e}")
                st.exception(traceback.format_exc())
    with c2:
        if st.button("Clear cached pipeline"):
            try:
                st.cache_resource.clear()
                st.success("Pipeline cache cleared. It will reinitialize on next query.")
            except Exception as e:
                st.error(f"Failed to clear pipeline cache: {e}")
                st.exception(traceback.format_exc())
# -------------------------------------------------------

# Render chat messages
def render_message(role, content):
    avatar = "👤" if role == "user" else "✨"
    with st.chat_message(role, avatar=avatar):
        st.markdown(content)

for msg in st.session_state.messages:
    render_message(msg["role"], msg["content"])

# Chat input
query = st.chat_input("Ask a question...")

if query:
    # Append and render user's message immediately
    st.session_state.messages.append({"role": "user", "content": query})
    render_message("user", query)

    # Run retrieval + LLM
    try:
        with st.spinner("Thinking — retrieving from docs and querying LLM..."):
            answer, docs = pipeline.ask(query)
    except Exception as e:
        error_msg = "Sorry — an error occurred when answering. Check logs."
        st.session_state.messages.append({"role": "assistant", "content": error_msg})
        render_message("assistant", error_msg)
        st.error(f"Error: {e}")
        st.exception(traceback.format_exc())
        docs = []
    else:
        # Append assistant response
        st.session_state.messages.append({"role": "assistant", "content": answer})
        render_message("assistant", answer)

        # Show retrieved source documents as compact expanders
        if docs:
            for i, d in enumerate(docs, start=1):
                src = d.metadata.get("source", "unknown") if isinstance(d.metadata, dict) else "unknown"
                preview = (d.page_content or "")[:600]
                with st.expander(f"Source {i}: {src}", expanded=False):
                    st.write(preview)
                    if len(d.page_content or "") > len(preview):
                        st.text_area(f"Full text — {src}", value=d.page_content or "", height=300, key=f"full_{i}_{hash(src)}")

# Optional debug view (hidden by default)
if st.checkbox("Show conversation history (debug)", value=False):
    for m in st.session_state.messages:
        st.write(f"{m['role']}: {m['content'][:400]}")
