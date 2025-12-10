# app.py — clean, sidebar-free chat UI
import streamlit as st
from rag import RAGPipeline
from ingest import run_ingestion_if_needed
import traceback

st.set_page_config(page_title="RAG Chat", layout="centered")
st.title("RAG Chat")
st.caption("Chat with your documents (Ollama Cloud models configured via Streamlit secrets)")

# --- Pipeline Initialization (cached) ---
@st.cache_resource
def initialize_pipeline():
    with st.spinner("Initializing retrieval pipeline..."):
        return RAGPipeline()

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

# Top controls (minimal)
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

        # Show retrieved source documents in a compact expander list
        if docs:
            for i, d in enumerate(docs, start=1):
                src = d.metadata.get("source", "unknown") if isinstance(d.metadata, dict) else "unknown"
                preview = (d.page_content or "")[:600]
                with st.expander(f"Source {i}: {src}", expanded=False):
                    st.write(preview)
                    if len(d.page_content or "") > len(preview):
                        st.text_area(f"Full text — {src}", value=d.page_content or "", height=300, key=f"full_{i}_{hash(src)}")

# Keep conversation history hidden but accessible for debugging if needed
if st.checkbox("Show conversation history (debug)", value=False):
    for m in st.session_state.messages:
        st.write(f"{m['role']}: {m['content'][:400]}")
