# app.py (Minimalist Chat Interface)
import streamlit as st
# NOTE: The import is set to 'newrag' to match your existing file name 'newrag.py'
from rag import RAGPipeline 

st.set_page_config(page_title="Local RAG Chat", layout="centered")
st.title("Offline RAG")

# --- Pipeline Initialization ---
# Ensure the pipeline initializes (and ingests docs if needed)
@st.cache_resource
def initialize_pipeline():
    with st.spinner("Initializing retrieval pipeline..."):
        return RAGPipeline()

pipeline = initialize_pipeline()

# --- Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
    
# Initial greeting message
if not st.session_state.messages:
    st.session_state.messages.append({"role": "assistant", "content": "Hello! I'm your offline RAG assistant. Ask me anything about your documents."})


# --- Message Rendering ---
def render_message(role, content):
    # Use standard Streamlit chat message components
    avatar = "👤" if role == "user" else "✨"
    with st.chat_message(role, avatar=avatar):
        st.markdown(content) # Use st.markdown for basic text formatting (e.g., paragraphs, bolding)

for msg in st.session_state.messages:
    render_message(msg["role"], msg["content"])

# --- Chat Input and Response ---
query = st.chat_input("Ask a question...")

if query:
    # 1. Display user query
    st.session_state.messages.append({"role": "user", "content": query})
    render_message("user", query)

    # 2. Get RAG answer
    with st.spinner("Thinking — retrieving from local docs..."):
        answer, docs = pipeline.ask(query)

    # 3. Handle personalization (removed name memory, but keeping it simple)
    final_answer = answer 

    # 4. Display assistant response
    st.session_state.messages.append({"role": "assistant", "content": final_answer})
    render_message("assistant", final_answer)