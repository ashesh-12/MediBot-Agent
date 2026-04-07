"""
app.py
MediBot — Agentic Medical Assistant
Streamlit UI entry point.

Run with:
    streamlit run app.py
"""

import streamlit as st
from dotenv import load_dotenv
from agent.medibot_agent import build_agent
from agent.memory import get_memory
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

load_dotenv()

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="MediBot — Medical Assistant",
    page_icon="🏥",
    layout="centered",
)

# ── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8fafc; }
    .stChatMessage { border-radius: 12px; margin-bottom: 8px; }
    .disclaimer {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 10px 16px;
        border-radius: 6px;
        font-size: 0.85rem;
        color: #856404;
        margin-bottom: 16px;
    }
    .tool-badge {
        background-color: #e8f4f8;
        border: 1px solid #bee3f8;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.75rem;
        color: #2b6cb0;
        margin-right: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────
st.title("🏥 MediBot")
st.caption("AI-powered medical information assistant · Powered by LangChain + Groq")

# ── Disclaimer ─────────────────────────────────────────────────
st.markdown("""
<div class="disclaimer">
    ⚠️ <strong>Medical Disclaimer:</strong> MediBot provides general medical
    information only. Always consult a qualified healthcare professional
    for personal medical advice, diagnosis, or treatment.
</div>
""", unsafe_allow_html=True)

# ── Tool info ──────────────────────────────────────────────────
with st.expander("ℹ️ What can MediBot do?"):
    st.markdown("""
    MediBot is an **agentic AI** that automatically picks the right tool for your question:

    - 🔍 **Symptom Checker** — Describe your symptoms and get possible conditions
    - 📖 **Disease Q&A** — Ask about causes, risk factors, and diagnosis
    - 💊 **Treatment Summary** — Get structured treatment options for any condition

    **Example questions:**
    - *"I have fever, headache and stiff neck — what could this be?"*
    - *"What causes type 2 diabetes?"*
    - *"How is hypertension treated?"*
    """)

# ── Session state ──────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent" not in st.session_state:
    with st.spinner("Loading MediBot..."):
        memory = get_memory(k=5)
        st.session_state.agent = build_agent(memory)
        st.session_state.memory = memory

# ── Chat history ───────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ── Chat input ─────────────────────────────────────────────────
if prompt := st.chat_input("Ask me about symptoms, diseases, or treatments..."):

    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.agent.invoke(
                    {"input": prompt}
                )
                answer = response.get("output", "I could not generate a response.")
            except Exception as e:
                answer = (
                    f"I encountered an error processing your request. "
                    f"Please try rephrasing your question.\n\n"
                    f"*Error: {str(e)}*"
                )

        st.markdown(answer)
        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    st.markdown("**Model:** llama-3.3-70b-versatile")
    st.markdown("**Vector Store:** FAISS (local)")
    st.markdown("**Embeddings:** all-MiniLM-L6-v2")
    st.markdown("**Memory:** Last 5 exchanges")

    st.divider()

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        memory = get_memory(k=5)
        st.session_state.agent = build_agent(memory)
        st.session_state.memory = memory
        st.rerun()

    st.divider()

    st.markdown("""
    **About MediBot**
    Built with LangChain, Groq, FAISS
    and Streamlit.
    Knowledge base: Medical Encyclopedia PDF
    """)