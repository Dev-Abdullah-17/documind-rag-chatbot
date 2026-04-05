import streamlit as st
import time
from pathlib import Path

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocuMind – RAG Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg-main: #0d0f14;
    --bg-card: #13161e;
    --bg-sidebar: #0a0c10;
    --accent: #7c6af7;
    --accent-glow: rgba(124, 106, 247, 0.25);
    --accent2: #4ecca3;
    --text-primary: #e8eaf0;
    --text-muted: #6b7280;
    --border: rgba(255,255,255,0.07);
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg-main);
    color: var(--text-primary);
}

#MainMenu, footer, header {visibility: hidden;}
.block-container {padding-top: 1.5rem; padding-bottom: 2rem;}

[data-testid="stSidebar"] {
    background: var(--bg-sidebar);
    border-right: 1px solid var(--border);
}

.app-header {
    display: flex; align-items: center; gap: 14px;
    padding: 0 0 1.2rem 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.5rem;
}
.app-header .logo {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem; font-weight: 700;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.app-header .tagline { font-size: 0.78rem; color: var(--text-muted); margin-top: 2px; }
.status-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--accent2); box-shadow: 0 0 8px var(--accent2);
    animation: pulse 2s infinite; flex-shrink: 0;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }

.stat-row { display: flex; gap: 10px; margin-bottom: 1rem; }
.stat-card {
    flex: 1; padding: 12px 14px; background: var(--bg-card);
    border: 1px solid var(--border); border-radius: 10px; text-align: center;
}
.stat-card .val {
    font-family: 'Space Mono', monospace; font-size: 1.4rem;
    color: var(--accent); font-weight: 700;
}
.stat-card .lbl { font-size: 0.7rem; color: var(--text-muted); margin-top: 2px; }

.doc-item {
    display: flex; align-items: center; gap: 8px;
    padding: 8px 12px; background: rgba(255,255,255,0.03);
    border: 1px solid var(--border); border-radius: 8px; margin-bottom: 6px;
    font-size: 0.82rem;
}

.welcome-card {
    text-align: center; padding: 3rem 2rem;
    background: var(--bg-card); border-radius: 16px;
    border: 1px solid var(--border);
}
.welcome-card h2 { font-family: 'Space Mono', monospace; color: var(--accent); font-size: 1.4rem; }
.welcome-card p { color: var(--text-muted); font-size: 0.9rem; line-height: 1.7; }

[data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 1px dashed rgba(124,106,247,0.4) !important;
    border-radius: 12px !important;
}

.stButton > button {
    background: transparent !important;
    border: 1px solid var(--border) !important;
    color: var(--text-muted) !important;
    border-radius: 8px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    background: var(--accent-glow) !important;
}
</style>
""", unsafe_allow_html=True)


# ─── Session State ────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "messages": [],
        "vectorstore": None,
        "doc_names": [],
        "total_chunks": 0,
        "api_key": "",
        "model_choice": "groq-llama3",
        "temperature": 0.3,
        "top_k": 4,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ─── Cached Libs ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_embeddings():
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# ─── Process PDFs ────────────────────────────────────────────────────────────
def process_pdfs(uploaded_files):
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS

    all_docs = []
    names = []
    tmp_dir = Path("/tmp/rag_docs")
    tmp_dir.mkdir(exist_ok=True)

    for f in uploaded_files:
        path = tmp_dir / f.name
        path.write_bytes(f.read())
        loader = PyPDFLoader(str(path))
        docs = loader.load()
        for d in docs:
            d.metadata["source_file"] = f.name
        all_docs.extend(docs)
        names.append(f.name)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=60
    )
    chunks = splitter.split_documents(all_docs)
    embeddings = load_embeddings()
    vs = FAISS.from_documents(chunks, embeddings)

    st.session_state.vectorstore = vs
    st.session_state.doc_names = names
    st.session_state.total_chunks = len(chunks)
    return len(chunks), names


# ─── Get LLM ─────────────────────────────────────────────────────────────────
def get_llm():
    choice = st.session_state.model_choice
    api_key = st.session_state.api_key
    temp = st.session_state.temperature

    if choice == "groq-llama3":
        from langchain_groq import ChatGroq
        return ChatGroq(model="llama-3.3-70b-versatile", temperature=temp, api_key=api_key)
    elif choice == "groq-llama3-8b":
        from langchain_groq import ChatGroq
        return ChatGroq(model="llama3-8b-8192", temperature=temp, api_key=api_key)
    elif choice == "openai-gpt4o-mini":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini", temperature=temp, api_key=api_key)
    elif choice == "gemini-flash":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=temp, google_api_key=api_key)


# ─── Stream Answer ───────────────────────────────────────────────────────────
def stream_answer(question: str):
    vs = st.session_state.vectorstore
    k = st.session_state.top_k

    docs_with_scores = vs.similarity_search_with_score(question, k=k)
    context_parts = []
    sources = {}

    for doc, score in docs_with_scores:
        fname = doc.metadata.get("source_file", "Document")
        page = doc.metadata.get("page", 0)
        context_parts.append(f"[Source: {fname}, Page {page+1}]\n{doc.page_content}")
        key = f"{fname} (p.{page+1})"
        sources[key] = round(float(score), 3)

    context = "\n\n---\n\n".join(context_parts)

    # Build conversation history (last 3 turns)
    history_text = ""
    for msg in st.session_state.messages[-6:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content']}\n"

    prompt = f"""You are DocuMind, a precise and helpful document assistant.
Answer the user's question based ONLY on the provided context.
If the answer is not in the context, say so clearly.
Be concise and accurate.

CONVERSATION HISTORY:
{history_text}

CONTEXT FROM DOCUMENTS:
{context}

USER QUESTION: {question}

ANSWER:"""

    llm = get_llm()
    full_response = ""
    stream_placeholder = st.empty()

    try:
        for chunk in llm.stream(prompt):
            token = chunk.content if hasattr(chunk, "content") else str(chunk)
            full_response += token
            stream_placeholder.markdown(full_response + "▌")
            time.sleep(0.01)
        stream_placeholder.markdown(full_response)
    except Exception as e:
        full_response = f"⚠️ Error: {str(e)}"
        stream_placeholder.warning(full_response)

    return full_response, sources


# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 DocuMind")
    st.markdown("*RAG-Powered Document Chat*")
    st.markdown("---")

    st.markdown("### ⚙️ Model Settings")
    st.session_state.model_choice = st.selectbox(
        "LLM",
        ["groq-llama3", "groq-llama3-8b", "openai-gpt4o-mini", "gemini-flash"],
        format_func=lambda x: {
            "groq-llama3":      "🦙 Llama 3.3 70B (Groq)",
            "groq-llama3-8b":   "🦙 Llama 3 8B (Groq - higher limits)",
            "openai-gpt4o-mini":"🤖 GPT-4o Mini (OpenAI)",
            "gemini-flash":     "✨ Gemini 1.5 Flash (Google - Free)",
        }[x],
        label_visibility="collapsed",
    )

    api_label = {
        "groq-llama3":       "Groq API Key",
        "groq-llama3-8b":    "Groq API Key",
        "openai-gpt4o-mini": "OpenAI API Key",
        "gemini-flash":      "Google AI Key → aistudio.google.com",
    }[st.session_state.model_choice]

    st.session_state.api_key = st.text_input(
        api_label, type="password", placeholder=f"Enter {api_label}..."
    )

    st.session_state.temperature = st.slider("🌡 Temperature", 0.0, 1.0, st.session_state.temperature, 0.05)
    st.session_state.top_k = st.slider("🔍 Top-K Chunks", 2, 8, st.session_state.top_k)

    st.markdown("---")
    st.markdown("### 📄 Upload Documents")

    uploaded_files = st.file_uploader(
        "PDFs", type=["pdf"], accept_multiple_files=True, label_visibility="collapsed"
    )

    col1, col2 = st.columns(2)
    with col1:
        process_btn = st.button("⚡ Process", use_container_width=True)
    with col2:
        if st.button("🗑 Clear All", use_container_width=True):
            st.session_state.messages = []
            st.session_state.vectorstore = None
            st.session_state.doc_names = []
            st.session_state.total_chunks = 0
            st.rerun()

    if process_btn and uploaded_files:
        with st.spinner("Embedding documents..."):
            try:
                n_chunks, names = process_pdfs(uploaded_files)
                st.success(f"✅ Indexed {n_chunks} chunks from {len(names)} file(s)")
            except Exception as e:
                st.error(f"Error: {e}")

    if st.session_state.doc_names:
        st.markdown("---")
        st.markdown("### 📚 Loaded Documents")
        st.markdown(f"""
        <div class="stat-row">
            <div class="stat-card"><div class="val">{len(st.session_state.doc_names)}</div><div class="lbl">Files</div></div>
            <div class="stat-card"><div class="val">{st.session_state.total_chunks}</div><div class="lbl">Chunks</div></div>
        </div>
        """, unsafe_allow_html=True)
        for name in st.session_state.doc_names:
            st.markdown(f'<div class="doc-item">📄 {name}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 💬 Chat")
    if st.button("🧹 Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    if st.session_state.messages:
        chat_text = "\n\n".join(
            [f"{'You' if m['role']=='user' else 'DocuMind'}: {m['content']}"
             for m in st.session_state.messages]
        )
        st.download_button("⬇️ Export Chat", chat_text, "chat.txt", use_container_width=True)

    st.markdown("---")
    st.markdown(
        '<div style="font-size:0.72rem;color:#4b5563;text-align:center;">DocuMind v3.0 · LangChain + FAISS</div>',
        unsafe_allow_html=True
    )


# ─── MAIN AREA ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <div class="status-dot"></div>
    <div>
        <div class="logo">DocuMind</div>
        <div class="tagline">Context-aware document intelligence · RAG + LangChain</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Welcome state
if not st.session_state.vectorstore:
    st.markdown("""
    <div class="welcome-card">
        <h2>📚 Upload Your Documents to Begin</h2>
        <p>Chat with multiple PDFs using Retrieval-Augmented Generation.<br>
        Every answer is grounded in your documents with source citations.</p>
        <p style="margin-top:1rem;">
        ① Select a model & enter API key &nbsp;|&nbsp;
        ② Upload PDF files &nbsp;|&nbsp;
        ③ Click ⚡ Process &nbsp;|&nbsp;
        ④ Ask questions!
        </p>
        <p style="font-size:0.8rem; color:#7c6af7; margin-top:1rem;">
        💡 Hit rate limit on Groq? Switch to <strong>Llama 3 8B</strong> (higher free limits)
        or <strong>Gemini Flash</strong> (free at aistudio.google.com)
        </p>
    </div>
    """, unsafe_allow_html=True)

# ── Chat History (native Streamlit — no HTML bugs) ────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(msg["content"])
            if msg.get("sources"):
                source_list = " · ".join([f"`{s}`" for s in msg["sources"].keys()])
                st.markdown(f"📎 **Sources:** {source_list}")

# ── Input ─────────────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask anything about your documents...")

if user_input and user_input.strip():
    if not st.session_state.vectorstore:
        st.warning("⚠️ Please upload and process documents first.")
    elif not st.session_state.api_key:
        st.warning("⚠️ Please enter your API key in the sidebar.")
    else:
        question = user_input.strip()

        # Save + show user message
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Stream assistant response
        with st.chat_message("assistant"):
            answer, sources = stream_answer(question)
            if sources:
                source_list = " · ".join([f"`{s}`" for s in sources.keys()])
                st.markdown(f"📎 **Sources:** {source_list}")

        # Save to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })
