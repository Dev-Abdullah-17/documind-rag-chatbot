import streamlit as st
import time
from pathlib import Path

# Page Config
st.set_page_config(
    page_title="DocuMind – RAG Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600;700&display=swap');

:root {
    --bg-deep: #0d0a14;
    --bg-card: rgba(255,255,255,0.045);
    --bg-card-solid: #15111f;
    --bg-sidebar: rgba(10, 8, 16, 0.55);
    --accent: #8b6af7;
    --accent2: #d66bf0;
    --accent-glow: rgba(139, 106, 247, 0.35);
    --text-primary: #efeaf9;
    --text-muted: #837d99;
    --border: rgba(255,255,255,0.08);
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: var(--text-primary);
}

/* ── Ambient purple-dusk background (mirrors the reference mockup) ── */
.stApp {
    background:
        radial-gradient(circle at 15% 8%, rgba(214,107,240,0.16), transparent 42%),
        radial-gradient(circle at 85% 0%, rgba(139,106,247,0.20), transparent 45%),
        radial-gradient(circle at 50% 100%, rgba(90,60,150,0.18), transparent 55%),
        linear-gradient(165deg, #241539 0%, #160f28 38%, #0d0a14 75%);
    background-attachment: fixed;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.stDeployButton {display: none !important;}
[data-testid="stToolbarActions"] {display: none !important;}

/* Fix: never hide/shrink the header itself and never touch its
   `display`/`overflow` — that's what was clipping the sidebar
   reopen arrow off-screen. Instead just recolor it so it blends
   with the dark theme, and hide only the deploy/menu clutter. */
header[data-testid="stHeader"] {
    background: rgba(13, 10, 20, 0.7) !important;
    backdrop-filter: blur(14px);
    border-bottom: 1px solid var(--border);
}
header[data-testid="stHeader"] svg { fill: var(--accent) !important; }
header[data-testid="stHeader"] button { color: var(--accent) !important; }

[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"] {
    visibility: visible !important;
    opacity: 1 !important;
}
[data-testid="stSidebarCollapsedControl"] svg,
[data-testid="collapsedControl"] svg {
    fill: var(--accent) !important;
}

/* The fixed bottom bar that wraps the chat input also ships with
   its own light background by default — recolor it too. */
[data-testid="stBottomBlockContainer"],
[data-testid="stBottom"] {
    background: transparent !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: var(--text-muted) !important;
}

.block-container {padding-top: 1.2rem; padding-bottom: 6rem; max-width: 900px;}

/* Floating glass panel feel for the main container */
section.main > div.block-container {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 22px;
    backdrop-filter: blur(18px);
    margin-top: 0.6rem;
}

[data-testid="stSidebar"] {
    background: var(--bg-sidebar);
    backdrop-filter: blur(18px);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] > div:first-child { padding-top: 1.2rem; }

/* ── Top app header ── */
.app-header {
    display: flex; align-items: center; justify-content: space-between;
    gap: 14px;
    padding: 0.4rem 0.6rem 1.2rem 0.6rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.6rem;
}
.app-header .left { display:flex; align-items:center; gap: 12px; }
.orb {
    width: 34px; height: 34px; border-radius: 50%; flex-shrink: 0;
    background: radial-gradient(circle at 32% 28%, #efe6ff 0%, var(--accent) 40%, var(--accent2) 100%);
    box-shadow: 0 0 18px var(--accent-glow), inset -3px -3px 8px rgba(0,0,0,0.35);
}
.app-header .logo {
    font-family: 'Space Mono', monospace;
    font-size: 1.35rem; font-weight: 700;
    background: linear-gradient(135deg, #efe6ff, var(--accent) 55%, var(--accent2));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    line-height: 1.1;
}
.app-header .tagline { font-size: 0.76rem; color: var(--text-muted); margin-top: 1px; }
.model-pill {
    font-size: 0.74rem; color: var(--text-muted);
    border: 1px solid var(--border); background: rgba(255,255,255,0.03);
    padding: 5px 12px; border-radius: 20px; display:flex; align-items:center; gap:6px;
}
.status-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--accent2); box-shadow: 0 0 8px var(--accent2);
    animation: pulse 2s infinite;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.35} }

/* ── Sidebar section labels ── */
.side-label {
    font-family:'Space Mono', monospace; font-size: 0.68rem; letter-spacing: 0.08em;
    text-transform: uppercase; color: var(--text-muted); margin: 1.1rem 0 0.5rem 0;
}

.stat-row { display: flex; gap: 10px; margin-bottom: 0.8rem; }
.stat-card {
    flex: 1; padding: 12px 10px; background: var(--bg-card-solid);
    border: 1px solid var(--border); border-radius: 12px; text-align: center;
}
.stat-card .val {
    font-family: 'Space Mono', monospace; font-size: 1.3rem;
    color: var(--accent); font-weight: 700;
}
.stat-card .lbl { font-size: 0.66rem; color: var(--text-muted); margin-top: 2px; }

.doc-item {
    display: flex; align-items: center; gap: 8px;
    padding: 7px 12px; background: rgba(255,255,255,0.03);
    border: 1px solid var(--border); border-radius: 8px; margin-bottom: 6px;
    font-size: 0.8rem;
}

/* ── Hero / welcome state ── */
.hero { text-align: center; padding: 2.2rem 1rem 1rem 1rem; }
.hero .orb-lg {
    width: 74px; height: 74px; border-radius: 50%; margin: 0 auto 1.1rem auto;
    background: radial-gradient(circle at 32% 28%, #f5eeff 0%, var(--accent) 42%, var(--accent2) 100%);
    box-shadow: 0 0 40px var(--accent-glow), inset -6px -6px 14px rgba(0,0,0,0.35);
}
.hero h2 {
    font-family: 'Space Mono', monospace; font-weight: 700; font-size: 1.5rem;
    color: var(--text-primary); margin-bottom: 0.5rem;
}
.hero p { color: var(--text-muted); font-size: 0.92rem; line-height: 1.7; max-width: 480px; margin: 0 auto; }

.chip-row { display:flex; gap:10px; justify-content:center; flex-wrap:wrap; margin: 1.4rem 0 1.8rem 0; }
.chip {
    font-size: 0.78rem; color: var(--text-primary);
    background: rgba(255,255,255,0.045); border: 1px solid var(--border);
    padding: 7px 16px; border-radius: 20px; display:flex; align-items:center; gap:6px;
}

.feature-row { display:flex; gap: 14px; margin-top: 1rem; flex-wrap: wrap; }
.feature-card {
    flex: 1; min-width: 190px; padding: 16px 16px 14px 16px;
    background: var(--bg-card-solid); border: 1px solid var(--border);
    border-radius: 14px; text-align: left;
}
.feature-card .icon { font-size: 1.25rem; }
.feature-card .tag {
    float: right; font-size: 0.62rem; color: var(--accent);
    border: 1px solid rgba(139,106,247,0.4); border-radius: 20px; padding: 2px 9px;
    background: rgba(139,106,247,0.08);
}
.feature-card h4 { font-size: 0.88rem; margin: 10px 0 4px 0; color: var(--text-primary); }
.feature-card p { font-size: 0.76rem; color: var(--text-muted); line-height: 1.5; margin:0; }

[data-testid="stFileUploader"] {
    background: var(--bg-card-solid) !important;
    border: 1px dashed rgba(139,106,247,0.45) !important;
    border-radius: 12px !important;
}

.stButton > button {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-muted) !important;
    border-radius: 10px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    background: var(--accent-glow) !important;
}

/* Chat input styled like the "Ask anything" pill */
[data-testid="stChatInput"] {
    background: var(--bg-card-solid);
    border: 1px solid var(--border);
    border-radius: 18px;
}
[data-testid="stChatInput"] textarea { color: var(--text-primary) !important; }
</style>
""", unsafe_allow_html=True)


# Session State
def init_state():
    defaults = {
        "messages": [],
        "vectorstore": None,
        "doc_names": [],
        "total_chunks": 0,
        "api_key": "",
        "sidebar_visible": True,
        "model_choice": "groq-llama3",
        "temperature": 0.3,
        "top_k": 4,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ── Manual sidebar visibility (does not rely on Streamlit's native
#    collapse arrow, whose testid changes across versions) ──
if not st.session_state.sidebar_visible:
    st.markdown("""
    <style>
    [data-testid="stSidebar"] { display: none !important; }
    </style>
    """, unsafe_allow_html=True)


# Cached Libs
@st.cache_resource(show_spinner=False)
def load_embeddings():
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# Process PDFs
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


# Get LLM
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


# Stream Answer
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
        full_response = f" Error: {str(e)}"
        stream_placeholder.warning(full_response)

    return full_response, sources


# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="display:flex; align-items:center; gap:10px; margin-bottom:0.3rem;">
        <div class="orb" style="width:28px;height:28px;"></div>
        <div class="logo" style="font-size:1.15rem;">DocuMind</div>
    </div>
    <div class="tagline" style="margin-bottom:0.6rem;">RAG-Powered Document Chat</div>
    """, unsafe_allow_html=True)

    if st.button("◀ Hide sidebar", use_container_width=True, key="hide_sidebar_btn"):
        st.session_state.sidebar_visible = False
        st.rerun()

    st.markdown('<div class="side-label">Model Settings</div>', unsafe_allow_html=True)
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

    st.markdown('<div class="side-label">📄 Upload Documents</div>', unsafe_allow_html=True)

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
                st.success(f" Indexed {n_chunks} chunks from {len(names)} file(s)")
            except Exception as e:
                st.error(f"Error: {e}")

    if st.session_state.doc_names:
        st.markdown('<div class="side-label">📚 Loaded Documents</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="stat-row">
            <div class="stat-card"><div class="val">{len(st.session_state.doc_names)}</div><div class="lbl">Files</div></div>
            <div class="stat-card"><div class="val">{st.session_state.total_chunks}</div><div class="lbl">Chunks</div></div>
        </div>
        """, unsafe_allow_html=True)
        for name in st.session_state.doc_names:
            st.markdown(f'<div class="doc-item">📄 {name}</div>', unsafe_allow_html=True)

    st.markdown('<div class="side-label">💬 Chat</div>', unsafe_allow_html=True)
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
        '<div style="font-size:0.7rem;color:#565068;text-align:center;">DocuMind v3.0 · LangChain + FAISS</div>',
        unsafe_allow_html=True
    )


# ─── MAIN AREA ────────────────────────────────────────────────────────────────
model_display = {
    "groq-llama3":       "Llama 3.3 70B",
    "groq-llama3-8b":    "Llama 3 8B",
    "openai-gpt4o-mini": "GPT-4o Mini",
    "gemini-flash":      "Gemini 1.5 Flash",
}[st.session_state.model_choice]

header_left, header_right = st.columns([0.06, 0.94]) if not st.session_state.sidebar_visible else (None, None)

if not st.session_state.sidebar_visible:
    with header_left:
        st.write("")
        if st.button("☰", key="show_sidebar_btn", help="Show sidebar"):
            st.session_state.sidebar_visible = True
            st.rerun()
    with header_right:
        st.markdown(f"""
        <div class="app-header">
            <div class="left">
                <div class="orb"></div>
                <div>
                    <div class="logo">DocuMind</div>
                    <div class="tagline">Context-aware document intelligence · RAG + LangChain</div>
                </div>
            </div>
            <div style="display:flex; align-items:center; gap:10px;">
                <div class="model-pill"><span class="status-dot"></span>{model_display}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="app-header">
        <div class="left">
            <div class="orb"></div>
            <div>
                <div class="logo">DocuMind</div>
                <div class="tagline">Context-aware document intelligence · RAG + LangChain</div>
            </div>
        </div>
        <div style="display:flex; align-items:center; gap:10px;">
            <div class="model-pill"><span class="status-dot"></span>{model_display}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Welcome / hero state
if not st.session_state.vectorstore:
    st.markdown("""
    <div class="hero">
        <div class="orb-lg"></div>
        <h2>Ready to explore your documents?</h2>
        <p>Chat with multiple PDFs using Retrieval-Augmented Generation.
        Every answer is grounded in your documents with source citations.</p>
    </div>
    <div class="chip-row">
        <div class="chip">🔑 1&nbsp; Select a model & enter API key</div>
        <div class="chip">📄 2&nbsp; Upload PDF files</div>
        <div class="chip">⚡ 3&nbsp; Click Process</div>
        <div class="chip">💬 4&nbsp; Ask questions</div>
    </div>
    <div class="feature-row">
        <div class="feature-card">
            <span class="icon">📚</span><span class="tag">Multi-file</span>
            <h4>Multi-PDF Chat</h4>
            <p>Upload and query several documents together in one conversation.</p>
        </div>
        <div class="feature-card">
            <span class="icon">🧠</span><span class="tag">4 engines</span>
            <h4>Multi-Model Engine</h4>
            <p>Switch between Llama, GPT-4o Mini and Gemini Flash anytime.</p>
        </div>
        <div class="feature-card">
            <span class="icon">🔎</span><span class="tag">Grounded</span>
            <h4>Cited Answers</h4>
            <p>Every response links back to the exact file and page it came from.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Chat History ──────────────────────────────────────────────────────────
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

# ── Input ─────────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask anything about your documents...")

if user_input and user_input.strip():
    if not st.session_state.vectorstore:
        st.warning("⚠️ Please upload and process documents first.")
    elif not st.session_state.api_key:
        st.warning("⚠️ Please enter your API key in the sidebar.")
    else:
        question = user_input.strip()

        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            answer, sources = stream_answer(question)
            if sources:
                source_list = " · ".join([f"`{s}`" for s in sources.keys()])
                st.markdown(f"📎 **Sources:** {source_list}")

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })
