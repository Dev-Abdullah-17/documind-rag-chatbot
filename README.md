# 🧠 DocuMind — Context-Aware RAG Chatbot

> Chat with multiple PDFs simultaneously using Retrieval-Augmented Generation (RAG), LangChain, and FAISS — with real-time streaming responses, source citations, and multi-LLM support.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![LangChain](https://img.shields.io/badge/LangChain-0.2+-green?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red?style=flat-square&logo=streamlit)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

---

## 🚀 Live Demo

> 🔗 **[Try it here → Live Demo](https://rag-chatbotapp.streamlit.app/)**

---

## 📸 Preview

![DocuMind Screenshot](assets/screenshot.png)

---

## ✨ Features

| Feature | Description |
|---|---|
| 📚 **Multi-PDF Support** | Upload and query multiple PDFs at the same time |
| 🔍 **FAISS Vector Search** | Fast semantic retrieval with configurable Top-K chunks |
| 🧠 **4 LLM Options** | Switch between Llama 3.3 70B, Llama 3.1 8B (Groq), GPT-4o Mini, Gemini 1.5 Flash |
| 💬 **Conversation Memory** | Maintains last 3 turns of context for follow-up questions |
| ⚡ **Streaming Responses** | Real-time token-by-token output as the model generates |
| 📎 **Source Citations** | Every answer shows exactly which document and page it came from |
| 🎨 **Clean Dark UI** | Professional Streamlit interface with sidebar controls |
| ⬇️ **Export Chat** | Download your full conversation as a `.txt` file |
| 🗑 **Clear Controls** | Reset documents or chat history independently |

---

## 🏗️ How It Works

```
                        ┌─────────────────────────────┐
  Upload PDFs    ──►    │   PyPDF Loader + Splitter    │
                        │  (RecursiveCharacterSplitter) │
                        └────────────┬────────────────┘
                                     │ text chunks
                                     ▼
                        ┌─────────────────────────────┐
                        │   HuggingFace Embeddings     │
                        │    (all-MiniLM-L6-v2)        │
                        └────────────┬────────────────┘
                                     │ vectors
                                     ▼
                        ┌─────────────────────────────┐
                        │      FAISS Vector Store      │  ◄── stored in memory
                        └────────────┬────────────────┘
                                     │
  User Question  ──►  similarity_search(question, k=4)
                                     │ top-k relevant chunks
                                     ▼
                        ┌─────────────────────────────┐
                        │      Prompt Builder          │
                        │  context + history + query   │
                        └────────────┬────────────────┘
                                     │
                                     ▼
                        ┌─────────────────────────────┐
                        │     LLM (streaming)          │
                        │  Groq / OpenAI / Gemini      │
                        └────────────┬────────────────┘
                                     │ streamed answer + sources
                                     ▼
                              Streamlit UI
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **UI** | Streamlit |
| **RAG Pipeline** | LangChain |
| **Vector Database** | FAISS (Facebook AI Similarity Search) |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` |
| **LLMs** | Groq (Llama 3.3 70B, Llama 3.1 8B), OpenAI GPT-4o Mini, Google Gemini 1.5 Flash |
| **PDF Loader** | PyPDF via LangChain |

---

## ⚙️ Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/documind-rag-chatbot
cd documind-rag-chatbot
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Get a free API key

| LLM | Where to get key | Cost |
|---|---|---|
| 🦙 Llama 3.3 70B (Groq) | [console.groq.com](https://console.groq.com) | ✅ Free |
| 🦙 Llama 3.1 8B (Groq) | [console.groq.com](https://console.groq.com) | ✅ Free |
| ✨ Gemini 1.5 Flash | [aistudio.google.com](https://aistudio.google.com) | ✅ Free |
| 🤖 GPT-4o Mini | [platform.openai.com](https://platform.openai.com) | Paid |

### 4. Run the app
```bash
streamlit run app.py
```

### 5. Use it
1. Select a model and paste your API key in the sidebar
2. Upload one or more PDF files
3. Click **⚡ Process** to build the vector index
4. Ask anything — answers stream in real time with source citations!

---

## 📁 Project Structure

```
documind-rag-chatbot/
├── app.py               # Main Streamlit application
├── requirements.txt     # Python dependencies
├── assets/
│   └── screenshot.png   # App screenshot (optional)
└── README.md
```

---

## 📦 Requirements

```
langchain>=0.2.0
langchain-community>=0.2.0
langchain-text-splitters>=0.2.0
langchain-groq>=0.1.0
langchain-openai>=0.1.0
langchain-google-genai>=1.0.0
faiss-cpu>=1.7.4
sentence-transformers>=2.2.0
streamlit>=1.35.0
pypdf>=4.0.0
```

---

## 🔧 Configuration

All settings are available in the sidebar UI at runtime:

| Setting | Default | Description |
|---|---|---|
| LLM | Llama 3.3 70B (Groq) | Language model for generating answers |
| Temperature | 0.3 | Creativity of responses (0 = precise, 1 = creative) |
| Top-K | 4 | Number of document chunks retrieved per query |

---

## 💡 Usage Tips

- **Hit rate limits on Groq 70B?** Switch to **Llama 3.1 8B Instant** — same API key, higher limits
- **Multiple PDFs?** Upload them all at once before clicking Process
- **Follow-up questions?** The bot remembers the last 3 turns of your conversation
- **Check sources** shown below each answer to verify which page the info came from

---

## 🗺️ Roadmap

- [ ] Support `.docx`, `.txt`, `.csv` files
- [ ] Persistent FAISS index (save/reload between sessions)
- [ ] User authentication
- [ ] Document summarization mode
- [ ] Query reformulation for better retrieval accuracy

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

---

## 📜 License

MIT License — free to use, modify, and distribute.

---

> **Built as part of my AI/ML portfolio.**  
> This project demonstrates end-to-end RAG pipeline design, FAISS vector search, multi-LLM integration, streaming responses, and production-quality Streamlit UI development.
