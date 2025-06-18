# ⚖️ Legal Document Q&A Chatbot

An intelligent, locally hosted **RAG + LLM** chatbot that answers queries based *strictly* on the contents of a **user-uploaded legal PDF**. Built with cutting-edge open-source tools and APIs for optimal performance, privacy, and relevance.

---

## 🚀 Features

- 🔍 **RAG Pipeline**: Retrieve-augmented generation ensures grounded answers.
- 📄 **PDF Upload Support**: Upload any legal PDF to initiate contextual Q&A.
- 🧠 **DeepSeek LLM & Embeddings**: Powered by DeepSeek for both vector generation and inference.
- 💬 **LangChain Integration**: Modular and scalable pipeline for prompt orchestration.
- 🧭 **Ollama + Groq**: Efficient local embedding with Ollama; blazing-fast LLM inference via Groq APIs.
- 📚 **FAISS Vector DB**: Enables fast and accurate similarity search on legal content.
- 🌐 **Streamlit UI**: Clean, user-friendly interface for real-time interaction.

---

## 📸 Demo

> 📽️ **Watch the full walkthrough:**  
[![Demo Video](https://img.youtube.com/vi/YOUR_VIDEO_ID/0.jpg)](https://www.youtube.com/watch?v=YOUR_VIDEO_ID)  
*Click the image above to watch the demo on YouTube.*

---

## 🧱 Tech Stack

| Layer              | Technology                         |
|-------------------|-------------------------------------|
| 🧠 LLM             | `DeepSeek` via `Groq API`           |
| 🔡 Embeddings      | `DeepSeek Embedding` via `Ollama`   |
| 🔍 Retriever       | `FAISS` Vector Store                |
| 🔗 Framework       | `LangChain`                         |
| 🖼️ Frontend        | `Streamlit`                         |
| 📦 Deployment      | `Localhost`                         |
| 📚 Input Type      | `.pdf` Legal Documents              |

---

## 🛠️ Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/legal-rag-chatbot.git
   cd legal-rag-chatbot
