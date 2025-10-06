
# 🧠 Career RAG Assistant

An AI-powered **Retrieval-Augmented Generation (RAG)** app that lets you interact with your **resume, LinkedIn profile, or professional portfolio** through natural language.  
It understands your projects, skills, and achievements — and can even generate **STAR interview stories**, project summaries, or skill-based responses — all using **local embeddings (FAISS)** with **no OpenAI API cost**.

---

## 🚀 Overview

The **Career RAG Assistant** is built to help professionals and job seekers quickly generate personalized answers for interviews, performance reviews, and applications.  
By combining **LangChain**, **FAISS**, and **Hugging Face Embeddings**, it retrieves relevant sections of your career documents and uses **Flan-T5** to generate human-like answers.

---

## 🧩 Features

✅ Chat with your resume and other files  
✅ Generate STAR (Situation-Task-Action-Result) stories  
✅ Summarize top projects or tools  
✅ No API key required — runs fully on CPU  
✅ Powered by open-source LLMs (Flan-T5 & Sentence-Transformers)  
✅ Interactive Streamlit UI

---

## 🏗️ Architecture

1. **Document Ingestion** — Loads PDFs (resume, LinkedIn exports, reports).  
2. **Text Splitting** — Breaks long text into semantically coherent chunks.  
3. **Embeddings** — Encodes text using `sentence-transformers/all-MiniLM-L6-v2`.  
4. **Vector Store (FAISS)** — Stores embeddings locally for fast retrieval.  
5. **Retrieval + Generation** — `LangChain RetrievalQA` with `Flan-T5`.  
6. **UI Layer** — Built in **Streamlit** for an elegant chat-style experience.

---
career-rag-assistant/
│
├── career_memory/ # FAISS vector store (auto-created)
├── build_faiss.py # Script to build FAISS DB from documents
├── career_rag_app.py # Streamlit app
├── requirements.txt # Project dependencies
├── resume.pdf # Example input resume
├── README.md # Project documentation
└── .gitignore
## 📁 Folder Structure


---

## 💻 How to Run Locally

```bash
# 1️⃣ Clone the repo
git clone https://github.com/<your-username>/career-rag-assistant.git
cd career-rag-assistant

# 2️⃣ Install dependencies
pip install -r requirements.txt

# 3️⃣ Build the FAISS vector database
python build_faiss.py

# 4️⃣ Run the Streamlit app
streamlit run career_rag_app.py


