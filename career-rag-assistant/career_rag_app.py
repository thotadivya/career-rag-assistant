import  streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS

from langchain_openai import OpenAIEmbeddings   # If you used OpenAI embeddings

# ===============================
# Load FAISS Vector DB
# ===============================
# IMPORTANT: Replace with the same embeddings you used when saving the DB
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_db = FAISS.load_local(
    "career_memory",
    embeddings,
    allow_dangerous_deserialization=True
)

# ===============================
# Load a lightweight model (Flan-T5 for CPU, fast)
# ===============================
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-large",
    max_new_tokens=512,
    temperature=0.7
)

llm = HuggingFacePipeline(pipeline=generator)

# ===============================
# Create Retrieval-QA chain
# ===============================
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vector_db.as_retriever(),
    chain_type="stuff"
)

# ===============================
# Streamlit UI
# ===============================
st.set_page_config(page_title="Career career-rag-assistant Assistant", page_icon="🧑‍💻", layout="centered")

st.title("🧑‍💻 Career career-rag-assistant Assistant")
st.write("Ask me anything about your career, projects, or skills!")

query = st.text_input("Enter your question:")

if st.button("Ask"):
    if query.strip():
        with st.spinner("Thinking... 🤔"):
            try:
                answer = qa.run(query)
                st.subheader("Answer")
                st.write(answer)
            except Exception as e:
                st.error(f"⚠️ Error: {e}")
    else:
        st.warning("Please enter a question first.")
