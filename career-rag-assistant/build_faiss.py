from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

# Load your resume (PDF)
loader = PyPDFLoader("resume.pdf")
docs = loader.load()

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = text_splitter.split_documents(docs)

# Use HuggingFace embeddings (no OpenAI key needed)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Build FAISS DB
vector_db = FAISS.from_documents(split_docs, embeddings)

# Save FAISS DB locally
vector_db.save_local("career_memory")

print("✅ FAISS DB created and saved in 'career_memory/'")
