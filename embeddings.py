from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
import os

# Define input files
languages = ["english.txt", "hindi.txt", "urdu.txt", "spanish.txt"]
data_dir = "data"
docs = []

# Load multilingual text data with UTF-8 encoding
try:
    for lang_file in languages:
        file_path = os.path.join(data_dir, lang_file)
        if not os.path.exists(file_path):
            print(f"⚠️ Warning: File {file_path} not found. Skipping.")
            continue
        loader = TextLoader(file_path, encoding="utf-8")
        loaded_docs = loader.load()
        # Add metadata to track source language
        for doc in loaded_docs:
            doc.metadata["language"] = lang_file.split(".")[0]
        docs.extend(loaded_docs)
except Exception as e:
    print(f"❌ Error loading documents: {e}")
    exit(1)

if not docs:
    print("❌ Error: No documents loaded. Exiting.")
    exit(1)

# Split text into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(docs)

# ✅ Load a multilingual embedding model
try:
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
except Exception as e:
    print(f"❌ Error loading embedding model: {e}")
    exit(1)

# ✅ Create FAISS vector store using LangChain
try:
    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embedding_model
    )
except Exception as e:
    print(f"❌ Error creating FAISS index: {e}")
    exit(1)

# ✅ Ensure the faiss_index directory exists
os.makedirs("faiss_index", exist_ok=True)

# ✅ Save FAISS index to disk
try:
    vector_store.save_local("faiss_index")
    print("✅ FAISS Vector Database Created and Saved Successfully!")
except Exception as e:
    print(f"❌ Error saving FAISS index: {e}")
    exit(1)
