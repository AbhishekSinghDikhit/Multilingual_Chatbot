import os
from google.generativeai import configure, GenerativeModel
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# ✅ Load environment variables (API keys)
load_dotenv()

# ✅ Initialize Gemini API
configure(api_key=os.getenv("GEMINI_API_KEY"))
model = GenerativeModel("gemini-2.0-flash")

# ✅ Ensure FAISS index exists before loading
faiss_path = "faiss_index"
if not os.path.exists(f"{faiss_path}/index.faiss"):
    raise FileNotFoundError(f"❌ FAISS index not found! Please run `embeddings.py` to generate it.")

# ✅ Load FAISS vector store with safe deserialization
vectorstore = FAISS.load_local(
    faiss_path,
    embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
    allow_dangerous_deserialization=True
)

# ✅ Use FAISS as a retriever
retriever = vectorstore.as_retriever()

def chatbot_response(user_input):
    """Retrieves relevant knowledge and generates a response using Gemini."""

    # ✅ Retrieve context from FAISS
    relevant_docs = retriever.get_relevant_documents(user_input)
    context = "\n".join([doc.page_content for doc in relevant_docs])

    # ✅ Generate response with Gemini API
    prompt = f"Use the following information to answer the question:\n{context}\n\nQuestion: {user_input}"
    response = model.generate_content(prompt)
    
    return response.text
