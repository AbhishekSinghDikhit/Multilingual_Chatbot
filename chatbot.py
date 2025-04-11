import os
from google import generativeai as genai
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langdetect import detect_langs, DetectorFactory
import functools

# Ensure consistent language detection results
DetectorFactory.seed = 0

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY not found in environment variables")

# Cache embeddings and vectorstore for performance
@functools.lru_cache(maxsize=1)
def get_embeddings():
    from langchain.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

@functools.lru_cache(maxsize=1)
def get_vectorstore():
    faiss_path = "faiss_index"
    if not os.path.exists(f"{faiss_path}/index.faiss"):
        raise FileNotFoundError(f"❌ FAISS index not found! Please run `embeddings.py` to generate it.")
    
    try:
        return FAISS.load_local(
            faiss_path,
            embeddings=get_embeddings(),
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        raise RuntimeError(f"❌ Failed to load FAISS index: {e}")

def get_gemini_model():
    genai.configure(api_key=GEMINI_API_KEY)
    try:
        # Updated to a valid model name (verify with Gemini API docs)
        return genai.GenerativeModel("gemini-2.0-flash")
    except Exception as e:
        raise RuntimeError(f"❌ Failed to initialize Gemini model: {e}")

def detect_languages(text):
    """Detects languages in a given text with confidence scores."""
    try:
        detected = detect_langs(text)
        # Filter languages with probability > 0.2 to avoid noise
        return [(lang.lang, lang.prob) for lang in detected if lang.prob > 0.2]
    except Exception as e:
        print(f"⚠️ Language detection error: {e}")
        return [("en", 1.0)]  # Default to English if detection fails

def select_response_language(detected_languages):
    """Selects a familiar language for the response, prioritizing English as a fallback."""
    if not detected_languages:
        return "en"
    
    # Supported languages (ISO 639-1 codes)
    supported_langs = {"en": "English", "hi": "Hindi", "ur": "Urdu", "es": "Spanish"}
    
    # Sort by probability and pick the highest-confidence supported language
    for lang, prob in sorted(detected_languages, key=lambda x: x[1], reverse=True):
        if lang in supported_langs:
            return lang
    
    # Fallback to English if no supported language is detected
    return "en"

def chatbot_response(user_input):
    """Generates a clear, user-friendly response in a familiar language."""
    try:
        # Detect languages in the input
        detected_languages = detect_languages(user_input)
        print(f"📝 Detected Languages: {detected_languages}")
        
        # Select the response language (prioritizes supported languages, falls back to English)
        response_lang = select_response_language(detected_languages)
        
        # Retrieve context from FAISS
        vectorstore = get_vectorstore()
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        relevant_docs = retriever.get_relevant_documents(user_input)
        context = "\n".join([doc.page_content for doc in relevant_docs]) if relevant_docs else "No relevant information found."

        # Initialize Gemini model
        model = get_gemini_model()
        
        # Language-specific instructions
        lang_instructions = {
            "en": "Respond in clear, concise English.",
            "hi": "जवाब स्पष्ट और संक्षिप्त हिंदी में दें।",
            "ur": "جواب واضح اور مختصر اردو میں دیں۔",
            "es": "Responde en español claro y conciso."
        }
        
        # Construct a focused prompt
        prompt = (
            "You are a helpful multilingual chatbot designed to provide clear and concise answers. "
            "The user may mix multiple languages in their question. Follow these rules:\n"
            f"1. Respond only in {lang_instructions.get(response_lang, 'clear, concise English')}.\n"
            "2. Use the provided context to answer accurately. If the context is insufficient, provide a general answer.\n"
            "3. Keep the response simple and avoid mixing languages to ensure clarity.\n"
            "4. If the question is unclear, politely ask for clarification in the chosen language.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{user_input}"
        )
        
        # Generate response
        response = model.generate_content(prompt)
        response_text = response.text.strip() if response.text else "Sorry, I couldn't generate a response."
        
        # If the response language is not English, optionally provide an English translation
        if response_lang != "en" and response_text:
            translation_prompt = (
                f"Translate the following text to English:\n\n{response_text}"
            )
            translation = model.generate_content(translation_prompt)
            if translation.text:
                response_text += f"\n\n(English translation: {translation.text.strip()})"
        
        return response_text

    except Exception as e:
        error_msg = f"❌ Error processing request: {str(e)}"
        print(error_msg)
        return "Sorry, something went wrong. Please try again."