import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
import time
import hashlib
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
import fnmatch
import random

# Load environment variables
load_dotenv()

# Set page configuration FIRST
st.set_page_config(
    layout="wide", 
    page_title="Advanced RAG Chatbot",
    page_icon="🤖",
    initial_sidebar_state="expanded"
)

# Greeting detection
GREETINGS = {
    "hello", "hi", "hey", "good morning", "good afternoon", "good evening",
    "สวัสดี", "สวัสดีครับ", "สวัสดีค่ะ", "หวัดดี", "hello!", "hi!", "hey!"
}

SMALL_TALK = {
    "how are you", "how are you doing", "what's up", "how's it going",
    "เป็นอย่างไรบ้าง", "สบายดีไหม", "ทำอะไรอยู่", "เป็นไงบ้าง"
}

# Import dependencies with proper error handling
@st.cache_data
def check_dependencies():
    """Check if all required packages are available"""
    missing_packages = []
    try:
        import google.generativeai as genai
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_community.document_loaders import PyPDFLoader, CSVLoader, UnstructuredExcelLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain.memory import ConversationBufferMemory
        from langchain_community.vectorstores import FAISS
        try:
            from mistralai import Mistral
        except ImportError:
            pass
    except ImportError as e:
        missing_package = str(e).split("'")[1] if "'" in str(e) else str(e)
        missing_packages.append(missing_package)
    return missing_packages

missing_deps = check_dependencies()
if missing_deps:
    st.error(f"Missing packages: {', '.join(missing_deps)}")
    st.info("""
    Install with:
    ```bash
    pip install -r requirements.txt
    ```
    """)
    st.stop()

# Import all required packages
try:
    import google.generativeai as genai
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.document_loaders import (
        PyPDFLoader, CSVLoader, TextLoader, UnstructuredExcelLoader
    )
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain.memory import ConversationBufferMemory
    from langchain.schema import Document
    import faiss
    from langchain_community.vectorstores import FAISS
    FAISS_AVAILABLE = True
    
    try:
        from mistralai import Mistral
        MISTRAL_AVAILABLE = True
    except ImportError:
        MISTRAL_AVAILABLE = False
        
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# Configuration - API Keys
GOOGLE_API_KEY = (
    st.secrets.get("GOOGLE_API_KEY") or 
    st.secrets.get("GEMINI_API_KEY") or 
    os.getenv("GOOGLE_API_KEY") or 
    os.getenv("GEMINI_API_KEY")
)

MISTRAL_API_KEY = (
    st.secrets.get("MISTRAL_API_KEY") or 
    os.getenv("MISTRAL_API_KEY")
)

if not GOOGLE_API_KEY and not MISTRAL_API_KEY:
    st.error("🚨 No API Keys found!")
    st.info("""
    **Setup API Keys:**
    **Gemini:** https://makersuite.google.com/app/apikey
    **Mistral:** https://console.mistral.ai/
    """)
    st.stop()

# Configure APIs
if GOOGLE_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    genai.configure(api_key=GOOGLE_API_KEY)

if MISTRAL_API_KEY and MISTRAL_AVAILABLE:
    mistral_client = Mistral(api_key=MISTRAL_API_KEY)

# Setup persistent storage
CACHE_DIR = Path("./vector_cache")
CACHE_DIR.mkdir(exist_ok=True)
APP_STATE_FILE = Path("./app_state.json")

def get_app_state():
    try:
        if APP_STATE_FILE.exists():
            with open(APP_STATE_FILE, 'r') as f:
                return json.load(f)
    except: pass
    return {"initialized": False, "last_load": 0}

def save_app_state(state):
    try:
        with open(APP_STATE_FILE, 'w') as f:
            json.dump(state, f)
    except: pass

# Translations
translations = {
    "en": {
        "title": "🤖 Advanced RAG Chatbot",
        "welcome": "Hello! I'm ready to chat about your documents.",
        "ask_placeholder": "Ask a question in Thai or English...",
        "thinking": "🧠 Generating response...",
        "error_response": "Sorry, I couldn't generate a response."
    },
    "th": {
        "title": "🤖 แชทบอท RAG ขั้นสูง",
        "welcome": "สวัสดี! ฉันพร้อมพูดคุยเกี่ยวกับเอกสารของคุณ",
        "ask_placeholder": "ถามคำถามเป็นภาษาไทยหรืออังกฤษ...",
        "thinking": "🧠 กำลังสร้างคำตอบ...",
        "error_response": "ขออภัย ฉันไม่สามารถสร้างคำตอบได้"
    }
}

def init_session_state():
    defaults = {
        "messages": [],
        "vectorstore": None,
        "language": "en",
        "uploaded_files": [],
        "local_files": [],
        "documents_processed": False,
        "document_chunks": 0,
        "debug_mode": False,
        "selected_model": "gemini",
        "similarity_threshold": 0.7,
        "max_tokens": 512,
        "temperature": 0.1,
        "initialization_complete": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

def safe_query(chain, question, max_retries=3):
    """Enhanced with exponential backoff"""
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                time.sleep((2 ** attempt) + random.uniform(0, 1))
            return query_with_analytics(chain, question)
        except Exception as e:
            if "RATE_LIMIT" in str(e) and attempt < max_retries - 1:
                continue
            raise e
    return {"answer": "Too many retries. Please try again later."}

def is_greeting(text: str) -> bool:
    return text.strip().lower() in GREETINGS

def is_small_talk(text: str) -> bool:
    text_lower = text.strip().lower()
    return any(phrase in text_lower for phrase in SMALL_TALK)

# [Include all your existing functions: scan_local_files, get_embeddings, 
# process_all_documents, setup_advanced_retrieval_chain, query_with_analytics, etc.]

def main():
    t = translations[st.session_state.language]
    
    st.markdown(f'<h1 style="text-align: center;">{t["title"]}</h1>', unsafe_allow_html=True)
    
    # Auto-load local files
    auto_load_local_files()
    
    if st.session_state.documents_processed:
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input(t["ask_placeholder"]):
            # Handle greetings and small talk
            if is_greeting(prompt):
                response = t["welcome"]
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.markdown(response)
            elif is_small_talk(prompt):
                response = "I'm doing great! How can I help you with your documents?" if st.session_state.language == "en" else "ฉันสบายดีค่ะ มีอะไรให้ช่วยไหมคะ?"
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.markdown(response)
            else:
                # Process with RAG
                retrieval_chain = setup_advanced_retrieval_chain()
                if retrieval_chain:
                    with st.spinner(t["thinking"]):
                        try:
                            response = safe_query(retrieval_chain, prompt)
                            answer = response.get('answer', t["error_response"])
                            st.markdown(answer)
                            st.session_state.messages.append({"role": "assistant", "content": answer})
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                else:
                    st.error("Could not set up retrieval system.")
    else:
        st.info("Waiting for documents to be processed...")

if __name__ == "__main__":
    main()
