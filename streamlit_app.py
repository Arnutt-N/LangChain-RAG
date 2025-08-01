# streamlit_app.py
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
    page_title="Gen AI : RAG Chatbot with Documents (Demo)",
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
        PyPDFLoader, CSVLoader, TextLoader, UnstructuredExcelLoader, UnstructuredWordDocumentLoader
    )
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain.memory import ConversationBufferMemory
    from langchain.schema import Document
    from langchain.chains import ConversationalRetrievalChain
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
        "title": "🤖 Gen AI : RAG Chatbot with Documents (Demo)",
        "upload_button": "Upload Additional Documents",
        "ask_placeholder": "Ask a question in Thai or English...",
        "processing": "Processing documents...",
        "welcome": "Hello! I'm ready to chat about various topics based on the documents.",
        "upload_success": lambda count: f"✅ {count} document(s) uploaded successfully!",
        "thinking": "🧠 Generating response...",
        "language": "Language / ภาษา",
        "clear_chat": "🗑️ Clear Chat",
        "clear_cache": "🗑️ Clear Cache",
        "reload_local": "🔄 Reload Local Files",
        "model_info": '<span class="emoji">🤖</span><span class="bold-text">Model:</span> Gemini Pro / Mistral Large | <span class="emoji">📊</span><span class="bold-text">Embedding:</span> MiniLM-L6-v2 | <span class="emoji">🗃️</span><span class="bold-text">Vector DB:</span> FAISS',
        "no_documents": "📄 No documents found. Please check the repository or upload files.",
        "error_processing": "❌ Error processing documents. Please try again.",
        "error_response": "🚨 Sorry, I encountered an error while generating response.",
        "checking_cache": "Checking cache...",
        "found_cached": "Found cached vectors",
        "saving_cache": "Saving to cache...",
        "local_files": "📁 Local Repository Files",
        "uploaded_files": "📤 Uploaded Files",
        "stats": "Statistics",
        "advanced_features": "Advanced Features",
        "auto_loaded": "Auto-loaded from repository",
        "processing_local": "Processing repository files...",
        "found_local_files": lambda count: f"Found {count} local files in repository",
        "system_ready": "System initialized and ready!",
        "loading_complete": "Loading complete",
    },
    "th": {
        "title": "🤖 Gen AI : RAG แชทกับเอกสาร (ตัวอย่าง)",
        "upload_button": "อัปโหลดเอกสารเพิ่มเติม",
        "ask_placeholder": "ถามคำถามเป็นภาษาไทยหรืออังกฤษ...",
        "processing": "กำลังประมวลผลเอกสาร...",
        "welcome": "สวัสดี! ฉันพร้อมพูดคุยเกี่ยวกับเอกสารต่างๆ",
        "upload_success": lambda count: f"✅ อัปโหลดเอกสาร {count} ฉบับสำเร็จ!",
        "thinking": "🧠 กำลังสร้างคำตอบ...",
        "language": "ภาษา / Language",
        "clear_chat": "🗑️ ล้างการแชท",
        "clear_cache": "🗑️ ล้าง Cache",
        "reload_local": "🔄 โหลดไฟล์ local ใหม่",
        "model_info": '<span class="emoji">🤖</span><span class="bold-text">โมเดล:</span> Gemini Pro / Mistral Large | <span class="emoji">📊</span><span class="bold-text">Embedding:</span> MiniLM-L6-v2 | <span class="emoji">🗃️</span><span class="bold-text">Vector DB:</span> FAISS',
        "no_documents": "📄 ไม่พบเอกสาร กรุณาตรวจสอบ repository หรืออัปโหลดไฟล์",
        "error_processing": "❌ เกิดข้อผิดพลาดในการประมวลผลเอกสาร",
        "error_response": "🚨 ขออภัย เกิดข้อผิดพลาดในการสร้างคำตอบ",
        "checking_cache": "ตรวจสอบ cache...",
        "found_cached": "พบ vectors ใน cache",
        "saving_cache": "บันทึกลง cache...",
        "local_files": "📁 ไฟล์ใน Repository",
        "uploaded_files": "📤 ไฟล์ที่อัปโหลด",
        "stats": "สถิติ",
        "advanced_features": "ฟีเจอร์ขั้นสูง",
        "auto_loaded": "โหลดอัตโนมัติจาก repository",
        "processing_local": "กำลังประมวลผลไฟล์ใน repository...",
        "found_local_files": lambda count: f"พบไฟล์ local {count} ไฟล์ใน repository",
        "system_ready": "ระบบเริ่มต้นและพร้อมใช้งาน!",
        "loading_complete": "โหลดเสร็จสิ้น",
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
        "auto_load_attempted": False,
        "show_loading_messages": True,
        "search_history": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# --- Core Application Logic ---

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

def load_gitignore_patterns():
    """Load .gitignore patterns for exclusion"""
    patterns = []
    gitignore_path = Path(".gitignore")
    if gitignore_path.exists():
        with open(gitignore_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    patterns.append(line)
    return patterns

def should_ignore_file(filepath: Path, ignore_patterns: List[str]) -> bool:
    """Check if a file should be ignored based on .gitignore and custom rules"""
    filename = filepath.name
    if filename == "requirements.txt":
        return True
    if filename.startswith('.'):
        return True
    for pattern in ignore_patterns:
        if fnmatch.fnmatch(str(filepath), pattern) or fnmatch.fnmatch(filename, pattern):
            return True
    return False

def scan_local_files() -> List[str]:
    """Scan for local files to process"""
    ignore_patterns = load_gitignore_patterns()
    supported_extensions = {".pdf", ".txt", ".csv", ".xlsx", ".xls", ".docx"}
    local_files = []
    for root, _, files in os.walk("."):
        for file in files:
            filepath = Path(root) / file
            if filepath.suffix.lower() in supported_extensions and not should_ignore_file(filepath, ignore_patterns):
                local_files.append(str(filepath.resolve()))
    return sorted(local_files)

def get_embeddings():
    """Initialize HuggingFace embeddings"""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def load_single_local_file(filepath: str) -> List[Document]:
    """Load a single local file"""
    try:
        if filepath.endswith(".pdf"):
            loader = PyPDFLoader(filepath)
        elif filepath.endswith(".txt"):
            loader = TextLoader(filepath, encoding="utf-8")
        elif filepath.endswith(".csv"):
            loader = CSVLoader(filepath)
        elif filepath.endswith((".xlsx", ".xls")):
            loader = UnstructuredExcelLoader(filepath)
        elif filepath.endswith(".docx"):
            loader = UnstructuredWordDocumentLoader(filepath)
        else:
            return []
        return loader.load()
    except Exception as e:
        st.warning(f"⚠️ Error loading {filepath}: {e}")
        return []

def process_all_documents(local_files: List[str], uploaded_files: List[Any], show_progress=True) -> bool:
    """Process all documents and create FAISS vectorstore"""
    try:
        all_documents = []
        total_files_to_process = len(local_files) + len(uploaded_files)
        if total_files_to_process == 0:
            return False

        if show_progress:
            progress_bar = st.progress(0)
            status_text = st.empty()
        else:
            progress_bar, status_text = None, None

        processed = 0
        for filepath in local_files:
            if show_progress and status_text:
                status_text.text(f"{translations[st.session_state.language]['processing_local']}: {os.path.basename(filepath)}...")
            if progress_bar:
                progress_bar.progress(processed / total_files_to_process * 0.5)
            
            docs = load_single_local_file(filepath)
            all_documents.extend(docs)
            processed += 1

        for uploaded_file in uploaded_files:
            if show_progress and status_text:
                status_text.text(f"Loading uploaded: {uploaded_file.name}...")
            if progress_bar:
                progress_bar.progress(processed / total_files_to_process * 0.5)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            try:
                docs = load_single_local_file(tmp_file_path)
                all_documents.extend(docs)
            finally:
                os.unlink(tmp_file_path)
            
            processed += 1

        if not all_documents:
            return False

        # Split documents
        if show_progress and status_text:
            status_text.text("Splitting documents...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_documents = text_splitter.split_documents(all_documents)

        if not split_documents:
            return False

        # Create embeddings and FAISS vectorstore
        if show_progress and status_text:
            status_text.text("Creating embeddings...")
        embeddings = get_embeddings()
        
        if show_progress and status_text:
            status_text.text("Building FAISS index...")
        vectorstore = FAISS.from_documents(split_documents, embeddings)
        
        # Save to cache
        if show_progress and status_text:
            status_text.text(translations[st.session_state.language]["saving_cache"])
        
        # Generate cache key based on file contents
        cache_string = "".join([f"{f}:{os.path.getmtime(f)}" for f in local_files])
        cache_string += "".join([f"{f.name}:{f.size}" for f in uploaded_files])
        cache_key = hashlib.md5(cache_string.encode()).hexdigest()[:16]
        cache_path = CACHE_DIR / f"vectors_{cache_key}"
        cache_path.mkdir(exist_ok=True)
        
        faiss.write_index(vectorstore.index, str(cache_path / "index.faiss"))
        vectorstore.save_local(str(cache_path), "index")
        
        # Save metadata
        metadata = {
            "timestamp": time.time(),
            "local_files": local_files,
            "uploaded_files": [f.name for f in uploaded_files],
            "num_chunks": len(split_documents)
        }
        with open(cache_path / "metadata.json", "w") as f:
            json.dump(metadata, f)

        # Update session state
        st.session_state.vectorstore = vectorstore
        st.session_state.document_chunks = len(split_documents)
        st.session_state.documents_processed = True
        
        if progress_bar:
            progress_bar.progress(1.0)
        if status_text:
            status_text.text(translations[st.session_state.language]["loading_complete"])
            time.sleep(0.5)
            status_text.empty()
        if progress_bar:
            progress_bar.empty()
            
        return True
    except Exception as e:
        if show_progress:
            st.error(f"Error processing documents: {e}")
        return False

def auto_load_local_files():
    """Automatically load local files on startup with better UX"""
    if st.session_state.auto_load_attempted:
        return
        
    st.session_state.auto_load_attempted = True
    app_state = get_app_state()

    try:
        local_files = scan_local_files()
        st.session_state.local_files = local_files

        if not local_files and not st.session_state.uploaded_files:
             st.info(translations[st.session_state.language]["no_documents"])
             return

        # Check cache first
        cache_string = "".join([f"{f}:{os.path.getmtime(f)}" for f in local_files])
        cache_key = hashlib.md5(cache_string.encode()).hexdigest()[:16]
        cache_path = CACHE_DIR / f"vectors_{cache_key}"

        if cache_path.exists():
            try:
                embeddings = get_embeddings()
                vectorstore = FAISS.load_local(str(cache_path), embeddings, "index", allow_dangerous_deserialization=True)
                with open(cache_path / "metadata.json", "r") as f:
                    metadata = json.load(f)
                
                st.session_state.vectorstore = vectorstore
                st.session_state.document_chunks = metadata.get("num_chunks", 0)
                st.session_state.documents_processed = True
                st.session_state.initialization_complete = True
                return
            except Exception as e:
                st.warning(f"Cache load failed, reprocessing: {e}")
                import shutil
                shutil.rmtree(cache_path, ignore_errors=True)

        # Process documents if not cached
        success = process_all_documents(local_files, [], show_progress=st.session_state.show_loading_messages)
        if success:
            app_state["initialized"] = True
            app_state["last_load"] = time.time()
            save_app_state(app_state)
            st.session_state.show_loading_messages = False
            st.session_state.initialization_complete = True
        else:
            st.error("❌ Failed to process local files. Try reloading manually.")
    except Exception as e:
        st.error(f"Error during auto-load: {e}")
        st.session_state.show_loading_messages = False

def setup_advanced_retrieval_chain():
    """Setup the LangChain retrieval chain with selected model"""
    if not st.session_state.vectorstore:
        return None

    try:
        # Configure retriever
        retriever = st.session_state.vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 5, "score_threshold": st.session_state.similarity_threshold}
        )

        # Configure LLM
        if st.session_state.selected_model == "mistral" and MISTRAL_API_KEY:
            from langchain_mistralai.chat_models import ChatMistralAI
            llm = ChatMistralAI(
                model="mistral-large-latest",
                temperature=st.session_state.temperature,
                max_tokens=st.session_state.max_tokens,
                api_key=MISTRAL_API_KEY
            )
        else: # Default to Gemini
            llm = ChatGoogleGenerativeAI(
                model="gemini-pro",
                temperature=st.session_state.temperature,
                max_output_tokens=st.session_state.max_tokens,
                convert_system_message_to_human=True
            )

        # Create chain
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='answer')
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            verbose=False
        )
        return chain
    except Exception as e:
        st.error(f"Error setting up retrieval chain: {e}")
        return None

def query_with_analytics(chain, question: str) -> Dict[str, Any]:
    """Query with analytics and error handling"""
    try:
        time.sleep(1) # Basic delay to help with rate limits

        st.session_state.search_history.append({"question": question, "timestamp": time.time()})
        if len(st.session_state.search_history) > 10:
            st.session_state.search_history = st.session_state.search_history[-10:]

        # Improved prompt
        prompt = f"""Answer the question in the same language as the question.
If the context does not contain enough information, use your general knowledge to provide a helpful and polite response.
Avoid saying "I don't know" unless absolutely necessary.

Question: {question}
Answer:"""
        
        result = chain.invoke({"question": prompt})
        return {
            "answer": result.get('answer', 'No answer generated.'),
            "source_documents": result.get('source_documents', [])
        }
    except Exception as e:
        error_msg = str(e).lower()
        if "quota" in error_msg or "rate" in error_msg or "429" in error_msg:
            raise Exception("RATE_LIMIT")
        elif "timeout" in error_msg:
            raise TimeoutError("Request timed out")
        else:
            raise e

# --- Main Application Function ---

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
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                # Handle greetings and small talk
                if is_greeting(prompt):
                    response = t["welcome"]
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                elif is_small_talk(prompt):
                    response = "I'm doing great! How can I help you with your documents?" if st.session_state.language == "en" else "ฉันสบายดีค่ะ มีอะไรให้ช่วยไหมคะ?"
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
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
                                st.session_state.messages.append({"role": "assistant", "content": t["error_response"]})
                    else:
                        st.error("Could not set up retrieval system.")
                        st.session_state.messages.append({"role": "assistant", "content": t["error_response"]})
    else:
        st.info("Waiting for documents to be processed...")

if __name__ == "__main__":
    main()
