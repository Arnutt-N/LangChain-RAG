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

# Load environment variables
load_dotenv()

# Set page configuration FIRST
st.set_page_config(
    layout="wide", 
    page_title="Gen AI : RAG Chatbot with Documents (Demo)",
    page_icon="🤖",
    initial_sidebar_state="expanded"
)

# Import dependencies with proper error handling
@st.cache_data
def check_dependencies():
    """Check if all required packages are available with correct versions"""
    missing_packages = []
    try:
        # Updated imports for latest versions
        import google.generativeai as genai
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_community.document_loaders import PyPDFLoader, CSVLoader, UnstructuredExcelLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain.memory import ConversationBufferMemory
        from langchain.chains import ConversationalRetrievalChain
        # Optional imports
        try:
            from mistralai import Mistral
        except ImportError:
            pass  # Mistral is optional
        # FAISS is required (lighter alternative to ChromaDB)
        import faiss
        from langchain_community.vectorstores import FAISS
    except ImportError as e:
        missing_package = str(e).split("'")[1] if "'" in str(e) else str(e)
        missing_packages.append(missing_package)
    
    return missing_packages

# Check dependencies early
missing_deps = check_dependencies()
if missing_deps:
    st.error(f"Missing packages: {', '.join(missing_deps)}")
    st.info("""
    Please install with:
    ```bash
    pip install streamlit langchain-google-genai langchain-huggingface langchain-community 
    pip install faiss-cpu  # For CPU version, or faiss-gpu for GPU
    pip install google-generativeai mistralai python-dotenv
    pip install pypdf unstructured[xlsx] openpyxl
    ```
    """)
    st.stop()

# Import all required packages
try:
    import google.generativeai as genai
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.document_loaders import (
        PyPDFLoader, 
        CSVLoader, 
        TextLoader,
        UnstructuredExcelLoader
    )
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain.memory import ConversationBufferMemory
    from langchain.chains import ConversationalRetrievalChain
    from langchain.schema import Document
    import pandas as pd
    import numpy as np
    
    # Optional Mistral import
    try:
        from mistralai import Mistral
        MISTRAL_AVAILABLE = True
    except ImportError:
        MISTRAL_AVAILABLE = False
    
    # FAISS import - required (replacing ChromaDB)
    try:
        import faiss
        from langchain_community.vectorstores import FAISS
        FAISS_AVAILABLE = True
    except ImportError:
        FAISS_AVAILABLE = False
        st.error("❌ FAISS is required but not installed!")
        st.info("Please install FAISS: `pip install faiss-cpu`")
        st.stop()
    
except ImportError as e:
    st.error(f"Import error: {e}")
    st.info("Please check your package versions and requirements.txt")
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

# Check if at least one API key is available
if not GOOGLE_API_KEY and not MISTRAL_API_KEY:
    st.error("🚨 No API Keys found!")
    st.info("""
    **Setup API Keys (choose one or both):**
    
    **Option 1: Google Gemini API**
    1. Go to https://makersuite.google.com/app/apikey
    2. Create API key
    3. Add to Streamlit secrets as GOOGLE_API_KEY
    
    **Option 2: Mistral AI API**
    1. Go to https://console.mistral.ai/
    2. Create API key  
    3. Add to Streamlit secrets as MISTRAL_API_KEY
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

# Global app state file to track initialization across users
APP_STATE_FILE = Path("./app_state.json")

def get_app_state():
    """Get global app state"""
    try:
        if APP_STATE_FILE.exists():
            with open(APP_STATE_FILE, 'r') as f:
                return json.load(f)
    except:
        pass
    return {"initialized": False, "last_load": 0}

def save_app_state(state):
    """Save global app state"""
    try:
        with open(APP_STATE_FILE, 'w') as f:
            json.dump(state, f)
    except:
        pass

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
        "model_info": '<span class="emoji">🤖</span><span class="bold-text">Model:</span> Gemini Flash / Mistral Large | <span class="emoji">📊</span><span class="bold-text">Embedding:</span> MiniLM-L6-v2 | <span class="emoji">🗃️</span><span class="bold-text">Vector DB:</span> FAISS',
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
        "model_info": '<span class="emoji">🤖</span><span class="bold-text">โมเดล:</span> Gemini Flash / Mistral Large | <span class="emoji">📊</span><span class="bold-text">Embedding:</span> MiniLM-L6-v2 | <span class="emoji">🗃️</span><span class="bold-text">Vector DB:</span> FAISS',
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

# Initialize session state
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
        "last_request_time": 0,
        "app_initialized": False,
        "file_analysis": {},
        "search_history": [],
        "similarity_threshold": 0.7,
        "max_tokens": 512,
        "temperature": 0.1,
        "auto_load_attempted": False,
        "show_loading_messages": True,
        "initialization_complete": False,
        "selected_model": "gemini", # Default model
        "selected_vector_db": "faiss", # Use FAISS only
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# Function to scan for local files
def load_gitignore_patterns():
    """Load .gitignore patterns to exclude files"""
    patterns = []
    try:
        if os.path.exists('.gitignore'):
            with open('.gitignore', 'r', encoding='utf-8') as f:
                patterns = [line.strip() for line in f.readlines() 
                           if line.strip() and not line.startswith('#')]
    except:
        pass
    
    # Default patterns to ignore
    default_patterns = [
        '.git/*', '*.pyc', '__pycache__/*', '.env', '*.log',
        'node_modules/*', '.vscode/*', '.idea/*', '*.tmp',
        'vector_cache/*', '.streamlit/*'
    ]
    
    return patterns + default_patterns

def should_ignore_file(filepath, patterns):
    """Check if file should be ignored based on patterns"""
    filename = os.path.basename(filepath)
    
    # Always ignore these (removed requirements.txt from auto-ignore)
    if filename.startswith('.'):
        return True
    
    for pattern in patterns:
        if fnmatch.fnmatch(filepath, pattern) or fnmatch.fnmatch(filename, pattern):
            return True
    
    return False

def scan_local_files():
    """Scan repository for document files"""
    supported_extensions = ('.pdf', '.txt', '.csv', '.xlsx', '.xls', '.docx')
    local_files = []
    ignore_patterns = load_gitignore_patterns()
    
    # Add requirements.txt to ignore patterns specifically
    ignore_patterns.append('requirements.txt')
    ignore_patterns.append('streamlit_app.py')
    ignore_patterns.append('*.py')
    
    try:
        current_dir = os.getcwd()
        
        # Scan current directory and subdirectories
        for root, dirs, files in os.walk('.'):
            # Skip hidden directories and common ignore directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules', 'vector_cache']]
            
            for file in files:
                if file.lower().endswith(supported_extensions):
                    filepath = os.path.join(root, file)
                    if not should_ignore_file(filepath, ignore_patterns):
                        local_files.append(filepath)
        
        # Remove duplicates and sort
        local_files = sorted(list(set(local_files)))
        
    except Exception as e:
        pass
    
    return local_files

# Optimized embeddings with latest HuggingFace integration
@st.cache_resource
def get_embeddings():
    """Initialize embeddings with latest langchain-huggingface"""
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Test embeddings
        test_embedding = embeddings.embed_query("test")
        if len(test_embedding) > 0:
            return embeddings
        else:
            return None
            
    except Exception as e:
        return None

# Enhanced caching system
def get_file_hash(filepath_or_content) -> str:
    """Generate hash from file path or content"""
    if isinstance(filepath_or_content, (str, Path)) and os.path.exists(filepath_or_content):
        # File path - read and hash
        with open(filepath_or_content, 'rb') as f:
            content = f.read()
    else:
        # Direct content
        content = filepath_or_content
        if isinstance(content, str):
            content = content.encode('utf-8')
    
    return hashlib.md5(content).hexdigest()

def get_cache_key(local_files: List[str], uploaded_files: List) -> str:
    """Generate cache key from all files"""
    file_hashes = {}
    
    # Hash local files
    for filepath in local_files:
        try:
            file_hashes[f"local_{filepath}"] = get_file_hash(filepath)
        except:
            continue
    
    # Hash uploaded files
    for uploaded_file in uploaded_files:
        try:
            file_hashes[f"upload_{uploaded_file.name}"] = get_file_hash(uploaded_file.getvalue())
        except:
            continue
    
    # FAISS identifier
    file_hashes["vector_db"] = "faiss"
    
    cache_string = json.dumps(sorted(file_hashes.items()), sort_keys=True)
    return hashlib.md5(cache_string.encode()).hexdigest()[:16]

def save_vectors_to_cache(vectorstore, cache_key: str, file_info: Dict):
    """Save FAISS vectorstore to cache"""
    try:
        cache_path = CACHE_DIR / f"vectors_{cache_key}"
        cache_path.mkdir(exist_ok=True)
        
        # Save FAISS index
        vectorstore.save_local(str(cache_path))
        
        # Save metadata
        metadata = {
            "file_info": file_info,
            "timestamp": time.time(),
            "cache_key": cache_key,
            "chunks": st.session_state.document_chunks,
            "vector_db_type": "faiss"
        }
        
        with open(cache_path / "metadata.json", "w", encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        return True
    except Exception as e:
        return False

def load_vectors_from_cache(cache_key: str):
    """Load FAISS vectorstore from cache"""
    try:
        cache_path = CACHE_DIR / f"vectors_{cache_key}"
        metadata_path = cache_path / "metadata.json"
        
        if not metadata_path.exists():
            return None, None
        
        # Load metadata
        with open(metadata_path, "r", encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Check if cache is not too old (24 hours)
        if time.time() - metadata["timestamp"] > 86400:
            return None, None
        
        # Load embeddings
        embeddings = get_embeddings()
        if not embeddings:
            return None, None
        
        # Load FAISS index
        vectorstore = FAISS.load_local(
            str(cache_path), 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        
        return vectorstore, metadata
        
    except Exception as e:
        return None, None

# Enhanced document processing with FAISS
def create_faiss_vectorstore(texts, embeddings):
    """Create FAISS vector store from texts"""
    try:
        # Create FAISS vectorstore
        vectorstore = FAISS.from_documents(texts, embeddings)
        return vectorstore
        
    except Exception as e:
        st.error(f"Error creating FAISS vector store: {e}")
        return None

def load_single_local_file(filepath: str) -> List[Document]:
    """Load a single local file"""
    try:
        file_extension = Path(filepath).suffix.lower()
        
        # Debug info
        if st.session_state.get('debug_mode', False):
            st.write(f"Loading file: {filepath} (extension: {file_extension})")
        
        # Use appropriate loader based on file type
        if file_extension == '.pdf':
            loader = PyPDFLoader(filepath)
        elif file_extension == '.csv':
            loader = CSVLoader(filepath)
        elif file_extension == '.txt':
            loader = TextLoader(filepath, encoding='utf-8')
        elif file_extension in ['.xlsx', '.xls']:
            loader = UnstructuredExcelLoader(filepath)
        elif file_extension == '.docx':
            try:
                from langchain_community.document_loaders import UnstructuredWordDocumentLoader
                loader = UnstructuredWordDocumentLoader(filepath)
            except:
                return []
        else:
            if st.session_state.get('debug_mode', False):
                st.write(f"Unsupported file type: {file_extension}")
            return []
        
        docs = loader.load()
        if st.session_state.get('debug_mode', False):
            st.write(f"Loaded {len(docs)} raw documents from {filepath}")
        
        cleaned_docs = []
        
        for i, doc in enumerate(docs):
            if hasattr(doc, 'page_content') and doc.page_content.strip():
                content = doc.page_content.replace('\n\n', '\n').strip()
                if len(content) > 20:  # Lower threshold to include more content
                    doc.page_content = content
                    # Enhanced metadata
                    doc.metadata.update({
                        'source_file': os.path.basename(filepath),
                        'file_path': filepath,
                        'file_hash': get_file_hash(filepath),
                        'file_type': 'local',
                        'chunk_id': i,
                        'content_length': len(content)
                    })
                    cleaned_docs.append(doc)
        
        if st.session_state.get('debug_mode', False):
            st.write(f"Cleaned to {len(cleaned_docs)} documents from {filepath}")
            if cleaned_docs:
                st.write(f"Sample content length: {len(cleaned_docs[0].page_content)}")
        
        return cleaned_docs
        
    except Exception as e:
        if st.session_state.get('debug_mode', False):
            st.error(f"Error loading {filepath}: {e}")
        return []

def load_single_uploaded_file(uploaded_file) -> List[Document]:
    """Load a single uploaded file"""
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name
        
        try:
            # Updated document loaders
            if file_extension == 'pdf':
                loader = PyPDFLoader(temp_file_path)
            elif file_extension == 'csv':
                loader = CSVLoader(temp_file_path)
            elif file_extension == 'txt':
                loader = TextLoader(temp_file_path, encoding='utf-8')
            elif file_extension in ['xlsx', 'xls']:
                loader = UnstructuredExcelLoader(temp_file_path)
            else:
                return []
            
            docs = loader.load()
            cleaned_docs = []
            
            for i, doc in enumerate(docs):
                if hasattr(doc, 'page_content') and doc.page_content.strip():
                    content = doc.page_content.replace('\n\n', '\n').strip()
                    if len(content) > 50:  # Only include meaningful content
                        doc.page_content = content
                        # Enhanced metadata
                        doc.metadata.update({
                            'source_file': uploaded_file.name,
                            'file_hash': get_file_hash(uploaded_file.getvalue()),
                            'file_type': 'uploaded',
                            'upload_time': time.time(),
                            'file_size': len(uploaded_file.getvalue()),
                            'chunk_id': i,
                            'content_length': len(content)
                        })
                        cleaned_docs.append(doc)
            
            return cleaned_docs
            
        finally:
            try:
                os.unlink(temp_file_path)
            except:
                pass
        
    except Exception as e:
        return []

def process_all_documents(local_files: List[str], uploaded_files: List, show_progress: bool = True) -> bool:
    """Process all documents with intelligent caching"""
    t = translations[st.session_state.language]
    
    total_files = len(local_files) + len(uploaded_files)
    if total_files == 0:
        return False
    
    # Generate cache key
    cache_key = get_cache_key(local_files, uploaded_files)
    
    # Try to load from cache
    if show_progress and st.session_state.show_loading_messages:
        cache_placeholder = st.empty()
        cache_placeholder.info(f"🔍 {t['checking_cache']}")
    
    cached_vectorstore, cached_metadata = load_vectors_from_cache(cache_key)
    
    if cached_vectorstore and cached_metadata:
        if show_progress and st.session_state.show_loading_messages:
            cache_placeholder.success(f"✅ {t['found_cached']}")
            time.sleep(0.3)
            cache_placeholder.empty()
        
        st.session_state.vectorstore = cached_vectorstore
        st.session_state.document_chunks = cached_metadata.get("chunks", 0)
        st.session_state.documents_processed = True
        return True
    
    if show_progress and st.session_state.show_loading_messages:
        cache_placeholder.empty()
    
    # Process documents if not in cache
    if show_progress and st.session_state.show_loading_messages:
        st.info(f"📝 {t['processing']}")
    
    all_documents = []
    progress_bar = st.progress(0) if show_progress and st.session_state.show_loading_messages else None
    status_text = st.empty() if show_progress and st.session_state.show_loading_messages else None
    
    try:
        total_files_to_process = len(local_files) + len(uploaded_files)
        processed = 0
        
        # Process local files
        for filepath in local_files:
            if show_progress and status_text:
                status_text.text(f"Loading local: {os.path.basename(filepath)}...")
            if progress_bar:
                progress_bar.progress(processed / total_files_to_process * 0.5)
            
            docs = load_single_local_file(filepath)
            all_documents.extend(docs)
            processed += 1
        
        # Process uploaded files
        for uploaded_file in uploaded_files:
            if show_progress and status_text:
                status_text.text(f"Loading uploaded: {uploaded_file.name}...")
            if progress_bar:
                progress_bar.progress(processed / total_files_to_process * 0.5)
            
            docs = load_single_uploaded_file(uploaded_file)
            all_documents.extend(docs)
            processed += 1
        
        if not all_documents:
            return False
        
        # Split documents with updated text splitter
        if show_progress and status_text:
            status_text.text("🔄 Splitting documents...")
        if progress_bar:
            progress_bar.progress(0.7)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )
        texts = text_splitter.split_documents(all_documents)
        
        if not texts:
            return False
        
        # Create embeddings and vectorstore
        if show_progress and status_text:
            status_text.text("📊 Creating embeddings...")
        if progress_bar:
            progress_bar.progress(0.9)
        
        embeddings = get_embeddings()
        if not embeddings:
            return False
        
        # Create FAISS vector store
        vectorstore = create_faiss_vectorstore(texts, embeddings)
        
        if not vectorstore:
            return False
        
        # Save to cache
        if show_progress and status_text:
            status_text.text(f"💾 {t['saving_cache']}")
        file_info = {
            "local_files": local_files,
            "uploaded_files": [f.name for f in uploaded_files],
            "total_documents": len(all_documents),
            "total_chunks": len(texts)
        }
        save_vectors_to_cache(vectorstore, cache_key, file_info)
        
        # Update session state
        st.session_state.vectorstore = vectorstore
        st.session_state.document_chunks = len(texts)
        st.session_state.documents_processed = True
        
        if progress_bar:
            progress_bar.progress(1.0)
            time.sleep(0.2)
            progress_bar.empty()
        if status_text:
            status_text.empty()
        
        return True
        
    except Exception as e:
        if progress_bar:
            progress_bar.empty()
        if status_text:
            status_text.empty()
        return False

# Auto-load local files on startup with improved UX
def auto_load_local_files():
    """Automatically load local files on startup with better UX"""
    app_state = get_app_state()
    
    # Always try to load if not attempted yet (don't skip based on app state)
    if st.session_state.auto_load_attempted:
        return
    
    st.session_state.auto_load_attempted = True
    
    # Scan for local files
    local_files = scan_local_files()
    st.session_state.local_files = local_files
    
    t = translations[st.session_state.language]
    
    if not local_files:
        st.info("📁 No local files found in repository. Upload documents using the sidebar.")
        return  # Exit early if no files
    
    # Always show file discovery message and process files
    st.info(t["found_local_files"](len(local_files)))
    
    # Force process files even if app was recently initialized
    try:
        success = process_all_documents(local_files, [], show_progress=True)
        
        if success:
            # Update app state to indicate successful initialization
            app_state["initialized"] = True
            app_state["last_load"] = time.time()
            save_app_state(app_state)
            
            st.session_state.initialization_complete = True
            st.success(f"✅ Successfully processed {len(local_files)} files with {st.session_state.document_chunks} chunks!")
        else:
            st.error("❌ Failed to process local files. Check file formats or try reloading manually.")
            # Debug info
            if st.session_state.debug_mode:
                st.write(f"Debug: Files found but processing failed for: {local_files}")
    except Exception as e:
        st.error(f"Error during auto-load: {e}")
        if st.session_state.debug_mode:
            st.exception(e)

# Custom Mistral LangChain wrapper with improved compatibility
class MistralLLM:
    def __init__(self, api_key: str, model: str = "mistral-large-latest", temperature: float = 0.1, max_tokens: int = 512):
        self.client = Mistral(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def invoke(self, input_data):
        """Invoke method to match LangChain interface"""
        try:
            # Handle different input formats
            if isinstance(input_data, dict):
                # Extract content from dict
                if 'input' in input_data:
                    content = input_data['input']
                elif 'question' in input_data:
                    content = input_data['question']
                else:
                    content = str(input_data)
            elif hasattr(input_data, 'content'):
                content = input_data.content
            else:
                content = str(input_data)
            
            # Create Mistral messages format
            mistral_messages = [{"role": "user", "content": content}]
            
            response = self.client.chat.complete(
                model=self.model,
                messages=mistral_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Create response object that matches LangChain format
            class Response:
                def __init__(self, content):
                    self.content = content
            
            return Response(response.choices[0].message.content)
            
        except Exception as e:
            st.error(f"Mistral API Error: {e}")
            return Response("Sorry, I encountered an error while generating response.")
    
    def _call(self, prompt: str, stop=None, run_manager=None):
        """Alternative method for LangChain compatibility"""
        try:
            response = self.client.chat.complete(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"
    
    @property
    def _llm_type(self):
        return "mistral"

# Enhanced query processing
def setup_advanced_retrieval_chain():
    """Setup retrieval chain with advanced features"""
    try:
        if not st.session_state.vectorstore:
            return None
        
        # Create retriever with similarity threshold
        retriever = st.session_state.vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 5,
                "score_threshold": st.session_state.similarity_threshold
            }
        )
        
        # Enhanced memory
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # Select model based on user choice
        if st.session_state.selected_model == "mistral" and MISTRAL_API_KEY and MISTRAL_AVAILABLE:
            # For Mistral, use a simplified approach due to LangChain compatibility issues
            return {
                "type": "mistral",
                "retriever": retriever,
                "memory": memory,
                "llm": MistralLLM(
                    api_key=MISTRAL_API_KEY,
                    model="mistral-large-latest",
                    temperature=st.session_state.temperature,
                    max_tokens=st.session_state.max_tokens
                )
            }
        else:
            # Default to Gemini model with full LangChain integration
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash-latest",
                temperature=st.session_state.temperature,
                max_tokens=st.session_state.max_tokens,
                google_api_key=GOOGLE_API_KEY,
            )
            
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                memory=memory,
                return_source_documents=True,
                verbose=False
            )
            
            return qa_chain
        
    except Exception as e:
        st.error(f"Error setting up retrieval chain: {e}")
        return None

def query_with_analytics(chain, question: str) -> Dict[str, Any]:
    """Query with analytics and error handling"""
    try:
        start_time = time.time()
        
        # Add delay to prevent rate limiting
        time.sleep(1)  # 1 second delay between requests
        
        # Store search history
        st.session_state.search_history.append({
            "question": question,
            "timestamp": time.time()
        })
        
        # Keep only last 10 searches
        if len(st.session_state.search_history) > 10:
            st.session_state.search_history = st.session_state.search_history[-10:]
        
        # Handle different chain types
        if isinstance(chain, dict) and chain.get("type") == "mistral":
            # Custom Mistral handling
            retriever = chain["retriever"]
            llm = chain["llm"]
            
            # Retrieve relevant documents
            relevant_docs = retriever.get_relevant_documents(question)
            
            # Prepare context from documents
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            
            # Create prompt for Mistral
            prompt = f"""Based on the following context, please answer the question. If the context doesn't contain enough information to answer the question, say so clearly.

Context:
{context}

Question: {question}

Answer:"""
            
            # Get response from Mistral
            response_obj = llm.invoke(prompt)
            answer = response_obj.content if hasattr(response_obj, 'content') else str(response_obj)
            
            # Create response object
            response = {
                "answer": answer,
                "source_documents": relevant_docs,
                "query_time": time.time() - start_time,
                "question": question,
                "model_used": "mistral"
            }
            
        else:
            # Standard LangChain handling (for Gemini)
            response = chain.invoke({"question": question})
            response["query_time"] = time.time() - start_time
            response["question"] = question
            response["model_used"] = st.session_state.selected_model
        
        return response
        
    except Exception as e:
        error_msg = str(e).lower()
        if "quota" in error_msg or "rate" in error_msg or "429" in error_msg:
            raise Exception("RATE_LIMIT")
        elif "timeout" in error_msg:
            raise TimeoutError("Request timed out")
        else:
            raise e

# UI Helper functions
def display_file_lists():
    """Display local and uploaded file lists"""
    t = translations[st.session_state.language]
    
    # Local files
    if st.session_state.local_files:
        st.markdown(f"**📁 {t['local_files']} ({len(st.session_state.local_files)})**")
        for filepath in st.session_state.local_files[:5]:  # Show first 5
            filename = os.path.basename(filepath)
            file_size = ""
            try:
                size_kb = os.path.getsize(filepath) / 1024
                file_size = f" ({size_kb:.1f}KB)"
            except:
                pass
            st.text(f"📄 {filename}{file_size}")
        
        if len(st.session_state.local_files) > 5:
            st.text(f"... and {len(st.session_state.local_files) - 5} more files")
    
    # Uploaded files
    if st.session_state.uploaded_files:
        st.markdown(f"**📤 {t['uploaded_files']} ({len(st.session_state.uploaded_files)})**")
        for uploaded_file in st.session_state.uploaded_files:
            file_size = len(uploaded_file.getvalue()) / 1024
            st.text(f"📄 {uploaded_file.name} ({file_size:.1f}KB)")

def display_advanced_settings():
    """Display advanced settings"""
    with st.expander("⚙️ Advanced Settings"):
        st.session_state.similarity_threshold = st.slider(
            "Similarity Threshold",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.similarity_threshold,
            step=0.1,
            help="Higher values = more strict matching"
        )
        
        st.session_state.max_tokens = st.slider(
            "Max Response Tokens",
            min_value=128,
            max_value=2048,
            value=st.session_state.max_tokens,
            step=128
        )
        
        st.session_state.temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.temperature,
            step=0.1,
            help="Higher values = more creative responses"
        )

def cleanup_cache():
    """Clean up old cache files"""
    try:
        current_time = time.time()
        for cache_dir in CACHE_DIR.glob("vectors_*"):
            metadata_path = cache_dir / "metadata.json"
            if metadata_path.exists():
                try:
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)
                    
                    # Delete if older than 7 days
                    if current_time - metadata["timestamp"] > 604800:
                        import shutil
                        shutil.rmtree(cache_dir)
                except:
                    continue
    except Exception as e:
        pass

# Main application
def main():
    try:
        # Handle language
        if "language" in st.query_params:
            st.session_state.language = st.query_params["language"]

        t = translations[st.session_state.language]

        # Enhanced CSS with compact styling - removed large white space
        st.markdown("""
            <style>
            .main .block-container {
                max-width: 1200px;
                padding-top: 1rem;
                padding-bottom: 1rem;
            }
            @media (max-width: 768px) {
                .main .block-container {
                    padding-top: 0.5rem;
                    padding-bottom: 1rem;
                    padding-left: 1rem;
                    padding-right: 1rem;
                }
            }
            .stTitle {
                text-align: center;
                color: #1f77b4;
                margin-top: 0.5rem;
                margin-bottom: 1rem;
                font-size: 2rem !important;
                font-weight: 700;
                line-height: 1.3 !important;
                padding: 0.5rem 0;
                display: flex;
                align-items: center;
                justify-content: center;
                min-height: 3rem;
                position: relative;
                z-index: 10;
            }
            .title-emoji {
                font-size: 2.5rem;
                margin-right: 0.4rem;
                filter: none;
                background: none;
                line-height: 1;
                display: inline-block;
                vertical-align: middle;
                padding: 0.2rem;
                margin-top: -0.1rem;
            }
            .title-text {
                background: linear-gradient(90deg, #1f77b4, #2ca02c);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                font-weight: 700;
                line-height: 1.2;
                display: inline-block;
                vertical-align: middle;
                padding: 0.1rem 0;
            }
            .model-info {
                background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
                border-left: 4px solid #2196f3;
                padding: 0.8rem;
                border-radius: 8px;
                margin: 0.5rem 0;
                font-size: 0.9rem;
                line-height: 1.4;
            }
            .model-info .emoji {
                font-size: 1.1rem;
                margin-right: 0.2rem;
                filter: none;
                background: none;
                line-height: 1;
                display: inline-block;
                vertical-align: middle;
                padding: 0.05rem;
            }
            .model-info .bold-text {
                font-weight: 700;
                color: #1976d2;
                vertical-align: middle;
            }
            .chat-container {
                background: #fafafa;
                border-radius: 12px;
                padding: 1rem;
                margin-top: 0.5rem;
                margin-bottom: 0.5rem;
                min-height: 40vh;
            }
            .chat-container-active {
                background: #fafafa;
                border-radius: 12px;
                padding: 1rem;
                margin-top: 0.5rem;
                margin-bottom: 0.5rem;
                min-height: 40vh;
            }
            .no-documents-container {
                background: #f8f9fa;
                border-radius: 8px;
                padding: 1rem;
                margin: 0.5rem 0;
                border: 1px solid #dee2e6;
            }
            @media (max-width: 768px) {
                .chat-container {
                    margin-bottom: 0.5rem;
                    padding: 0.8rem;
                    min-height: 35vh;
                }
            }
            .status-columns {
                margin: 0.5rem 0;
            }
            .compact-info {
                padding: 0.3rem 0;
                margin: 0.2rem 0;
            }
            .footer {
                display: none;
            }
            /* Hide Streamlit default padding */
            .block-container {
                padding-top: 1rem !important;
                padding-bottom: 0.5rem !important;
            }
            /* Compact status indicators */
            .stColumns > div {
                padding: 0.2rem;
            }
            </style>
        """, unsafe_allow_html=True)

        # Title first - prominent display
        st.markdown(f'''
            <div class="stTitle">
                <span class="title-emoji">🤖</span>
                <span class="title-text">{t["title"].replace("🤖 ", "")}</span>
            </div>
        ''', unsafe_allow_html=True)

        if not st.session_state.app_initialized:
            st.session_state.app_initialized = True
            cleanup_cache()

        # Model info with better styling - updated to show FAISS
        st.markdown(f'<div class="model-info">{t["model_info"]}</div>', unsafe_allow_html=True)

        # Document loading status - compact version
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.session_state.local_files:
                st.success(f"📁 {len(st.session_state.local_files)} files")
            else:
                st.info("📁 No files")
                
        with col2:
            if st.session_state.documents_processed and st.session_state.vectorstore:
                chunk_count = st.session_state.document_chunks
                if chunk_count > 0:
                    st.success(f"🗃️ FAISS ({chunk_count})")
                else:
                    st.error(f"🗃️ FAISS (0 chunks!)")
            else:
                st.warning("🗃️ Not ready")
                
        with col3:
            if st.session_state.initialization_complete:
                # Show current model
                current_model = "Mistral" if st.session_state.selected_model == "mistral" else "Gemini"
                st.success(f"✅ {current_model}")
            else:
                st.info("⏳ Loading...")

        # Sidebar
        with st.sidebar:
            # Language selection
            st.markdown(f'<div class="emoji-text"><span class="emoji-inline">🌐</span><span class="bold-text">{t["language"].replace("🌐 ", "")}</span></div>', unsafe_allow_html=True)
            selected_lang = st.selectbox(
                "Select Language",
                options=["ไทย", "English"],
                index=1 if st.session_state.language == "en" else 0,
                label_visibility="collapsed"
            )
            
            new_language = "th" if selected_lang == "ไทย" else "en"
            if new_language != st.session_state.language:
                st.session_state.language = new_language

            st.markdown("---")

            # Model selection
            st.markdown(f'<div class="emoji-text"><span class="emoji-inline">🤖</span><span class="bold-text">AI Model</span></div>', unsafe_allow_html=True)
            available_models = []
            if GOOGLE_API_KEY:
                available_models.append("Gemini Flash")
            if MISTRAL_API_KEY and MISTRAL_AVAILABLE:
                available_models.append("Mistral Large")
            
            if available_models:
                model_choice = st.selectbox(
                    "Choose AI Model",
                    options=available_models,
                    index=0 if st.session_state.selected_model == "gemini" else (1 if "Mistral Large" in available_models else 0),
                    label_visibility="collapsed"
                )
                
                if model_choice == "Mistral Large":
                    st.session_state.selected_model = "mistral"
                else:
                    st.session_state.selected_model = "gemini"
            else:
                st.error("No AI models available")

            st.markdown("---")

            # Debug mode
            st.session_state.debug_mode = st.checkbox("🔍 Debug Mode")

            # Reload local files button
            if st.button(t["reload_local"], use_container_width=True):
                # Clear session state
                st.session_state.local_files = []
                st.session_state.auto_load_attempted = False
                st.session_state.documents_processed = False
                st.session_state.vectorstore = None
                st.session_state.show_loading_messages = True
                st.session_state.initialization_complete = False
                # Reset app state to allow messages to show again
                save_app_state({"initialized": False, "last_load": 0})
                st.rerun()

            # File uploader
            st.markdown(f'<div class="emoji-text"><span class="emoji-inline">📤</span><span class="bold-text">{t["upload_button"]}</span></div>', unsafe_allow_html=True)
            uploaded_files = st.file_uploader(
                "Upload Additional Documents",
                accept_multiple_files=True,
                type=['pdf', 'csv', 'txt', 'xlsx', 'xls'],
                label_visibility="collapsed"
            )

            # Process additional uploaded files
            if uploaded_files and uploaded_files != st.session_state.uploaded_files:
                st.session_state.uploaded_files = uploaded_files
                with st.spinner(t["processing"]):
                    success = process_all_documents(st.session_state.local_files, uploaded_files, show_progress=True)
                    if success:
                        st.success(t["upload_success"](len(uploaded_files)))
                    else:
                        st.error(t["error_processing"])

            st.markdown("---")

            # Display file lists
            display_file_lists()

            # Advanced settings
            display_advanced_settings()

            # Debug information
            if st.session_state.debug_mode and st.session_state.documents_processed:
                with st.expander("🔍 Debug Information"):
                    st.write(f"**Documents Status:**")
                    st.write(f"- Local files: {len(st.session_state.local_files)}")
                    st.write(f"- Uploaded files: {len(st.session_state.uploaded_files)}")
                    st.write(f"- Document chunks: {st.session_state.document_chunks}")
                    st.write(f"- Vector store: {'✅ Ready' if st.session_state.vectorstore else '❌ Not ready'}")
                    st.write(f"- Initialization complete: {'✅ Yes' if st.session_state.initialization_complete else '❌ No'}")
                    st.write(f"- Search history: {len(st.session_state.search_history)} queries")
                    
                    cache_files = list(CACHE_DIR.glob("vectors_*"))
                    st.write(f"- Cache files: {len(cache_files)}")
                    
                    if st.session_state.vectorstore:
                        st.write(f"**Vector Database Info:**")
                        try:
                            # Get FAISS info
                            vectorstore_type = type(st.session_state.vectorstore).__name__
                            st.write(f"- Vector store type: {vectorstore_type}")
                            st.write(f"- Database: FAISS (Facebook AI Similarity Search)")
                            
                            # FAISS index info
                            if hasattr(st.session_state.vectorstore, 'index'):
                                st.write(f"- FAISS index size: {st.session_state.vectorstore.index.ntotal}")
                                st.write(f"- Vector dimension: {st.session_state.vectorstore.index.d}")
                            else:
                                st.write(f"- FAISS status: Available")
                        except Exception as e:
                            st.write(f"- Vector store info: {str(e)}")
                    
                    # Test vector store functionality
                    if st.button("🧪 Test Vector Store"):
                        if st.session_state.vectorstore:
                            try:
                                test_query = "test query"
                                retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
                                results = retriever.get_relevant_documents(test_query)
                                st.write(f"✅ Vector store working! Found {len(results)} documents")
                                for i, doc in enumerate(results[:2]):
                                    st.write(f"Doc {i+1}: {doc.page_content[:100]}...")
                            except Exception as e:
                                st.error(f"❌ Vector store test failed: {e}")
                        else:
                            st.error("❌ No vector store available")

            # Statistics
            if st.session_state.debug_mode and st.session_state.documents_processed:
                st.markdown(f'<div class="emoji-text"><span class="emoji-inline">📊</span><span class="bold-text">{t["stats"]}</span></div>', unsafe_allow_html=True)
                st.write(f"📁 Local files: {len(st.session_state.local_files)}")
                st.write(f"📤 Uploaded: {len(st.session_state.uploaded_files)}")
                st.write(f"🔢 Chunks: {st.session_state.document_chunks}")
                st.write(f"🔍 Searches: {len(st.session_state.search_history)}")
                
                cache_files = list(CACHE_DIR.glob("vectors_*"))
                st.write(f"💾 Cache files: {len(cache_files)}")

            # Clear buttons
            if st.button(t["clear_chat"], use_container_width=True):
                st.session_state.messages = []
                st.success("✅ Chat cleared!")
            
            if st.button("🗑️ Clear FAISS Cache", use_container_width=True):
                try:
                    import shutil
                    # Clear cache directory
                    if CACHE_DIR.exists():
                        shutil.rmtree(CACHE_DIR)
                        CACHE_DIR.mkdir(exist_ok=True)
                    st.session_state.vectorstore = None
                    st.session_state.documents_processed = False
                    st.session_state.auto_load_attempted = False
                    st.session_state.show_loading_messages = True
                    # Reset app state
                    save_app_state({"initialized": False, "last_load": 0})
                    st.success("✅ FAISS cache cleared!")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error clearing FAISS cache: {e}")
            
            # Show FAISS status
            if st.session_state.vectorstore:
                st.success("📊 FAISS ready")
                try:
                    if hasattr(st.session_state.vectorstore, 'index'):
                        st.info(f"Vectors: {st.session_state.vectorstore.index.ntotal}")
                        st.caption(f"Dimension: {st.session_state.vectorstore.index.d}")
                except Exception as e:
                    st.caption(f"FAISS info: {str(e)}")
            else:
                st.info("📊 FAISS not ready")

        # Main content area
        # Auto-load local files if not done yet
        auto_load_local_files()

        # Main content
        if st.session_state.documents_processed:
            # Chat interface in container - ONLY when documents are processed
            st.markdown('<div class="chat-container-active">', unsafe_allow_html=True)
            
            # Display chat messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Chat input
            if prompt := st.chat_input(t["ask_placeholder"]):
                # Add user message
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                # Generate response
                with st.chat_message("assistant"):
                    retrieval_chain = setup_advanced_retrieval_chain()
                    if retrieval_chain:
                        with st.spinner(t["thinking"]):
                            try:
                                response = query_with_analytics(retrieval_chain, prompt)
                                answer = response.get('answer', 'No answer generated')
                                
                                st.markdown(answer)
                                st.session_state.messages.append({"role": "assistant", "content": answer})
                                
                                # Enhanced source display
                                if 'source_documents' in response and response['source_documents']:
                                    with st.expander(f"📚 Sources ({len(response['source_documents'])})"):
                                        for i, doc in enumerate(response['source_documents']):
                                            st.markdown(f'<span class="bold-text">Source {i+1}:</span>', unsafe_allow_html=True)
                                            content = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                                            st.markdown(content)
                                            
                                            if hasattr(doc, 'metadata') and doc.metadata:
                                                meta = doc.metadata
                                                source_type = "📁 Local" if meta.get('file_type') == 'local' else "📤 Uploaded"
                                                st.caption(f"{source_type}: {meta.get('source_file', 'Unknown')}")
                                                if st.session_state.debug_mode:
                                                    st.caption(f"🔢 Chunk: {meta.get('chunk_id', 'N/A')} | Length: {meta.get('content_length', 'N/A')}")
                                            st.markdown("---")
                                
                                # Query analytics
                                if st.session_state.debug_mode:
                                    st.caption(f"⏱️ Query time: {response.get('query_time', 0):.2f}s")
                                
                            except TimeoutError:
                                st.error("⏱️ Request timed out. Please try a shorter question.")
                            except Exception as e:
                                if "RATE_LIMIT" in str(e):
                                    st.error("⏳ API rate limit reached. Please wait and try again.")
                                else:
                                    st.error(f"🚨 Error: {str(e)}")
                    else:
                        st.error("⚠️ Could not set up the retrieval system.")
            
            st.markdown('</div>', unsafe_allow_html=True)

        else:
            # NO LARGE CONTAINER - Just show compact status info
            st.markdown('<div class="no-documents-container">', unsafe_allow_html=True)
            st.markdown("📄 **Status:** Loading documents or waiting for upload...")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Show debug info about file scanning - more compact
            if st.session_state.debug_mode:
                with st.expander("🔍 Debug Info", expanded=False):
                    st.write(f"- Auto load: {st.session_state.auto_load_attempted}")
                    st.write(f"- Local files: {len(st.session_state.local_files)}")
                    st.write(f"- Processed: {st.session_state.documents_processed}")
                    st.write(f"- Directory: {os.getcwd()}")
                    
                    # Manual file scan for debugging
                    if st.button("🔍 Debug Scan"):
                        debug_files = []
                        try:
                            for root, dirs, files in os.walk('.'):
                                if not root.startswith('.'):
                                    for file in files:
                                        if not file.startswith('.'):
                                            debug_files.append(os.path.join(root, file))
                            st.write(f"- All files: {debug_files[:10]}")  # Show first 10
                            
                            # Check specifically for document files
                            doc_files = [f for f in debug_files if f.lower().endswith(('.pdf', '.txt', '.csv', '.xlsx', '.xls', '.docx'))]
                            st.write(f"- Documents: {doc_files}")
                            
                        except Exception as e:
                            st.write(f"- Error: {e}")
                    
                    # Create test file button
                    if st.button("📝 Create Test File"):
                        try:
                            test_content = """# Test Document for RAG System

This document tests the RAG chatbot functionality.

## Key Features
- Document loading
- Vector embeddings  
- Question answering
- Source retrieval

Use this content to verify the system works correctly.
"""
                            with open("test_document.txt", "w", encoding="utf-8") as f:
                                f.write(test_content)
                            st.success("✅ Created test_document.txt")
                            time.sleep(1)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")
            
            # Show loading progress - compact
            if st.session_state.auto_load_attempted and not st.session_state.documents_processed:
                if st.session_state.local_files:
                    st.warning(f"⚠️ Found {len(st.session_state.local_files)} files but processing failed.")
                else:
                    st.info("💡 No local files found. Upload documents via sidebar.")
            
            # Compact help section
            with st.expander("ℹ️ Quick Help", expanded=False):
                if st.session_state.language == "en":
                    st.markdown("""
                    **🚀 Quick Start:**
                    - Auto-detects PDF, TXT, CSV, XLSX files in repository
                    - Upload additional files via sidebar
                    - Uses Gemini Flash + FAISS for fast search
                    - Smart caching for quick reloads
                    
                    **📊 Supported Files:** PDF, TXT, CSV, XLSX, DOCX
                    """)
                else:
                    st.markdown("""
                    **🚀 เริ่มต้นใช้งาน:**
                    - ตรวจจับไฟล์ PDF, TXT, CSV, XLSX อัตโนมัติ
                    - อัปโหลดไฟล์เพิ่มผ่านแถบด้านข้าง
                    - ใช้ Gemini Flash + FAISS ค้นหาเร็ว
                    - ระบบ cache อัจฉริยะ
                    
                    **📊 ไฟล์ที่รองรับ:** PDF, TXT, CSV, XLSX, DOCX
                    """)

        # Always show chat input area at the bottom - no extra margin
        pass
        
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        if st.session_state.debug_mode:
            st.code(str(e))

if __name__ == "__main__":
    main()
