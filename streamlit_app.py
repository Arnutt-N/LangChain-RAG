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
    page_title="Gen AI : Advanced RAG Chatbot",
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
        from langchain_community.vectorstores import FAISS
        from langchain_community.document_loaders import PyPDFLoader, CSVLoader, UnstructuredExcelLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain.memory import ConversationBufferMemory
        from langchain.chains import ConversationalRetrievalChain
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
    pip install -r requirements.txt
    ```
    """)
    st.stop()

# Import all required packages
try:
    import google.generativeai as genai
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
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
    
except ImportError as e:
    st.error(f"Import error: {e}")
    st.info("Please check your package versions and requirements.txt")
    st.stop()

# Configuration
GOOGLE_API_KEY = (
    st.secrets.get("GOOGLE_API_KEY") or 
    st.secrets.get("GEMINI_API_KEY") or 
    os.getenv("GOOGLE_API_KEY") or 
    os.getenv("GEMINI_API_KEY")
)

if not GOOGLE_API_KEY:
    st.error("🚨 Google API Key not found!")
    st.info("""
    **Setup Google API Key:**
    1. Go to https://makersuite.google.com/app/apikey
    2. Create API key
    3. Add to Streamlit secrets as GOOGLE_API_KEY
    """)
    st.stop()

# Configure Google AI
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
genai.configure(api_key=GOOGLE_API_KEY)

# Setup persistent storage
CACHE_DIR = Path("./vector_cache")
CACHE_DIR.mkdir(exist_ok=True)

# Translations
translations = {
    "en": {
        "title": "🤖 Advanced RAG Chatbot",
        "upload_button": "Upload Additional Documents",
        "ask_placeholder": "Ask a question in Thai or English...",
        "processing": "Processing documents...",
        "welcome": "👋 Hello! I'm ready to chat about various topics based on the documents.",
        "upload_success": lambda count: f"✅ {count} document(s) uploaded successfully!",
        "thinking": "🧠 Generating response...",
        "language": "🌐 Language / ภาษา",
        "clear_chat": "🗑️ Clear Chat",
        "clear_cache": "🗑️ Clear Cache",
        "reload_local": "🔄 Reload Local Files",
        "model_info": "🤖 **Model:** Gemini Pro | 📊 **Embedding:** MiniLM-L6-v2 | 🗃️ **Vector DB:** FAISS",
        "no_documents": "📄 No documents found. Please check the repository or upload files.",
        "error_processing": "❌ Error processing documents. Please try again.",
        "error_response": "🚨 Sorry, I encountered an error while generating response.",
        "checking_cache": "🔍 Checking cache...",
        "found_cached": "✅ Found cached vectors",
        "saving_cache": "💾 Saving to cache...",
        "local_files": "📁 Local Repository Files",
        "uploaded_files": "📤 Uploaded Files",
        "stats": "Statistics",
        "advanced_features": "Advanced Features",
        "auto_loaded": "✅ Auto-loaded from repository",
        "processing_local": "📂 Processing repository files...",
        "found_local_files": lambda count: f"📁 Found {count} local files in repository",
    },
    "th": {
        "title": "🤖 แชทบอท RAG ขั้นสูง",
        "upload_button": "อัปโหลดเอกสารเพิ่มเติม",
        "ask_placeholder": "ถามคำถามเป็นภาษาไทยหรืออังกฤษ...",
        "processing": "กำลังประมวลผลเอกสาร...",
        "welcome": "👋 สวัสดี! ฉันพร้อมพูดคุยเกี่ยวกับเอกสารต่างๆ",
        "upload_success": lambda count: f"✅ อัปโหลดเอกสาร {count} ฉบับสำเร็จ!",
        "thinking": "🧠 กำลังสร้างคำตอบ...",
        "language": "🌐 ภาษา / Language",
        "clear_chat": "🗑️ ล้างการแชท",
        "clear_cache": "🗑️ ล้าง Cache",
        "reload_local": "🔄 โหลดไฟล์ local ใหม่",
        "model_info": "🤖 **โมเดล:** Gemini Pro | 📊 **Embedding:** MiniLM-L6-v2 | 🗃️ **Vector DB:** FAISS",
        "no_documents": "📄 ไม่พบเอกสาร กรุณาตรวจสอบ repository หรืออัปโหลดไฟล์",
        "error_processing": "❌ เกิดข้อผิดพลาดในการประมวลผลเอกสาร",
        "error_response": "🚨 ขออภัย เกิดข้อผิดพลาดในการสร้างคำตอบ",
        "checking_cache": "🔍 ตรวจสอบ cache...",
        "found_cached": "✅ พบ vectors ใน cache",
        "saving_cache": "💾 บันทึกลง cache...",
        "local_files": "📁 ไฟล์ใน Repository",
        "uploaded_files": "📤 ไฟล์ที่อัปโหลด",
        "stats": "สถิติ",
        "advanced_features": "ฟีเจอร์ขั้นสูง",
        "auto_loaded": "✅ โหลดอัตโนมัติจาก repository",
        "processing_local": "📂 กำลังประมวลผลไฟล์ใน repository...",
        "found_local_files": lambda count: f"📁 พบไฟล์ local {count} ไฟล์ใน repository",
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
    
    # Always ignore these
    if filename.startswith('.') or filename == 'requirements.txt':
        return True
    
    for pattern in patterns:
        if fnmatch.fnmatch(filepath, pattern) or fnmatch.fnmatch(filename, pattern):
            return True
    
    return False

@st.cache_data
def scan_local_files():
    """Scan repository for document files"""
    supported_extensions = ('.pdf', '.txt', '.csv', '.xlsx', '.xls', '.docx')
    local_files = []
    ignore_patterns = load_gitignore_patterns()
    
    try:
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
        st.warning(f"Error scanning local files: {e}")
    
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
            st.error("❌ Embeddings model failed to generate vectors")
            return None
            
    except Exception as e:
        st.error(f"Error initializing embeddings: {e}")
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
    
    cache_string = json.dumps(sorted(file_hashes.items()), sort_keys=True)
    return hashlib.md5(cache_string.encode()).hexdigest()[:16]

def save_vectors_to_cache(vectorstore: FAISS, cache_key: str, file_info: Dict):
    """Save vectorstore to cache"""
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
            "chunks": st.session_state.document_chunks
        }
        
        with open(cache_path / "metadata.json", "w", encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        return True
    except Exception as e:
        st.warning(f"Could not save to cache: {e}")
        return False

def load_vectors_from_cache(cache_key: str):
    """Load vectorstore from cache"""
    try:
        cache_path = CACHE_DIR / f"vectors_{cache_key}"
        metadata_path = cache_path / "metadata.json"
        
        if not cache_path.exists() or not metadata_path.exists():
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
        
        # Load FAISS vectorstore
        vectorstore = FAISS.load_local(
            str(cache_path), 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        
        return vectorstore, metadata
        
    except Exception as e:
        st.warning(f"Could not load from cache: {e}")
        return None, None

# Enhanced document processing
def load_single_local_file(filepath: str) -> List[Document]:
    """Load a single local file"""
    try:
        file_extension = Path(filepath).suffix.lower()
        
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
            # Try to load docx if available
            try:
                from langchain_community.document_loaders import UnstructuredWordDocumentLoader
                loader = UnstructuredWordDocumentLoader(filepath)
            except:
                st.warning(f"Cannot load .docx file: {filepath}")
                return []
        else:
            st.warning(f"Unsupported file type: {file_extension}")
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
                        'source_file': os.path.basename(filepath),
                        'file_path': filepath,
                        'file_hash': get_file_hash(filepath),
                        'file_type': 'local',
                        'chunk_id': i,
                        'content_length': len(content)
                    })
                    cleaned_docs.append(doc)
        
        return cleaned_docs
        
    except Exception as e:
        st.warning(f"Error loading local file {filepath}: {e}")
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
                st.warning(f"Unsupported file type: {file_extension}")
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
        st.error(f"Error loading file {uploaded_file.name}: {e}")
        return []

def process_all_documents(local_files: List[str], uploaded_files: List) -> bool:
    """Process all documents with intelligent caching"""
    t = translations[st.session_state.language]
    
    total_files = len(local_files) + len(uploaded_files)
    if total_files == 0:
        return False
    
    # Generate cache key
    cache_key = get_cache_key(local_files, uploaded_files)
    
    # Try to load from cache
    st.info(f"🔍 {t['checking_cache']}")
    cached_vectorstore, cached_metadata = load_vectors_from_cache(cache_key)
    
    if cached_vectorstore and cached_metadata:
        st.success(f"✅ {t['found_cached']}")
        st.session_state.vectorstore = cached_vectorstore
        st.session_state.document_chunks = cached_metadata.get("chunks", 0)
        st.session_state.documents_processed = True
        return True
    
    # Process documents if not in cache
    st.info(f"📝 {t['processing']}")
    
    all_documents = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        total_files_to_process = len(local_files) + len(uploaded_files)
        processed = 0
        
        # Process local files
        for filepath in local_files:
            status_text.text(f"Loading local: {os.path.basename(filepath)}...")
            progress_bar.progress(processed / total_files_to_process * 0.5)
            
            docs = load_single_local_file(filepath)
            all_documents.extend(docs)
            processed += 1
        
        # Process uploaded files
        for uploaded_file in uploaded_files:
            status_text.text(f"Loading uploaded: {uploaded_file.name}...")
            progress_bar.progress(processed / total_files_to_process * 0.5)
            
            docs = load_single_uploaded_file(uploaded_file)
            all_documents.extend(docs)
            processed += 1
        
        if not all_documents:
            st.warning("No documents could be processed")
            return False
        
        # Split documents with updated text splitter
        status_text.text("🔄 Splitting documents...")
        progress_bar.progress(0.7)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )
        texts = text_splitter.split_documents(all_documents)
        
        if not texts:
            st.warning("No text content found after splitting")
            return False
        
        # Create embeddings and vectorstore
        status_text.text("📊 Creating embeddings...")
        progress_bar.progress(0.9)
        
        embeddings = get_embeddings()
        if not embeddings:
            return False
        
        vectorstore = FAISS.from_documents(texts, embeddings)
        
        # Save to cache
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
        
        progress_bar.progress(1.0)
        progress_bar.empty()
        status_text.empty()
        
        return True
        
    except Exception as e:
        if 'progress_bar' in locals():
            progress_bar.empty()
        if 'status_text' in locals():
            status_text.empty()
        st.error(f"Error processing documents: {e}")
        return False

# Auto-load local files on startup
def auto_load_local_files():
    """Automatically load local files on startup"""
    if st.session_state.auto_load_attempted:
        return
    
    st.session_state.auto_load_attempted = True
    
    # Scan for local files
    local_files = scan_local_files()
    st.session_state.local_files = local_files
    
    t = translations[st.session_state.language]
    
    if local_files:
        st.info(t["found_local_files"](len(local_files)))
        
        # Auto-process if we have local files
        with st.spinner(t["processing_local"]):
            success = process_all_documents(local_files, [])
            if success:
                st.success(f"✅ {t['auto_loaded']}")
                st.success(f"📊 Processed {len(local_files)} local files into {st.session_state.document_chunks} chunks!")

# Enhanced query processing (same as before)
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
        
        # Updated Gemini model with latest API
        llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
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
        
        # Store search history
        st.session_state.search_history.append({
            "question": question,
            "timestamp": time.time()
        })
        
        # Keep only last 10 searches
        if len(st.session_state.search_history) > 10:
            st.session_state.search_history = st.session_state.search_history[-10:]
        
        response = chain.invoke({"question": question})
        
        # Add analytics
        response["query_time"] = time.time() - start_time
        response["question"] = question
        
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
        st.warning(f"Error cleaning cache: {e}")

# Main application
def main():
    try:
        # Handle language
        if "language" in st.query_params:
            st.session_state.language = st.query_params["language"]

        t = translations[st.session_state.language]

        if not st.session_state.app_initialized:
            st.session_state.app_initialized = True
            cleanup_cache()

        # Enhanced CSS
        st.markdown("""
            <style>
            .main .block-container {
                max-width: 1200px;
                padding-top: 2rem;
                padding-bottom: 3rem;
            }
            .stTitle {
                text-align: center;
                color: #1f77b4;
                margin-bottom: 2rem;
            }
            .metric-card {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 8px;
                padding: 1rem;
                margin: 0.5rem 0;
            }
            .footer {
                position: fixed;
                left: 50%;
                bottom: 0;
                transform: translateX(-50%);
                text-align: center;
                padding: 10px 0;
                font-size: 14px;
                color: #545454;
                background-color: white;
                width: 100%;
                border-top: 1px solid #eee;
                z-index: 999;
            }
            </style>
        """, unsafe_allow_html=True)

        # Auto-load local files on first run
        auto_load_local_files()

        # Sidebar
        with st.sidebar:
            # Language selection
            st.markdown(f"**{t['language']}**")
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

            # Debug mode
            st.session_state.debug_mode = st.checkbox("🔍 Debug Mode")

            # Reload local files button
            if st.button(t["reload_local"], use_container_width=True):
                st.session_state.local_files = scan_local_files()
                st.session_state.auto_load_attempted = False
                st.session_state.documents_processed = False
                st.session_state.vectorstore = None
                st.rerun()

            # File uploader
            st.markdown(f"**{t['upload_button']}**")
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
                    success = process_all_documents(st.session_state.local_files, uploaded_files)
                    if success:
                        st.success(t["upload_success"](len(uploaded_files)))
                    else:
                        st.error(t["error_processing"])

            st.markdown("---")

            # Display file lists
            display_file_lists()

            # Advanced settings
            display_advanced_settings()

            # Statistics
            if st.session_state.debug_mode and st.session_state.documents_processed:
                st.markdown(f"**📊 {t['stats']}**")
                st.write(f"📁 Local files: {len(st.session_state.local_files)}")
                st.write(f"📤 Uploaded: {len(st.session_state.uploaded_files)}")
                st.write(f"🔢 Chunks: {st.session_state.document_chunks}")
                st.write(f"🔍 Searches: {len(st.session_state.search_history)}")
                
                cache_files = list(CACHE_DIR.glob("vectors_*"))
                st.write(f"💾 Cache files: {len(cache_files)}")

            # Clear buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button(t["clear_chat"], use_container_width=True):
                    st.session_state.messages = []
                    st.success("✅ Chat cleared!")
            
            with col2:
                if st.button(t["clear_cache"], use_container_width=True):
                    try:
                        import shutil
                        if CACHE_DIR.exists():
                            shutil.rmtree(CACHE_DIR)
                            CACHE_DIR.mkdir(exist_ok=True)
                        st.session_state.vectorstore = None
                        st.session_state.documents_processed = False
                        st.session_state.auto_load_attempted = False
                        st.success("✅ Cache cleared!")
                    except Exception as e:
                        st.error(f"Error clearing cache: {e}")

        # Main content
        st.title(t["title"])
        st.info(t["model_info"])

        # Chat interface
        if st.session_state.documents_processed:
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
                                            st.markdown(f"**Source {i+1}:**")
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

        else:
            # Welcome message
            st.markdown(f"### {t['welcome']}")
            
            with st.expander("ℹ️ How to use / วิธีใช้งาน", expanded=True):
                if st.session_state.language == "en":
                    st.markdown("""
                    **🚀 Advanced RAG Chatbot with Auto-Load:**
                    
                    **📁 Auto-Detection:**
                    - Automatically scans repository for PDF, TXT, CSV, XLSX files
                    - Respects .gitignore patterns
                    - Loads documents on startup
                    
                    **📤 Additional Upload:**
                    - Upload more documents using the sidebar
                    - Combines with auto-detected files
                    - Smart caching for fast reloads
                    
                    **🤖 AI Features:**
                    - Gemini Pro language model
                    - MiniLM-L6-v2 embeddings for semantic search
                    - Adjustable similarity threshold
                    - Configurable response parameters
                    
                    **💾 Smart Caching:**
                    - Automatic vector caching
                    - Fast reload for same documents
                    - Cache cleanup and management
                    - Persistent storage across sessions
                    
                    **📊 File Types Supported:**
                    - 📄 PDF files
                    - 📝 Text files (.txt)
                    - 📊 CSV files
                    - 📈 Excel files (.xlsx, .xls)
                    - 📄 Word documents (.docx)
                    """)
                else:
                    st.markdown("""
                    **🚀 Advanced RAG Chatbot พร้อม Auto-Load:**
                    
                    **📁 การตรวจจับอัตโนมัติ:**
                    - สแกนหาไฟล์ PDF, TXT, CSV, XLSX ใน repository อัตโนมัติ
                    - เคารพรูปแบบ .gitignore
                    - โหลดเอกสารตอนเริ่มต้น
                    
                    **📤 อัปโหลดเพิ่มเติม:**
                    - อัปโหลดเอกสารเพิ่มผ่านแถบด้านข้าง
                    - รวมกับไฟล์ที่ตรวจจับอัตโนมัติ
                    - Smart caching สำหรับโหลดเร็ว
                    
                    **🤖 ฟีเจอร์ AI:**
                    - โมเดลภาษา Gemini Pro
                    - MiniLM-L6-v2 embeddings สำหรับค้นหาความหมาย
                    - ปรับระดับความคล้ายได้
                    - ตั้งค่าพารามิเตอร์การตอบได้
                    
                    **💾 Smart Caching:**
                    - Cache vectors อัตโนมัติ
                    - โหลดเร็วสำหรับเอกสารเดิม
                    - ทำความสะอาดและจัดการ cache
                    - เก็บข้อมูลถาวรข้ามเซสชั่น
                    
                    **📊 ประเภทไฟล์ที่รองรับ:**
                    - 📄 ไฟล์ PDF
                    - 📝 ไฟล์ข้อความ (.txt)
                    - 📊 ไฟล์ CSV
                    - 📈 ไฟล์ Excel (.xlsx, .xls)
                    - 📄 เอกสาร Word (.docx)
                    """)

            total_files = len(st.session_state.local_files) + len(st.session_state.uploaded_files)
            if total_files == 0:
                st.info(t["no_documents"])
            else:
                st.info(f"📁 Found {len(st.session_state.local_files)} local files. Click 'Reload Local Files' if files were added recently.")

        # Footer
        st.markdown(
            '<div class="footer">Advanced RAG Chatbot v2.0 with Auto-Load | Created by Arnutt Noitumyae, 2024</div>',
            unsafe_allow_html=True
        )
        
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        if st.session_state.debug_mode:
            st.code(str(e))

if __name__ == "__main__":
    main()
