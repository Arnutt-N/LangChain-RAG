"""
Gen AI RAG Chatbot with Qwen Embeddings and Pre-built Vector Support
Complete Streamlit Application
"""

import streamlit as st
import os
import sys
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
    page_title="Gen AI : RAG Chatbot with Qwen Embeddings",
    page_icon="🤖",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# EARLY DEBUG INFO
# ==============================================================================
if 'debug_mode' not in st.session_state:
    st.session_state.debug_mode = False

if st.session_state.get('debug_mode', False):
    st.write(f"🐍 Python version: {sys.version}")
    st.write("🔄 App is starting...")

# ==============================================================================
# PRE-BUILT VECTOR SUPPORT
# ==============================================================================

PREBUILT_VECTORS_DIR = Path("prebuilt_vectors")
CACHE_DIR = Path("./vector_cache")
CACHE_DIR.mkdir(exist_ok=True)

def check_prebuilt_vectors():
    """Check if pre-built vectors exist in repository"""
    if not PREBUILT_VECTORS_DIR.exists():
        return False, None
    
    index_path = PREBUILT_VECTORS_DIR / "faiss_index"
    metadata_path = PREBUILT_VECTORS_DIR / "metadata.json"
    qwen_config_path = PREBUILT_VECTORS_DIR / "qwen_config.json"
    
    if index_path.exists() and metadata_path.exists():
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Check if using Qwen
            if qwen_config_path.exists():
                with open(qwen_config_path, 'r') as f:
                    qwen_config = json.load(f)
                    metadata['qwen_config'] = qwen_config
                    
            return True, metadata
        except Exception as e:
            st.error(f"Error reading pre-built metadata: {e}")
            return False, None
    return False, None

# ==============================================================================
# DEPENDENCY CHECKS
# ==============================================================================

@st.cache_data
def check_dependencies():
    """Check if all required packages are available"""
    missing_packages = []
    try:
        import google.generativeai as genai
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_community.document_loaders import PyPDFLoader, CSVLoader, TextLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain.memory import ConversationBufferMemory
        from langchain.chains import ConversationalRetrievalChain
        import faiss
        from langchain_community.vectorstores import FAISS
        from huggingface_hub import InferenceClient
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
    pip install streamlit langchain langchain-community 
    pip install faiss-cpu google-generativeai
    pip install huggingface-hub pypdf
    ```
    """)
    st.stop()

# ==============================================================================
# IMPORTS
# ==============================================================================

try:
    import google.generativeai as genai
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_community.document_loaders import (
        PyPDFLoader, 
        CSVLoader, 
        TextLoader,
        UnstructuredExcelLoader
    )
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.memory import ConversationBufferMemory
    from langchain.chains import ConversationalRetrievalChain
    from langchain.schema import Document
    import pandas as pd
    import numpy as np
    from huggingface_hub import InferenceClient
    import faiss
    from langchain_community.vectorstores import FAISS
    
except ImportError as e:
    st.error(f"Import error: {e}")
    st.info("Please check your requirements.txt")
    st.stop()

# ==============================================================================
# QWEN EMBEDDINGS CLASS
# ==============================================================================

from langchain.embeddings.base import Embeddings

class QwenEmbeddings(Embeddings):
    """Qwen embeddings wrapper for LangChain using HF Inference API"""
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        model_name: str = "Qwen/Qwen3-Embedding-8B",
        provider: str = "nebius",
        batch_size: int = 16,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        self.api_key = api_key or os.environ.get("HF_TOKEN")
        if not self.api_key:
            raise ValueError("HF_TOKEN not found. Set it in environment or pass api_key parameter")
        
        self.model_name = model_name
        self.provider = provider
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        self.client = InferenceClient(
            provider=self.provider,
            api_key=self.api_key,
        )
        
    def _embed_with_retry(self, text: str) -> List[float]:
        """Embed a single text with retry logic"""
        for attempt in range(self.max_retries):
            try:
                result = self.client.feature_extraction(
                    text,
                    model=self.model_name,
                )
                if isinstance(result, np.ndarray):
                    return result.tolist()
                return result
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise e
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            for text in batch:
                if len(text) > 8000:
                    text = text[:8000]
                
                embedding = self._embed_with_retry(text)
                embeddings.append(embedding)
                
                if len(texts) > 1:
                    time.sleep(0.1)
        
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text"""
        if len(text) > 8000:
            text = text[:8000]
            
        return self._embed_with_retry(text)

# ==============================================================================
# CONFIGURATION - API KEYS
# ==============================================================================

GOOGLE_API_KEY = (
    st.secrets.get("GOOGLE_API_KEY") or 
    st.secrets.get("GEMINI_API_KEY") or 
    os.getenv("GOOGLE_API_KEY") or 
    os.getenv("GEMINI_API_KEY")
)

HF_TOKEN = (
    st.secrets.get("HF_TOKEN") or
    st.secrets.get("HUGGINGFACE_TOKEN") or
    os.getenv("HF_TOKEN") or
    os.getenv("HUGGINGFACE_TOKEN")
)

MISTRAL_API_KEY = (
    st.secrets.get("MISTRAL_API_KEY") or 
    os.getenv("MISTRAL_API_KEY")
)

# Check if at least Google API key is available
if not GOOGLE_API_KEY:
    st.error("🚨 No Google API Key found!")
    st.info("""
    **Setup Google Gemini API:**
    1. Go to https://makersuite.google.com/app/apikey
    2. Create API key
    3. Add to Streamlit secrets as GOOGLE_API_KEY
    """)
    st.stop()

# Configure APIs
if GOOGLE_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    genai.configure(api_key=GOOGLE_API_KEY)

# ==============================================================================
# TRANSLATIONS
# ==============================================================================

translations = {
    "en": {
        "title": "🤖 Gen AI : RAG Chatbot with Qwen Embeddings",
        "upload_button": "Upload Additional Documents",
        "ask_placeholder": "Ask a question in Thai or English...",
        "processing": "Processing documents...",
        "thinking": "🧠 Generating response...",
        "language": "Language / ภาษา",
        "clear_chat": "🗑️ Clear Chat",
        "model_info": "Model: Gemini Flash | Embeddings: Qwen3-8B | Vector DB: FAISS",
        "using_prebuilt": "✅ Using pre-built vector database",
        "using_qwen": "🚀 Using Qwen3-Embedding-8B (1536 dims)",
        "using_minilm": "📊 Using MiniLM-L6-v2 (384 dims)",
        "no_hf_token": "⚠️ HF_TOKEN not found - Using fallback embeddings",
    },
    "th": {
        "title": "🤖 Gen AI : RAG แชทกับเอกสาร (Qwen Embeddings)",
        "upload_button": "อัปโหลดเอกสารเพิ่มเติม",
        "ask_placeholder": "ถามคำถามเป็นภาษาไทยหรืออังกฤษ...",
        "processing": "กำลังประมวลผลเอกสาร...",
        "thinking": "🧠 กำลังสร้างคำตอบ...",
        "language": "ภาษา / Language",
        "clear_chat": "🗑️ ล้างการแชท",
        "model_info": "โมเดล: Gemini Flash | Embeddings: Qwen3-8B | Vector DB: FAISS",
        "using_prebuilt": "✅ ใช้ vector database ที่สร้างไว้แล้ว",
        "using_qwen": "🚀 ใช้ Qwen3-Embedding-8B (1536 มิติ)",
        "using_minilm": "📊 ใช้ MiniLM-L6-v2 (384 มิติ)",
        "no_hf_token": "⚠️ ไม่พบ HF_TOKEN - ใช้ embeddings สำรอง",
    }
}

# ==============================================================================
# SESSION STATE INITIALIZATION
# ==============================================================================

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
        "similarity_threshold": 0.7,
        "max_tokens": 512,
        "temperature": 0.1,
        "using_prebuilt": False,
        "prebuilt_metadata": None,
        "embeddings_type": None,
        "embeddings": None,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# ==============================================================================
# EMBEDDINGS MANAGEMENT
# ==============================================================================

@st.cache_resource
def get_embeddings(use_qwen: bool = True):
    """Initialize embeddings - Qwen if HF_TOKEN available, else fallback"""
    
    if use_qwen and HF_TOKEN:
        try:
            embeddings = QwenEmbeddings(
                api_key=HF_TOKEN,
                model_name="Qwen/Qwen3-Embedding-8B",
                provider="nebius",
                batch_size=16,
                max_retries=3
            )
            st.session_state.embeddings_type = "qwen"
            return embeddings
        except Exception as e:
            st.warning(f"Failed to initialize Qwen embeddings: {e}")
            st.info("Falling back to HuggingFace embeddings...")
    
    # Fallback to HuggingFace embeddings
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        st.session_state.embeddings_type = "minilm"
        return embeddings
    except Exception as e:
        st.error(f"Failed to initialize embeddings: {e}")
        return None

@st.cache_resource(show_spinner=False)
def load_prebuilt_vectors():
    """Load pre-built vectors from repository"""
    try:
        # Check if Qwen was used for pre-built vectors
        qwen_config_path = PREBUILT_VECTORS_DIR / "qwen_config.json"
        use_qwen = qwen_config_path.exists()
        
        # Get appropriate embeddings
        embeddings = get_embeddings(use_qwen=use_qwen)
        if not embeddings:
            return None, None
        
        # Load FAISS index
        index_path = str(PREBUILT_VECTORS_DIR / "faiss_index")
        vectorstore = FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
        
        return vectorstore, embeddings
    except Exception as e:
        st.error(f"Failed to load pre-built vectors: {e}")
        return None, None

# ==============================================================================
# DOCUMENT PROCESSING
# ==============================================================================

def scan_local_files():
    """Scan repository for document files"""
    supported_extensions = ('.pdf', '.txt', '.csv', '.xlsx', '.xls', '.docx')
    local_files = []
    
    excluded_files = ['requirements.txt', 'README.md', '.env', 'streamlit_app.py']
    excluded_dirs = ['.git', '__pycache__', 'venv', 'env', '.streamlit', 
                     'vector_cache', 'prebuilt_vectors']
    
    try:
        for root, dirs, files in os.walk('.'):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in excluded_dirs]
            
            for file in files:
                if file.lower().endswith(supported_extensions):
                    filepath = os.path.join(root, file)
                    if not any(ex in filepath for ex in excluded_dirs):
                        if file not in excluded_files and not file.endswith('.py'):
                            local_files.append(filepath)
        
        return sorted(list(set(local_files)))
        
    except Exception as e:
        st.error(f"Error scanning files: {e}")
        return []

def load_document(filepath: str) -> List[Document]:
    """Load a single document file"""
    try:
        file_extension = Path(filepath).suffix.lower()
        
        if file_extension == '.pdf':
            loader = PyPDFLoader(filepath)
        elif file_extension == '.csv':
            loader = CSVLoader(filepath, encoding='utf-8')
        elif file_extension in ['.txt', '.md']:
            loader = TextLoader(filepath, encoding='utf-8')
        elif file_extension in ['.xlsx', '.xls']:
            loader = UnstructuredExcelLoader(filepath)
        else:
            return []
        
        docs = loader.load()
        
        # Add metadata
        for doc in docs:
            doc.metadata['source'] = filepath
            doc.metadata['file_type'] = 'local'
        
        return docs
        
    except Exception as e:
        st.error(f"Error loading {filepath}: {e}")
        return []

def process_documents(files: List[str], show_progress: bool = True) -> bool:
    """Process documents and create vector store"""
    
    if not files:
        return False
    
    # Check for pre-built vectors first
    has_prebuilt, metadata = check_prebuilt_vectors()
    
    if has_prebuilt and not st.session_state.uploaded_files:
        if show_progress:
            with st.spinner("🔄 Loading pre-built vector database..."):
                vectorstore, embeddings = load_prebuilt_vectors()
                
                if vectorstore:
                    st.session_state.vectorstore = vectorstore
                    st.session_state.embeddings = embeddings
                    st.session_state.document_chunks = metadata.get("total_chunks", 0)
                    st.session_state.documents_processed = True
                    st.session_state.using_prebuilt = True
                    st.session_state.prebuilt_metadata = metadata
                    
                    t = translations[st.session_state.language]
                    st.success(t["using_prebuilt"])
                    
                    # Show embedding type
                    if metadata.get('qwen_config'):
                        st.info(t["using_qwen"])
                    else:
                        st.info(t["using_minilm"])
                    
                    return True
    
    # Process documents dynamically
    if show_progress:
        st.info("📝 Processing documents...")
    
    all_documents = []
    progress_bar = st.progress(0) if show_progress else None
    
    try:
        for idx, filepath in enumerate(files):
            if progress_bar:
                progress_bar.progress((idx + 1) / len(files))
            
            docs = load_document(filepath)
            all_documents.extend(docs)
        
        if not all_documents:
            st.error("No documents loaded!")
            return False
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        )
        
        texts = text_splitter.split_documents(all_documents)
        
        if not texts:
            st.error("No text chunks created!")
            return False
        
        # Create embeddings and vector store
        embeddings = get_embeddings(use_qwen=(HF_TOKEN is not None))
        if not embeddings:
            return False
        
        st.session_state.embeddings = embeddings
        
        # Create FAISS vector store
        with st.spinner("Creating vector database..."):
            vectorstore = FAISS.from_documents(texts, embeddings)
        
        st.session_state.vectorstore = vectorstore
        st.session_state.document_chunks = len(texts)
        st.session_state.documents_processed = True
        
        if progress_bar:
            progress_bar.empty()
        
        st.success(f"✅ Processed {len(texts)} chunks from {len(files)} documents")
        
        # Show embedding type
        t = translations[st.session_state.language]
        if st.session_state.embeddings_type == "qwen":
            st.info(t["using_qwen"])
        else:
            st.info(t["using_minilm"])
        
        return True
        
    except Exception as e:
        st.error(f"Error processing documents: {e}")
        if progress_bar:
            progress_bar.empty()
        return False

# ==============================================================================
# AUTO-LOAD DOCUMENTS
# ==============================================================================

def auto_load_documents():
    """Automatically load documents on startup"""
    if 'auto_loaded' in st.session_state and st.session_state.auto_loaded:
        return
    
    st.session_state.auto_loaded = True
    
    # Check for pre-built vectors first
    has_prebuilt, metadata = check_prebuilt_vectors()
    
    if has_prebuilt:
        with st.spinner("Loading pre-built vectors..."):
            vectorstore, embeddings = load_prebuilt_vectors()
            
            if vectorstore:
                st.session_state.vectorstore = vectorstore
                st.session_state.embeddings = embeddings
                st.session_state.document_chunks = metadata.get("total_chunks", 0)
                st.session_state.documents_processed = True
                st.session_state.using_prebuilt = True
                st.session_state.prebuilt_metadata = metadata
                
                t = translations[st.session_state.language]
                st.success(t["using_prebuilt"])
                
                if metadata.get('qwen_config'):
                    st.info(f"🚀 Embeddings: Qwen3-8B ({metadata['qwen_config']['dimension']} dims)")
                else:
                    st.info(f"📊 Embeddings: {metadata.get('model', 'Unknown')}")
                return
    
    # Scan for local files
    local_files = scan_local_files()
    st.session_state.local_files = local_files
    
    if local_files:
        st.info(f"Found {len(local_files)} local documents")
        if st.button("📚 Load Documents", key="auto_load_btn"):
            success = process_documents(local_files, show_progress=True)
            if success:
                st.rerun()

# ==============================================================================
# QUERY PROCESSING
# ==============================================================================

def setup_qa_chain():
    """Setup QA chain with vector store"""
    if not st.session_state.vectorstore:
        return None
    
    try:
        # Initialize LLM (Gemini)
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-latest",
            temperature=st.session_state.temperature,
            max_tokens=st.session_state.max_tokens,
            google_api_key=GOOGLE_API_KEY,
        )
        
        # Create retriever
        retriever = st.session_state.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        # Memory
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # Create QA chain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            verbose=False
        )
        
        return qa_chain
        
    except Exception as e:
        st.error(f"Error setting up QA chain: {e}")
        return None

# ==============================================================================
# UI COMPONENTS
# ==============================================================================

def display_sidebar():
    """Display sidebar with controls"""
    t = translations[st.session_state.language]
    
    with st.sidebar:
        # Language selection
        st.markdown("### 🌐 Language / ภาษา")
        selected_lang = st.selectbox(
            "Select",
            options=["English", "ไทย"],
            index=0 if st.session_state.language == "en" else 1,
            label_visibility="collapsed"
        )
        st.session_state.language = "en" if selected_lang == "English" else "th"
        
        st.divider()
        
        # Debug mode
        st.session_state.debug_mode = st.checkbox("🔍 Debug Mode")
        
        # Display vector info
        if st.session_state.using_prebuilt:
            with st.expander("🚀 Pre-built Vectors", expanded=True):
                if st.session_state.prebuilt_metadata:
                    meta = st.session_state.prebuilt_metadata
                    st.write(f"**Created:** {meta.get('created_at', 'Unknown')[:19]}")
                    st.write(f"**Chunks:** {meta.get('total_chunks', 0)}")
                    st.write(f"**Docs:** {len(meta.get('documents', []))}")
                    
                    if meta.get('qwen_config'):
                        st.success("Using Qwen Embeddings")
                        st.write(f"Dimension: {meta['qwen_config']['dimension']}")
                    else:
                        st.info(f"Model: {meta.get('model', 'Unknown')}")
        
        # HF Token status
        if HF_TOKEN:
            st.success("✅ HF_TOKEN configured")
        else:
            st.warning(t["no_hf_token"])
        
        st.divider()
        
        # File upload
        st.markdown(f"### 📤 {t['upload_button']}")
        uploaded_files = st.file_uploader(
            "Upload",
            accept_multiple_files=True,
            type=['pdf', 'csv', 'txt', 'xlsx', 'xls'],
            label_visibility="collapsed"
        )
        
        if uploaded_files:
            st.warning("Note: Uploaded files will override pre-built vectors")
            if st.button("Process Uploads"):
                # Process uploaded files logic here
                st.info("Upload processing not implemented in this version")
        
        # Settings
        with st.expander("⚙️ Settings"):
            st.session_state.temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.temperature,
                step=0.1
            )
            
            st.session_state.max_tokens = st.slider(
                "Max Tokens",
                min_value=128,
                max_value=2048,
                value=st.session_state.max_tokens,
                step=128
            )
        
        # Clear chat
        if st.session_state.messages:
            if st.button(t["clear_chat"], use_container_width=True):
                st.session_state.messages = []
                st.rerun()
        
        # Statistics
        if st.session_state.documents_processed:
            st.divider()
            st.markdown("### 📊 Statistics")
            st.write(f"Chunks: {st.session_state.document_chunks}")
            if st.session_state.embeddings_type:
                st.write(f"Embeddings: {st.session_state.embeddings_type}")
            st.write(f"Messages: {len(st.session_state.messages)}")

# ==============================================================================
# MAIN APPLICATION
# ==============================================================================

def main():
    """Main application function"""
    
    t = translations[st.session_state.language]
    
    # CSS styling
    st.markdown("""
        <style>
        .main .block-container {
            max-width: 1200px;
            padding-top: 2rem;
        }
        .stTitle {
            text-align: center;
            color: #1f77b4;
            margin-bottom: 2rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Title
    st.title(t["title"])
    
    # Model info
    st.info(f"ℹ️ {t['model_info']}")
    
    # Display sidebar
    display_sidebar()
    
    # Auto-load documents
    auto_load_documents()
    
    # Main chat interface
    if st.session_state.documents_processed and st.session_state.document_chunks > 0:
        
        # Setup QA chain
        if 'qa_chain' not in st.session_state or st.session_state.qa_chain is None:
            st.session_state.qa_chain = setup_qa_chain()
        
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
                if st.session_state.qa_chain:
                    with st.spinner(t["thinking"]):
                        try:
                            response = st.session_state.qa_chain.invoke({"question": prompt})
                            answer = response.get('answer', 'No answer generated')
                            
                            st.markdown(answer)
                            st.session_state.messages.append({"role": "assistant", "content": answer})
                            
                            # Show sources
                            if response.get("source_documents"):
                                with st.expander(f"📚 Sources ({len(response['source_documents'])})"):
                                    for i, doc in enumerate(response['source_documents']):
                                        st.markdown(f"**Source {i+1}:**")
                                        st.text(doc.page_content[:300] + "...")
                                        if doc.metadata:
                                            st.caption(f"From: {Path(doc.metadata.get('source', 'Unknown')).name}")
                            
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                else:
                    st.error("QA chain not initialized")
    
    else:
        # No documents loaded
        st.warning("📄 No documents loaded yet")
        
        if not st.session_state.local_files:
            st.info("""
            **To get started:**
            1. Add documents to your repository
            2. Or create pre-built vectors with `prebuild_vectors_qwen.py`
            3. Or upload documents using the sidebar
            """)
        
        # Show debug info if enabled
        if st.session_state.debug_mode:
            with st.expander("🔍 Debug Information"):
                st.write(f"Local files: {len(st.session_state.local_files)}")
                st.write(f"Documents processed: {st.session_state.documents_processed}")
                st.write(f"Chunks: {st.session_state.document_chunks}")
                st.write(f"Vector store: {'Yes' if st.session_state.vectorstore else 'No'}")
                st.write(f"HF_TOKEN: {'Yes' if HF_TOKEN else 'No'}")
                st.write(f"Embeddings type: {st.session_state.embeddings_type}")
                
                # Check pre-built
                has_prebuilt, meta = check_prebuilt_vectors()
                st.write(f"Pre-built vectors: {'Yes' if has_prebuilt else 'No'}")
                if has_prebuilt and meta:
                    st.json(meta)

# ==============================================================================
# RUN APPLICATION
# ==============================================================================

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        if st.session_state.get('debug_mode', False):
            import traceback
            st.code(traceback.format_exc())