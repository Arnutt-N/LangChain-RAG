"""
Gen AI RAG Chatbot with Multiple AI Models Support
Supports: OpenAI, Gemini, Mistral, DeepSeek, Qwen, Kimi
Complete Streamlit Application with Qwen Embeddings
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
from typing import Optional, List, Dict, Any, Union
import fnmatch
from dataclasses import dataclass
from enum import Enum

# Load environment variables
load_dotenv()

# Set page configuration FIRST
st.set_page_config(
    layout="wide", 
    page_title="Gen AI : Multi-Model RAG Chatbot",
    page_icon="🤖",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# MODEL CONFIGURATIONS
# ==============================================================================

class ModelProvider(Enum):
    OPENAI = "openai"
    GEMINI = "gemini"
    MISTRAL = "mistral"
    DEEPSEEK = "deepseek"
    QWEN = "qwen"
    KIMI = "kimi"

@dataclass
class ModelConfig:
    provider: ModelProvider
    model_name: str
    display_name: str
    api_key_env: str
    max_tokens: int = 2048
    temperature: float = 0.1
    requires_special_handling: bool = False

# Model Configurations
MODEL_CONFIGS = {
    ModelProvider.OPENAI: ModelConfig(
        provider=ModelProvider.OPENAI,
        model_name="gpt-4-turbo-preview",
        display_name="🎯 OpenAI GPT-4",
        api_key_env="OPENAI_API_KEY",
        max_tokens=4096
    ),
    ModelProvider.GEMINI: ModelConfig(
        provider=ModelProvider.GEMINI,
        model_name="gemini-1.5-flash-latest",
        display_name="⚡ Google Gemini Flash",
        api_key_env="GOOGLE_API_KEY",
        max_tokens=2048
    ),
    ModelProvider.MISTRAL: ModelConfig(
        provider=ModelProvider.MISTRAL,
        model_name="mistral-large-latest",
        display_name="🌊 Mistral Large",
        api_key_env="MISTRAL_API_KEY",
        max_tokens=2048
    ),
    ModelProvider.DEEPSEEK: ModelConfig(
        provider=ModelProvider.DEEPSEEK,
        model_name="deepseek-chat",
        display_name="🔍 DeepSeek Chat",
        api_key_env="DEEPSEEK_API_KEY",
        max_tokens=4096
    ),
    ModelProvider.QWEN: ModelConfig(
        provider=ModelProvider.QWEN,
        model_name="qwen-max",
        display_name="🐼 Qwen Max",
        api_key_env="QWEN_API_KEY",
        max_tokens=8192,
        requires_special_handling=True
    ),
    ModelProvider.KIMI: ModelConfig(
        provider=ModelProvider.KIMI,
        model_name="moonshot-v1-8k",
        display_name="🌙 Kimi Moonshot",
        api_key_env="KIMI_API_KEY",
        max_tokens=8000,
        requires_special_handling=True
    ),
}

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
    required_packages = {
        'google.generativeai': 'google-generativeai',
        'openai': 'openai',
        'mistralai': 'mistralai',
        'langchain': 'langchain',
        'langchain_community': 'langchain-community',
        'faiss': 'faiss-cpu',
        'huggingface_hub': 'huggingface-hub',
        'requests': 'requests',
    }
    
    for module, package in required_packages.items():
        try:
            __import__(module)
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

# Check dependencies
missing_deps = check_dependencies()
if missing_deps:
    st.error(f"Missing packages: {', '.join(missing_deps)}")
    st.info(f"""
    Please install with:
    ```bash
    pip install {' '.join(missing_deps)}
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
    from langchain.embeddings.base import Embeddings
    from langchain.llms.base import LLM
    import pandas as pd
    import numpy as np
    from huggingface_hub import InferenceClient
    import faiss
    from langchain_community.vectorstores import FAISS
    import requests
    
    # Optional imports with graceful fallback
    try:
        import openai
        from langchain_openai import ChatOpenAI
        OPENAI_AVAILABLE = True
    except ImportError:
        OPENAI_AVAILABLE = False
    
    try:
        from mistralai import Mistral
        MISTRAL_AVAILABLE = True
    except ImportError:
        MISTRAL_AVAILABLE = False
    
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# ==============================================================================
# API KEY MANAGEMENT
# ==============================================================================

def get_api_key(env_name: str) -> Optional[str]:
    """Get API key from environment or secrets"""
    return (
        st.secrets.get(env_name) or 
        os.getenv(env_name)
    )

# Load all API keys
API_KEYS = {
    ModelProvider.OPENAI: get_api_key("OPENAI_API_KEY"),
    ModelProvider.GEMINI: get_api_key("GOOGLE_API_KEY") or get_api_key("GEMINI_API_KEY"),
    ModelProvider.MISTRAL: get_api_key("MISTRAL_API_KEY"),
    ModelProvider.DEEPSEEK: get_api_key("DEEPSEEK_API_KEY"),
    ModelProvider.QWEN: get_api_key("QWEN_API_KEY") or get_api_key("DASHSCOPE_API_KEY"),
    ModelProvider.KIMI: get_api_key("KIMI_API_KEY") or get_api_key("MOONSHOT_API_KEY"),
}

HF_TOKEN = get_api_key("HF_TOKEN") or get_api_key("HUGGINGFACE_TOKEN")

# ==============================================================================
# QWEN EMBEDDINGS CLASS
# ==============================================================================

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
        self.api_key = api_key or HF_TOKEN
        if not self.api_key:
            raise ValueError("HF_TOKEN not found")
        
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
# LLM WRAPPERS FOR DIFFERENT PROVIDERS
# ==============================================================================

class DeepSeekLLM(LLM):
    """DeepSeek LLM wrapper for LangChain"""
    
    def __init__(self, api_key: str, model: str = "deepseek-chat", temperature: float = 0.1, max_tokens: int = 4096):
        super().__init__()
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_url = "https://api.deepseek.com/v1/chat/completions"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content']
        except Exception as e:
            return f"Error calling DeepSeek API: {str(e)}"
    
    @property
    def _llm_type(self) -> str:
        return "deepseek"

class QwenLLM(LLM):
    """Qwen LLM wrapper for LangChain"""
    
    def __init__(self, api_key: str, model: str = "qwen-max", temperature: float = 0.1, max_tokens: int = 8192):
        super().__init__()
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "input": {
                "messages": [{"role": "user", "content": prompt}]
            },
            "parameters": {
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            return result['output']['text']
        except Exception as e:
            return f"Error calling Qwen API: {str(e)}"
    
    @property
    def _llm_type(self) -> str:
        return "qwen"

class KimiLLM(LLM):
    """Kimi/Moonshot LLM wrapper for LangChain"""
    
    def __init__(self, api_key: str, model: str = "moonshot-v1-8k", temperature: float = 0.1, max_tokens: int = 8000):
        super().__init__()
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_url = "https://api.moonshot.cn/v1/chat/completions"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content']
        except Exception as e:
            return f"Error calling Kimi API: {str(e)}"
    
    @property
    def _llm_type(self) -> str:
        return "kimi"

class MistralLLM(LLM):
    """Mistral LLM wrapper for LangChain"""
    
    def __init__(self, api_key: str, model: str = "mistral-large-latest", temperature: float = 0.1, max_tokens: int = 2048):
        super().__init__()
        if not MISTRAL_AVAILABLE:
            raise ImportError("mistralai package not installed")
        self.client = Mistral(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        try:
            response = self.client.chat.complete(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error calling Mistral API: {str(e)}"
    
    @property
    def _llm_type(self) -> str:
        return "mistral"

# ==============================================================================
# LLM FACTORY
# ==============================================================================

def create_llm(provider: ModelProvider, config: ModelConfig) -> Optional[Union[LLM, Any]]:
    """Factory function to create appropriate LLM based on provider"""
    
    api_key = API_KEYS.get(provider)
    if not api_key:
        st.error(f"❌ No API key found for {config.display_name}")
        st.info(f"Please set {config.api_key_env} in environment variables or Streamlit secrets")
        return None
    
    try:
        if provider == ModelProvider.OPENAI:
            if not OPENAI_AVAILABLE:
                st.error("OpenAI package not installed. Run: pip install openai langchain-openai")
                return None
            return ChatOpenAI(
                model=config.model_name,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                openai_api_key=api_key
            )
        
        elif provider == ModelProvider.GEMINI:
            os.environ["GOOGLE_API_KEY"] = api_key
            genai.configure(api_key=api_key)
            return ChatGoogleGenerativeAI(
                model=config.model_name,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                google_api_key=api_key
            )
        
        elif provider == ModelProvider.MISTRAL:
            return MistralLLM(
                api_key=api_key,
                model=config.model_name,
                temperature=config.temperature,
                max_tokens=config.max_tokens
            )
        
        elif provider == ModelProvider.DEEPSEEK:
            return DeepSeekLLM(
                api_key=api_key,
                model=config.model_name,
                temperature=config.temperature,
                max_tokens=config.max_tokens
            )
        
        elif provider == ModelProvider.QWEN:
            return QwenLLM(
                api_key=api_key,
                model=config.model_name,
                temperature=config.temperature,
                max_tokens=config.max_tokens
            )
        
        elif provider == ModelProvider.KIMI:
            return KimiLLM(
                api_key=api_key,
                model=config.model_name,
                temperature=config.temperature,
                max_tokens=config.max_tokens
            )
        
        else:
            st.error(f"Unsupported provider: {provider}")
            return None
            
    except Exception as e:
        st.error(f"Error creating LLM for {config.display_name}: {str(e)}")
        return None

# ==============================================================================
# TRANSLATIONS
# ==============================================================================

translations = {
    "en": {
        "title": "🤖 Multi-Model RAG Chatbot",
        "subtitle": "Supports OpenAI, Gemini, Mistral, DeepSeek, Qwen, Kimi",
        "upload_button": "Upload Additional Documents",
        "ask_placeholder": "Ask a question in any language...",
        "processing": "Processing documents...",
        "thinking": "🧠 Generating response...",
        "language": "Language / ภาษา",
        "clear_chat": "🗑️ Clear Chat",
        "select_model": "Select AI Model",
        "model_status": "Model Status",
        "available_models": "Available Models",
        "missing_api_key": "Missing API Key",
        "embeddings_info": "Embeddings: Qwen3-8B (1536 dims)",
        "using_prebuilt": "✅ Using pre-built vector database",
        "no_models": "❌ No AI models configured. Please add API keys.",
    },
    "th": {
        "title": "🤖 RAG แชทบอทหลายโมเดล",
        "subtitle": "รองรับ OpenAI, Gemini, Mistral, DeepSeek, Qwen, Kimi",
        "upload_button": "อัปโหลดเอกสารเพิ่มเติม",
        "ask_placeholder": "ถามคำถามได้ทุกภาษา...",
        "processing": "กำลังประมวลผลเอกสาร...",
        "thinking": "🧠 กำลังสร้างคำตอบ...",
        "language": "ภาษา / Language",
        "clear_chat": "🗑️ ล้างการแชท",
        "select_model": "เลือกโมเดล AI",
        "model_status": "สถานะโมเดล",
        "available_models": "โมเดลที่ใช้ได้",
        "missing_api_key": "ไม่มี API Key",
        "embeddings_info": "Embeddings: Qwen3-8B (1536 มิติ)",
        "using_prebuilt": "✅ ใช้ vector database ที่สร้างไว้แล้ว",
        "no_models": "❌ ไม่มีโมเดล AI กรุณาเพิ่ม API keys",
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
        "selected_model": ModelProvider.GEMINI,
        "temperature": 0.1,
        "max_tokens": 2048,
        "using_prebuilt": False,
        "prebuilt_metadata": None,
        "embeddings_type": None,
        "embeddings": None,
        "available_models": [],
        "current_llm": None,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# ==============================================================================
# CHECK AVAILABLE MODELS
# ==============================================================================

def check_available_models() -> List[ModelProvider]:
    """Check which models have API keys configured"""
    available = []
    for provider, api_key in API_KEYS.items():
        if api_key:
            # Additional checks for package availability
            if provider == ModelProvider.OPENAI and not OPENAI_AVAILABLE:
                continue
            if provider == ModelProvider.MISTRAL and not MISTRAL_AVAILABLE:
                continue
            available.append(provider)
    return available

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
    
    # Fallback to sentence-transformers
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
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
        qwen_config_path = PREBUILT_VECTORS_DIR / "qwen_config.json"
        use_qwen = qwen_config_path.exists()
        
        embeddings = get_embeddings(use_qwen=use_qwen)
        if not embeddings:
            return None, None
        
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
    supported_extensions = ('.pdf', '.txt', '.csv', '.xlsx', '.xls', '.docx', '.md')
    local_files = []
    
    excluded_files = ['requirements.txt', '.env', 'streamlit_app.py']
    excluded_dirs = ['.git', '__pycache__', 'venv', 'env', '.streamlit', 
                     'vector_cache', 'prebuilt_vectors', 'models']
    
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
    """Setup QA chain with selected model"""
    if not st.session_state.vectorstore:
        return None
    
    try:
        # Get selected model config
        config = MODEL_CONFIGS[st.session_state.selected_model]
        
        # Create LLM
        llm = create_llm(st.session_state.selected_model, config)
        if not llm:
            return None
        
        st.session_state.current_llm = llm
        
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
        
        # Model selection
        st.markdown(f"### 🤖 {t['select_model']}")
        
        available_models = check_available_models()
        st.session_state.available_models = available_models
        
        if available_models:
            model_options = [MODEL_CONFIGS[m].display_name for m in available_models]
            model_values = list(available_models)
            
            # Default selection
            default_index = 0
            if st.session_state.selected_model in available_models:
                default_index = available_models.index(st.session_state.selected_model)
            
            selected_display = st.selectbox(
                "Choose Model",
                options=model_options,
                index=default_index,
                label_visibility="collapsed"
            )
            
            # Update selected model
            selected_index = model_options.index(selected_display)
            st.session_state.selected_model = model_values[selected_index]
            
            # Show model info
            config = MODEL_CONFIGS[st.session_state.selected_model]
            st.info(f"""
            **Model:** {config.model_name}
            **Max Tokens:** {config.max_tokens}
            """)
        else:
            st.error(t["no_models"])
        
        # Model status
        with st.expander(f"📊 {t['model_status']}", expanded=False):
            st.markdown(f"**{t['available_models']}:**")
            for provider in ModelProvider:
                config = MODEL_CONFIGS[provider]
                if API_KEYS.get(provider):
                    st.success(f"✅ {config.display_name}")
                else:
                    st.error(f"❌ {config.display_name}")
                    st.caption(f"Set {config.api_key_env}")
        
        st.divider()
        
        # Debug mode
        st.session_state.debug_mode = st.checkbox("🔍 Debug Mode")
        
        # Display vector info
        if st.session_state.using_prebuilt:
            with st.expander("🚀 Pre-built Vectors", expanded=False):
                if st.session_state.prebuilt_metadata:
                    meta = st.session_state.prebuilt_metadata
                    st.write(f"**Chunks:** {meta.get('total_chunks', 0)}")
                    st.write(f"**Docs:** {len(meta.get('documents', []))}")
        
        # HF Token status
        if HF_TOKEN:
            st.success("✅ HF_TOKEN configured")
            st.caption(t["embeddings_info"])
        else:
            st.warning("⚠️ No HF_TOKEN (using fallback embeddings)")
        
        st.divider()
        
        # File upload
        st.markdown(f"### 📤 {t['upload_button']}")
        uploaded_files = st.file_uploader(
            "Upload",
            accept_multiple_files=True,
            type=['pdf', 'csv', 'txt', 'xlsx', 'xls', 'docx', 'md'],
            label_visibility="collapsed"
        )
        
        if uploaded_files:
            st.warning("Note: Uploaded files will override pre-built vectors")
            if st.button("Process Uploads"):
                st.info("Upload processing not implemented in this version")
        
        # Settings
        with st.expander("⚙️ Settings"):
            config = MODEL_CONFIGS[st.session_state.selected_model]
            
            st.session_state.temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=config.temperature,
                step=0.1
            )
            
            st.session_state.max_tokens = st.slider(
                "Max Tokens",
                min_value=128,
                max_value=config.max_tokens,
                value=min(2048, config.max_tokens),
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
            margin-bottom: 1rem;
        }
        .subtitle {
            text-align: center;
            color: #666;
            font-size: 0.9rem;
            margin-bottom: 2rem;
        }
        .model-badge {
            display: inline-block;
            padding: 0.25rem 0.5rem;
            margin: 0.2rem;
            border-radius: 0.25rem;
            font-size: 0.85rem;
            font-weight: 500;
        }
        .model-available {
            background-color: #d4edda;
            color: #155724;
        }
        .model-unavailable {
            background-color: #f8d7da;
            color: #721c24;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Title
    st.markdown(f'<h1 class="stTitle">{t["title"]}</h1>', unsafe_allow_html=True)
    st.markdown(f'<p class="subtitle">{t["subtitle"]}</p>', unsafe_allow_html=True)
    
    # Display available models as badges
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        available_models = check_available_models()
        badges_html = ""
        for provider in ModelProvider:
            config = MODEL_CONFIGS[provider]
            if provider in available_models:
                badges_html += f'<span class="model-badge model-available">{config.display_name.split()[1]}</span>'
            else:
                badges_html += f'<span class="model-badge model-unavailable">{config.display_name.split()[1]}</span>'
        st.markdown(badges_html, unsafe_allow_html=True)
    
    # Display sidebar
    display_sidebar()
    
    # Auto-load documents
    auto_load_documents()
    
    # Main chat interface
    if st.session_state.documents_processed and st.session_state.document_chunks > 0:
        
        # Check if we have available models
        if not st.session_state.available_models:
            st.error(t["no_models"])
            st.stop()
        
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
                # Check if we need to recreate the chain (model changed)
                if st.session_state.qa_chain is None:
                    st.session_state.qa_chain = setup_qa_chain()
                
                if st.session_state.qa_chain:
                    with st.spinner(t["thinking"]):
                        try:
                            # Show which model is being used
                            config = MODEL_CONFIGS[st.session_state.selected_model]
                            st.caption(f"Using: {config.display_name}")
                            
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
                            # Try to provide more specific error info
                            if "rate limit" in str(e).lower():
                                st.info("💡 Try switching to a different model or wait a moment")
                            elif "api key" in str(e).lower():
                                st.info(f"💡 Check your API key for {config.display_name}")
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
            
            **Available Models:** Check the sidebar for model status
            """)
        
        # Show debug info if enabled
        if st.session_state.debug_mode:
            with st.expander("🔍 Debug Information"):
                st.write(f"Local files: {len(st.session_state.local_files)}")
                st.write(f"Documents processed: {st.session_state.documents_processed}")
                st.write(f"Chunks: {st.session_state.document_chunks}")
                st.write(f"Vector store: {'Yes' if st.session_state.vectorstore else 'No'}")
                st.write(f"HF_TOKEN: {'Yes' if HF_TOKEN else 'No'}")
                st.write(f"Available models: {[m.value for m in st.session_state.available_models]}")
                
                # API Keys status
                st.write("\n**API Keys Status:**")
                for provider, key in API_KEYS.items():
                    st.write(f"- {provider.value}: {'✅' if key else '❌'}")

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