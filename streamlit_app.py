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
    page_title="Advanced RAG Chatbot",
    page_icon="🤖",
    initial_sidebar_state="expanded"
)

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
    
    # Optional Mistral import
    try:
        from mistralai import Mistral
        MISTRAL_AVAILABLE = True
    except ImportError:
        MISTRAL_AVAILABLE = False
    
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

# Initialize session state
def init_session_state():
    defaults = {
        "messages": [],
        "vectorstore": None,
        "language": "en",
        "local_files": [],
        "documents_processed": False,
        "document_chunks": 0,
        "debug_mode": False,
        "selected_model": "gemini",
        "similarity_threshold": 0.7,
        "max_tokens": 512,
        "temperature": 0.1,
        "system_ready": False,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# Simple file scanner
def scan_for_documents():
    """Simple document scanner"""
    supported_extensions = ('.pdf', '.txt', '.csv', '.xlsx', '.xls', '.docx')
    found_files = []
    
    try:
        # Check current directory only
        for file in os.listdir('.'):
            if file.lower().endswith(supported_extensions) and not file.startswith('.'):
                found_files.append(file)
        
        # If no files found, create a test file
        if not found_files:
            test_content = """# Welcome to RAG Chatbot

This is a test document created automatically.

## Features
- Document processing and analysis
- Question answering with context
- Vector-based semantic search

## How to use
1. Upload your documents or place them in the repository
2. Ask questions about the content
3. Get accurate answers with source citations

## Supported formats
- PDF files
- Text files (.txt)
- CSV files
- Excel files (.xlsx, .xls)
- Word documents (.docx)

You can replace this file with your own documents!
"""
            try:
                with open('README_RAG.txt', 'w', encoding='utf-8') as f:
                    f.write(test_content)
                found_files.append('README_RAG.txt')
                st.info("📄 Created demo document: README_RAG.txt")
            except Exception as e:
                st.warning(f"Could not create demo file: {e}")
        
    except Exception as e:
        st.error(f"Error scanning files: {e}")
    
    return found_files

# Simple document loader
@st.cache_data
def load_documents(file_list):
    """Load and process documents"""
    all_docs = []
    
    for filename in file_list:
        try:
            file_path = Path(filename)
            if not file_path.exists():
                continue
                
            # Determine loader based on extension
            if filename.endswith('.pdf'):
                loader = PyPDFLoader(str(file_path))
            elif filename.endswith('.csv'):
                loader = CSVLoader(str(file_path))
            elif filename.endswith('.txt'):
                loader = TextLoader(str(file_path), encoding='utf-8')
            elif filename.endswith(('.xlsx', '.xls')):
                loader = UnstructuredExcelLoader(str(file_path))
            else:
                continue
            
            docs = loader.load()
            for doc in docs:
                doc.metadata['source_file'] = filename
                doc.metadata['file_type'] = 'local'
            all_docs.extend(docs)
            
        except Exception as e:
            st.warning(f"Could not load {filename}: {e}")
    
    return all_docs

# Create vector store
@st.cache_resource
def create_vector_store(documents):
    """Create vector store from documents"""
    if not documents:
        return None, 0
    
    try:
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        
        if not chunks:
            return None, 0
        
        # Create embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Create vector store
        vectorstore = FAISS.from_documents(chunks, embeddings)
        
        return vectorstore, len(chunks)
        
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None, 0

# Custom Mistral wrapper
class MistralLLM:
    def __init__(self, api_key: str, model: str = "mistral-large-latest", temperature: float = 0.1, max_tokens: int = 512):
        self.client = Mistral(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def invoke(self, messages):
        try:
            if hasattr(messages, 'content'):
                mistral_messages = [{"role": "user", "content": messages.content}]
            elif isinstance(messages, str):
                mistral_messages = [{"role": "user", "content": messages}]
            else:
                mistral_messages = [{"role": "user", "content": str(messages)}]
            
            response = self.client.chat.complete(
                model=self.model,
                messages=mistral_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            class Response:
                def __init__(self, content):
                    self.content = content
            
            return Response(response.choices[0].message.content)
            
        except Exception as e:
            return Response(f"Error: {e}")

# Setup QA chain
def setup_qa_chain():
    """Setup QA chain"""
    if not st.session_state.vectorstore:
        return None
    
    try:
        # Create retriever
        retriever = st.session_state.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        
        # Create memory
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # Select model
        if st.session_state.selected_model == "mistral" and MISTRAL_API_KEY and MISTRAL_AVAILABLE:
            llm = MistralLLM(api_key=MISTRAL_API_KEY)
        elif GOOGLE_API_KEY:
            llm = ChatGoogleGenerativeAI(
                model="gemini-pro",
                temperature=st.session_state.temperature,
                max_tokens=st.session_state.max_tokens,
                google_api_key=GOOGLE_API_KEY,
            )
        else:
            return None
        
        # Create chain
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

def initialize_system():
    """Initialize the RAG system"""
    if st.session_state.system_ready:
        return
    
    with st.spinner("🚀 Initializing RAG system..."):
        # Scan for documents
        found_files = scan_for_documents()
        st.session_state.local_files = found_files
        
        if found_files:
            st.success(f"📁 Found {len(found_files)} files: {', '.join(found_files)}")
            
            # Load documents
            documents = load_documents(found_files)
            
            if documents:
                st.success(f"📄 Loaded {len(documents)} documents")
                
                # Create vector store
                vectorstore, chunks = create_vector_store(documents)
                
                if vectorstore:
                    st.session_state.vectorstore = vectorstore
                    st.session_state.document_chunks = chunks
                    st.session_state.documents_processed = True
                    st.session_state.system_ready = True
                    
                    st.success(f"✅ System ready! Created {chunks} chunks for search.")
                else:
                    st.error("❌ Failed to create vector store")
            else:
                st.error("❌ No documents could be loaded")
        else:
            st.warning("📁 No documents found")

# Main app
def main():
    # CSS
    st.markdown("""
        <style>
        .main .block-container {
            max-width: 1200px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }
        .status-container {
            background: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
        }
        .chat-container {
            background: #fafafa;
            border-radius: 0.5rem;
            padding: 1rem;
            margin: 1rem 0;
            min-height: 400px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Title
    st.title("🤖 Advanced RAG Chatbot")
    
    # Show Mistral availability
    if not MISTRAL_AVAILABLE:
        st.info("💡 Mistral AI not available. Install with: `pip install mistralai`")
    
    # Model info
    model_text = "Gemini Pro"
    if MISTRAL_API_KEY and MISTRAL_AVAILABLE:
        model_text += " / Mistral Large"
    
    st.info(f"🤖 **Model:** {model_text} | 📊 **Embedding:** MiniLM-L6-v2 | 🗃️ **Vector DB:** FAISS")
    
    # Initialize system on first run
    if not st.session_state.system_ready:
        initialize_system()
    
    # Status display
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.session_state.local_files:
            st.success(f"📁 {len(st.session_state.local_files)} files found")
        else:
            st.info("📁 No files")
    
    with col2:
        if st.session_state.documents_processed and st.session_state.vectorstore:
            st.success(f"🗃️ Vector DB ready ({st.session_state.document_chunks} chunks)")
        else:
            st.warning("🗃️ Vector DB not ready")
    
    with col3:
        if st.session_state.system_ready:
            st.success("✅ System ready")
        else:
            st.info("⏳ Loading...")
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🌐 Language")
        language = st.selectbox("Select", ["English", "ไทย"], label_visibility="collapsed")
        
        st.markdown("---")
        
        # Model selection
        if GOOGLE_API_KEY and MISTRAL_API_KEY and MISTRAL_AVAILABLE:
            st.markdown("### 🤖 AI Model")
            model_choice = st.selectbox("Choose Model", ["Gemini Pro", "Mistral Large"], label_visibility="collapsed")
            st.session_state.selected_model = "mistral" if model_choice == "Mistral Large" else "gemini"
        
        st.markdown("---")
        
        # Debug mode
        st.session_state.debug_mode = st.checkbox("🔍 Debug Mode")
        
        # Reload button
        if st.button("🔄 Reload System", use_container_width=True):
            # Clear everything
            for key in ["system_ready", "documents_processed", "vectorstore", "local_files", "document_chunks", "messages"]:
                if key in st.session_state:
                    del st.session_state[key]
            
            # Clear caches
            st.cache_data.clear()
            st.cache_resource.clear()
            
            st.success("🔄 System cleared! Reloading...")
            time.sleep(1)
            st.rerun()
        
        # Clear chat
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.success("✅ Chat cleared!")
        
        # File list
        if st.session_state.local_files:
            st.markdown("### 📁 Files")
            for file in st.session_state.local_files:
                st.text(f"📄 {file}")
        
        # Debug info
        if st.session_state.debug_mode:
            st.markdown("### 🔍 Debug")
            st.json({
                "files_found": len(st.session_state.local_files),
                "documents_processed": st.session_state.documents_processed,
                "chunks": st.session_state.document_chunks,
                "system_ready": st.session_state.system_ready,
                "messages": len(st.session_state.messages)
            })
    
    # Main content
    if st.session_state.system_ready:
        # Chat interface
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        # Display messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("⏳ System is initializing. Please wait...")
    
    # Single chat input at the bottom
    if st.session_state.system_ready:
        placeholder = "Ask a question about your documents..."
    else:
        placeholder = "System is loading, please wait..."
    
    # SINGLE CHAT INPUT
    if prompt := st.chat_input(placeholder):
        if not st.session_state.system_ready:
            st.error("❌ System not ready yet. Please wait for initialization to complete.")
            st.rerun()
        
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Get QA chain
        qa_chain = setup_qa_chain()
        
        if qa_chain:
            try:
                # Get response
                with st.spinner("🤔 Thinking..."):
                    response = qa_chain.invoke({"question": prompt})
                    answer = response.get('answer', 'No response generated')
                
                # Add assistant message
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
                # Show sources
                if 'source_documents' in response and response['source_documents']:
                    sources_text = f"\n\n**Sources:**\n"
                    for i, doc in enumerate(response['source_documents']):
                        sources_text += f"{i+1}. {doc.metadata.get('source_file', 'Unknown')}\n"
                    st.session_state.messages[-1]["content"] += sources_text
                
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
        else:
            st.session_state.messages.append({"role": "assistant", "content": "Sorry, the system is not available."})
        
        # Rerun to show new messages
        st.rerun()

if __name__ == "__main__":
    main()