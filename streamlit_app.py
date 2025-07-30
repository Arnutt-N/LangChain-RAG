import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
import time
import hashlib
import json
from typing import Optional

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    layout="wide", 
    page_title="Gen AI : RAG with ChromaDB Cloud",
    page_icon="🤖"
)

# Check dependencies
@st.cache_data
def check_dependencies():
    missing_packages = []
    try:
        import chromadb
        from langchain_community.vectorstores import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as e:
        if "chromadb" in str(e):
            missing_packages.append("chromadb")
        if "langchain" in str(e):
            missing_packages.append("langchain-community langchain-google-genai langchain-huggingface")
    return missing_packages

missing_deps = check_dependencies()
if missing_deps:
    st.error(f"Missing packages: {', '.join(missing_deps)}")
    st.info("Install with: pip install chromadb python-dotenv langchain-community langchain-google-genai langchain-huggingface")
    st.stop()

# Import required packages
import chromadb
from chromadb.api import ClientAPI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, TextLoader, UnstructuredExcelLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configuration
CHROMA_API_KEY = st.secrets.get("CHROMA_API_KEY") or os.getenv("CHROMA_API_KEY")
CHROMA_TENANT = st.secrets.get("CHROMA_TENANT") or os.getenv("CHROMA_TENANT")
CHROMA_DATABASE = st.secrets.get("CHROMA_DATABASE") or os.getenv("CHROMA_DATABASE")
GOOGLE_API_KEY = (
    st.secrets.get("GOOGLE_API_KEY") or 
    st.secrets.get("GEMINI_API_KEY") or 
    os.getenv("GOOGLE_API_KEY") or 
    os.getenv("GEMINI_API_KEY")
)

# Check required credentials
if not CHROMA_API_KEY or not CHROMA_TENANT or not CHROMA_DATABASE:
    st.error("🚨 ChromaDB Cloud credentials not found!")
    st.info("""
    **Setup ChromaDB Cloud:**
    1. Go to https://www.trychroma.com/
    2. Create free account & get API key
    3. Create database and get tenant ID
    4. Add to Streamlit secrets:
       - CHROMA_API_KEY
       - CHROMA_TENANT 
       - CHROMA_DATABASE
    """)
    st.stop()

if not GOOGLE_API_KEY:
    st.error("🚨 Google API Key not found!")
    st.info("Add GOOGLE_API_KEY to Streamlit secrets")
    st.stop()

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# ChromaDB Cloud Connection
_client: Optional[ClientAPI] = None
_collection = None

def get_chroma_client() -> ClientAPI:
    """Get or create ChromaDB Cloud client"""
    global _client
    if _client is None:
        try:
            _client = chromadb.CloudClient(
                api_key=CHROMA_API_KEY,
                tenant=CHROMA_TENANT,
                database=CHROMA_DATABASE
            )
            st.success("✅ Connected to ChromaDB Cloud")
        except Exception as e:
            st.error(f"❌ Failed to connect to ChromaDB Cloud: {e}")
            return None
    return _client

def get_chroma_collection(collection_name="rag_documents"):
    """Get or create ChromaDB collection"""
    global _collection
    if _collection is None:
        client = get_chroma_client()
        if client:
            try:
                _collection = client.get_or_create_collection(
                    name=collection_name,
                    metadata={"description": "RAG documents collection"}
                )
                st.info(f"📂 Using collection: {collection_name}")
            except Exception as e:
                st.error(f"❌ Failed to get collection: {e}")
                return None
    return _collection

# Translations
translations = {
    "en": {
        "title": "🤖 RAG Chatbot with ChromaDB Cloud",
        "upload_button": "Upload Documents",
        "ask_placeholder": "Ask a question in Thai or English...",
        "processing": "Processing documents...",
        "welcome": "👋 Hello! I'm ready to chat about various topics based on the documents.",
        "upload_success": lambda count: f"✅ {count} document(s) uploaded successfully!",
        "thinking": "🧠 Generating response...",
        "language": "🌐 Language / ภาษา",
        "clear_chat": "🗑️ Clear Chat",
        "clear_database": "🗑️ Clear Database",
        "model_info": "🤖 **Model:** Gemini Pro | 📊 **Embedding:** MiniLM-L6 | ☁️ **Vector DB:** ChromaDB Cloud",
        "no_documents": "📄 No documents uploaded yet. Please upload some documents to start chatting!",
        "error_processing": "❌ Error processing documents. Please try again.",
        "error_response": "🚨 Sorry, I encountered an error while generating response.",
        "checking_existing": "🔍 Checking existing documents...",
        "found_existing": "✅ Found existing documents in cloud database",
        "saving_to_db": "☁️ Saving to ChromaDB Cloud...",
        "db_stats": "Cloud Database Statistics",
        "connection_status": "Connection Status",
    },
    "th": {
        "title": "🤖 แชทบอท RAG กับ ChromaDB Cloud",
        "upload_button": "อัปโหลดเอกสาร",
        "ask_placeholder": "ถามคำถามเป็นภาษาไทยหรืออังกฤษ...",
        "processing": "กำลังประมวลผลเอกสาร...",
        "welcome": "👋 สวัสดี! ฉันพร้อมพูดคุยเกี่ยวกับเอกสารต่างๆ",
        "upload_success": lambda count: f"✅ อัปโหลดเอกสาร {count} ฉบับสำเร็จ!",
        "thinking": "🧠 กำลังสร้างคำตอบ...",
        "language": "🌐 ภาษา / Language",
        "clear_chat": "🗑️ ล้างการแชท",
        "clear_database": "🗑️ ล้างฐานข้อมูล",
        "model_info": "🤖 **โมเดล:** Gemini Pro | 📊 **Embedding:** MiniLM-L6 | ☁️ **Vector DB:** ChromaDB Cloud",
        "no_documents": "📄 ยังไม่มีเอกสารอัปโหลด กรุณาอัปโหลดเอกสารเพื่อเริ่มแชท!",
        "error_processing": "❌ เกิดข้อผิดพลาดในการประมวลผลเอกสาร",
        "error_response": "🚨 ขออภัย เกิดข้อผิดพลาดในการสร้างคำตอบ",
        "checking_existing": "🔍 ตรวจสอบเอกสารที่มีอยู่...",
        "found_existing": "✅ พบเอกสารที่มีอยู่ในฐานข้อมูลคลาวด์",
        "saving_to_db": "☁️ บันทึกลง ChromaDB Cloud...",
        "db_stats": "สถิติฐานข้อมูลคลาวด์",
        "connection_status": "สถานะการเชื่อมต่อ",
    }
}

# Initialize session state
def init_session_state():
    defaults = {
        "messages": [],
        "vectorstore": None,
        "language": "en",
        "uploaded_files": [],
        "documents_processed": False,
        "document_chunks": 0,
        "debug_mode": False,
        "last_request_time": 0,
        "app_initialized": False,
        "chroma_connected": False,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# Initialize embeddings
@st.cache_resource
def get_embeddings():
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
            st.error("Embeddings model failed to generate vectors")
            return None
            
    except Exception as e:
        st.error(f"Error initializing embeddings: {e}")
        return None

# Database functions
def get_file_hash(content):
    """Generate hash from file content"""
    if isinstance(content, str):
        content = content.encode('utf-8')
    return hashlib.md5(content).hexdigest()

def check_document_exists(collection, file_hash):
    """Check if document already exists in ChromaDB Cloud"""
    try:
        results = collection.get(
            where={"file_hash": file_hash},
            limit=1
        )
        return len(results['ids']) > 0
    except Exception as e:
        st.warning(f"Error checking document existence: {e}")
        return False

def get_database_stats(collection):
    """Get database statistics from ChromaDB Cloud"""
    try:
        count = collection.count()
        
        # Get unique files
        results = collection.get(include=["metadatas"])
        unique_files = set()
        for metadata in results.get('metadatas', []):
            if metadata and 'source_file' in metadata:
                unique_files.add(metadata['source_file'])
        
        return {
            "documents": len(unique_files),
            "chunks": count,
            "status": "connected"
        }
    except Exception as e:
        st.warning(f"Error getting database stats: {e}")
        return {"documents": 0, "chunks": 0, "status": "error"}

def clear_database(collection):
    """Clear all documents from ChromaDB Cloud"""
    try:
        # Get all IDs and delete them
        all_data = collection.get()
        if all_data['ids']:
            collection.delete(ids=all_data['ids'])
        return True
    except Exception as e:
        st.error(f"Error clearing database: {e}")
        return False

# Document processing functions
def load_single_uploaded_file(uploaded_file):
    """Load a single uploaded file"""
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name
        
        try:
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
            
            for doc in docs:
                if hasattr(doc, 'page_content') and doc.page_content.strip():
                    content = doc.page_content.replace('\n\n', '\n').strip()
                    if len(content) > 50:
                        doc.page_content = content
                        # Add metadata
                        doc.metadata.update({
                            'source_file': uploaded_file.name,
                            'file_hash': get_file_hash(uploaded_file.getvalue()),
                            'upload_time': str(time.time()),
                            'file_size': len(uploaded_file.getvalue())
                        })
                        cleaned_docs.append(doc)
            
            return cleaned_docs
            
        finally:
            try:
                os.unlink(temp_file_path)
            except:
                pass
        
    except Exception as e:
        raise Exception(f"Error loading uploaded file {uploaded_file.name}: {str(e)}")

def process_documents_to_cloud(uploaded_files):
    """Process documents and save to ChromaDB Cloud"""
    t = translations[st.session_state.language]
    
    if not uploaded_files:
        return False
    
    # Get ChromaDB collection
    collection = get_chroma_collection()
    if not collection:
        st.error("Could not connect to ChromaDB Cloud")
        return False
    
    all_documents = []
    new_files = 0
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Process each file
        for i, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Processing {uploaded_file.name}...")
            progress_bar.progress((i + 1) / len(uploaded_files) * 0.6)
            
            # Check if file already exists
            file_hash = get_file_hash(uploaded_file.getvalue())
            
            if check_document_exists(collection, file_hash):
                st.info(f"📄 {uploaded_file.name} already exists in cloud database")
                continue
            
            try:
                docs = load_single_uploaded_file(uploaded_file)
                if docs:
                    all_documents.extend(docs)
                    new_files += 1
                    
            except Exception as e:
                st.warning(f"Could not process {uploaded_file.name}: {e}")
                continue
        
        if not all_documents:
            progress_bar.empty()
            status_text.empty()
            if new_files == 0:
                st.info(t["found_existing"])
                return True
            else:
                st.warning("No new documents to process")
                return False
        
        # Split documents
        status_text.text("🔄 Splitting documents into chunks...")
        progress_bar.progress(0.7)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        texts = text_splitter.split_documents(all_documents)
        
        if not texts:
            progress_bar.empty()
            status_text.empty()
            st.warning("No text content found")
            return False
        
        # Save to ChromaDB Cloud
        status_text.text(f"☁️ {t['saving_to_db']}")
        progress_bar.progress(0.9)
        
        # Prepare data for ChromaDB Cloud
        ids = [f"doc_{i}_{int(time.time())}" for i in range(len(texts))]
        documents = [text.page_content for text in texts]
        metadatas = [text.metadata for text in texts]
        
        # Add documents to ChromaDB Cloud
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        
        st.session_state.document_chunks = len(texts)
        st.session_state.documents_processed = True
        st.session_state.chroma_connected = True
        
        progress_bar.progress(1.0)
        progress_bar.empty()
        status_text.empty()
        
        return True
        
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"Error processing documents: {e}")
        return False

# Rate limiting
def check_rate_limit():
    current_time = time.time()
    time_since_last = current_time - st.session_state.last_request_time
    min_interval = 3
    
    if time_since_last < min_interval:
        wait_time = min_interval - time_since_last
        time.sleep(wait_time)
    
    st.session_state.last_request_time = time.time()

# Custom ChromaDB retriever for LangChain
class ChromaCloudRetriever:
    """Custom retriever for ChromaDB Cloud"""
    
    def __init__(self, collection, embeddings, k=3):
        self.collection = collection
        self.embeddings = embeddings
        self.k = k
    
    def get_relevant_documents(self, query):
        """Get relevant documents from ChromaDB Cloud"""
        try:
            # Generate query embedding
            query_embedding = self.embeddings.embed_query(query)
            
            # Query ChromaDB Cloud
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=self.k,
                include=["documents", "metadatas"]
            )
            
            # Convert to LangChain Document format
            documents = []
            for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
                from langchain.schema import Document
                documents.append(Document(page_content=doc, metadata=metadata or {}))
            
            return documents
            
        except Exception as e:
            st.error(f"Error retrieving documents: {e}")
            return []

# Query functions
def setup_retrieval_chain():
    """Setup retrieval chain with ChromaDB Cloud"""
    try:
        collection = get_chroma_collection()
        embeddings = get_embeddings()
        
        if not collection or not embeddings:
            return None
        
        # Create custom retriever
        retriever = ChromaCloudRetriever(collection, embeddings, k=3)
        
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=0.1,
            max_tokens=512,
            google_api_key=GOOGLE_API_KEY,
            request_timeout=30,
            max_retries=1,
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

def query_with_timeout(chain, question, timeout=30):
    try:
        check_rate_limit()
        response = chain.invoke({"question": question})
        return response
    except Exception as e:
        error_msg = str(e).lower()
        if "quota" in error_msg or "rate" in error_msg or "429" in error_msg:
            raise Exception("RATE_LIMIT")
        elif "timeout" in error_msg or "timed out" in error_msg:
            raise TimeoutError("Request timed out")
        else:
            raise e

# Main application
def main():
    try:
        if "language" in st.query_params:
            st.session_state.language = st.query_params["language"]

        t = translations[st.session_state.language]

        if not st.session_state.app_initialized:
            st.session_state.app_initialized = True

        # Custom CSS
        st.markdown("""
            <style>
            .main .block-container {
                max-width: 1200px;
                padding-top: 2rem;
                padding-bottom: 2rem;
            }
            .stTitle {
                text-align: center;
                color: #1f77b4;
            }
            .connection-status {
                background-color: #e8f5e8;
                border: 1px solid #4caf50;
                border-radius: 5px;
                padding: 10px;
                margin: 10px 0;
                font-size: 14px;
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
            }
            .debug-info {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 5px;
                padding: 10px;
                margin: 10px 0;
                font-size: 12px;
            }
            </style>
        """, unsafe_allow_html=True)

        # Sidebar
        with st.sidebar:
            # Language selection
            st.markdown(f"**{t['language']}**")
            selected_lang = st.selectbox(
                "Select Language",
                options=["ไทย", "English"],
                index=1 if st.session_state.language == "en" else 0,
                key="language_selection",
                label_visibility="collapsed"
            )
            
            new_language = "th" if selected_lang == "ไทย" else "en"
            if new_language != st.session_state.language:
                st.session_state.language = new_language

            st.markdown("---")

            # Connection status
            st.markdown(f"**{t['connection_status']}**")
            client = get_chroma_client()
            if client:
                st.markdown('<div class="connection-status">🟢 Connected to ChromaDB Cloud</div>', unsafe_allow_html=True)
                st.session_state.chroma_connected = True
            else:
                st.markdown('<div class="connection-status">🔴 Connection Failed</div>', unsafe_allow_html=True)
                st.session_state.chroma_connected = False

            # Debug mode toggle
            st.session_state.debug_mode = st.checkbox("🔍 Debug Mode", value=st.session_state.debug_mode)

            # File uploader
            st.markdown(f"**{t['upload_button']}**")
            uploaded_files = st.file_uploader(
                "Upload Documents",
                accept_multiple_files=True,
                type=['pdf', 'csv', 'txt', 'xlsx', 'xls'],
                key="file_uploader",
                label_visibility="collapsed",
                disabled=not st.session_state.chroma_connected
            )

            # Process uploaded files
            if uploaded_files and uploaded_files != st.session_state.uploaded_files and st.session_state.chroma_connected:
                st.session_state.uploaded_files = uploaded_files
                with st.spinner(t["processing"]):
                    success = process_documents_to_cloud(uploaded_files)
                    if success:
                        st.success(t["upload_success"](len(uploaded_files)))
                    else:
                        st.error(t["error_processing"])

            st.markdown("---")

            # Database statistics
            if st.session_state.debug_mode and st.session_state.chroma_connected:
                st.markdown(f"**{t['db_stats']}**")
                collection = get_chroma_collection()
                if collection:
                    db_stats = get_database_stats(collection)
                    st.write(f"📄 Documents: {db_stats['documents']}")
                    st.write(f"🔢 Chunks: {db_stats['chunks']}")
                    st.write(f"🔗 Status: {db_stats['status']}")
                    
                    # Show ChromaDB Cloud info
                    st.markdown("**Cloud Info:**")
                    st.write(f"🏢 Tenant: {CHROMA_TENANT[:8]}...")
                    st.write(f"💾 Database: {CHROMA_DATABASE}")

            # Clear buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button(t["clear_chat"], use_container_width=True):
                    st.session_state.messages = []
                    st.success("✅ Chat cleared!")
            
            with col2:
                if st.button(t["clear_database"], use_container_width=True, disabled=not st.session_state.chroma_connected):
                    collection = get_chroma_collection()
                    if collection and clear_database(collection):
                        st.session_state.documents_processed = False
                        st.session_state.uploaded_files = []
                        st.success("✅ Cloud database cleared!")
                    else:
                        st.error("❌ Failed to clear database")

        # Main content
        st.title(t["title"])
        st.info(t["model_info"])
        
        if st.session_state.chroma_connected:
            st.success("🌐 Connected to ChromaDB Cloud - your documents are stored securely in the cloud!")
        else:
            st.error("❌ Could not connect to ChromaDB Cloud. Please check your credentials.")

        # Check for existing documents
        if st.session_state.chroma_connected and not st.session_state.documents_processed:
            collection = get_chroma_collection()
            if collection:
                stats = get_database_stats(collection)
                if stats["chunks"] > 0:
                    st.session_state.documents_processed = True
                    st.info(t["found_existing"])

        # Chat interface
        if st.session_state.chroma_connected and st.session_state.documents_processed:
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
                    retrieval_chain = setup_retrieval_chain()
                    if retrieval_chain:
                        with st.spinner(t["thinking"]):
                            try:
                                enhanced_prompt = f"Based on the provided documents, please answer: {prompt}"
                                response = query_with_timeout(retrieval_chain, enhanced_prompt)
                                answer = response.get('answer', 'No answer generated')
                                
                                st.markdown(answer)
                                st.session_state.messages.append({"role": "assistant", "content": answer})
                                
                                # Show sources
                                if 'source_documents' in response and response['source_documents']:
                                    with st.expander("📚 Sources"):
                                        for i, doc in enumerate(response['source_documents']):
                                            st.markdown(f"**Source {i+1}:**")
                                            content = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                                            st.markdown(content)
                                            if hasattr(doc, 'metadata') and doc.metadata:
                                                st.caption(f"File: {doc.metadata.get('source_file', 'Unknown')}")
                                            st.markdown("---")
                                
                            except TimeoutError:
                                st.error("⏱️ Request timed out. Please try asking a shorter question.")
                            except Exception as e:
                                if "RATE_LIMIT" in str(e):
                                    st.error("⏳ API rate limit reached. Please wait a moment and try again.")
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
                    **How to use this ChromaDB Cloud RAG Chatbot:**
                    1. ☁️ **Cloud Connection**: Automatically connects to your ChromaDB Cloud database
                    2. 📁 **Upload Documents**: Use the sidebar to upload PDF, TXT, CSV, or XLSX files
                    3. 🌐 **Cloud Storage**: Documents are stored securely in ChromaDB Cloud
                    4. 🚀 **Global Access**: Access your documents from anywhere with internet
                    5. 💬 **Start Chatting**: Ask questions about your documents
                    6. 🔍 **Debug Mode**: See database statistics and debug information
                    
                    **ChromaDB Cloud Benefits:**
                    - ✅ **Managed service** - no infrastructure to manage
                    - ✅ **Global availability** - access from anywhere
                    - ✅ **Auto-scaling** - handles traffic spikes automatically
                    - ✅ **Secure** - enterprise-grade security
                    - ✅ **Fast** - optimized for vector operations
                    - ✅ **Duplicate detection** - same files won't be processed twice
                    
                    **Setup Requirements:**
                    - ChromaDB Cloud account
                    - API key, tenant ID, and database name
                    - Add credentials to Streamlit secrets
                    
                    **Supported File Types:**
                    - 📄 PDF files
                    - 📝 Text files (.txt)
                    - 📊 CSV files  
                    - 📈 Excel files (.xlsx, .xls)
                    """)
                else:
                    st.markdown("""
                    **วิธีใช้งาน ChromaDB Cloud RAG Chatbot:**
                    1. ☁️ **เชื่อมต่อคลาวด์**: เชื่อมต่อฐานข้อมูล ChromaDB Cloud อัตโนมัติ
                    2. 📁 **อัปโหลดเอกสาร**: ใช้แถบด้านข้างอัปโหลดไฟล์ PDF, TXT, CSV หรือ XLSX
                    3. 🌐 **เก็บข้อมูลคลาวด์**: เอกสารถูกเก็บอย่างปลอดภัยใน ChromaDB Cloud
                    4. 🚀 **เข้าถึงจากทุกที่**: เข้าถึงเอกสารจากทุกที่ที่มีอินเทอร์เน็ต
                    5. 💬 **เริ่มแชท**: ถามคำถามเกี่ยวกับเอกสาร
                    6. 🔍 **โหมดดีบัก**: ดูสถิติฐานข้อมูลและข้อมูลดีบัก
                    
                    **ข้อดี ChromaDB Cloud:**
                    - ✅ **บริการที่ได้รับการจัดการ** - ไม่ต้องจัดการ infrastructure
                    - ✅ **พร้อมใช้งานทั่วโลก** - เข้าถึงได้จากทุกที่
                    - ✅ **ปรับขนาดอัตโนมัติ** - รองรับการใช้งานที่เพิ่มขึ้น
                    - ✅ **ปลอดภัย** - ความปลอดภัยระดับองค์กร
                    - ✅ **เร็ว** - เหมาะสำหรับการดำเนินการเวกเตอร์
                    - ✅ **ตรวจจับไฟล์ซ้ำ** - ไฟล์เดิมไม่ถูกประมวลผลซ้ำ
                    
                    **ข้อกำหนดการตั้งค่า:**
                    - บัญชี ChromaDB Cloud
                    - API key, tenant ID และชื่อฐานข้อมูล
                    - เพิ่ม credentials ใน Streamlit secrets
                    
                    **ประเภทไฟล์ที่รองรับ:**
                    - 📄 ไฟล์ PDF
                    - 📝 ไฟล์ข้อความ (.txt)
                    - 📊 ไฟล์ CSV
                    - 📈 ไฟล์ Excel (.xlsx, .xls)
                    """)

            if not uploaded_files and st.session_state.chroma_connected:
                st.info(t["no_documents"])
            elif not st.session_state.chroma_connected:
                st.warning("⚠️ Please check your ChromaDB Cloud credentials to start using the chatbot.")

        # Footer
        st.markdown(
            '<div class="footer">Created by Arnutt Noitumyae, 2024 | ChromaDB Cloud + Gemini AI</div>',
            unsafe_allow_html=True
        )
        
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        if st.session_state.debug_mode:
            st.code(str(e))

if __name__ == "__main__":
    main()
