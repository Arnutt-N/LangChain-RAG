import streamlit as st
import os
import fnmatch
import io
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Set page configuration FIRST - before any other streamlit commands
st.set_page_config(
    layout="wide", 
    page_title="Gen AI : RAG Chatbot with Documents",
    page_icon="🤖",
    initial_sidebar_state="expanded"
)

# Fixed imports with better error handling
@st.cache_data
def check_dependencies():
    """Check if all required packages are available"""
    missing_packages = []
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
        except ImportError:
            missing_packages.append("langchain-huggingface or langchain-community")
    
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError:
        missing_packages.append("langchain-google-genai")
    
    return missing_packages

# Check dependencies early
missing_deps = check_dependencies()
if missing_deps:
    st.error(f"Missing packages: {', '.join(missing_deps)}")
    st.info("Please install the required packages and restart the app.")
    st.stop()

# Now import everything
try:
    # Use updated HuggingFace embeddings to avoid deprecation
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.memory import ConversationBufferMemory
    from langchain.chains import ConversationalRetrievalChain
    
    # Document loaders - ใช้ community imports
    from langchain_community.document_loaders import (
        PyPDFLoader, 
        CSVLoader, 
        TextLoader, 
        UnstructuredExcelLoader
    )
    
    # Text splitter - try multiple import paths
    try:
        from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
    except ImportError:
        try:
            from langchain.text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
        except ImportError:
            from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
    
    # Use FAISS instead of ChromaDB for better compatibility
    from langchain_community.vectorstores import FAISS
    import tempfile
    
except ImportError as e:
    st.error(f"Import error: {e}")
    st.info("Some required packages are missing. Please check your requirements.txt file.")
    st.stop()

# Early API key check
api_key = (
    st.secrets.get("GOOGLE_API_KEY") or 
    st.secrets.get("GEMINI_API_KEY") or 
    os.getenv("GOOGLE_API_KEY") or 
    os.getenv("GEMINI_API_KEY")
)

if api_key is None:
    st.error("🚨 GOOGLE_API_KEY or GEMINI_API_KEY is not set. Please set it in the Streamlit Cloud secrets or environment variables.")
    st.info("👉 You can get a free API key from: https://makersuite.google.com/app/apikey")
    st.stop()
else:
    os.environ["GOOGLE_API_KEY"] = api_key

# Translations (unchanged)
translations = {
    "en": {
        "title": "🤖 Gen AI : RAG Chatbot with Documents",
        "upload_button": "Upload Documents",
        "browse_files": "Browse files",
        "ask_placeholder": "Ask a question in Thai or English...",
        "processing": "Processing documents...",
        "welcome": "👋 Hello! I'm ready to chat about various topics based on the documents. How can I assist you today?",
        "upload_success": lambda count: f"✅ {count} new document(s) uploaded and processed successfully!",
        "local_knowledge": "📚 My Documents",
        "thinking": "🧠 Generating response...",
        "language": "🌐 Language / ภาษา",
        "clear_chat": "🗑️ Clear Chat",
        "model_info": "🤖 **Model:** Gemini Pro | 📊 **Embedding:** MiniLM-L6 | 🗃️ **Vector DB:** FAISS",
        "no_documents": "📄 No documents uploaded yet. Please upload some documents to start chatting!",
        "error_processing": "❌ Error processing documents. Please try again.",
        "error_response": "🚨 Sorry, I encountered an error while generating response.",
        "error_setup": "⚠️ Sorry, I couldn't set up the retrieval system.",
        "debug_info": "🔍 Debug Info",
        "chunk_count": "Document chunks",
        "retrieval_results": "Retrieved documents",
        "rate_limit_error": "⏳ API rate limit reached. Please wait a moment and try again.",
        "timeout_error": "⏱️ Request timed out. Please try asking a shorter question.",
        "initializing": "🔧 Initializing system...",
        "loading_embeddings": "📥 Loading embeddings model...",
    },
    "th": {
        "title": "🤖 Gen AI : แชทบอทเอกสาร RAG",
        "upload_button": "อัปโหลดเอกสาร",
        "browse_files": "เลือกไฟล์",
        "ask_placeholder": "ถามคำถามเป็นภาษาไทยหรืออังกฤษ...",
        "processing": "กำลังประมวลผลเอกสาร...",
        "welcome": "👋 สวัสดี! ฉันพร้อมที่จะพูดคุยเกี่ยวกับหัวข้อต่างๆ ตามเอกสารที่มี วันนี้ฉันจะช่วยคุณอย่างไรดี?",
        "upload_success": lambda count: f"✅ อัปโหลดและประมวลผลเอกสารใหม่ {count} ฉบับสำเร็จแล้ว!",
        "local_knowledge": "📚 คลังข้อมูลของฉัน",
        "thinking": "🧠 กำลังสร้างคำตอบ...",
        "language": "🌐 ภาษา / Language",
        "clear_chat": "🗑️ ล้างการแชท",
        "model_info": "🤖 **โมเดล:** Gemini Pro | 📊 **Embedding:** MiniLM-L6 | 🗃️ **Vector DB:** FAISS",
        "no_documents": "📄 ยังไม่มีเอกสารอัปโหลด กรุณาอัปโหลดเอกสารเพื่อเริ่มแชท!",
        "error_processing": "❌ เกิดข้อผิดพลาดในการประมวลผลเอกสาร กรุณาลองใหม่อีกครั้ง",
        "error_response": "🚨 ขออภัย เกิดข้อผิดพลาดในการสร้างคำตอบ",
        "error_setup": "⚠️ ขออภัย ไม่สามารถตั้งค่าระบบค้นหาได้",
        "debug_info": "🔍 ข้อมูลดีบัก",
        "chunk_count": "ส่วนของเอกสาร",
        "retrieval_results": "เอกสารที่ค้นพบ",
        "rate_limit_error": "⏳ ถึงขีดจำกัดการใช้งาน API กรุณารอสักครู่แล้วลองใหม่",
        "timeout_error": "⏱️ คำขอใช้เวลานานเกินไป กรุณาลองถามคำถามสั้นๆ",
        "initializing": "🔧 กำลังเตรียมระบบ...",
        "loading_embeddings": "📥 กำลังโหลดโมเดล embeddings...",
    }
}

# Initialize session state
def init_session_state():
    """Initialize session state variables"""
    defaults = {
        "messages": [],
        "vectorstore": None,
        "local_files": [],
        "language": "en",
        "uploaded_files": [],
        "documents_processed": False,
        "document_chunks": 0,
        "debug_mode": False,
        "last_request_time": 0,
        "embeddings_loaded": False,
        "embeddings_model": None,
        "app_initialized": False,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Call initialization
init_session_state()

# Show loading message if not initialized
if not st.session_state.app_initialized:
    t = translations[st.session_state.language]
    loading_placeholder = st.empty()
    loading_placeholder.info(f"{t['initializing']} กำลังเตรียมระบบ...")

# Function to load .gitignore patterns
@st.cache_data
def load_gitignore():
    patterns = []
    try:
        if os.path.exists('.gitignore'):
            encodings = ['utf-8', 'cp874', 'tis-620', 'windows-1252', 'latin-1']
            for encoding in encodings:
                try:
                    with open('.gitignore', 'r', encoding=encoding) as file:
                        patterns = file.read().splitlines()
                    break
                except UnicodeDecodeError:
                    continue
    except Exception as e:
        st.warning(f"Could not load .gitignore: {str(e)}")
    return patterns

# Function to check if a file should be ignored
def should_ignore(filename, patterns):
    if filename == 'requirements.txt':
        return True
    for pattern in patterns:
        if fnmatch.fnmatch(filename, pattern):
            return True
    return False

# Rate limiting function
def check_rate_limit():
    """Check if we should wait before making another request"""
    current_time = time.time()
    time_since_last = current_time - st.session_state.last_request_time
    min_interval = 3  # Minimum 3 seconds between requests
    
    if time_since_last < min_interval:
        wait_time = min_interval - time_since_last
        time.sleep(wait_time)
    
    st.session_state.last_request_time = time.time()

# Initialize optimized embeddings with better caching and error handling
@st.cache_resource(show_spinner=False)
def get_embeddings():
    """Initialize and cache embeddings - optimized for speed"""
    try:
        # Use minimal configuration to avoid parameter conflicts
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Test the embeddings with a simple query
        test_embedding = embeddings.embed_query("test")
        if len(test_embedding) > 0:
            st.session_state.embeddings_loaded = True
            return embeddings
        else:
            st.error("Embeddings model failed to generate vectors")
            return None
            
    except Exception as e:
        st.error(f"Error initializing embeddings: {str(e)}")
        if st.session_state.debug_mode:
            st.code(str(e))
        return None

# Optimized document loading with better timeout handling
def load_documents(file_paths, uploaded_files):
    documents = []
    processed_count = 0
    max_files = 10  # Limit number of files to prevent timeout
    
    # Combine and limit files
    all_files = list(file_paths) + [(f, True) for f in uploaded_files]
    if len(all_files) > max_files:
        st.warning(f"Too many files ({len(all_files)}). Processing first {max_files} files.")
        all_files = all_files[:max_files]
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        for i, file_info in enumerate(all_files):
            if isinstance(file_info, tuple):
                uploaded_file, is_uploaded = file_info
                file_name = uploaded_file.name
            else:
                file_path = file_info
                file_name = file_path
                is_uploaded = False
            
            if file_name == 'requirements.txt':
                continue
                
            status_text.text(f"Processing {file_name}...")
            progress_bar.progress((i + 1) / len(all_files))
            
            try:
                if is_uploaded:
                    docs = load_single_uploaded_file(uploaded_file)
                else:
                    docs = load_single_local_file(file_path)
                
                if docs:
                    documents.extend(docs)
                    processed_count += 1
                    
            except Exception as e:
                st.warning(f"Could not load file {file_name}: {str(e)}")
                continue
                
            # Prevent timeout by processing in chunks
            if processed_count >= 5:
                time.sleep(0.1)  # Small delay to prevent blocking
        
        progress_bar.empty()
        status_text.empty()
        
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"Error during document loading: {str(e)}")
    
    return documents

def load_single_local_file(file_path):
    """Load a single local file"""
    try:
        file_extension = file_path.split('.')[-1].lower()
        
        if file_extension == 'pdf':
            loader = PyPDFLoader(file_path)
        elif file_extension == 'csv':
            loader = CSVLoader(file_path)
        elif file_extension == 'txt':
            loader = TextLoader(file_path, encoding='utf-8')
        elif file_extension in ['xlsx', 'xls']:
            loader = UnstructuredExcelLoader(file_path)
        else:
            return []
        
        docs = loader.load()
        cleaned_docs = []
        
        for doc in docs:
            if hasattr(doc, 'page_content') and doc.page_content.strip():
                # Clean the content
                content = doc.page_content.replace('\n\n', '\n').strip()
                if len(content) > 50:  # Only include meaningful content
                    doc.page_content = content
                    cleaned_docs.append(doc)
        
        return cleaned_docs
        
    except Exception as e:
        raise Exception(f"Error loading {file_path}: {str(e)}")

def load_single_uploaded_file(uploaded_file):
    """Load a single uploaded file"""
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        # Create temporary file
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
                    # Clean the content
                    content = doc.page_content.replace('\n\n', '\n').strip()
                    if len(content) > 50:  # Only include meaningful content
                        doc.page_content = content
                        cleaned_docs.append(doc)
            
            return cleaned_docs
            
        finally:
            # Always clean up temporary file
            try:
                os.unlink(temp_file_path)
            except:
                pass
        
    except Exception as e:
        raise Exception(f"Error loading uploaded file {uploaded_file.name}: {str(e)}")

# Optimized document processing with progress tracking
def process_documents(documents):
    if not documents:
        return None

    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Text splitting
        status_text.text("🔄 Splitting documents into chunks...")
        progress_bar.progress(0.2)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,  # Smaller chunks for better retrieval
            chunk_overlap=50,  # Reduced overlap
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )
        texts = text_splitter.split_documents(documents)
        
        if not texts:
            progress_bar.empty()
            status_text.empty()
            st.warning("No text content found in documents")
            return None
        
        # Store chunk count
        st.session_state.document_chunks = len(texts)
        
        # Step 2: Get embeddings
        status_text.text("📊 Loading embeddings model...")
        progress_bar.progress(0.4)
        
        embeddings = get_embeddings()
        if not embeddings:
            progress_bar.empty()
            status_text.empty()
            st.error("Could not initialize embeddings")
            return None
        
        # Step 3: Create vector store
        status_text.text("🗃️ Creating vector database...")
        progress_bar.progress(0.6)
        
        # Process in smaller batches to prevent timeout
        batch_size = 50
        if len(texts) > batch_size:
            # Create initial vectorstore with first batch
            vectorstore = FAISS.from_documents(
                documents=texts[:batch_size],
                embedding=embeddings
            )
            
            # Add remaining documents in batches
            for i in range(batch_size, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                batch_vs = FAISS.from_documents(documents=batch, embedding=embeddings)
                vectorstore.merge_from(batch_vs)
                
                # Update progress
                progress = 0.6 + (0.3 * (i / len(texts)))
                progress_bar.progress(min(progress, 0.9))
                status_text.text(f"🗃️ Processing batch {i//batch_size + 1}...")
        else:
            vectorstore = FAISS.from_documents(
                documents=texts,
                embedding=embeddings
            )
        
        progress_bar.progress(1.0)
        status_text.text("✅ Vector database created successfully!")
        
        # Clean up UI
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()
        
        return vectorstore
        
    except Exception as e:
        if 'progress_bar' in locals():
            progress_bar.empty()
        if 'status_text' in locals():
            status_text.empty()
        st.error(f"Error processing documents: {str(e)}")
        if st.session_state.debug_mode:
            st.code(str(e))
        return None

# Optimized retrieval chain setup
def setup_retrieval_chain(vectorstore):
    try:
        # Optimized retriever
        retriever = vectorstore.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": 3}  # Reduced for faster processing
        )
        
        # Simpler memory setup
        memory = ConversationBufferMemory(
            memory_key="chat_history", 
            return_messages=True,
            output_key="answer"
        )
        
        # Initialize Gemini model with conservative settings
        llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=0.1,
            max_tokens=512,
            google_api_key=api_key,
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
        st.error(f"Error setting up retrieval chain: {str(e)}")
        if st.session_state.debug_mode:
            st.code(str(e))
        return None

# Enhanced query function with better timeout handling
def query_with_timeout(chain, question, timeout=30):
    """Query with timeout and better error handling"""
    try:
        check_rate_limit()
        
        # Use invoke instead of __call__
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

# Utility functions
def clear_uploaded_files():
    st.session_state.uploaded_files = []
    st.session_state.vectorstore = None
    st.session_state.documents_processed = False
    st.session_state.document_chunks = 0

@st.cache_data
def refresh_local_files():
    try:
        ignore_patterns = load_gitignore()
        return [
            f for f in os.listdir('.') 
            if f.endswith(('.pdf', '.csv', '.txt', '.xlsx', '.xls')) 
            and not should_ignore(f, ignore_patterns) 
            and f != 'requirements.txt'
        ]
    except Exception as e:
        st.warning(f"Could not refresh local files: {str(e)}")
        return []

def main():
    try:
        # Handle language from query params
        if "language" in st.query_params:
            st.session_state.language = st.query_params["language"]

        t = translations[st.session_state.language]

        # Mark app as initialized after initial setup
        if not st.session_state.app_initialized:
            st.session_state.app_initialized = True

        # Custom CSS
        st.markdown("""
            <style>
            .sidebar .sidebar-content {
                background-color: #f0f2f6;
            }
            .main .block-container {
                max-width: 1200px;
                padding-top: 2rem;
                padding-bottom: 2rem;
            }
            .stTitle {
                text-align: center;
                color: #1f77b4;
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
            .compact-container {
                margin-bottom: 1rem;
            }
            .spacer {
                margin-bottom: 1rem;
            }
            .sidebar-label {
                font-size: 14px;
                font-weight: normal;
                margin-bottom: 0.5rem;
            }
            .uploaded-docs-header {
                font-size: 16px;
                font-weight: bold;
                margin-top: 0.5rem;
                margin-bottom: 0.5rem;
            }
            .stChatInputContainer {
                max-width: 800px;
                margin: 0 auto;
            }
            .stButton > button {
                margin-top: 1.0rem;
                border-radius: 10px;
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
            st.markdown(f"<div class='sidebar-label'>{t['language']}</div>", unsafe_allow_html=True)
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
                st.query_params["language"] = new_language
                # Use experimental_rerun instead of rerun to avoid RerunData error
                st.experimental_rerun()
            
            st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)

            # Debug mode toggle
            st.session_state.debug_mode = st.checkbox("🔍 Debug Mode", value=st.session_state.debug_mode)
            
            st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)

            # File uploader
            st.markdown(f"<div class='sidebar-label'>{t['upload_button']}</div>", unsafe_allow_html=True)
            uploaded_files = st.file_uploader(
                "Upload Documents", 
                accept_multiple_files=True, 
                type=['pdf', 'csv', 'txt', 'xlsx', 'xls'], 
                key="file_uploader",
                label_visibility="collapsed",
                help=t["upload_button"]
            )

            # Handle file uploads
            if uploaded_files and uploaded_files != st.session_state.uploaded_files:
                st.session_state.uploaded_files = uploaded_files
                st.session_state.vectorstore = None
                st.session_state.documents_processed = False
                st.success(t["upload_success"](len(uploaded_files)))

            # Show debug info if enabled
            if st.session_state.debug_mode and st.session_state.document_chunks > 0:
                st.markdown(f"<div class='debug-info'>{t['chunk_count']}: {st.session_state.document_chunks}</div>", unsafe_allow_html=True)

            # Clear chat button
            if st.button(t["clear_chat"], use_container_width=True):
                st.session_state.messages = []
                clear_uploaded_files()
                # Use experimental_rerun instead of rerun
                st.experimental_rerun()

        # Main content
        st.title(t["title"])

        # Display model information
        st.info(t["model_info"])

        # Show API usage warning
        st.warning("⚠️ **Note**: This app uses Gemini API free tier (50 requests/day). Please use sparingly.")

        # Display local knowledge base
        st.session_state.local_files = refresh_local_files()
        if st.session_state.local_files:
            st.markdown(f"<div class='uploaded-docs-header'>{t['local_knowledge']}</div>", unsafe_allow_html=True)
            for file in st.session_state.local_files:
                st.write(f"📄 {file}")

        # Process documents if needed
        total_documents = len(st.session_state.local_files) + len(st.session_state.uploaded_files)
        
        if total_documents > 0 and not st.session_state.documents_processed:
            with st.spinner(t["processing"]):
                try:
                    documents = load_documents(st.session_state.local_files, st.session_state.uploaded_files)
                    if documents:
                        st.session_state.vectorstore = process_documents(documents)
                        if st.session_state.vectorstore:
                            st.session_state.documents_processed = True
                            st.success(f"✅ Processed {len(documents)} documents into {st.session_state.document_chunks} chunks!")
                        else:
                            st.error(t["error_processing"])
                    else:
                        st.warning("No documents found to process")
                except Exception as e:
                    st.error(f"{t['error_processing']}: {str(e)}")
                    if st.session_state.debug_mode:
                        st.code(str(e))

        # Chat interface
        if st.session_state.vectorstore:
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
                    retrieval_chain = setup_retrieval_chain(st.session_state.vectorstore)
                    if retrieval_chain:
                        with st.spinner(t["thinking"]):
                            try:
                                # Enhanced prompt for better results
                                enhanced_prompt = f"Based on the provided documents, please answer the following question in detail: {prompt}"
                                
                                response = query_with_timeout(retrieval_chain, enhanced_prompt, timeout=30)
                                answer = response.get('answer', 'No answer generated')
                                
                                st.markdown(answer)
                                st.session_state.messages.append({"role": "assistant", "content": answer})
                                
                                # Show sources and debug info
                                if 'source_documents' in response and response['source_documents']:
                                    with st.expander("📚 Sources"):
                                        for i, doc in enumerate(response['source_documents']):
                                            st.markdown(f"**Source {i+1}:**")
                                            content = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                                            st.markdown(content)
                                            if hasattr(doc, 'metadata') and doc.metadata:
                                                st.caption(f"Metadata: {doc.metadata}")
                                            st.markdown("---")
                                    
                                    # Debug info
                                    if st.session_state.debug_mode:
                                        st.markdown(f"<div class='debug-info'>{t['retrieval_results']}: {len(response['source_documents'])}</div>", unsafe_allow_html=True)
                                else:
                                    if st.session_state.debug_mode:
                                        st.warning("No source documents retrieved - this might indicate an issue with document processing or retrieval.")
                                
                            except TimeoutError:
                                error_msg = t["timeout_error"]
                                st.error(error_msg)
                                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                            except Exception as e:
                                if "RATE_LIMIT" in str(e):
                                    error_msg = t["rate_limit_error"]
                                    st.error(error_msg)
                                    st.info("💡 **Tip**: Try again in a few minutes, or use a shorter question.")
                                else:
                                    error_msg = f"{t['error_response']}: {str(e)}"
                                    st.error(error_msg)
                                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                                if st.session_state.debug_mode:
                                    st.code(str(e))
                    else:
                        error_msg = t["error_setup"]
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})

        else:
            # Welcome message when no documents are processed
            if total_documents == 0:
                st.info(t["no_documents"])
            st.markdown(f"### {t['welcome']}")
            
            # Instructions
            with st.expander("ℹ️ How to use / วิธีใช้งาน", expanded=True):
                if st.session_state.language == "en":
                    st.markdown("""
                    **How to use this RAG Chatbot:**
                    1. 📁 **Upload Documents**: Use the sidebar to upload PDF, TXT, CSV, or XLSX files
                    2. ⏳ **Wait for Processing**: The system will process your documents automatically
                    3. 💬 **Start Chatting**: Ask questions about your documents in Thai or English
                    4. 🔍 **Debug Mode**: Enable debug mode to see detailed processing information
                    5. 🌐 **Change Language**: Use the language selector in the sidebar
                    
                    **Important Notes:**
                    - ⚠️ **Free API Limit**: 50 requests per day - use wisely!
                    - ⏱️ **Be Patient**: Responses may take 10-30 seconds
                    - 📝 **Keep Questions Short**: Shorter questions get faster responses
                    - 🔄 **Rate Limiting**: Wait a few seconds between questions
                    - 📁 **File Limit**: Maximum 10 files to prevent timeouts
                    
                    **Tips for better results:**
                    - Ask specific questions about the document content
                    - Use clear and complete sentences
                    - If the bot doesn't find relevant information, try rephrasing your question
                    
                    **Supported File Types:**
                    - 📄 PDF files
                    - 📝 Text files (.txt)
                    - 📊 CSV files
                    - 📈 Excel files (.xlsx, .xls)
                    """)
                else:
                    st.markdown("""
                    **วิธีใช้งาน RAG Chatbot:**
                    1. 📁 **อัปโหลดเอกสาร**: ใช้แถบด้านข้างเพื่ออัปโหลดไฟล์ PDF, TXT, CSV หรือ XLSX
                    2. ⏳ **รอการประมวลผล**: ระบบจะประมวลผลเอกสารของคุณโดยอัตโนมัติ
                    3. 💬 **เริ่มแชท**: ถามคำถามเกี่ยวกับเอกสารของคุณเป็นภาษาไทยหรืออังกฤษ
                    4. 🔍 **โหมดดีบัก**: เปิดโหมดดีบักเพื่อดูข้อมูลการประมวลผลโดยละเอียด
                    5. 🌐 **เปลี่ยนภาษา**: ใช้ตัวเลือกภาษาในแถบด้านข้าง
                    
                    **ข้อสำคัญ:**
                    - ⚠️ **ขีดจำกัด API ฟรี**: 50 คำขอต่อวัน - ใช้อย่างประหยัด!
                    - ⏱️ **อดทนรอ**: การตอบอาจใช้เวลา 10-30 วินาที
                    - 📝 **ถามคำถามสั้นๆ**: คำถามสั้นจะได้คำตอบเร็วกว่า
                    - 🔄 **จำกัดอัตรา**: รอสักครู่ระหว่างคำถาม
                    - 📁 **จำกัดไฟล์**: สูงสุด 10 ไฟล์เพื่อป้องกันการค้าง
                    
                    **เคล็ดลับสำหรับผลลัพธ์ที่ดีขึ้น:**
                    - ถามคำถามที่เฉพาะเจาะจงเกี่ยวกับเนื้อหาในเอกสาร
                    - ใช้ประโยคที่ชัดเจนและสมบูรณ์
                    - หากบอทหาข้อมูลที่เกี่ยวข้องไม่เจอ ลองเปลี่ยนคำถาม
                    
                    **ประเภทไฟล์ที่รองรับ:**
                    - 📄 ไฟล์ PDF
                    - 📝 ไฟล์ข้อความ (.txt)
                    - 📊 ไฟล์ CSV
                    - 📈 ไฟล์ Excel (.xlsx, .xls)
                    """)

        # Footer
        st.markdown(
            '<div class="footer">Created by Arnutt Noitumyae, 2024 | Rate-Limited Gemini & FAISS</div>',
            unsafe_allow_html=True
        )
        
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        if st.session_state.debug_mode:
            st.code(str(e))
        st.info("Please refresh the page and try again.")

if __name__ == "__main__":
    main()
