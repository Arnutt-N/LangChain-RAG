import streamlit as st
import os
import fnmatch
import io
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Fixed imports - ใช้ community imports
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

from langchain.text_splitters import CharacterTextSplitter  # Fixed import
from langchain_community.vectorstores import Chroma
import tempfile
import chromadb
from chromadb.config import Settings

# Set page configuration
st.set_page_config(
    layout="wide", 
    page_title="Gen AI : RAG Chatbot with Documents",
    page_icon="🤖",
    initial_sidebar_state="expanded"
)

# Retrieve the API key from Streamlit secrets or environment
api_key = (
    st.secrets.get("GOOGLE_API_KEY") or 
    st.secrets.get("GEMINI_API_KEY") or 
    os.getenv("GOOGLE_API_KEY") or 
    os.getenv("GEMINI_API_KEY")
)

# Ensure the script stops execution if the API key is not set
if api_key is None:
    st.error("🚨 GOOGLE_API_KEY or GEMINI_API_KEY is not set. Please set it in the Streamlit Cloud secrets or environment variables.")
    st.info("👉 You can get a free API key from: https://makersuite.google.com/app/apikey")
    st.stop()
else:
    os.environ["GOOGLE_API_KEY"] = api_key

# Translations
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
        "thinking": "🧠 Thinking...",
        "language": "🌐 Language / ภาษา",
        "clear_chat": "🗑️ Clear Chat",
        "model_info": "🤖 **Model:** Gemini 2.5 Flash | 📊 **Embedding:** BGE-M3 | 🗃️ **Vector DB:** ChromaDB",
        "no_documents": "📄 No documents uploaded yet. Please upload some documents to start chatting!",
        "error_processing": "❌ Error processing documents. Please try again.",
        "error_response": "🚨 Sorry, I encountered an error while generating response.",
        "error_setup": "⚠️ Sorry, I couldn't set up the retrieval system.",
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
        "thinking": "🧠 กำลังคิด...",
        "language": "🌐 ภาษา / Language",
        "clear_chat": "🗑️ ล้างการแชท",
        "model_info": "🤖 **โมเดล:** Gemini 2.5 Flash | 📊 **Embedding:** BGE-M3 | 🗃️ **Vector DB:** ChromaDB",
        "no_documents": "📄 ยังไม่มีเอกสารอัปโหลด กรุณาอัปโหลดเอกสารเพื่อเริ่มแชท!",
        "error_processing": "❌ เกิดข้อผิดพลาดในการประมวลผลเอกสาร กรุณาลองใหม่อีกครั้ง",
        "error_response": "🚨 ขออภัย เกิดข้อผิดพลาดในการสร้างคำตอบ",
        "error_setup": "⚠️ ขออภัย ไม่สามารถตั้งค่าระบบค้นหาได้",
    }
}

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "local_files" not in st.session_state:
    st.session_state.local_files = []
if "language" not in st.session_state:
    st.session_state.language = "en"
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "chroma_client" not in st.session_state:
    st.session_state.chroma_client = None
if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = False

# Function to load .gitignore patterns
def load_gitignore():
    patterns = []
    try:
        if os.path.exists('.gitignore'):
            encodings = ['utf-8', 'cp874', 'tis-620', 'windows-1252', 'latin-1']
            for encoding in encodings:
                try:
                    with open('.gitignore', 'r', encoding=encoding) as file:
                        patterns = file.read().splitlines()
                    break  # If successful, exit the loop
                except UnicodeDecodeError:
                    continue  # If unsuccessful, try the next encoding
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

# Initialize BGE-M3 embeddings
@st.cache_resource
def get_embeddings():
    """Initialize and cache BGE-M3 embeddings"""
    try:
        model_name = "BAAI/bge-m3"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': True}
        
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        return embeddings
    except Exception as e:
        st.error(f"Error initializing embeddings: {str(e)}")
        # Fallback to a smaller model
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            return embeddings
        except Exception as e2:
            st.error(f"Error with fallback embeddings: {str(e2)}")
            return None

# Initialize ChromaDB client
@st.cache_resource
def get_chroma_client():
    """Initialize and cache ChromaDB client"""
    try:
        # Create a persistent ChromaDB client
        client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        return client
    except Exception as e:
        st.error(f"Error initializing ChromaDB: {str(e)}")
        # Fallback to in-memory client
        try:
            client = chromadb.Client()
            return client
        except Exception as e2:
            st.error(f"Error with fallback ChromaDB: {str(e2)}")
            return None

# Load documents
def load_documents(file_paths, uploaded_files):
    documents = []
    
    # Load local files
    for file_path in file_paths:
        if file_path == 'requirements.txt':
            continue
        try:
            if file_path.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            elif file_path.endswith('.csv'):
                loader = CSVLoader(file_path)
            elif file_path.endswith('.txt'):
                loader = TextLoader(file_path, encoding='utf-8')
            elif file_path.endswith(('.xlsx', '.xls')):
                loader = UnstructuredExcelLoader(file_path)
            else:
                continue
            
            docs = loader.load()
            documents.extend(docs)
            
        except Exception as e:
            st.warning(f"Could not load file {file_path}: {str(e)}")
            continue
    
    # Load uploaded files
    for uploaded_file in uploaded_files:
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_file_path = temp_file.name
            
            # Load based on file type
            if uploaded_file.name.endswith('.pdf'):
                loader = PyPDFLoader(temp_file_path)
            elif uploaded_file.name.endswith('.csv'):
                loader = CSVLoader(temp_file_path)
            elif uploaded_file.name.endswith('.txt'):
                loader = TextLoader(temp_file_path, encoding='utf-8')
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                loader = UnstructuredExcelLoader(temp_file_path)
            else:
                os.unlink(temp_file_path)
                continue
            
            docs = loader.load()
            documents.extend(docs)
            
            # Clean up temporary file
            os.unlink(temp_file_path)
            
        except Exception as e:
            st.warning(f"Could not load uploaded file {uploaded_file.name}: {str(e)}")
            continue
    
    return documents

# Process documents
def process_documents(documents):
    if not documents:
        return None

    try:
        # Split documents into chunks
        text_splitter = CharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200,
            separator="\n"
        )
        texts = text_splitter.split_documents(documents)
        
        if not texts:
            st.warning("No text content found in documents")
            return None
        
        # Get embeddings
        embeddings = get_embeddings()
        if not embeddings:
            return None
        
        # Get ChromaDB client
        chroma_client = get_chroma_client()
        if not chroma_client:
            return None
        
        # Create collection name
        collection_name = "rag_documents"
        
        # Delete existing collection if it exists
        try:
            chroma_client.delete_collection(name=collection_name)
        except:
            pass
        
        # Create vector store with ChromaDB
        vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            client=chroma_client,
            collection_name=collection_name,
            persist_directory="./chroma_db"
        )
        
        return vectorstore
        
    except Exception as e:
        st.error(f"Error processing documents: {str(e)}")
        return None

# Setup retrieval chain
def setup_retrieval_chain(vectorstore):
    try:
        retriever = vectorstore.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": 3}
        )
        
        memory = ConversationBufferMemory(
            memory_key="chat_history", 
            return_messages=True,
            output_key="answer"
        )
        
        # Initialize Gemini model with fallback
        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0.1,
                google_api_key=api_key
            )
        except:
            # Fallback to gemini-pro if 2.5-flash is not available
            llm = ChatGoogleGenerativeAI(
                model="gemini-pro",
                temperature=0.1,
                google_api_key=api_key
            )
        
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm, 
            retriever=retriever, 
            memory=memory,
            return_source_documents=True
        )
        
        return qa_chain
        
    except Exception as e:
        st.error(f"Error setting up retrieval chain: {str(e)}")
        return None

# Function to clear all uploaded files and reset vectorstore
def clear_uploaded_files():
    st.session_state.uploaded_files = []
    st.session_state.vectorstore = None
    st.session_state.documents_processed = False
    # Clear ChromaDB collection
    try:
        chroma_client = get_chroma_client()
        if chroma_client:
            chroma_client.delete_collection(name="rag_documents")
    except:
        pass

# Function to refresh local files
def refresh_local_files():
    try:
        ignore_patterns = load_gitignore()
        st.session_state.local_files = [
            f for f in os.listdir('.') 
            if f.endswith(('.pdf', '.csv', '.txt', '.xlsx', '.xls')) 
            and not should_ignore(f, ignore_patterns) 
            and f != 'requirements.txt'
        ]
    except Exception as e:
        st.warning(f"Could not refresh local files: {str(e)}")
        st.session_state.local_files = []

def main():
    # Handle language from query params
    if "language" in st.query_params:
        st.session_state.language = st.query_params["language"]

    t = translations[st.session_state.language]

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
        .success-message {
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            border-radius: 5px;
            padding: 10px;
            margin: 10px 0;
        }
        </style>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        # Language selection moved to the top
        st.markdown(f"<div class='sidebar-label'>{t['language']}</div>", unsafe_allow_html=True)
        selected_lang = st.selectbox(
            "", 
            options=["ไทย", "English"], 
            index=1 if st.session_state.language == "en" else 0, 
            key="language_selection",
            label_visibility="collapsed"
        )
        
        new_language = "th" if selected_lang == "ไทย" else "en"
        
        if new_language != st.session_state.language:
            st.session_state.language = new_language
            st.query_params["language"] = new_language
            st.rerun()
        
        # Spacer between language selection and file uploader
        st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)

        # Add "Upload Documents" text above the file uploader
        st.markdown(f"<div class='sidebar-label'>{t['upload_button']}</div>", unsafe_allow_html=True)

        # File uploader
        uploaded_files = st.file_uploader(
            "", 
            accept_multiple_files=True, 
            type=['pdf', 'csv', 'txt', 'xlsx', 'xls'], 
            key="file_uploader",
            label_visibility="collapsed",
            help=t["upload_button"]
        )

        # Handle file uploads
        if uploaded_files and uploaded_files != st.session_state.uploaded_files:
            st.session_state.uploaded_files = uploaded_files
            st.session_state.vectorstore = None  # Reset vectorstore to force reprocessing
            st.session_state.documents_processed = False
            st.success(t["upload_success"](len(uploaded_files)))

        # Clear chat button
        if st.button(t["clear_chat"], use_container_width=True):
            st.session_state.messages = []
            clear_uploaded_files()
            st.rerun()

    # Main content
    st.title(t["title"])

    # Display model information
    st.info(t["model_info"])

    # Display local knowledge base
    refresh_local_files()
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
                        st.success(f"✅ Processed {len(documents)} document chunks successfully!")
                    else:
                        st.error(t["error_processing"])
                else:
                    st.warning("No documents found to process")
            except Exception as e:
                st.error(f"{t['error_processing']}: {str(e)}")

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
                            response = retrieval_chain({"question": prompt})
                            answer = response.get('answer', 'No answer generated')
                            
                            st.markdown(answer)
                            st.session_state.messages.append({"role": "assistant", "content": answer})
                            
                            # Show sources if available
                            if 'source_documents' in response and response['source_documents']:
                                with st.expander("📚 Sources"):
                                    for i, doc in enumerate(response['source_documents']):
                                        st.markdown(f"**Source {i+1}:**")
                                        content = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                                        st.markdown(content)
                                        if hasattr(doc, 'metadata') and doc.metadata:
                                            st.caption(f"Metadata: {doc.metadata}")
                                        st.markdown("---")
                                        
                        except Exception as e:
                            error_msg = f"{t['error_response']}: {str(e)}"
                            st.error(error_msg)
                            st.session_state.messages.append({"role": "assistant", "content": error_msg})
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
                4. 🌐 **Change Language**: Use the language selector in the sidebar
                
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
                4. 🌐 **เปลี่ยนภาษา**: ใช้ตัวเลือกภาษาในแถบด้านข้าง
                
                **ประเภทไฟล์ที่รองรับ:**
                - 📄 ไฟล์ PDF
                - 📝 ไฟล์ข้อความ (.txt)
                - 📊 ไฟล์ CSV
                - 📈 ไฟล์ Excel (.xlsx, .xls)
                """)

    # Footer
    st.markdown(
        '<div class="footer">Created by Arnutt Noitumyae, 2024 | Updated with Gemini & ChromaDB</div>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    try:
        if "language" in st.query_params:
            st.session_state.language = st.query_params["language"]
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.info("Please refresh the page and try again.")
