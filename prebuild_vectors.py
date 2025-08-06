#!/usr/bin/env python3
"""
Pre-build Vector Cache for GitHub Deployment
รันในเครื่อง local เพื่อสร้าง vectors ล่วงหน้า
"""

import os
import json
import hashlib
from datetime import datetime
from pathlib import Path

# Import required libraries (same as streamlit_app.py)
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    UnstructuredExcelLoader,
    UnstructuredWordDocumentLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def scan_documents(directories=None):
    """Scan for documents in specified directories"""
    if directories is None:
        directories = ["documents", "data", "."]
    
    supported_extensions = ['.pdf', '.txt', '.csv', '.xlsx', '.xls', '.docx', '.md']
    excluded_files = ['requirements.txt', 'README.md', '.env']
    excluded_dirs = ['.git', '__pycache__', 'venv', '.streamlit', 'vector_cache', 'prebuilt_vectors']
    
    found_files = []
    
    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            continue
            
        for ext in supported_extensions:
            for file_path in dir_path.rglob(f"*{ext}"):
                # Skip excluded directories
                if any(excluded in str(file_path) for excluded in excluded_dirs):
                    continue
                # Skip excluded files and Python files
                if file_path.name in excluded_files or file_path.suffix == '.py':
                    continue
                    
                found_files.append(str(file_path))
    
    return sorted(set(found_files))

def get_file_hash(filepath):
    """Generate hash for file content"""
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()[:16]

def load_document(filepath):
    """Load document based on file type"""
    filepath = str(filepath)
    
    try:
        if filepath.endswith('.pdf'):
            loader = PyPDFLoader(filepath)
        elif filepath.endswith(('.txt', '.md')):
            loader = TextLoader(filepath, encoding='utf-8')
        elif filepath.endswith('.csv'):
            loader = CSVLoader(filepath, encoding='utf-8')
        elif filepath.endswith(('.xlsx', '.xls')):
            loader = UnstructuredExcelLoader(filepath)
        elif filepath.endswith('.docx'):
            loader = UnstructuredWordDocumentLoader(filepath)
        else:
            return []
        
        documents = loader.load()
        return documents
        
    except Exception as e:
        print(f"⚠️ Error loading {filepath}: {e}")
        return []

def main():
    print("=" * 60)
    print("🚀 PRE-BUILD VECTOR CACHE FOR STREAMLIT DEPLOYMENT")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path("prebuilt_vectors")
    output_dir.mkdir(exist_ok=True)
    
    # Scan for documents
    print("\n🔍 Scanning for documents...")
    documents_list = scan_documents()
    
    if not documents_list:
        print("❌ No documents found!")
        print("\n💡 Tips:")
        print("1. Add documents to your project directory")
        print("2. Supported formats: PDF, TXT, CSV, XLSX, DOCX")
        return
    
    print(f"\n📁 Found {len(documents_list)} documents:")
    for doc in documents_list:
        print(f"  - {doc}")
    
    # Initialize components
    print("\n🔧 Initializing embeddings model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        length_function=len
    )
    
    # Process documents
    all_texts = []
    metadata = {
        "documents": [],
        "created_at": datetime.now().isoformat(),
        "total_chunks": 0,
        "model": "sentence-transformers/all-MiniLM-L6-v2"
    }
    
    for filepath in documents_list:
        print(f"\n📄 Processing: {filepath}")
        
        # Load document
        documents = load_document(filepath)
        if not documents:
            continue
        
        # Split into chunks
        texts = text_splitter.split_documents(documents)
        
        # Add metadata
        for text in texts:
            text.metadata['source'] = filepath
            text.metadata['file_hash'] = get_file_hash(filepath)
        
        all_texts.extend(texts)
        
        # Update metadata
        metadata["documents"].append({
            "filepath": filepath,
            "hash": get_file_hash(filepath),
            "chunks": len(texts),
            "size": os.path.getsize(filepath)
        })
        
        print(f"  ✅ Created {len(texts)} chunks")
    
    if not all_texts:
        print("\n❌ No text content extracted!")
        return
    
    # Create FAISS vector store
    print(f"\n🔨 Building FAISS index with {len(all_texts)} chunks...")
    vectorstore = FAISS.from_documents(all_texts, embeddings)
    
    # Update metadata
    metadata["total_chunks"] = len(all_texts)
    
    # Save vector store
    print("\n💾 Saving vector store...")
    vectorstore.save_local(str(output_dir / "faiss_index"))
    
    # Save metadata
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Verify
    print("\n🔍 Verifying...")
    try:
        test_store = FAISS.load_local(
            str(output_dir / "faiss_index"),
            embeddings,
            allow_dangerous_deserialization=True
        )
        print("✅ Verification successful!")
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return
    
    print("\n" + "=" * 60)
    print("✅ SUCCESS! Pre-built vectors created!")
    print("=" * 60)
    print("\n📋 Next steps:")
    print("1. Commit to GitHub:")
    print("   git add prebuilt_vectors/")
    print("   git commit -m 'Add pre-built vectors'")
    print("   git push")
    print("\n2. Deploy to Streamlit Cloud")
    print("\n✨ Users will have instant access without processing!")

if __name__ == "__main__":
    main()