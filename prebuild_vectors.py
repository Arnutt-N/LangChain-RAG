#!/usr/bin/env python3
"""
Pre-build Vector Cache for GitHub Deployment
FIXED VERSION - Excludes virtual environment and unnecessary files
"""

import os
import json
import hashlib
from datetime import datetime
from pathlib import Path

# Import required libraries
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    UnstructuredExcelLoader,
    UnstructuredWordDocumentLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Try both import methods for compatibility
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def scan_documents(directories=None):
    """Scan for documents in specified directories - EXCLUDING env and other unnecessary folders"""
    if directories is None:
        # Only scan specific directories, NOT the entire project
        directories = ["documents", "data"]
        
        # Add current directory ONLY if documents folder doesn't exist
        if not Path("documents").exists() and not Path("data").exists():
            directories.append(".")
    
    supported_extensions = ['.pdf', '.txt', '.csv', '.xlsx', '.xls', '.docx', '.md']
    
    # CRITICAL: Exclude these files and directories
    excluded_files = [
        'requirements.txt', 
        'README.md', 
        '.env',
        'LICENSE',
        'Dockerfile',
        '.gitignore',
        'setup.py',
        'setup.cfg',
        'pyproject.toml'
    ]
    
    excluded_dirs = [
        'env',           # Virtual environment
        'venv',          # Virtual environment
        '.venv',         # Virtual environment  
        '__pycache__',   # Python cache
        '.git',          # Git directory
        '.github',       # GitHub configs
        '.streamlit',    # Streamlit config
        'vector_cache',  # Cache directory
        'prebuilt_vectors', # Output directory
        'node_modules',  # Node.js
        '.pytest_cache', # Pytest cache
        '.mypy_cache',   # Mypy cache
        'dist',          # Distribution
        'build',         # Build directory
        '.idea',         # IDE configs
        '.vscode',       # IDE configs
        'temp',          # Temporary files
        'tmp',           # Temporary files
        'test',          # Test files
        'tests',         # Test files
    ]
    
    found_files = []
    
    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            continue
        
        # If scanning current directory, be extra careful
        if directory == ".":
            # Only look for documents in root, not subdirectories
            for ext in supported_extensions:
                for file_path in dir_path.glob(f"*{ext}"):
                    # Make sure it's a file, not in a subdirectory
                    if file_path.is_file():
                        filename = file_path.name
                        # Skip if it's an excluded file
                        if filename in excluded_files or filename.endswith('.py'):
                            continue
                        found_files.append(str(file_path))
        else:
            # For specific directories like 'documents', scan recursively
            for ext in supported_extensions:
                for file_path in dir_path.rglob(f"*{ext}"):
                    # Check if path contains any excluded directory
                    path_str = str(file_path)
                    skip = False
                    for excluded in excluded_dirs:
                        if excluded in path_str.split(os.sep):
                            skip = True
                            break
                    
                    if skip:
                        continue
                    
                    # Skip excluded files and Python files
                    if file_path.name in excluded_files or file_path.suffix == '.py':
                        continue
                    
                    found_files.append(str(file_path))
    
    # Remove duplicates and sort
    found_files = sorted(set(found_files))
    
    # Final safety check - remove any file that looks suspicious
    clean_files = []
    for file in found_files:
        # Additional safety checks
        if any(excluded in file for excluded in ['site-packages', 'dist-info', '__pycache__', '/env/', '/venv/']):
            continue
        clean_files.append(file)
    
    return clean_files

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
    
    # Check if documents folder exists
    if not Path("documents").exists() and not Path("data").exists():
        print("\n📁 Creating 'documents' folder...")
        Path("documents").mkdir(exist_ok=True)
        print("✅ Created 'documents' folder")
        print("\n⚠️ Please add your documents to the 'documents' folder and run again!")
        
        # Create a sample file
        sample_file = Path("documents") / "sample.txt"
        if not sample_file.exists():
            sample_file.write_text("""Sample Document for RAG Chatbot

This is a sample document to test the RAG system.

Key Features:
- Document processing
- Vector embeddings
- Semantic search
- Question answering

Add your own PDF, TXT, CSV, XLSX, or DOCX files to this folder!
""")
            print(f"📝 Created sample file: {sample_file}")
        return
    
    # Scan for documents
    print("\n🔍 Scanning for documents...")
    print("   Excluding: env/, venv/, __pycache__, .git/, etc.")
    documents_list = scan_documents()
    
    if not documents_list:
        print("\n❌ No documents found!")
        print("\n💡 Please add documents to:")
        print("   - documents/ folder")
        print("   - data/ folder")
        print("\n📊 Supported formats: PDF, TXT, CSV, XLSX, DOCX, MD")
        return
    
    print(f"\n✅ Found {len(documents_list)} documents:")
    for doc in documents_list:
        # Show relative path for clarity
        display_path = doc.replace(os.getcwd() + os.sep, "")
        print(f"  📄 {display_path}")
    
    # Confirm before processing
    print(f"\n📊 Will process {len(documents_list)} files")
    
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
        print(f"\n📄 Processing: {Path(filepath).name}")
        
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
        print("💡 Check that your documents contain readable text")
        return
    
    # Create FAISS vector store
    print(f"\n🔨 Building FAISS index with {len(all_texts)} chunks...")
    print("   This may take a moment...")
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
    
    # Check output size
    import shutil
    total_size = sum(f.stat().st_size for f in Path(output_dir).rglob('*') if f.is_file())
    size_mb = total_size / (1024 * 1024)
    
    print("\n" + "=" * 60)
    print("✅ SUCCESS! Pre-built vectors created!")
    print("=" * 60)
    print(f"\n📊 Statistics:")
    print(f"  - Documents processed: {len(metadata['documents'])}")
    print(f"  - Total chunks: {metadata['total_chunks']}")
    print(f"  - Output size: {size_mb:.2f} MB")
    
    if size_mb > 50:
        print(f"\n⚠️ Warning: Output is {size_mb:.2f} MB")
        print("  Consider reducing chunk_size if too large for GitHub")
    
    print("\n📋 Next steps:")
    print("1. Commit to GitHub:")
    print("   git add prebuilt_vectors/")
    print("   git commit -m 'Add pre-built vectors'")
    print("   git push")
    print("\n2. Deploy to Streamlit Cloud")
    print("\n✨ Users will have instant access without processing!")

if __name__ == "__main__":
    main()