#!/usr/bin/env python3
"""
Pre-build Vector Cache with Qwen Embeddings
Using Hugging Face Inference API
"""

import os
import json
import hashlib
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Import required libraries
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    UnstructuredExcelLoader,
    UnstructuredWordDocumentLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Import Qwen embeddings
from qwen_embeddings import QwenEmbeddings

# Load environment variables
load_dotenv()

def check_hf_token():
    """Check if HF_TOKEN is available"""
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not hf_token:
        print("❌ HF_TOKEN not found!")
        print("\n💡 Please set your Hugging Face token:")
        print("   export HF_TOKEN='your-token-here'")
        print("   or add to .env file:")
        print("   HF_TOKEN=your-token-here")
        print("\nGet your token from: https://huggingface.co/settings/tokens")
        return None
    return hf_token

def scan_documents(directories=None):
    """Scan for documents in specified directories"""
    if directories is None:
        directories = ["documents", "data"]
        if not Path("documents").exists() and not Path("data").exists():
            directories.append(".")
    
    supported_extensions = ['.pdf', '.txt', '.csv', '.xlsx', '.xls', '.docx', '.md']
    excluded_files = [
        'requirements.txt', 'README.md', '.env',
        'LICENSE', 'Dockerfile', '.gitignore'
    ]
    excluded_dirs = [
        'env', 'venv', '.venv', '__pycache__',
        '.git', '.github', '.streamlit',
        'vector_cache', 'prebuilt_vectors'
    ]
    
    found_files = []
    
    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            continue
        
        if directory == ".":
            for ext in supported_extensions:
                for file_path in dir_path.glob(f"*{ext}"):
                    if file_path.is_file():
                        filename = file_path.name
                        if filename in excluded_files or filename.endswith('.py'):
                            continue
                        found_files.append(str(file_path))
        else:
            for ext in supported_extensions:
                for file_path in dir_path.rglob(f"*{ext}"):
                    path_str = str(file_path)
                    skip = False
                    for excluded in excluded_dirs:
                        if excluded in path_str.split(os.sep):
                            skip = True
                            break
                    
                    if skip:
                        continue
                    
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
    print("🚀 PRE-BUILD VECTOR CACHE WITH QWEN EMBEDDINGS")
    print("=" * 60)
    
    # Check for HF token
    hf_token = check_hf_token()
    if not hf_token:
        return
    
    print("✅ HF_TOKEN found")
    
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
            sample_file.write_text("""Sample Document for RAG Chatbot with Qwen Embeddings

This document tests the RAG system using Qwen3-Embedding-8B model.

Key Features:
- Advanced 8B parameter embedding model
- 1536-dimensional embeddings
- Superior multilingual support
- Better semantic understanding

The Qwen embedding model provides state-of-the-art performance for:
- Document retrieval
- Semantic search
- Question answering
- Cross-lingual tasks

Add your own PDF, TXT, CSV, XLSX, or DOCX files to this folder!
""")
            print(f"📝 Created sample file: {sample_file}")
        return
    
    # Scan for documents
    print("\n🔍 Scanning for documents...")
    documents_list = scan_documents()
    
    if not documents_list:
        print("\n❌ No documents found!")
        print("\n💡 Please add documents to:")
        print("   - documents/ folder")
        print("   - data/ folder")
        return
    
    print(f"\n✅ Found {len(documents_list)} documents:")
    for doc in documents_list:
        display_path = doc.replace(os.getcwd() + os.sep, "")
        print(f"  📄 {display_path}")
    
    print(f"\n📊 Will process {len(documents_list)} files")
    
    # Initialize Qwen embeddings
    print("\n🔧 Initializing Qwen embeddings...")
    print("   Model: Qwen/Qwen3-Embedding-8B")
    print("   Provider: Nebius (via Hugging Face)")
    
    try:
        embeddings = QwenEmbeddings(
            api_key=hf_token,
            model_name="Qwen/Qwen3-Embedding-8B",
            provider="nebius",
            batch_size=16,  # Smaller batch for API calls
            max_retries=3
        )
        print("✅ Qwen embeddings initialized")
        
        # Test embeddings
        test_text = "Test embedding"
        test_result = embeddings.embed_query(test_text)
        print(f"✅ Test embedding successful: {len(test_result)} dimensions")
        
    except Exception as e:
        print(f"❌ Failed to initialize Qwen embeddings: {e}")
        print("\n💡 Make sure your HF_TOKEN is valid and has access to the model")
        return
    
    # Text splitter with optimized settings for Qwen
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Larger chunks for better context
        chunk_overlap=150,  # More overlap for continuity
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        length_function=len
    )
    
    # Process documents
    all_texts = []
    metadata = {
        "documents": [],
        "created_at": datetime.now().isoformat(),
        "total_chunks": 0,
        "model": "Qwen/Qwen3-Embedding-8B",
        "embedding_dimension": 1536,
        "provider": "nebius"
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
        return
    
    # Create FAISS vector store with Qwen embeddings
    print(f"\n🔨 Building FAISS index with {len(all_texts)} chunks...")
    print("⏳ This may take a while due to API calls...")
    
    try:
        # Process in smaller batches to show progress
        batch_size = 50
        vectorstore = None
        
        for i in range(0, len(all_texts), batch_size):
            batch = all_texts[i:min(i+batch_size, len(all_texts))]
            print(f"   Processing batch {i//batch_size + 1}/{(len(all_texts)-1)//batch_size + 1}...")
            
            if vectorstore is None:
                # Create initial vectorstore
                vectorstore = FAISS.from_documents(batch, embeddings)
            else:
                # Add to existing vectorstore
                batch_vectorstore = FAISS.from_documents(batch, embeddings)
                vectorstore.merge_from(batch_vectorstore)
        
        print("✅ FAISS index created successfully!")
        
    except Exception as e:
        print(f"❌ Failed to create FAISS index: {e}")
        print("\n💡 Check your API rate limits and token validity")
        return
    
    # Update metadata
    metadata["total_chunks"] = len(all_texts)
    
    # Save vector store
    print("\n💾 Saving vector store...")
    vectorstore.save_local(str(output_dir / "faiss_index"))
    
    # Save metadata
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save Qwen embeddings config
    qwen_config = {
        "model_name": "Qwen/Qwen3-Embedding-8B",
        "provider": "nebius",
        "dimension": 1536,
        "requires_hf_token": True
    }
    with open(output_dir / "qwen_config.json", 'w') as f:
        json.dump(qwen_config, f, indent=2)
    
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
    total_size = sum(f.stat().st_size for f in Path(output_dir).rglob('*') if f.is_file())
    size_mb = total_size / (1024 * 1024)
    
    print("\n" + "=" * 60)
    print("✅ SUCCESS! Pre-built vectors with Qwen embeddings created!")
    print("=" * 60)
    print(f"\n📊 Statistics:")
    print(f"  - Model: Qwen/Qwen3-Embedding-8B")
    print(f"  - Embedding dimension: 1536")
    print(f"  - Documents processed: {len(metadata['documents'])}")
    print(f"  - Total chunks: {metadata['total_chunks']}")
    print(f"  - Output size: {size_mb:.2f} MB")
    
    if size_mb > 50:
        print(f"\n⚠️ Warning: Output is {size_mb:.2f} MB")
        print("  Consider reducing chunk_size if too large for GitHub")
    
    print("\n📋 Next steps:")
    print("1. Copy qwen_embeddings.py to your project")
    print("2. Update streamlit_app.py to use QwenEmbeddings")
    print("3. Commit to GitHub:")
    print("   git add prebuilt_vectors/ qwen_embeddings.py")
    print("   git commit -m 'Add Qwen embeddings and pre-built vectors'")
    print("   git push")
    print("\n4. Add HF_TOKEN to Streamlit Secrets")
    print("\n✨ Users will have instant access with superior Qwen embeddings!")

if __name__ == "__main__":
    main()