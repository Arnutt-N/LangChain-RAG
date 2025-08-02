#!/usr/bin/env python3
"""
Quick script to create test files for your RAG chatbot
Run this in your project directory: python create_test_files.py
"""

def create_test_files():
    """Create test files with substantial content"""
    
    # Test TXT file
    test_txt_content = """# RAG Chatbot Documentation

This is a comprehensive guide to the RAG (Retrieval-Augmented Generation) chatbot system.

## System Overview
The RAG chatbot combines document retrieval with AI generation to provide accurate, context-aware responses based on your documents.

## Key Features
1. **Multi-format Support**: Processes PDF, TXT, CSV, XLSX, and DOCX files
2. **FAISS Vector Database**: Uses Facebook's efficient similarity search
3. **Smart Caching**: Intelligent caching system for faster reloads
4. **Dual AI Models**: Supports both Gemini Flash and Mistral Large
5. **Multilingual**: Works in Thai and English

## Technical Architecture
- **Document Loading**: Uses LangChain document loaders
- **Text Chunking**: RecursiveCharacterTextSplitter for optimal chunks
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2 model
- **Vector Storage**: FAISS for fast similarity search
- **LLM Integration**: ChatGoogleGenerativeAI and Mistral API

## How It Works
1. Documents are automatically scanned from the repository
2. Content is extracted and split into manageable chunks
3. Text chunks are converted to vector embeddings
4. Embeddings are stored in FAISS vector database
5. User queries are processed to find relevant document chunks
6. AI models generate responses based on retrieved context

## Usage Instructions
- Place your documents in the project directory
- The system will auto-detect supported file formats
- Use the sidebar to upload additional documents
- Ask questions about your document content
- Enable debug mode for detailed processing information

## Troubleshooting
- Check that files contain readable text content
- Verify file formats are supported (PDF, TXT, CSV, XLSX, DOCX)
- Use debug mode to see detailed processing steps
- Create test files if no documents are found

This document should generate multiple text chunks when processed.
"""

    # Test CSV file
    test_csv_content = """Component,Description,Technology,Purpose
Document Loader,Processes various file formats,LangChain Community,Extract text from files
Text Splitter,Divides documents into chunks,RecursiveCharacterTextSplitter,Create manageable text pieces
Embeddings,Converts text to vectors,HuggingFace Transformers,Enable similarity search
Vector Database,Stores text embeddings,FAISS,Fast similarity retrieval
AI Model,Generates responses,Gemini Flash / Mistral Large,Create contextual answers
Caching System,Stores processed data,Local file system,Improve performance
User Interface,Web application,Streamlit,Provide interactive chat
Memory System,Maintains conversation,ConversationBufferMemory,Context awareness
Retrieval Chain,Combines components,LangChain,End-to-end processing
Debug Mode,Development tools,Built-in debugging,Troubleshoot issues"""

    # Test markdown file
    test_md_content = """# Technical Implementation Guide

## Document Processing Pipeline

### Stage 1: File Discovery
The system automatically scans for supported document types:
- PDF files using PyPDFLoader
- Text files using TextLoader with encoding detection
- CSV files using CSVLoader with multiple encoding attempts
- Excel files using UnstructuredExcelLoader
- Word documents using UnstructuredWordDocumentLoader

### Stage 2: Content Extraction
Each file type uses specialized loaders:
- **PDF**: Extracts text preserving document structure
- **CSV**: Processes tabular data into searchable text
- **TXT/MD**: Handles plain text with proper encoding
- **XLSX**: Converts spreadsheet content to text format
- **DOCX**: Extracts formatted document content

### Stage 3: Text Preprocessing
Content undergoes several cleaning steps:
- Remove excessive whitespace
- Normalize line endings
- Preserve document structure
- Filter out empty or meaningless content

### Stage 4: Chunk Creation
Documents are split using RecursiveCharacterTextSplitter:
- Chunk size: 800-1000 characters
- Overlap: 100-200 characters for context preservation
- Smart separators: paragraphs, sentences, phrases
- Metadata preservation for source tracking

### Stage 5: Vector Embedding
Text chunks are converted to numerical representations:
- Model: sentence-transformers/all-MiniLM-L6-v2
- Dimension: 384-dimensional vectors
- Normalization: L2 normalized for cosine similarity
- Batch processing for efficiency

### Stage 6: Vector Storage
Embeddings are stored in FAISS database:
- Index type: Flat (exact search)
- Similarity metric: Cosine similarity
- Persistent storage with metadata
- Fast retrieval capabilities

This comprehensive pipeline ensures reliable document processing and accurate information retrieval.
"""

    try:
        # Create test_document.txt
        with open("test_document.txt", "w", encoding="utf-8") as f:
            f.write(test_txt_content)
        print("✅ Created test_document.txt")
        
        # Create test_data.csv
        with open("test_data.csv", "w", encoding="utf-8") as f:
            f.write(test_csv_content)
        print("✅ Created test_data.csv")
        
        # Create technical_guide.md
        with open("technical_guide.md", "w", encoding="utf-8") as f:
            f.write(test_md_content)
        print("✅ Created technical_guide.md")
        
        print("\n🎉 Test files created successfully!")
        print("📁 Files created:")
        print("  - test_document.txt (comprehensive documentation)")
        print("  - test_data.csv (component information)")
        print("  - technical_guide.md (implementation details)")
        print("\n🚀 Now run your Streamlit app and it should process these files!")
        
    except Exception as e:
        print(f"❌ Error creating test files: {e}")

if __name__ == "__main__":
    create_test_files()
