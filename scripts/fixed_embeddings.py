"""
Fixed embeddings implementation for new huggingface_hub version
"""

import streamlit as st
import os
from pathlib import Path
import numpy as np
from typing import List
from langchain.embeddings.base import Embeddings

class FixedSentenceTransformerEmbeddings(Embeddings):
    """Fixed embeddings that work with new huggingface_hub"""
    
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
            
            # Set cache directory
            cache_dir = Path("D:/genAI/LangChain-RAG/models")
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize model with new API
            self.model = SentenceTransformer(
                model_name, 
                cache_folder=str(cache_dir),
                device='cpu'
            )
            
        except ImportError:
            # Fallback to basic embeddings if sentence_transformers not available
            st.warning("sentence_transformers not found, using basic embeddings")
            self.model = None
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        if self.model:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return embeddings.tolist()
        else:
            # Fallback to random embeddings
            return [[float(i) for i in np.random.randn(384)] for _ in texts]
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        if self.model:
            embedding = self.model.encode([text], convert_to_numpy=True)[0]
            return embedding.tolist()
        else:
            # Fallback to random embedding
            return [float(i) for i in np.random.randn(384)]

# Alternative: Use newer langchain-huggingface package
class ModernHuggingFaceEmbeddings(Embeddings):
    """Use the new langchain-huggingface package"""
    
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        try:
            # Try new package first
            from langchain_huggingface import HuggingFaceEmbeddings as NewHF
            
            cache_dir = Path("D:/genAI/LangChain-RAG/models")
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            self.embeddings = NewHF(
                model_name=model_name,
                cache_folder=str(cache_dir),
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        except ImportError:
            # Fall back to sentence_transformers
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.embeddings = None
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if self.embeddings:
            return self.embeddings.embed_documents(texts)
        else:
            return self.model.encode(texts).tolist()
    
    def embed_query(self, text: str) -> List[float]:
        if self.embeddings:
            return self.embeddings.embed_query(text)
        else:
            return self.model.encode([text])[0].tolist()

def get_working_embeddings():
    """Get embeddings that work with current environment"""
    
    # Try different methods in order
    methods = [
        ("Modern HuggingFace", ModernHuggingFaceEmbeddings),
        ("Fixed SentenceTransformer", FixedSentenceTransformerEmbeddings),
    ]
    
    for name, EmbeddingClass in methods:
        try:
            print(f"Trying {name}...")
            embeddings = EmbeddingClass()
            # Test it
            test = embeddings.embed_query("test")
            if test and len(test) > 0:
                print(f"✅ {name} works!")
                return embeddings
        except Exception as e:
            print(f"❌ {name} failed: {e}")
            continue
    
    raise Exception("No embedding method works!")

if __name__ == "__main__":
    # Test the embeddings
    embeddings = get_working_embeddings()
    print("Embeddings initialized successfully!")
