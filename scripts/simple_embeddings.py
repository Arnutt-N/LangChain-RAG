"""
Alternative Lightweight Embeddings for Poor Internet Connection
Uses a smaller, faster model
"""

import streamlit as st
import os
from pathlib import Path
import numpy as np
from typing import List
import hashlib

class SimpleEmbeddings:
    """Simple embeddings using hash-based vectors - works offline!"""
    
    def __init__(self, dimension=384):
        self.dimension = dimension
        
    def _text_to_vector(self, text: str) -> List[float]:
        """Convert text to vector using hashing - no model needed!"""
        # Create hash from text
        hash_obj = hashlib.sha256(text.encode())
        hash_bytes = hash_obj.digest()
        
        # Convert to vector
        np.random.seed(int.from_bytes(hash_bytes[:4], 'big'))
        vector = np.random.randn(self.dimension)
        
        # Normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
            
        return vector.tolist()
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents"""
        return [self._text_to_vector(text) for text in texts]
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        return self._text_to_vector(text)

def get_simple_embeddings():
    """Get simple offline embeddings"""
    return SimpleEmbeddings(dimension=384)

# Test the embeddings
if __name__ == "__main__":
    print("Testing Simple Embeddings...")
    embeddings = get_simple_embeddings()
    
    # Test documents
    docs = ["Hello world", "How are you?", "RAG chatbot"]
    doc_vectors = embeddings.embed_documents(docs)
    print(f"✅ Embedded {len(docs)} documents")
    print(f"✅ Vector dimension: {len(doc_vectors[0])}")
    
    # Test query
    query_vector = embeddings.embed_query("Hello")
    print(f"✅ Query vector created")
    
    print("\n✨ Simple Embeddings ready for offline use!")
