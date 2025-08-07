"""
Standalone embeddings that work without langchain-huggingface
"""

from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
from pathlib import Path
import numpy as np
from typing import List

class StandaloneSentenceTransformerEmbeddings(Embeddings):
    """Embeddings using sentence-transformers directly, no huggingface_hub issues"""
    
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        # Set cache directory
        self.cache_dir = Path("D:/genAI/LangChain-RAG/models")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        try:
            self.model = SentenceTransformer(
                model_name,
                cache_folder=str(self.cache_dir),
                device='cpu'
            )
            print(f"✅ Loaded model: {model_name}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        if not texts:
            return []
        
        # Encode in batches for efficiency
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=32
        )
        
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        embedding = self.model.encode(
            [text],
            convert_to_numpy=True,
            show_progress_bar=False
        )[0]
        
        return embedding.tolist()

def test_embeddings():
    """Test the standalone embeddings"""
    print("=" * 50)
    print("Testing Standalone Embeddings")
    print("=" * 50)
    
    try:
        # Create embeddings
        embeddings = StandaloneSentenceTransformerEmbeddings()
        
        # Test document embedding
        docs = ["Hello world", "How are you?", "LangChain RAG"]
        doc_embeddings = embeddings.embed_documents(docs)
        print(f"✅ Embedded {len(docs)} documents")
        print(f"   Embedding dimension: {len(doc_embeddings[0])}")
        
        # Test query embedding
        query = "Hello"
        query_embedding = embeddings.embed_query(query)
        print(f"✅ Embedded query")
        print(f"   Query embedding dimension: {len(query_embedding)}")
        
        # Test similarity
        from numpy import dot
        from numpy.linalg import norm
        
        # Calculate cosine similarity
        a = np.array(doc_embeddings[0])  # "Hello world"
        b = np.array(query_embedding)    # "Hello"
        cos_sim = dot(a, b)/(norm(a)*norm(b))
        print(f"✅ Similarity test")
        print(f"   Cosine similarity ('Hello world' vs 'Hello'): {cos_sim:.3f}")
        
        print("\n✨ Embeddings working perfectly!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_embeddings()
    if not success:
        print("\nPlease install: pip install sentence-transformers")
