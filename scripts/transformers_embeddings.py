"""
Alternative embeddings using transformers library directly
No dependency on sentence-transformers
"""

from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from typing import List
from langchain.embeddings.base import Embeddings
from pathlib import Path

class TransformersEmbeddings(Embeddings):
    """Embeddings using transformers library directly - no sentence-transformers needed"""
    
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        cache_dir = Path("D:/genAI/LangChain-RAG/models")
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=str(cache_dir)
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            cache_dir=str(cache_dir)
        )
        self.model.eval()
        
    def _mean_pooling(self, model_output, attention_mask):
        """Mean pooling - Take attention mask into account for correct averaging"""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def _encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings"""
        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=512
        )
        
        # Get embeddings
        with torch.no_grad():
            model_output = self.model(**encoded)
        
        # Mean pooling
        embeddings = self._mean_pooling(model_output, encoded['attention_mask'])
        
        # Normalize
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings.numpy()
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed list of documents"""
        if not texts:
            return []
        
        # Process in batches
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            embeddings = self._encode(batch)
            all_embeddings.extend(embeddings.tolist())
        
        return all_embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed single query"""
        embeddings = self._encode([text])
        return embeddings[0].tolist()

def test_transformers_embeddings():
    """Test the transformers embeddings"""
    print("Testing Transformers Embeddings...")
    
    try:
        embeddings = TransformersEmbeddings()
        
        # Test
        docs = ["Hello world", "How are you?"]
        doc_embs = embeddings.embed_documents(docs)
        print(f"✅ Embedded {len(docs)} documents")
        print(f"   Dimension: {len(doc_embs[0])}")
        
        query_emb = embeddings.embed_query("Hello")
        print(f"✅ Embedded query")
        print(f"   Dimension: {len(query_emb)}")
        
        print("\n✨ Transformers embeddings working!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_transformers_embeddings()
