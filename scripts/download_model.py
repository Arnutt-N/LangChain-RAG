"""
Script to pre-download embedding model for offline use
This will download the model once and cache it locally
"""

import os
from sentence_transformers import SentenceTransformer
import sys

print("🔄 Downloading MiniLM-L6-v2 model...")
print("This will take a few minutes on first run, but will be cached for future use.")
print("-" * 50)

# Set cache directory
cache_dir = "D:\\genAI\\LangChain-RAG\\models"
os.makedirs(cache_dir, exist_ok=True)

try:
    # Download and cache the model
    model = SentenceTransformer(
        'sentence-transformers/all-MiniLM-L6-v2',
        cache_folder=cache_dir
    )
    
    print("✅ Model downloaded successfully!")
    print(f"📁 Cached at: {cache_dir}")
    
    # Test the model
    test_embedding = model.encode("Test sentence")
    print(f"✅ Model test successful! Embedding dimension: {len(test_embedding)}")
    
except Exception as e:
    print(f"❌ Error downloading model: {e}")
    sys.exit(1)

print("\n✨ You can now run the Streamlit app without downloading delays!")
